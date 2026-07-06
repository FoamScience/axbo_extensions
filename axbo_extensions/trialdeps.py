"""Trial registry + dependency resolution (generic port of foamBO metrics.py).

The registry is pure derived state — rebuildable from the saved Client — so it works in
detached ask/run/tell mode with no live job objects (plan §3). Selectors land here in
Feature B (#16); this module starts with the shared primitive both B and region screening
need: rebuilding the registry from the loaded Client.
"""

from __future__ import annotations

import logging
import math
import shlex
import subprocess

log = logging.getLogger("axbo_extensions.trialdeps")

# The executor stamps each trial's output location here in run_metadata (plan §1.2).
# The brain reads it (never copies artifacts); deps/region scalars resolve from it.
ARTIFACT_REF_KEY = "artifact_ref"

_CUSTOM_CMD_TIMEOUT = 30


def rebuild_trial_registry(client) -> dict[int, dict]:
    """Derive ``{trial_index: {status, parameters, artifact_ref}}`` from the Client."""
    registry: dict[int, dict] = {}
    for idx, trial in client._experiment.trials.items():
        arm = getattr(trial, "arm", None)
        registry[idx] = {
            "status": trial.status.name,
            "parameters": dict(arm.parameters) if arm is not None else {},
            "artifact_ref": (trial.run_metadata or {}).get(ARTIFACT_REF_KEY),
        }
    return registry


def trial_objectives(client, objective_names: list[str]) -> dict[int, float]:
    """Per-trial scalar objective (sum of objective-metric means; lower=better) from the
    store — what the ``best`` selector needs (foamBO never populated this → its bug, §3)."""
    df = client._experiment.lookup_data().df
    df = df[df["metric_name"].isin(objective_names)]
    return {int(idx): float(g["mean"].sum()) for idx, g in df.groupby("trial_index")}


def _fidelity_filtered(completed: dict, selector, target_params: dict, fidelity_param):
    """Keep only candidates whose fidelity matches the selector's fidelity_filter."""
    if not fidelity_param:
        return completed
    fid_filter = getattr(selector, "fidelity_filter", None) or "same"
    if fid_filter == "any":
        return completed
    if fid_filter == "same":
        match_val = target_params.get(fidelity_param)
    else:
        try:
            match_val = float(fid_filter)
        except (ValueError, TypeError):
            match_val = fid_filter

    def _match(params):
        v = params.get(fidelity_param)
        if isinstance(v, float) and isinstance(match_val, (int, float)):
            return abs(v - match_val) < 1e-6
        return v == match_val

    return {k: v for k, v in completed.items() if _match(v.get("parameters", {}))}


def resolve_source_trial(
    selector,
    target_params: dict,
    registry: dict[int, dict],
    *,
    fidelity_param: str | None = None,
    parameter_groups: dict[str, list[str]] | None = None,
    objectives: dict[int, float] | None = None,
) -> tuple[int, object] | None:
    """Resolve ``(source_trial_index, artifact_ref)`` for a dependency selector, or None.

    Generic port of foamBO ``_resolve_source_trial`` (metrics.py): registry + args instead
    of ``self``; ``artifact_ref`` instead of ``case_path``. The ``best`` strategy needs the
    ``objectives`` map — foamBO read a never-written ``objective_value`` and silently
    degraded to the lowest index (§3 bug); here it skips loudly instead.
    """
    completed = {
        k: v for k, v in registry.items()
        if v.get("status") == "COMPLETED" and v.get("artifact_ref") is not None
    }
    completed = _fidelity_filtered(completed, selector, target_params, fidelity_param)
    if not completed:
        return None

    strategy = selector.strategy
    if strategy == "baseline":
        e = completed.get(0)
        return (0, e["artifact_ref"]) if e else None
    if strategy == "by_index":
        e = completed.get(selector.index)
        return (selector.index, e["artifact_ref"]) if e else None
    if strategy == "latest":
        idx = max(completed)
        return (idx, completed[idx]["artifact_ref"])
    if strategy == "best":
        if objectives is None:
            log.warning("'best' selector needs an objectives map; skipping")
            return None
        idx = min(completed, key=lambda k: objectives.get(k, float("inf")))
        return (idx, completed[idx]["artifact_ref"])
    if strategy == "nearest":
        ranges: dict[str, list[float]] = {}
        for e in completed.values():
            for k, v in e.get("parameters", {}).items():
                if isinstance(v, (int, float)):
                    r = ranges.setdefault(k, [v, v])
                    r[0], r[1] = min(r[0], v), max(r[1], v)
        span = {k: (hi - lo) or 1.0 for k, (lo, hi) in ranges.items()}

        def _dist(a, b):
            total = 0.0
            for key, va in a.items():
                vb = b.get(key)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    total += ((va - vb) / span.get(key, 1.0)) ** 2
            return math.sqrt(total)

        idx = min(completed, key=lambda k: _dist(target_params, completed[k].get("parameters", {})))
        thr = getattr(selector, "similarity_threshold", None)
        if thr is not None and _dist(target_params, completed[idx].get("parameters", {})) > thr:
            return None
        return (idx, completed[idx]["artifact_ref"])
    if strategy == "matching_group":
        group = getattr(selector, "group", None)
        groups = parameter_groups or {}
        group_params = [n for n, gs in groups.items() if group in (gs or [])]
        if not group or not group_params:
            return None
        for idx in sorted(completed, reverse=True):
            ep = completed[idx].get("parameters", {})
            if all(target_params.get(p) == ep.get(p) for p in group_params):
                return (idx, completed[idx]["artifact_ref"])
        return None
    if strategy == "custom":
        cmd = selector.command
        try:
            out = subprocess.check_output(
                shlex.split(cmd) if isinstance(cmd, str) else cmd, timeout=_CUSTOM_CMD_TIMEOUT
            )
            idx = int(out.decode().strip())
            e = completed.get(idx)
            return (idx, e["artifact_ref"]) if e else None
        except Exception as exc:  # noqa: BLE001
            log.warning("custom trial selector failed: %s", exc)
            return None
    return None


def build_depends_on(
    dependencies, target_params: dict, registry: dict[int, dict], **kwargs
) -> list[dict]:
    """Resolve enabled trial dependencies into the generic ``depends_on`` sidecar (§1.1).

    ``kwargs`` are forwarded to :func:`resolve_source_trial` (fidelity_param, parameter_groups,
    objectives). ``fallback='error'`` on an unresolved dependency raises; ``'skip'`` records
    ``{"resolved": false}``.
    """
    out: list[dict] = []
    for dep in dependencies:
        if not getattr(dep, "enabled", True):
            continue
        res = resolve_source_trial(dep.source, target_params, registry, **kwargs)
        if res is None:
            if getattr(dep.source, "fallback", "skip") == "error":
                raise RuntimeError(f"trial dependency {dep.name!r} could not be resolved")
            out.append({"name": dep.name, "resolved": False})
            continue
        idx, ref = res
        out.append({
            "name": dep.name, "resolved": True,
            "source_trial_index": idx, "source_artifact_ref": ref,
        })
    return out
