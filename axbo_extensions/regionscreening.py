"""Region screening — inactivity decision core (port of foamBO _maybe_screen_regions).

Only the **pure decision** is ported: given a region's per-trial scalars, is its spread
below the inactivity threshold, and how long has the streak run. That math is durable and
reusable.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

log = logging.getLogger("axbo_extensions.regionscreening")


def region_inactive(
    scalars: Iterable[float],
    min_delta: Optional[float] = None,
    min_delta_frac: Optional[float] = None,
) -> dict:
    """Is the region's objective-scalar spread below its inactivity threshold?

    ``min_delta`` = absolute threshold; else ``min_delta_frac`` is relative to the largest
    magnitude observed. Returns ``{samples, spread, threshold, inactive}``.
    """
    vals = list(scalars)
    spread = (max(vals) - min(vals)) if vals else 0.0
    ref = max((abs(v) for v in vals), default=0.0)
    threshold = min_delta if min_delta is not None else (min_delta_frac or 0.0) * ref
    return {
        "samples": len(vals),
        "spread": spread,
        "threshold": threshold,
        "inactive": bool(vals) and spread < threshold,
    }


def next_streak(streak: int, inactive: bool) -> int:
    """Consecutive-inactive streak: increment when inactive, reset to 0 otherwise."""
    return streak + 1 if inactive else 0


def shrunk_bounds(lo: float, hi: float, center: float, shrink_factor: float) -> tuple[float, float]:
    """New ``[lo, hi]`` after a shrink: a ``shrink_factor * range`` window around ``center``,
    clamped to the original bounds. Pure — the shrink geometry."""
    half = (shrink_factor * (hi - lo)) / 2.0
    return max(lo, center - half), min(hi, center + half)


def _group_params(parameter_groups: dict, group: str) -> list[str]:
    return [n for n, gs in (parameter_groups or {}).items() if group in (gs or [])]


def maybe_screen_regions(
    client, cfg, scalars_by_region: dict, parameter_groups: dict, state: dict
) -> None:
    """Advise/shrink parameter groups whose objective-scalar spread stays below threshold for
    ``confirm_passes`` consecutive passes (plan §2.A2). ``scalars_by_region``: ``{group:
    {trial_idx: scalar}}`` (caller reads these from the executor's results — the generic
    result-field provider). ``state`` persists streaks / shrunk groups / original bounds /
    advisories across the detached ask/run/tell boundary. Shrink mutates the search space
    in place (Ax-private update_parameter, gated)."""
    if not getattr(cfg, "enabled", False) or not cfg.regions:
        return
    streaks = state.setdefault("streaks", {})
    shrunk = state.setdefault("shrunk_groups", [])
    advisories = state.setdefault("advisories", [])

    ready = []
    for region in cfg.regions:
        vals = list(scalars_by_region.get(region.group, {}).values())
        if len(vals) < cfg.after_trials:
            continue
        inactive = region_inactive(vals, region.min_delta, region.min_delta_frac)["inactive"]
        streaks[region.group] = next_streak(streaks.get(region.group, 0), inactive)
        if streaks[region.group] >= cfg.confirm_passes and region.group not in shrunk:
            ready.append(region.group)
    if not ready:
        return

    if cfg.mode == "advise":
        for g in ready:
            advisories.append({"group": g, "params": _group_params(parameter_groups, g)})
            shrunk.append(g)  # recorded so we advise once
            log.info("region screening (advise): group %s inactive", g)
        return

    # shrink mode — narrow each group param's range around the best point, once per group.
    try:
        from ax.core.parameter import RangeParameter

        exp = client._experiment
        orig_bounds = state.setdefault("original_bounds", {})
        try:
            best = client.get_best_parameterization(use_model_predictions=False)[0]
        except Exception:  # noqa: BLE001
            best = {}
        n_total = len(exp.search_space.parameters)
        budget = max(min(cfg.max_shrinkable or n_total, n_total - 1) - len(orig_bounds), 0)
        for g in ready:
            for pname in _group_params(parameter_groups, g):
                if budget <= 0:
                    break
                p = exp.search_space.parameters.get(pname)
                if not isinstance(p, RangeParameter):
                    continue
                lo, hi = float(p.lower), float(p.upper)
                orig_bounds.setdefault(pname, (lo, hi))
                center = best.get(pname, (lo + hi) / 2.0)
                new_lo, new_hi = shrunk_bounds(lo, hi, center, cfg.shrink_factor)
                p.update_range(lower=new_lo, upper=new_hi)
                budget -= 1
                log.info("region screening: shrank %s -> [%.4g, %.4g]", pname, new_lo, new_hi)
            shrunk.append(g)
    except Exception as e:  # noqa: BLE001 - Ax-private mutation path, stay non-fatal
        log.warning("region screening shrink failed: %s", e)
