"""Sobol-sensitivity dimensionality reduction (generic port of foamBO
``_maybe_reduce_dimensions``).

After the GP is fit, fix low-influence parameters at a constant to shrink the search space.
The latch lives in a caller-owned ``state`` dict so it survives the detached ask/run/tell
boundary (foamBO used an in-memory ``nonlocal``). Must run AFTER ``get_next_trials`` — Sobol
needs a fitted model.
"""

from __future__ import annotations

import logging

log = logging.getLogger("axbo_extensions.dimreduction")


def select_params_to_fix(
    importance: dict[str, float], min_importance: float, max_fix_fraction: float
) -> list[str]:
    """Low-influence params to fix: importance < threshold, ascending, capped so >=1 stays
    active. Pure (no model) — the testable core of the reduction decision."""
    names = list(importance)
    cap = min(int(len(names) * max_fix_fraction), len(names) - 1)
    ordered = sorted(names, key=lambda n: importance[n])
    return [n for n in ordered if importance[n] < min_importance][:cap]


def _cv_fit_ok(gs) -> bool:
    """False if model fit is too poor to trust Sobol (mean CV error > 50% of error range)."""
    try:
        import numpy as np
        from ax.adapter.cross_validation import cross_validate

        errs = []
        for r in cross_validate(adapter=gs.adapter):
            for m, obs in r.observed.data.means_dict.items():
                errs.append(abs(obs - r.predicted.means_dict.get(m, obs)))
        if errs:
            rng = max(errs)
            if rng > 0 and float(np.mean(errs)) / rng > 0.5:
                return False
    except Exception:  # noqa: BLE001 - CV unavailable -> proceed with Sobol
        pass
    return True


def _sobol_importance(gs, exp) -> dict[str, float]:
    """Total-order Sobol indices of the fitted GP mean, per range parameter (Ax-private)."""
    import torch
    from ax.utils.sensitivity.sobol_measures import SobolSensitivityGPMean

    model = gs.adapter.generator.surrogate.model
    names = [n for n, p in exp.search_space.parameters.items() if hasattr(p, "lower")]
    bounds = torch.zeros(2, len(names), dtype=torch.float64)
    bounds[1] = 1.0  # Ax models operate in normalized [0,1] space
    total = SobolSensitivityGPMean(
        model=model, bounds=bounds, num_mc_samples=1000
    ).total_order_indices()
    return {n: float(total[i].abs()) for i, n in enumerate(names)}


def maybe_reduce_dimensions(client, cfg, state: dict) -> None:
    """Fix low-influence parameters once enough trials are in. Mutates the experiment search
    space in place (persists when the Client is saved); idempotent via ``state['done']``.

    cfg: a DimensionalityReductionConfig-like; state: ``{"done": bool, "attempts": int}``
    (the caller persists it). Wrapped in a 3-strike try/except over the Ax-private paths (§6).
    """
    if state.get("done") or not getattr(cfg, "enabled", False):
        return
    from ax.core.base_trial import TrialStatus

    exp = client._experiment
    n_done = sum(1 for t in exp.trials.values() if t.status == TrialStatus.COMPLETED)
    if n_done < cfg.after_trials:
        return
    gs = client._generation_strategy
    if gs is None or getattr(gs, "adapter", None) is None:
        return

    try:
        if not _cv_fit_ok(gs):
            log.info("dim-reduction deferred: model fit too poor")
            return
        importance = _sobol_importance(gs, exp)
    except Exception as e:  # noqa: BLE001
        state["attempts"] = state.get("attempts", 0) + 1
        if state["attempts"] >= 3:
            state["done"] = True
            log.warning("dim-reduction disabled after 3 failed attempts: %s", e)
        return

    to_fix = select_params_to_fix(importance, cfg.min_importance, cfg.max_fix_fraction)
    if not to_fix:
        state["done"] = True
        return

    best: dict = {}
    if cfg.fix_at == "best":
        try:
            best = client.get_best_parameterization(use_model_predictions=False)[0]
        except Exception:  # noqa: BLE001 - fall back to center
            best = {}

    from ax.core.parameter import FixedParameter

    for n in to_fix:
        p = exp.search_space.parameters[n]
        if cfg.fix_at == "best" and best.get(n) is not None:
            val = best[n]
        elif hasattr(p, "lower"):
            val = (p.lower + p.upper) / 2
        elif hasattr(p, "values"):
            val = p.values[0]
        else:
            continue
        exp.search_space.update_parameter(
            FixedParameter(name=n, parameter_type=p.parameter_type, value=val)
        )
        log.info("dim-reduction: fixed %s=%s (importance=%.4f)", n, val, importance[n])
    state["done"] = True
