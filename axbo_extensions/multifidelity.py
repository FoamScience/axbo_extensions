"""Multi-fidelity GLUE for Ax + BoTorch (ported from foamBO ``robustness.py`` /
``optimize.py``).

- ``MultiFidHVKGAcquisition``: Ax ``Acquisition`` subclass that strips ``X_pending``
  (qMultiFidHVKG's input constructor rejects it).
- ``update_cost_state``: recompute ``cost_intercept`` / ``fidelity_weights`` for the
  ``AffineFidelityCostModel`` qMultiFidHVKG builds, from the observed is_cost metric. The
  ±0.5 positivity-floor + normalization math is load-bearing — copied verbatim.
- ``apply_fidelity_parameters``: stamp ``_is_fidelity`` / ``_target_value`` onto Ax
  search-space parameters after ``configure_experiment``.
- ``augment_specs_for_multifidelity``: inject ``SingleTaskMultiFidelityGP`` surrogate
  + qMultiFidHVKG acqf + ``MultiFidHVKGAcquisition`` into a Client's generation-strategy nodes.

Pinned to ax-platform 1.3.x / botorch 0.18.x.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

from ax.generators.torch.botorch_modular.acquisition import Acquisition

log = logging.getLogger("axbo_extensions.multifidelity")


class MultiFidHVKGAcquisition(Acquisition):
    """Acquisition subclass that strips kwargs incompatible with qMultiFidHVKG.

    ``construct_inputs_qMultiFidHVKG`` doesn't accept ``X_pending`` (unlike most other
    BoTorch input constructors). This subclass clears it before the parent passes
    acqf_options to the input constructor.
    """

    def _construct_botorch_acquisition(self, botorch_acqf_class, botorch_acqf_options, model):
        _saved = self.X_pending
        self.X_pending = None
        try:
            return super()._construct_botorch_acquisition(
                botorch_acqf_class=botorch_acqf_class,
                botorch_acqf_options=botorch_acqf_options,
                model=model,
            )
        finally:
            self.X_pending = _saved


def _fidelity_feature_index(client, fid_param_name: str) -> int:
    """Feature index of the fidelity parameter in the search space."""
    param_names = list(client._experiment.search_space.parameters.keys())
    return param_names.index(fid_param_name)


def update_cost_state(client, multifid_cost: dict, log=None) -> None:
    """Recompute cost-model params from observed is_cost metric data.

    Reads observed values of the cost metric, computes per-fidelity means, then derives
    ``cost_intercept`` and ``fidelity_weights`` for the ``AffineFidelityCostModel`` that
    qMultiFidHVKG's input constructor builds internally. Writes directly into the acqf_opts
    dict (``multifid_cost['acqf_opts_ref']``) so the next acquisition construction picks up
    updated values.

    AffineFidelityCostModel: cost(x) = cost_intercept + Σ(w_i × s_i). With two fidelity
    levels (s=0 cheap, s=1 expensive): cost_intercept = mean_cost_at_s0,
    fidelity_weight = mean_cost_at_s1 - mean_cost_at_s0.
    """
    from ax.core.base_trial import TrialStatus as _TS

    cost_metric_name = multifid_cost["metric"]
    fidelity_param_name = multifid_cost["fidelity_param"]
    # Collect observed costs per fidelity level (keyed by raw fidelity value)
    sums: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    df = client._experiment.lookup_data().df
    for trial in client._experiment.trials.values():
        if trial.status not in (_TS.COMPLETED, _TS.EARLY_STOPPED):
            continue
        fid_val = trial.arm.parameters.get(fidelity_param_name)
        if fid_val is None:
            continue
        sub = df[(df.trial_index == trial.index) & (df.metric_name == cost_metric_name)]
        if sub.empty:
            continue
        sums[fid_val] += float(sub["mean"].iloc[-1])
        counts[fid_val] += 1
    per_f = {fv: sums[fv] / counts[fv] for fv in sums if counts[fv] > 0}
    if not per_f or per_f == multifid_cost["state"]["per_fidelity"]:
        return  # no change
    multifid_cost["state"]["per_fidelity"] = per_f

    # Build numeric mapping for AffineFidelityCostModel.
    # For numeric fidelity: use values directly.
    # For categorical (str) fidelity: map to ordinal indices sorted by cost.
    raw_keys = sorted(per_f.keys(), key=lambda k: per_f[k])
    is_numeric = all(isinstance(k, (int, float)) for k in raw_keys)
    if is_numeric:
        fid_vals = sorted(float(k) for k in raw_keys)
        costs = [per_f[k] for k in sorted(per_f.keys())]
    else:
        # Map categorical levels to 0..N-1 ordered by ascending cost
        fid_vals = list(range(len(raw_keys)))
        costs = [per_f[k] for k in raw_keys]

    cost_intercept = min(costs)  # cheapest fidelity floor
    # Weights: slope per fidelity unit. For discrete {0,1}: w = cost_high - cost_low.
    # For continuous: linear fit.
    fid_range = fid_vals[-1] - fid_vals[0] if len(fid_vals) > 1 else 0
    if fid_range > 0:
        fid_weight = (max(costs) - min(costs)) / fid_range
    else:
        fid_weight = 1.0
    # Positivity floor: BoTorch's InverseCostWeightedUtility requires cost(X) > 0 over
    # the entire acqf domain. Ax extends INT-fidelity model-space bounds by ±0.5, so
    # AffineFidelityCostModel sees fid ∈ [-0.5, 1.5]; cost(X) = intercept + weight × X
    # must stay > 0 across that range → intercept ≥ 0.5 × weight + ε. Normalize to keep
    # costs O(1) (qMultiFidHVKG tolerates large absolute values poorly).
    max_cost = max(costs) if max(costs) > 0 else 1.0
    cost_intercept_norm = cost_intercept / max_cost
    fid_weight_norm = fid_weight / max_cost
    cost_intercept_safe = max(cost_intercept_norm, 0.5 * fid_weight_norm + 1e-3)
    opts = multifid_cost["acqf_opts_ref"]
    opts["cost_intercept"] = max(cost_intercept_safe, 1e-6)
    if log:
        log.debug(
            "MultiFid cost normalized: raw intercept=%.3f weight=%.3f max_cost=%.3f → "
            "intercept=%.3f weight=%.3f",
            cost_intercept,
            fid_weight,
            max_cost,
            cost_intercept_safe,
            fid_weight_norm,
        )
    fid_feature_idx = _fidelity_feature_index(client, multifid_cost["fidelity_param"])
    opts["fidelity_weights"] = {fid_feature_idx: fid_weight_norm}
    if log:
        log.info(
            "Multi-fidelity cost updated: intercept=%.2f, weight=%.2f "
            "(from %d fidelity levels: %s)",
            cost_intercept,
            fid_weight,
            len(per_f),
            list(per_f.keys()),
        )


def apply_fidelity_parameters(client, fidelity_params: dict, log=None) -> None:
    """Stamp ``_is_fidelity`` / ``_target_value`` onto Ax search-space parameters.

    ``fidelity_params`` maps parameter name -> fidelity_target. Call after
    ``configure_experiment`` (Ax's parameter configs don't carry these attrs).
    """
    from ax.core.parameter import FixedParameter

    for pname, target_val in fidelity_params.items():
        p = client._experiment.search_space.parameters.get(pname)
        if p is None or isinstance(p, FixedParameter):
            continue
        p._is_fidelity = True
        p._target_value = target_val
        if log:
            log.info("Fidelity parameter: %s (fidelity_target=%s)", pname, target_val)


def augment_specs_for_multifidelity(
    client, cost_metric_name: Optional[str] = None, log=None
) -> Optional[dict]:
    """Inject MultiFid surrogate + qMultiFidHVKG acqf + ``MultiFidHVKGAcquisition`` into the Client's
    generation-strategy nodes.

    Requires fidelity parameters already stamped (``apply_fidelity_parameters``) and a
    generation strategy already set. Returns the ``multifid_cost`` state dict (to feed
    ``update_cost_state`` each batch) when ``cost_metric_name`` is given, else ``None``.
    """
    from ax.adapter.registry import Generators
    from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec
    from ax.generators.torch.botorch_modular.utils import ModelConfig
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qMultiFidelityHypervolumeKnowledgeGradient,
    )
    from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

    fidelity_params = [
        p
        for p in client._experiment.search_space.parameters.values()
        if getattr(p, "is_fidelity", False)
    ]
    if not fidelity_params:
        return None

    gs = client._generation_strategy
    multifid_cost: Optional[dict] = None
    for node in gs._nodes:
        for spec in node.generator_specs:
            if spec.generator_enum != Generators.BOTORCH_MODULAR:
                continue
            gk = spec.generator_kwargs
            if "botorch_acqf_class" not in gk:
                gk["botorch_acqf_class"] = qMultiFidelityHypervolumeKnowledgeGradient
                if log:
                    log.info("Multi-fidelity: auto-selected qMultiFidHVKG")
            # qMultiFidHVKG's fantasy-based acqf optimizer diverges on raw multi-scale bounds
            # (out-of-bounds candidates / hangs). MBM deliberately DROPS UnitX to keep discrete
            # params discrete, leaving the acqf in raw parameter space — fine for qLogNEHVI/MARS,
            # fatal for KG. Force UnitX-inclusive transforms so the acqf optimizes in [0,1].
            # Idempotent; range-int params get continuous-relaxed + rounded (acceptable for KG).
            from ax.adapter.registry import Cont_X_trans, Y_trans

            gk["transforms"] = Cont_X_trans + Y_trans
            # Inject surrogate only if not already set (preserves an explicit one and is
            # idempotent on a reloaded client whose strategy already carries it).
            if gk.get("surrogate_spec") is None:
                # allow_batched_models=False forces a ModelListGP of per-output STMFGPs
                # (qMultiFidHVKG requirement — load-bearing).
                gk["surrogate_spec"] = SurrogateSpec(
                    model_configs=[ModelConfig(botorch_model_class=SingleTaskMultiFidelityGP)],
                    allow_batched_models=False,
                )
                gk["acquisition_class"] = MultiFidHVKGAcquisition
                if log:
                    log.info(
                        "Multi-fidelity: SingleTaskMultiFidelityGP surrogate (fidelity param: %s)",
                        fidelity_params[0].name,
                    )
            # Build cost state regardless of fresh/reloaded surrogate so the batch loop's
            # update_cost_state has a live acqf_opts_ref after a JSON reload.
            if cost_metric_name is not None:
                multifid_cost = {
                    "state": {"per_fidelity": {}},
                    "metric": cost_metric_name,
                    "fidelity_param": fidelity_params[0].name,
                    "acqf_opts_ref": gk.setdefault("botorch_acqf_options", {}),
                }
                multifid_cost["acqf_opts_ref"].setdefault("cost_intercept", 1.0)
    return multifid_cost


# ---------------------------------------------------------------------------
# Ax JSON serialization registry — run at import so saved MultiFid experiments serialize.
# MultiFidHVKGAcquisition is an Ax Acquisition wrapper; qMultiFidHVKG/MOMF are BoTorch
# AcquisitionFunctions not in Ax's default registry. Without this, save_to_json_file
# raises "Class qMultiFidelityHypervolumeKnowledgeGradient not in registry".
# ---------------------------------------------------------------------------
try:
    from ax.storage.botorch_modular_registry import (
        CLASS_TO_REGISTRY,
        CLASS_TO_REVERSE_REGISTRY,
        register_acquisition,
    )
    from botorch.acquisition import AcquisitionFunction as _AcqFn

    register_acquisition(MultiFidHVKGAcquisition)
    for _mod, _name in [
        (
            "botorch.acquisition.multi_objective.hypervolume_knowledge_gradient",
            "qMultiFidelityHypervolumeKnowledgeGradient",
        ),
        ("botorch.acquisition.multi_objective.multi_fidelity", "MOMF"),
    ]:
        try:
            _cls = getattr(__import__(_mod, fromlist=[_name]), _name)
            CLASS_TO_REGISTRY[_AcqFn][_cls] = _name
            CLASS_TO_REVERSE_REGISTRY[_AcqFn][_name] = _cls
        except (ImportError, AttributeError):  # pragma: no cover
            pass
except ImportError:  # pragma: no cover - Ax storage registry path drift
    log.warning("Ax storage registry unavailable; MultiFid JSON (de)serialization skipped")
