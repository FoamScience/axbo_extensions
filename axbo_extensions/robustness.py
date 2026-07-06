"""Robust-optimization GLUE for Ax + BoTorch (ported from foamBO ``robustness.py``).

Self-contained, decoupled from foamBO's config objects: the context helpers take a
``params`` sequence whose items expose ``.name`` / ``.bounds`` / ``.parameter_type`` /
``.groups``.

Pieces:
- ``RobustOptimizationConfig``: pydantic config (vestigial ``risk_measure`` dropped — §0;
  CVaR vs MARS is chosen by objective count, not a config field).
- ``SubstituteContextFeatures``: ModelListGP-safe ``InputTransform`` (the irreplaceable
  primitive — replaces context columns in place, keeping dim ``d`` constant).
- ``resolve_context_points`` / ``build_context_tensor``: Sobol scenario generation.
- ``RobustAcquisition``: Ax ``Acquisition`` subclass injecting CVaR (SOO) / MARS (MOO).

The dormant MARS-in-KG subsystem (RobustContextValueFunction / RobustMCObjective) is NOT
ported — foamBO never shipped it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor

from botorch.acquisition.risk_measures import CVaR
from botorch.models.transforms.input import InputTransform

from ax.generators.torch.botorch_modular.acquisition import Acquisition

log = logging.getLogger("axbo_extensions.robustness")


class RobustOptimizationConfig(BaseModel):
    """Robust optimization across environmental/context variables."""

    context_groups: list[str] = Field(
        description="Parameter group names treated as context (environmental) variables"
    )
    robustness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "0=risk-neutral, 1=most conservative. Maps to CVaR/MARS alpha "
            "(clamped >= 0.05); higher alpha optimises the worst (1-alpha) fraction."
        ),
    )
    context_points: Optional[list[dict[str, float]]] = Field(
        default=None, description="Explicit context scenarios; if None, auto-generated via Sobol"
    )
    context_samples: int = Field(
        default=10, ge=2, description="Number of Sobol samples when auto-generating context points"
    )
    context_constraints: list[str] = Field(
        default=[], description="Inequality filters on context points (e.g. 'flowRate >= 0.01')"
    )


def alpha_from_robustness(robustness: float) -> float:
    """CVaR/MARS alpha: robustness clamped to >= 0.05 (§5.1)."""
    return max(robustness, 0.05)


# ---------------------------------------------------------------------------
# SubstituteContextFeatures — ModelListGP-compatible input transform
# ---------------------------------------------------------------------------


class SubstituteContextFeatures(InputTransform, torch.nn.Module):
    """Replace context dims with n_w scenarios. Keeps tensor dim ``d`` unchanged.

    Unlike ``AppendFeatures`` (which changes ``d`` and breaks ``ModelListGP``), this
    replaces the context columns in-place, replicating each input ``q`` times ``n_w`` to
    produce ``(q * n_w, d)`` outputs.
    """

    is_one_to_many: bool = True

    def __init__(self, context_indices: list[int], feature_set: Tensor, **kwargs: Any) -> None:
        super().__init__()
        self.transform_on_train = False
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        self.register_buffer("ctx_indices", torch.tensor(context_indices, dtype=torch.long))
        self.register_buffer("feature_set", feature_set)  # (n_w, d_ctx)
        self._n_w = feature_set.shape[0]

    @property
    def n_w(self) -> int:
        return self._n_w

    def transform(self, X: Tensor) -> Tensor:
        # X: (..., q, d) → (..., q * n_w, d)
        shape = X.shape
        q, d = shape[-2], shape[-1]
        batch = shape[:-2]
        X_exp = X.unsqueeze(-2).expand(*batch, q, self._n_w, d).clone()
        for i, idx in enumerate(self.ctx_indices):
            X_exp[..., idx] = self.feature_set[:, i]
        return X_exp.reshape(*batch, q * self._n_w, d)

    def untransform(self, X: Tensor) -> Tensor:
        raise NotImplementedError("SubstituteContextFeatures is not reversible")


# Register with Ax's input_transform_argparse so ModelConfig can construct this transform
# via input_transform_classes / input_transform_options (Ax-version-sensitive — §8).
try:
    from ax.generators.torch.botorch_modular.input_constructors.input_transforms import (
        input_transform_argparse,
    )

    @input_transform_argparse.register(SubstituteContextFeatures)
    def _argparse_substitute_context(
        input_transform_class,
        dataset=None,
        search_space_digest=None,
        input_transform_options=None,
        **kwargs,
    ):
        return input_transform_options or {}

except ImportError:  # pragma: no cover - Ax registry path drift
    log.warning("input_transform_argparse not found; SubstituteContextFeatures argparse skipped")


# When the surrogate carries a one-to-many context transform (SubstituteContextFeatures),
# Ax's in-sample best_point is ill-defined: model.predict expands X_obs to (n_obs * n_w),
# so best_in_sample_point's argmax indexes past X_obs (IndexError). For MOO Ax already
# raises NotImplementedError here (best point undefined) and the adapter swallows it — do
# the same for robust SOO. Robustness is realised by the CVaR acqf, not an in-sample point.
try:
    from ax.generators.torch.botorch_modular.surrogate import Surrogate as _Surrogate

    _orig_best_in_sample = _Surrogate.best_in_sample_point

    def _best_in_sample_point(self, *args, **kwargs):
        model = getattr(self, "model", None)
        if getattr(getattr(model, "input_transform", None), "is_one_to_many", False):
            raise NotImplementedError(
                "best in-sample point undefined under a one-to-many context transform"
            )
        return _orig_best_in_sample(self, *args, **kwargs)

    _Surrogate.best_in_sample_point = _best_in_sample_point
except ImportError:  # pragma: no cover - Ax surrogate path drift
    log.warning("ax Surrogate unavailable; robust SOO best_point shim skipped")


# ---------------------------------------------------------------------------
# Context scenario generation
# ---------------------------------------------------------------------------


def _param_bounds(p) -> tuple[float, float]:
    """Extract (lower, upper) from a ParameterConfig-like or Ax RangeParameter."""
    if hasattr(p, "bounds") and p.bounds is not None:
        return float(p.bounds[0]), float(p.bounds[1])
    return float(p.lower), float(p.upper)


def _param_groups(p) -> list[str]:
    return list(getattr(p, "groups", None) or [])


def context_param_names(robust_cfg: RobustOptimizationConfig, params: Sequence) -> list[str]:
    """Ordered names of parameters tagged with any of ``context_groups``."""
    return [p.name for p in params if any(g in robust_cfg.context_groups for g in _param_groups(p))]


def context_dim_indices(robust_cfg: RobustOptimizationConfig, params: Sequence) -> list[int]:
    """Integer indices of context parameters in the feature tensor."""
    all_names = [p.name for p in params]
    return [all_names.index(n) for n in context_param_names(robust_cfg, params)]


def resolve_context_points(robust_cfg: RobustOptimizationConfig, params: Sequence) -> None:
    """Auto-generate ``context_points`` via Sobol if not explicit (mutates in place)."""
    if robust_cfg.context_points is not None:
        return

    from torch.quasirandom import SobolEngine

    names = context_param_names(robust_cfg, params)
    d = len(names)
    if d == 0:
        raise ValueError(
            f"No parameters found in context_groups {robust_cfg.context_groups}. "
            "Check that parameter 'groups' tags match context_groups names."
        )

    param_map = {p.name: p for p in params}
    bounds_lo, bounds_hi, is_int = [], [], []
    for cn in names:
        p = param_map[cn]
        lo, hi = _param_bounds(p)
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        is_int.append(getattr(p, "parameter_type", "float") == "int")

    n_raw = max(robust_cfg.context_samples * 3, 20)  # oversample for filtering
    raw = SobolEngine(dimension=d, scramble=True).draw(n_raw).numpy()
    bounds_lo = np.array(bounds_lo)
    bounds_hi = np.array(bounds_hi)
    scaled = bounds_lo + raw * (bounds_hi - bounds_lo)
    for j, is_i in enumerate(is_int):
        if is_i:
            scaled[:, j] = np.round(scaled[:, j])

    points = [{cn: float(scaled[i, j]) for j, cn in enumerate(names)} for i in range(n_raw)]

    if robust_cfg.context_constraints:
        import sympy

        symbols = [sympy.Symbol(n) for n in names]
        compiled = []
        for expr_str in robust_cfg.context_constraints:
            normalized = _parse_inequality(expr_str)
            compiled.append(sympy.lambdify(symbols, normalized, modules=["numpy"]))
        points = [pt for pt in points if all(fn(*[pt[cn] for cn in names]) >= 0 for fn in compiled)]

    if len(points) < 2:
        raise ValueError(
            f"Only {len(points)} context points survived constraint filtering (need >= 2). "
            "Relax context_constraints or increase context_samples."
        )

    robust_cfg.context_points = points[: robust_cfg.context_samples]
    log.debug(
        "Generated %d context points via Sobol from %s",
        len(robust_cfg.context_points),
        robust_cfg.context_groups,
    )


def _parse_inequality(expr: str):
    """Parse 'a >= b' / 'a <= b' / 'a > b' / 'a < b' into a sympy expr that is >= 0 when
    the inequality holds (LHS-RHS for >=/>, RHS-LHS for <=/<)."""
    import sympy

    for op, sign in ((">=", 1), (">", 1), ("<=", -1), ("<", -1)):
        if op in expr:
            lhs, rhs = expr.split(op, 1)
            diff = sympy.sympify(lhs) - sympy.sympify(rhs)
            return diff if sign > 0 else -diff
    raise ValueError(f"Unsupported inequality (need one of >=,>,<=,<): {expr!r}")


def build_context_tensor(robust_cfg: RobustOptimizationConfig, params: Sequence) -> Tensor:
    """Build a normalized ``(n_w, d_ctx)`` tensor from ``context_points``."""
    names = context_param_names(robust_cfg, params)
    param_map = {p.name: p for p in params}
    bounds_lo, bounds_hi = [], []
    for cn in names:
        lo, hi = _param_bounds(param_map[cn])
        bounds_lo.append(lo)
        bounds_hi.append(hi)
    bounds_lo = np.array(bounds_lo)
    bounds_hi = np.array(bounds_hi)
    raw = np.array([[pt[cn] for cn in names] for pt in robust_cfg.context_points])
    normalized = (raw - bounds_lo) / (bounds_hi - bounds_lo + 1e-12)
    return torch.tensor(normalized, dtype=torch.double)


# ---------------------------------------------------------------------------
# RobustAcquisition — CVaR (SOO) / MARS (MOO)
# ---------------------------------------------------------------------------


class RobustAcquisition(Acquisition):
    """Acquisition subclass that wraps the objective in a risk measure.

    SOO: CVaR wraps the scalar objective. MOO: MARS (random Chebyshev scalarization +
    VaR). Both swap the acqf to qLogNEI (the one-to-many context transform is incompatible
    with best_f-from-Y acqfs like qLogEI/qLogNEHVI). CVaR vs MARS is chosen by ``is_moo``
    in the passed ``risk_config`` — there is no risk_measure config field (§0.1).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        opts = kwargs.get("options") or {}
        self._risk_config: dict[str, Any] = opts.pop("risk_config", {})
        super().__init__(*args, **kwargs)

    def _construct_botorch_acquisition(self, botorch_acqf_class, botorch_acqf_options, model):
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement

        is_moo = self._risk_config.get("is_moo", False)
        # Risk measures (CVaR/MARS) use a one-to-many input transform (context substitution),
        # so the acqf's q-batch expands to q*n_w internally. qLogEI's input constructor computes
        # best_f from raw training Y (q-batch 1) -> shape mismatch. qLogNEI instead evaluates the
        # model at X_baseline (transform applied consistently), so swap to it for BOTH CVaR (SOO)
        # and MARS (MOO), regardless of the acqf the strategy configured (qLogEI / qLogNEHVI /
        # qLogNParEGO for >7 obj). foamBO only ever exercised the MOO swap.
        botorch_acqf_class = qLogNoisyExpectedImprovement
        if is_moo:
            # MARS scalarization + constraint transforms mismatch in prune_inferior_points.
            _saved_oc = self._outcome_constraints
            self._outcome_constraints = None
            try:
                return super()._construct_botorch_acquisition(
                    botorch_acqf_class=botorch_acqf_class,
                    botorch_acqf_options=botorch_acqf_options,
                    model=model,
                )
            finally:
                self._outcome_constraints = _saved_oc
        return super()._construct_botorch_acquisition(
            botorch_acqf_class=botorch_acqf_class,
            botorch_acqf_options=botorch_acqf_options,
            model=model,
        )

    def get_botorch_objective_and_transform(
        self,
        botorch_acqf_class,
        model,
        objective_weights,
        outcome_constraints=None,
        X_observed=None,
        learned_objective_preference_model=None,
    ):
        is_moo = self._risk_config.get("is_moo", False)
        if is_moo:
            from botorch.acquisition.logei import qLogNoisyExpectedImprovement

            botorch_acqf_class = qLogNoisyExpectedImprovement
            if outcome_constraints is not None:
                outcome_constraints = None  # MARS drops constraints (shape mismatch)

        from ax.generators.torch.utils import get_botorch_objective_and_transform as _get_obj

        objective, posterior_transform = _get_obj(
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            learned_objective_preference_model=learned_objective_preference_model,
        )

        alpha = self._risk_config.get("alpha", 0.5)
        n_w = self._risk_config.get("n_w", 10)
        if is_moo:
            return self._build_mars(model, objective_weights, X_observed, alpha, n_w)
        risk_obj = CVaR(alpha=alpha, n_w=n_w, preprocessing_function=objective)
        return risk_obj, posterior_transform

    def _build_mars(self, model, objective_weights, X_observed, alpha, n_w):
        from botorch.acquisition.multi_objective.multi_output_risk_measures import MARS
        from botorch.utils.sampling import sample_simplex

        flat_w = objective_weights.sum(dim=0) if objective_weights.dim() > 1 else objective_weights
        n_obj = int((flat_w != 0).sum().item())
        weights = sample_simplex(d=n_obj, n=1, dtype=torch.double).squeeze(0)
        ref_point = torch.zeros(n_obj, dtype=torch.double)
        nonzero_mask = flat_w != 0
        signs = torch.sign(flat_w[nonzero_mask])

        def _align_fn(samples: Tensor, X=None) -> Tensor:
            if samples.shape[-1] > n_obj:
                samples = samples[..., :n_obj]
            return samples * signs.to(samples)

        mars = MARS(
            alpha=alpha,
            n_w=n_w,
            chebyshev_weights=weights,
            ref_point=ref_point,
            preprocessing_function=_align_fn,
        )
        if X_observed is not None:
            try:
                mars.set_baseline_Y(model=model, X_baseline=X_observed)
            except Exception as e:  # pragma: no cover
                log.warning("Failed to set MARS baseline_Y: %s", e)
        return mars, None


# ---------------------------------------------------------------------------
# Generator augmentation — the wiring spine
# ---------------------------------------------------------------------------


# BO generator enums whose specs we augment with robust/MultiFid wiring.
def _bo_generators():
    from ax.adapter.registry import Generators

    return {Generators.BOTORCH_MODULAR, Generators.BO_MIXED, Generators.SAASBO}


def augment_generator_specs(
    client, params: Sequence, robust_cfg: RobustOptimizationConfig, multifidelity: bool = False
) -> dict[str, Any]:
    """Wire robust (and optionally MultiFid) config into the client's generation strategy.

    Non-destructive: only mutates BO generator specs, adding the context input transform
    (non-MultiFid), the risk-measure acquisition class, and context fixed_features.

    ``multifidelity=True`` composes with MultiFid (active behavior, §6): SingleTaskMultiFidelityGP
    surrogate + qMultiFidHVKG + MultiFidHVKGAcquisition, ``Normalize``-only transform (NO
    SubstituteContextFeatures — qMultiFidHVKG can't handle one-to-many); robustness is implicit
    via context-spanning training data + rotating fixed_features. Returns a state dict for
    ``cycle_context``.
    """
    from ax.core.observation import ObservationFeatures
    from ax.core.optimization_config import MultiObjectiveOptimizationConfig
    from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
    from botorch.models.transforms.input import Normalize

    from .multifidelity import MultiFidHVKGAcquisition

    resolve_context_points(robust_cfg, params)
    ctx_names = context_param_names(robust_cfg, params)
    ctx_indices = context_dim_indices(robust_cfg, params)
    feature_set = build_context_tensor(robust_cfg, params)  # normalized [0,1] (surrogate space)
    # RAW context values (parameter space) — fixed_features/ObservationFeatures are raw params;
    # Ax normalizes them itself. Pinning the normalized feature_set here is only correct when
    # context bounds are [0,1]; for other bounds it pins out-of-range values. Use raw.
    raw_context = torch.tensor(
        [[float(pt[cn]) for cn in ctx_names] for pt in robust_cfg.context_points],
        dtype=torch.double,
    )
    n_w = feature_set.shape[0]
    alpha = alpha_from_robustness(robust_cfg.robustness)

    # ax 1.3.x: optimization_config.objective is a plain Objective even for MOO; detect
    # MOO via the optimization-config class. (foamBO's objective-isinstance check is stale
    # on 1.3.1 — it would mis-route MOO to CVaR instead of MARS.)
    is_moo = isinstance(client._experiment.optimization_config, MultiObjectiveOptimizationConfig)
    risk_config = {
        "alpha": alpha,
        "n_w": n_w,
        "is_moo": is_moo,
        "context_names": ctx_names,
        "context_indices": ctx_indices,
    }

    if multifidelity:
        from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

        # NO explicit input transforms — let Ax pick its multi-fidelity defaults (which handle
        # the fidelity dim specially). Forcing a bare Normalize normalizes the fidelity column
        # too, breaking qMultiFidHVKG's target-fidelity projection → out-of-bounds candidates.
        # This mirrors the working MF-only path (augment_specs_for_multifidelity). NO SCF either
        # (qMultiFidHVKG can't do a one-to-many transform); robustness stays implicit via
        # context-spanning training data + rotating fixed_features.
        model_config = ModelConfig(botorch_model_class=SingleTaskMultiFidelityGP)
    else:
        model_config = ModelConfig(
            input_transform_classes=[Normalize, SubstituteContextFeatures],
            input_transform_options={
                "SubstituteContextFeatures": {
                    "context_indices": ctx_indices,
                    "feature_set": feature_set,
                },
            },
        )

    surrogate_spec = SurrogateSpec(
        model_configs=[model_config],
        allow_batched_models=not multifidelity,  # MultiFid: qMultiFidHVKG needs ModelListGP
    )

    first_ctx_raw = {ctx_names[i]: float(raw_context[0, i]) for i in range(len(ctx_names))}
    bo_generators = _bo_generators()
    augmented = 0
    for node in client._generation_strategy._nodes:
        for spec in node.generator_specs:
            if spec.generator_enum not in bo_generators:
                continue
            spec.generator_kwargs["surrogate_spec"] = surrogate_spec
            if multifidelity:
                from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
                    qMultiFidelityHypervolumeKnowledgeGradient,
                )

                spec.generator_kwargs["botorch_acqf_class"] = (
                    qMultiFidelityHypervolumeKnowledgeGradient
                )
                spec.generator_kwargs["acquisition_class"] = MultiFidHVKGAcquisition
            else:
                spec.generator_kwargs["acquisition_class"] = RobustAcquisition
                spec.generator_kwargs["acquisition_options"] = {"risk_config": risk_config}
            spec.fixed_features = ObservationFeatures(parameters=first_ctx_raw)
            augmented += 1

    log.info(
        "%s: %s(alpha=%.2f, n_w=%d), context=%s, augmented %d specs",
        "MultiFid+robust" if multifidelity else "robust",
        "MARS" if is_moo else "CVaR",
        alpha,
        n_w,
        ctx_names,
        augmented,
    )
    return {
        "robust_cfg": robust_cfg,
        "ctx_names": ctx_names,
        "ctx_indices": ctx_indices,
        "feature_set": feature_set,
        "raw_context": raw_context,
        "n_w": n_w,
        "risk_config": risk_config,
        "is_moo": is_moo,
        "multifidelity": multifidelity,
    }


def cycle_context(client, robust_state: dict, trial_index: int) -> None:
    """Round-robin fixed_features to context point ``trial_index % n_w`` before each gen.

    Merges with existing fixed_features so upstream pinning (e.g. MultiFid fidelity) is preserved.
    """
    from ax.core.observation import ObservationFeatures

    # RAW context values — fixed_features are raw params (see augment_generator_specs). Falls
    # back to the normalized feature_set for states saved before raw_context existed.
    raw_context = robust_state.get("raw_context", robust_state["feature_set"])
    ctx_names = robust_state["ctx_names"]
    n_w = robust_state["n_w"]
    ctx_idx = trial_index % n_w
    ctx_values = {ctx_names[i]: float(raw_context[ctx_idx, i]) for i in range(len(ctx_names))}

    for node in client._generation_strategy._nodes:
        for spec in node.generator_specs:
            if spec.generator_enum in _bo_generators():
                existing = spec.fixed_features
                merged = dict(existing.parameters) if existing is not None else {}
                merged.update(ctx_values)
                spec.fixed_features = ObservationFeatures(parameters=merged)


# ---------------------------------------------------------------------------
# Ax JSON serialization registry — run at import so saved robust/MultiFid experiments
# can be (de)serialized. (Dormant RobustContextValueFunction NOT registered — §0.2.)
# ---------------------------------------------------------------------------
try:
    from ax.storage.botorch_modular_registry import (
        CLASS_TO_REGISTRY,
        CLASS_TO_REVERSE_REGISTRY,
        register_acquisition,
    )
    from botorch.models.transforms.input import InputTransform as _InputTransform

    from .multifidelity import MultiFidHVKGAcquisition as _MultiFidHVKGAcquisition

    CLASS_TO_REGISTRY[_InputTransform][SubstituteContextFeatures] = "SubstituteContextFeatures"
    CLASS_TO_REVERSE_REGISTRY[_InputTransform]["SubstituteContextFeatures"] = (
        SubstituteContextFeatures
    )
    register_acquisition(RobustAcquisition)
    register_acquisition(_MultiFidHVKGAcquisition)
except ImportError:  # pragma: no cover - Ax storage registry path drift
    log.warning("Ax storage registry unavailable; robust/MultiFid JSON (de)serialization skipped")
