"""Probability-based global stopping for Ax (ported from foamBO ``analysis.py``).

Ax ships one global stopping strategy, ``ImprovementGlobalStoppingStrategy``, which is
purely *retrospective*: it compares the running optimum (SOO) or hypervolume (MOO) against
its value ``window_size`` observed trials ago. It never consults the surrogate, so it can't
tell "plateau because converged" from "plateau because BO is exploring", and a single lucky
noise draw resets its window.

``ProbabilisticGlobalStoppingStrategy`` is *prospective*: it asks the fitted GP how likely
any of the next ``horizon`` trials is to improve on the incumbent, and stops when that
probability falls below ``probability_bar``.

The estimate is a decay extrapolation, not a rollout::

    max_pi_t  = max_x P(f(x) improves over incumbent)      # one optimize_acqf per check
    PI_t      ~ exp(a + b*t)                               # log-linear fit over history
    P_H       = 1 - prod_{t=T+1}^{T+H} (1 - PI_t)          # P(>=1 improvement in next H)
    stop      <=> P_H < probability_bar

A fantasy rollout to H (EI argmax -> rsample -> condition_on_observations, H times, per
objective, per check) would be more faithful to real BO dynamics but costs minutes-to-hours
per stopping check and is stochastic, so stopping decisions would be nondeterministic.

Ax hands the strategy only the experiment (``orchestrator.py``:
``gss.should_stop_optimization(experiment=self.experiment)``) -- never the generation
strategy -- so the surrogate has to be reached through an injected ``gs_provider``.

Pinned to ax-platform 1.3.x / botorch 0.18.x.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy

log = logging.getLogger("axbo_extensions.stopping")


# ---------------------------------------------------------------------------
# Model / search-space introspection
# ---------------------------------------------------------------------------


def objective_models(gs) -> tuple[Optional[list], list[str]]:
    """Per-objective sub-models from the surrogate (``ModelListGP`` or single)."""
    surr = getattr(getattr(gs, "adapter", None), "generator", None)
    surr = getattr(surr, "surrogate", None)
    if surr is None or surr.model is None:
        return None, []
    model = surr.model
    outcomes = list(getattr(surr, "_outcomes", None) or [])
    if hasattr(model, "models"):
        return list(model.models), outcomes
    return [model], outcomes


def model_bounds(sub_model) -> list[list[float]]:
    """Bounds from the model's training data, widened 1%.

    Training data rather than the search space: Ax transforms can add features (e.g.
    ``step``), so search-space bounds would have the wrong dimension.
    """
    X = sub_model.train_inputs[0]
    lo = X.min(dim=0).values.tolist()
    hi = X.max(dim=0).values.tolist()
    return [
        [l - 0.01 * max(abs(h - l), 1e-6), h + 0.01 * max(abs(h - l), 1e-6)]
        for l, h in zip(lo, hi)
    ]


def fidelity_info(experiment: Experiment) -> tuple[Optional[int], Optional[float]]:
    """``(feature_index, target_value)`` of the fidelity parameter, or ``(None, None)``.

    Index is in parameter order, matching ``update_cost_state`` and Ax's
    ``fidelity_features`` search-space digest convention.
    """
    for idx, (_name, p) in enumerate(experiment.search_space.parameters.items()):
        if getattr(p, "is_fidelity", False):
            return idx, p.target_value
    return None, None


def denoised_incumbent(sub_model, minimize: bool, fid_idx=None, fid_target=None) -> float:
    """GP-denoised incumbent: best posterior mean at the training inputs.

    Raw ``train_targets`` carry observation noise -- one spuriously good draw inflates PI
    against a target that was never really achieved. Averaging through the kernel matches
    Frazier (2018) §4.1 noise-aware incumbent and BoTorch's best_point convention.

    Under multi-fidelity the incumbent is restricted to target-fidelity rows: cheap-fidelity
    samples sit where the posterior mean is smoother and often artificially better, which
    would bias PI low and stop early.
    """
    import torch

    X = sub_model.train_inputs[0]
    if fid_idx is not None and fid_target is not None:
        mask = torch.isclose(
            X[:, fid_idx],
            torch.tensor(float(fid_target), dtype=X.dtype),
            atol=1e-6,
        )
        if mask.any():
            X = X[mask]
        # else: no target-fidelity rows yet -> fall back to all rows
    with torch.no_grad():
        mu = sub_model.posterior(X).mean.squeeze(-1)
    return float(mu.min().item()) if minimize else float(mu.max().item())


# ---------------------------------------------------------------------------
# Context transforms (robust mode)
# ---------------------------------------------------------------------------


def _context_transforms(sub_model) -> list:
    """``SubstituteContextFeatures`` instances on a sub-model's input transform."""
    from .robustness import SubstituteContextFeatures

    tf = getattr(sub_model, "input_transform", None)
    if tf is None:
        return []
    # ChainedInputTransform exposes its children via .items(); a bare transform doesn't.
    items = tf.items() if hasattr(tf, "items") else [(None, tf)]
    return [t for _name, t in items if isinstance(t, SubstituteContextFeatures)]


def set_context_transforms(sub_model, enabled: bool) -> list[tuple[Any, bool]]:
    """Toggle ``transform_on_eval`` on context transforms. Returns restore tuples."""
    saved = []
    for t in _context_transforms(sub_model):
        saved.append((t, t.transform_on_eval))
        t.transform_on_eval = enabled
    return saved


def restore_context_transforms(saved: list[tuple[Any, bool]]) -> None:
    for t, flag in saved:
        t.transform_on_eval = flag


# ---------------------------------------------------------------------------
# max PI -- nominal and robust
# ---------------------------------------------------------------------------


def max_pi_nominal(
    sub_model, bounds, best_f: float, minimize: bool, fixed_features=None,
    n_restarts: int = 8, raw_samples: int = 64,
) -> Optional[float]:
    """``max_x P(f(x) improves over best_f)`` via analytic PI.

    Context transforms are disabled for the duration: they are one-to-many, so an enabled
    transform makes the posterior expand ``q -> q*n_w`` and the analytic acqf shape blows up.
    """
    import torch
    from botorch.acquisition.analytic import ProbabilityOfImprovement
    from botorch.optim import optimize_acqf

    bounds_t = torch.tensor(bounds, dtype=torch.double).T  # (2, d)
    if fixed_features:
        for dim_idx, val in fixed_features.items():
            bounds_t[0, dim_idx] = val
            bounds_t[1, dim_idx] = val

    saved = set_context_transforms(sub_model, enabled=False)
    try:
        acqf = ProbabilityOfImprovement(model=sub_model, best_f=best_f, maximize=not minimize)
        _, max_pi = optimize_acqf(
            acq_function=acqf, bounds=bounds_t, q=1,
            num_restarts=n_restarts, raw_samples=raw_samples,
            fixed_features=fixed_features,
        )
        return float(max_pi)
    except Exception as e:
        log.debug("nominal PI optimization failed: %s", e)
        return None
    finally:
        restore_context_transforms(saved)


def _cvar_objective(alpha: float, n_w: int, minimize: bool):
    """CVaR over the context dimension, in BoTorch's larger-is-better convention.

    BoTorch risk measures take the *worst* ``1-alpha`` fraction to mean the lowest samples,
    so a minimization objective is negated on the way in and the incumbent is carried in the
    same negated space.
    """
    from botorch.acquisition.risk_measures import CVaR

    sign = -1.0 if minimize else 1.0

    def _prep(Y, X=None):
        # (batch x m) -> (batch,); single-output sub-models, so m == 1.
        return sign * Y.squeeze(-1)

    return CVaR(alpha=alpha, n_w=n_w, preprocessing_function=_prep)


def robust_incumbent(sub_model, alpha: float, minimize: bool) -> Optional[float]:
    """Best CVaR-over-context value among observed designs, in CVaR (maximize) space.

    The context transform is left *enabled*: it expands each training design into its
    ``n_w`` context scenarios, which is exactly the set CVaR reduces over.
    """
    import torch

    ctx = _context_transforms(sub_model)
    if not ctx:
        return None
    n_w = ctx[0].n_w

    saved = set_context_transforms(sub_model, enabled=True)
    try:
        X = sub_model.train_inputs[0]
        with torch.no_grad():
            # posterior applies the transform: (n, d) -> (n*n_w, d), grouped by design.
            mu = sub_model.posterior(X).mean  # (n*n_w, 1)
            cvar = _cvar_objective(alpha, n_w, minimize)(mu.unsqueeze(0))  # (1, n)
        return float(cvar.max().item())
    except Exception as e:
        log.debug("robust incumbent failed: %s", e)
        return None
    finally:
        restore_context_transforms(saved)


def max_pi_robust(
    sub_model, bounds, best_f: float, alpha: float, minimize: bool, fixed_features=None,
    n_restarts: int = 8, raw_samples: int = 64, mc_samples: int = 256,
) -> Optional[float]:
    """``max_x P(CVaR_alpha(f(x, W)) improves over best_f)``.

    Under robust optimization the quantity being optimized is CVaR over context, not
    nominal ``f``. Nominal PI would answer the wrong question: it can read "still improving"
    while the robust quantile has flattened, or the reverse.

    No custom acqf is needed. ``SubstituteContextFeatures`` already expands ``X`` into its
    ``n_w`` context scenarios, and BoTorch's ``CVaR`` is an ``MCAcquisitionObjective``, so MC
    qPI with CVaR as its objective *is* the robust functional.

    ``best_f`` is in CVaR (maximize) space -- see ``robust_incumbent``.

    Note: robust MOO generates candidates with MARS, whose Chebyshev weights are resampled
    per call. Applying CVaR per objective instead keeps the stopping signal deterministic;
    a MARS-based signal would jitter with the weights and make stopping a coin flip.
    """
    import torch
    from botorch.acquisition.monte_carlo import qProbabilityOfImprovement
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler

    ctx = _context_transforms(sub_model)
    if not ctx:
        return None
    n_w = ctx[0].n_w

    bounds_t = torch.tensor(bounds, dtype=torch.double).T
    if fixed_features:
        for dim_idx, val in fixed_features.items():
            bounds_t[0, dim_idx] = val
            bounds_t[1, dim_idx] = val
    # Context dims are substituted by the transform, so their bounds are inert; pinning them
    # to the training mean keeps optimize_acqf from wasting restarts on a dead subspace.
    for i, idx in enumerate(ctx[0].ctx_indices.tolist()):
        mid = float(sub_model.train_inputs[0][:, idx].mean())
        bounds_t[0, idx] = mid
        bounds_t[1, idx] = mid

    saved = set_context_transforms(sub_model, enabled=True)
    try:
        acqf = qProbabilityOfImprovement(
            model=sub_model,
            best_f=best_f,
            objective=_cvar_objective(alpha, n_w, minimize),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples])),
            tau=1e-3,
        )
        _, max_pi = optimize_acqf(
            acq_function=acqf, bounds=bounds_t, q=1,
            num_restarts=n_restarts, raw_samples=raw_samples,
        )
        return float(max_pi)
    except Exception as e:
        log.debug("robust PI optimization failed: %s", e)
        return None
    finally:
        restore_context_transforms(saved)


# ---------------------------------------------------------------------------
# Decay extrapolation
# ---------------------------------------------------------------------------


_DECAY_TOL = 1e-9


def improvement_probability_in_horizon(
    pi_history: list[float], horizon: int
) -> Optional[float]:
    """``P(>=1 improvement in the next `horizon` trials)`` from a PI decay fit.

    Fits ``log PI_t = a + b*t`` over history, then treats each future trial as an independent
    Bernoulli(PI_t)::

        P_H = 1 - prod_{t=T+1}^{T+H} (1 - PI_t)

    Returns ``None`` when the fit can't support a claim: fewer than 3 positive observations,
    or PI flat/rising, so there is no decay to extrapolate and no grounds to stop. Flatness
    is a tolerance, not ``b >= 0``: a flat history fits ``b ~ +/-1e-17`` and the sign is
    float noise, which would make the branch taken nondeterministic. Slopes shallower than
    the tolerance decay by <1e-7 over a 100-trial horizon -- flat by any measure, and they
    yield ``P_H ~ 1`` and refuse to stop regardless, so nothing is lost by rejecting them.

    The independence assumption is the load-bearing approximation. Real BO trials are
    correlated -- each observation reshapes the posterior the next PI is computed from -- so
    this over-counts distinct chances to improve and reads high. It errs toward continuing.
    """
    vals = np.asarray(pi_history, dtype=float)
    mask = vals > 0
    if mask.sum() < 3:
        return None
    x = np.arange(len(vals))[mask]
    y = np.log(vals[mask])
    try:
        b, a = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if b >= -_DECAY_TOL:
        return None

    T = len(vals) - 1
    t = np.arange(T + 1, T + 1 + horizon)
    pi_t = np.clip(np.exp(a + b * t), 0.0, 1.0 - 1e-12)
    return float(1.0 - np.exp(np.log1p(-pi_t).sum()))


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


class ProbabilisticGlobalStoppingStrategy(BaseGlobalStoppingStrategy):
    """Stop when the GP says the next ``horizon`` trials are unlikely to improve.

    Args:
        min_trials: Minimum completed trials before the strategy can fire (base-class gate).
        gs_provider: Zero-arg callable returning the generation strategy. Ax passes only the
            experiment to a stopping strategy, so the surrogate has to arrive this way, e.g.
            ``gs_provider=lambda: client._generation_strategy``.
        horizon: Lookahead length ``H``, in trials.
        probability_bar: Stop when ``P(>=1 improvement in next H) < probability_bar``.
        robust_alpha: If set, PI is computed over the CVaR-of-context functional at this
            alpha (see ``alpha_from_robustness``) rather than over nominal ``f``.
        observe_only: Compute and record verdicts but never stop on them. Use this to see
            what the rule *would* have done on a real study before trusting it.
        delegate: Another stopping strategy that makes the actual decision. PI is still
            computed and recorded every check. Pass Ax's ``ImprovementGlobalStoppingStrategy``
            to keep stock behaviour while collecting PI data alongside it.
        inactive_when_pending_trials: Base-class gate; don't stop while trials are running.

    PI decides only when ``observe_only`` is False and ``delegate`` is None. Otherwise it
    observes: every check still appends to :attr:`pi_history`, appends a verdict to
    :attr:`verdicts`, and logs, so a dry run yields the same data a live run would.

    Under multi-fidelity, only target-fidelity trials advance the history and the horizon:
    ``H`` counts target-fidelity trials. Cheap-fidelity trials leave target-fidelity PI flat,
    and feeding those flat stretches to the decay fit would read as convergence and stop
    early.
    """

    def __init__(
        self,
        min_trials: int,
        gs_provider: Callable[[], Any],
        horizon: int = 100,
        probability_bar: float = 0.05,
        robust_alpha: Optional[float] = None,
        observe_only: bool = False,
        delegate: Optional[BaseGlobalStoppingStrategy] = None,
        inactive_when_pending_trials: bool = True,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if not 0.0 < probability_bar < 1.0:
            raise ValueError("probability_bar must be in (0, 1).")
        super().__init__(
            min_trials=min_trials,
            inactive_when_pending_trials=inactive_when_pending_trials,
        )
        self.gs_provider = gs_provider
        self.horizon = horizon
        self.probability_bar = probability_bar
        self.robust_alpha = robust_alpha
        self.observe_only = observe_only
        self.delegate = delegate
        self.pi_history: dict[str, list[float]] = {}
        self.verdicts: list[dict[str, Any]] = []
        self.last_verdict: Optional[dict[str, Any]] = None
        self._verdict_trial: Optional[int] = None

    def __repr__(self) -> str:
        return super().__repr__() + (
            f" min_trials={self.min_trials} "
            f"horizon={self.horizon} "
            f"probability_bar={self.probability_bar} "
            f"robust_alpha={self.robust_alpha} "
            f"observe_only={self.observe_only} "
            f"delegate={self.delegate} "
            f"inactive_when_pending_trials={self.inactive_when_pending_trials}"
        )

    @property
    def pi_decides(self) -> bool:
        """Whether the PI verdict is the one Ax acts on."""
        return self.delegate is None and not self.observe_only

    def state_dict(self) -> dict[str, Any]:
        """JSON-serializable history, for drivers that rebuild the strategy every ask.

        Ax's Orchestrator holds one strategy object for a whole study, so in-memory history
        suffices there. A detached driver runs one ask per process and reconstructs the
        strategy each time; without rehydration ``pi_history`` would reset on every
        invocation and the decay fit would never see more than a single point.
        """
        return {
            "pi_history": {k: list(v) for k, v in self.pi_history.items()},
            "verdicts": list(self.verdicts),
            "verdict_trial": self._verdict_trial,
        }

    def load_state_dict(self, state: Optional[dict[str, Any]]) -> None:
        """Restore history produced by :meth:`state_dict`. Empty/None state is a no-op."""
        if not state:
            return
        self.pi_history = {k: list(v) for k, v in (state.get("pi_history") or {}).items()}
        self.verdicts = list(state.get("verdicts") or [])
        self._verdict_trial = state.get("verdict_trial")
        self.last_verdict = self.verdicts[-1] if self.verdicts else None

    def _at_target_fidelity(self, experiment: Experiment, trial_index: int) -> bool:
        """Whether every arm of a trial sits at the fidelity parameter's target value."""
        fid_name = next(
            (n for n, p in experiment.search_space.parameters.items()
             if getattr(p, "is_fidelity", False)),
            None,
        )
        if fid_name is None:
            return True
        target = experiment.search_space.parameters[fid_name].target_value
        arms = experiment.trials[trial_index].arms
        return all(
            abs(float(a.parameters.get(fid_name, target)) - float(target)) < 1e-6
            for a in arms
        )

    def collect(self, experiment: Experiment) -> Optional[dict[str, Any]]:
        """Compute max PI and the horizon probability for every objective; record a verdict.

        Deliberately *not* behind the base class's ``min_trials`` / pending-trials gates.
        Those gate the stopping *decision*; gating collection too would mean the decay fit
        can't produce its first verdict until ``min_trials + 3``. The cost is one
        ``optimize_acqf`` per objective per check -- cheap next to the trials being decided
        about.

        Returns the verdict dict (also appended to :attr:`verdicts` and cached on
        :attr:`last_verdict`), or ``None`` when no verdict is possible: no completed trials,
        no fitted surrogate, or a latest trial below target fidelity.

        Cached per trial index, so repeated checks within a trial neither recompute the acqf
        nor double-append to the history -- a duplicate would compress the fitted decay.
        """
        completed = experiment.trial_indices_by_status[TrialStatus.COMPLETED]
        if not completed:
            return None
        latest = max(completed)
        if self._verdict_trial == latest:
            return self.last_verdict

        if not self._at_target_fidelity(experiment, latest):
            return None

        gs = self.gs_provider()
        models, outcomes = objective_models(gs)
        if models is None:
            return None

        opt_config = experiment.optimization_config
        if opt_config is None:
            return None
        # metric_weights: negative weight == minimize, matching Ax's internal convention.
        obj_info = {name: (w < 0) for name, w in opt_config.objective.metric_weights}

        fid_idx, fid_target = fidelity_info(experiment)
        fixed_features = (
            {fid_idx: float(fid_target)}
            if fid_idx is not None and fid_target is not None
            else None
        )

        max_pis: dict[str, float] = {}
        probabilities: dict[str, float] = {}
        for sub_model, outcome_name in zip(models, outcomes):
            if outcome_name not in obj_info:
                continue
            minimize = obj_info[outcome_name]
            bounds = model_bounds(sub_model)

            if self.robust_alpha is not None:
                best_f = robust_incumbent(sub_model, self.robust_alpha, minimize)
                max_pi = (
                    None if best_f is None
                    else max_pi_robust(
                        sub_model, bounds, best_f, self.robust_alpha, minimize,
                        fixed_features=fixed_features,
                    )
                )
            else:
                best_f = denoised_incumbent(
                    sub_model, minimize, fid_idx=fid_idx, fid_target=fid_target
                )
                max_pi = max_pi_nominal(
                    sub_model, bounds, best_f, minimize, fixed_features=fixed_features
                )

            if max_pi is None:
                continue

            max_pis[outcome_name] = max_pi
            history = self.pi_history.setdefault(outcome_name, [])
            history.append(max_pi)

            p_h = improvement_probability_in_horizon(history, self.horizon)
            if p_h is not None:
                probabilities[outcome_name] = p_h

        if not max_pis:
            return None

        verdict = {
            "trial": latest,
            "max_pi": max_pis,
            "p_horizon": probabilities,
            "pi_would_stop": False,
            "reason": "",
        }

        pending = [name for name in max_pis if name not in probabilities]
        if pending:
            verdict["reason"] = (
                f"PI history does not support a decay fit yet for "
                f"{', '.join(sorted(pending))}."
            )
        else:
            # Any objective still plausibly improving keeps the optimization alive: for MOO
            # the Pareto front is still expanding as long as one objective can move.
            worst_name = max(probabilities, key=lambda k: probabilities[k])
            worst_p = probabilities[worst_name]
            detail = ", ".join(f"{k}={v:.4f}" for k, v in sorted(probabilities.items()))
            if worst_p >= self.probability_bar:
                verdict["reason"] = (
                    f"P(improvement within {self.horizon} trials) for {worst_name} "
                    f"(={worst_p:.4f}) is at or above probability_bar "
                    f"(={self.probability_bar})."
                )
            else:
                verdict["pi_would_stop"] = True
                verdict["reason"] = (
                    f"The extrapolated probability of any improvement within the next "
                    f"{self.horizon} trials is below probability_bar "
                    f"(={self.probability_bar}) for every objective ({detail})."
                )

        self.last_verdict = verdict
        self._verdict_trial = latest
        self.verdicts.append(verdict)
        log.info(
            "PI stopping verdict @ trial %d: would_stop=%s. %s",
            latest, verdict["pi_would_stop"], verdict["reason"],
        )
        return verdict

    def should_stop_optimization(self, experiment: Experiment, **kwargs: Any) -> tuple[bool, str]:
        """Collect a PI verdict, then decide -- or let someone else decide.

        Collection happens on every check in every mode, so an observe-only run produces the
        same :attr:`verdicts` a deciding run would.
        """
        verdict = self.collect(experiment)

        if self.delegate is not None:
            stop, message = self.delegate.should_stop_optimization(experiment, **kwargs)
            return stop, f"{message} [PI observing: {self._observation(verdict)}]"

        if self.observe_only:
            return False, f"Observe-only; not stopping. PI: {self._observation(verdict)}"

        return super().should_stop_optimization(experiment, **kwargs)

    def _observation(self, verdict: Optional[dict[str, Any]]) -> str:
        if verdict is None:
            return "no verdict available."
        return f"would_stop={verdict['pi_would_stop']}. {verdict['reason']}"

    def _should_stop_optimization(self, experiment: Experiment, **kwargs: Any) -> tuple[bool, str]:
        verdict = self.collect(experiment)
        if verdict is None:
            return False, (
                "No PI verdict available: no completed trials, no fitted surrogate, or the "
                "latest trial is below target fidelity."
            )
        return verdict["pi_would_stop"], verdict["reason"]
