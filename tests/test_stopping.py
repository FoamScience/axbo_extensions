"""Unit tests for axbo_extensions.stopping.

The two things worth pinning down: the decay extrapolation math, and the claim that MC qPI
with a CVaR objective composes with SubstituteContextFeatures' one-to-many expansion (the
robust path rests entirely on that shape contract holding).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Decay extrapolation
# ---------------------------------------------------------------------------


class TestImprovementProbabilityInHorizon:
    def test_needs_three_positive_points(self):
        from axbo_extensions.stopping import improvement_probability_in_horizon

        assert improvement_probability_in_horizon([0.5, 0.4], 100) is None
        assert improvement_probability_in_horizon([0.5, 0.0, 0.0], 100) is None

    def test_flat_history_is_inconclusive(self):
        """No decay -> no grounds to extrapolate a stop."""
        from axbo_extensions.stopping import improvement_probability_in_horizon

        assert improvement_probability_in_horizon([0.3] * 10, 100) is None

    def test_rising_history_is_inconclusive(self):
        from axbo_extensions.stopping import improvement_probability_in_horizon

        assert improvement_probability_in_horizon([0.1, 0.2, 0.3, 0.4], 100) is None

    def test_fast_decay_gives_low_probability(self):
        from axbo_extensions.stopping import improvement_probability_in_horizon

        history = [float(np.exp(-1.5 * t)) for t in range(12)]
        p = improvement_probability_in_horizon(history, 100)
        assert p is not None and p < 0.01

    def test_slow_decay_keeps_probability_high(self):
        from axbo_extensions.stopping import improvement_probability_in_horizon

        history = [float(np.exp(-0.01 * t)) for t in range(12)]
        p = improvement_probability_in_horizon(history, 100)
        assert p is not None and p > 0.9

    def test_matches_explicit_product(self):
        """P_H == 1 - prod(1 - PI_t) over the extrapolated tail."""
        from axbo_extensions.stopping import improvement_probability_in_horizon

        a, b, n, H = np.log(0.5), -0.3, 8, 20
        history = [float(np.exp(a + b * t)) for t in range(n)]

        expected = 1.0 - np.prod(
            [1.0 - np.exp(a + b * t) for t in range(n, n + H)]
        )
        got = improvement_probability_in_horizon(history, H)
        assert got == pytest.approx(expected, rel=1e-6)

    def test_longer_horizon_is_monotone(self):
        from axbo_extensions.stopping import improvement_probability_in_horizon

        history = [float(np.exp(-0.4 * t)) for t in range(10)]
        ps = [improvement_probability_in_horizon(history, h) for h in (10, 50, 100)]
        assert ps == sorted(ps)


# ---------------------------------------------------------------------------
# Robust PI -- the qPI + CVaR + one-to-many transform shape contract
# ---------------------------------------------------------------------------


def _robust_gp(minimize: bool = False, n_w: int = 4):
    """SingleTaskGP over 3 dims (2 design, 1 context) carrying a context transform."""
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from axbo_extensions.robustness import SubstituteContextFeatures

    torch.manual_seed(0)
    X = torch.rand(14, 3, dtype=torch.double)
    Y = (X[:, :1] ** 2 + 0.3 * X[:, 2:3]).double()

    feature_set = torch.linspace(0, 1, n_w, dtype=torch.double).unsqueeze(-1)
    tf = SubstituteContextFeatures(context_indices=[2], feature_set=feature_set)

    model = SingleTaskGP(X, Y, input_transform=tf)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


class TestRobustPI:
    def test_incumbent_is_finite(self):
        from axbo_extensions.stopping import robust_incumbent

        got = robust_incumbent(_robust_gp(), alpha=0.5, minimize=False)
        assert got is not None and np.isfinite(got)

    def test_max_pi_robust_is_a_probability(self):
        """The load-bearing claim: CVaR-as-objective survives the q -> q*n_w expansion."""
        from axbo_extensions.stopping import max_pi_robust, model_bounds, robust_incumbent

        model = _robust_gp()
        best_f = robust_incumbent(model, alpha=0.5, minimize=False)
        pi = max_pi_robust(
            model, model_bounds(model), best_f, alpha=0.5, minimize=False,
            n_restarts=2, raw_samples=8, mc_samples=64,
        )
        assert pi is not None, "robust PI returned None -- shape contract broke"
        assert 0.0 <= pi <= 1.0

    def test_transform_state_restored(self):
        """PI must not leave transform_on_eval flipped -- candidate generation reads it."""
        from axbo_extensions.stopping import (
            max_pi_nominal, max_pi_robust, model_bounds, robust_incumbent,
        )

        model = _robust_gp()
        tf = model.input_transform
        assert tf.transform_on_eval is True

        best_f = robust_incumbent(model, alpha=0.5, minimize=False)
        assert tf.transform_on_eval is True

        max_pi_robust(
            model, model_bounds(model), best_f, alpha=0.5, minimize=False,
            n_restarts=2, raw_samples=8, mc_samples=64,
        )
        assert tf.transform_on_eval is True

        # The nominal path disables the transform; it must put it back.
        max_pi_nominal(model, model_bounds(model), 0.5, minimize=False,
                       n_restarts=2, raw_samples=8)
        assert tf.transform_on_eval is True

    def test_minimize_negates_consistently(self):
        """A minimize incumbent lives in negated (maximize) space -> PI stays valid."""
        from axbo_extensions.stopping import max_pi_robust, model_bounds, robust_incumbent

        model = _robust_gp()
        best_f = robust_incumbent(model, alpha=0.5, minimize=True)
        pi = max_pi_robust(
            model, model_bounds(model), best_f, alpha=0.5, minimize=True,
            n_restarts=2, raw_samples=8, mc_samples=64,
        )
        assert pi is not None and 0.0 <= pi <= 1.0


# ---------------------------------------------------------------------------
# Nominal PI
# ---------------------------------------------------------------------------


def _plain_gp():
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.manual_seed(0)
    X = torch.rand(12, 2, dtype=torch.double)
    Y = (X[:, :1] ** 2).double()
    model = SingleTaskGP(X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


class TestNominalPI:
    def test_max_pi_is_a_probability(self):
        from axbo_extensions.stopping import max_pi_nominal, model_bounds

        model = _plain_gp()
        pi = max_pi_nominal(model, model_bounds(model), best_f=0.5, minimize=False,
                            n_restarts=2, raw_samples=8)
        assert pi is not None and 0.0 <= pi <= 1.0

    def test_unreachable_incumbent_drives_pi_down(self):
        """best_f far above anything the GP believes in -> PI ~ 0."""
        from axbo_extensions.stopping import max_pi_nominal, model_bounds

        model = _plain_gp()
        pi = max_pi_nominal(model, model_bounds(model), best_f=1e6, minimize=False,
                            n_restarts=2, raw_samples=8)
        assert pi is not None and pi < 1e-3

    def test_denoised_incumbent_beats_no_row_filter(self):
        from axbo_extensions.stopping import denoised_incumbent

        model = _plain_gp()
        got = denoised_incumbent(model, minimize=False)
        assert np.isfinite(got)

    def test_denoised_incumbent_restricts_to_target_fidelity(self):
        """Cheap-fidelity rows must not set the incumbent."""
        from axbo_extensions.stopping import denoised_incumbent

        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        torch.manual_seed(0)
        # dim 1 is fidelity: 0.0 (cheap, inflated Y) or 1.0 (target).
        X = torch.cat([
            torch.cat([torch.rand(8, 1, dtype=torch.double),
                       torch.zeros(8, 1, dtype=torch.double)], dim=1),
            torch.cat([torch.rand(8, 1, dtype=torch.double),
                       torch.ones(8, 1, dtype=torch.double)], dim=1),
        ])
        Y = torch.where(X[:, 1:2] == 0.0, 10.0, 1.0).double()
        model = SingleTaskGP(X, Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        model.eval()

        target_only = denoised_incumbent(model, minimize=False, fid_idx=1, fid_target=1.0)
        unfiltered = denoised_incumbent(model, minimize=False)
        assert target_only < unfiltered, "cheap-fidelity rows leaked into the incumbent"


# ---------------------------------------------------------------------------
# Strategy wiring
# ---------------------------------------------------------------------------


class TestCollect:
    """collect() against a real GP behind a mocked generation strategy."""

    def _experiment_and_gs(self, completed=(0, 1, 2)):
        from unittest.mock import MagicMock

        from ax.core.trial_status import TrialStatus

        model = _plain_gp()
        gs = MagicMock()
        gs.adapter.generator.surrogate.model = model
        gs.adapter.generator.surrogate._outcomes = ["obj"]

        exp = MagicMock()
        exp.trial_indices_by_status = {TrialStatus.COMPLETED: set(completed)}
        exp.search_space.parameters = {}  # no fidelity parameter
        exp.optimization_config.objective.metric_weights = [("obj", 1.0)]
        return exp, gs

    def test_records_history_and_verdict(self):
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        exp, gs = self._experiment_and_gs()
        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: gs)

        verdict = gss.collect(exp)
        assert verdict is not None
        assert verdict["trial"] == 2
        assert 0.0 <= verdict["max_pi"]["obj"] <= 1.0
        assert len(gss.pi_history["obj"]) == 1
        assert len(gss.verdicts) == 1

    def test_caches_per_trial_and_does_not_double_append(self):
        """A duplicate PI sample would compress the fitted decay."""
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        exp, gs = self._experiment_and_gs()
        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: gs)

        first = gss.collect(exp)
        second = gss.collect(exp)

        assert second is first
        assert len(gss.pi_history["obj"]) == 1
        assert len(gss.verdicts) == 1

    def test_new_trial_appends(self):
        from ax.core.trial_status import TrialStatus
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        exp, gs = self._experiment_and_gs()
        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: gs)

        gss.collect(exp)
        exp.trial_indices_by_status = {TrialStatus.COMPLETED: {0, 1, 2, 3}}
        gss.collect(exp)

        assert len(gss.pi_history["obj"]) == 2
        assert [v["trial"] for v in gss.verdicts] == [2, 3]

    def test_collects_below_min_trials(self):
        """Collection is not gated: the decay fit needs history from the start."""
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        exp, gs = self._experiment_and_gs()
        gss = ProbabilisticGlobalStoppingStrategy(min_trials=999, gs_provider=lambda: gs)

        assert gss.collect(exp) is not None
        assert len(gss.pi_history["obj"]) == 1

    def test_observe_only_run_still_collects(self):
        from ax.core.trial_status import TrialStatus
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        exp, gs = self._experiment_and_gs()
        exp.trials_by_status = {TrialStatus.RUNNING: [], TrialStatus.COMPLETED: [0, 1, 2]}
        gss = ProbabilisticGlobalStoppingStrategy(
            min_trials=1, gs_provider=lambda: gs, observe_only=True)

        stop, _ = gss.should_stop_optimization(exp)
        assert stop is False
        assert len(gss.verdicts) == 1, "observe-only must yield the same data a live run does"


class TestStatePersistence:
    """Detached drivers (upo) rebuild the strategy each ask -- history must survive."""

    def test_round_trip(self):
        import json

        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        a = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)
        a.pi_history = {"obj": [0.5, 0.3, 0.1]}
        a.verdicts = [{"trial": 2, "max_pi": {"obj": 0.1}, "p_horizon": {},
                       "pi_would_stop": False, "reason": "x"}]
        a._verdict_trial = 2

        # Must survive a JSON round trip: upo persists this inside strategy_state.
        state = json.loads(json.dumps(a.state_dict()))

        b = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)
        b.load_state_dict(state)

        assert b.pi_history == {"obj": [0.5, 0.3, 0.1]}
        assert b.verdicts == a.verdicts
        assert b._verdict_trial == 2
        assert b.last_verdict == a.verdicts[-1]

    def test_load_empty_state_is_noop(self):
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)
        gss.load_state_dict(None)
        gss.load_state_dict({})
        assert gss.pi_history == {}
        assert gss.verdicts == []

    def test_rehydrated_history_accumulates_across_processes(self):
        """The point of persistence: ask N+1 extends ask N's history rather than restarting."""
        from ax.core.trial_status import TrialStatus
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        exp, gs = TestCollect()._experiment_and_gs()

        first = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: gs)
        first.collect(exp)
        state = first.state_dict()

        # A fresh process: new strategy object, same persisted state.
        second = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: gs)
        second.load_state_dict(state)
        exp.trial_indices_by_status = {TrialStatus.COMPLETED: {0, 1, 2, 3}}
        second.collect(exp)

        assert len(second.pi_history["obj"]) == 2


class TestStrategy:
    def test_rejects_bad_args(self):
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        with pytest.raises(ValueError):
            ProbabilisticGlobalStoppingStrategy(
                min_trials=5, gs_provider=lambda: None, horizon=0)
        with pytest.raises(ValueError):
            ProbabilisticGlobalStoppingStrategy(
                min_trials=5, gs_provider=lambda: None, probability_bar=1.5)

    def test_no_surrogate_does_not_stop(self):
        from unittest.mock import MagicMock

        from ax.core.trial_status import TrialStatus
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)
        exp = MagicMock()
        exp.trial_indices_by_status = {TrialStatus.COMPLETED: {0, 1}}
        exp.search_space.parameters = {}

        stop, msg = gss._should_stop_optimization(exp)
        assert stop is False
        assert "surrogate" in msg

    def test_pi_decides_by_default(self):
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)
        assert gss.pi_decides is True

    def test_observe_only_and_delegate_disable_pi_decision(self):
        from unittest.mock import MagicMock

        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        observing = ProbabilisticGlobalStoppingStrategy(
            min_trials=1, gs_provider=lambda: None, observe_only=True)
        assert observing.pi_decides is False

        delegating = ProbabilisticGlobalStoppingStrategy(
            min_trials=1, gs_provider=lambda: None, delegate=MagicMock())
        assert delegating.pi_decides is False

    def test_no_completed_trials_does_not_stop(self):
        from unittest.mock import MagicMock

        from ax.core.trial_status import TrialStatus
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)
        exp = MagicMock()
        exp.trial_indices_by_status = {TrialStatus.COMPLETED: set()}

        stop, _ = gss._should_stop_optimization(exp)
        assert stop is False

    def _stopping_verdict(self):
        return {
            "trial": 7, "max_pi": {"obj": 1e-6}, "p_horizon": {"obj": 1e-4},
            "pi_would_stop": True, "reason": "decayed.",
        }

    def test_pi_decides_when_no_mode_set(self, monkeypatch):
        """Baseline: an unmodified strategy acts on its own verdict."""
        from unittest.mock import MagicMock

        from ax.core.trial_status import TrialStatus
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)
        monkeypatch.setattr(gss, "collect", lambda exp: self._stopping_verdict())

        exp = MagicMock()
        exp.trials_by_status = {TrialStatus.RUNNING: [], TrialStatus.COMPLETED: [1, 2, 3]}

        stop, msg = gss.should_stop_optimization(exp)
        assert stop is True
        assert "decayed" in msg

    def test_observe_only_never_stops_despite_stop_verdict(self, monkeypatch):
        """The whole point of the mode: verdict says stop, optimization continues."""
        from unittest.mock import MagicMock

        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(
            min_trials=1, gs_provider=lambda: None, observe_only=True)
        monkeypatch.setattr(gss, "collect", lambda exp: self._stopping_verdict())

        stop, msg = gss.should_stop_optimization(MagicMock())
        assert stop is False
        assert "would_stop=True" in msg, "the verdict must still be reported"

    def test_delegate_decides_and_pi_only_observes(self, monkeypatch):
        from unittest.mock import MagicMock

        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        delegate = MagicMock()
        delegate.should_stop_optimization.return_value = (False, "delegate says continue")

        gss = ProbabilisticGlobalStoppingStrategy(
            min_trials=1, gs_provider=lambda: None, delegate=delegate)
        monkeypatch.setattr(gss, "collect", lambda exp: self._stopping_verdict())

        exp = MagicMock()
        stop, msg = gss.should_stop_optimization(exp)

        # PI wanted to stop; the delegate's verdict is the one that counts.
        assert stop is False
        assert "delegate says continue" in msg
        assert "would_stop=True" in msg
        delegate.should_stop_optimization.assert_called_once_with(exp)

    def test_delegate_stop_is_honoured(self, monkeypatch):
        from unittest.mock import MagicMock

        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        delegate = MagicMock()
        delegate.should_stop_optimization.return_value = (True, "no improvement")

        gss = ProbabilisticGlobalStoppingStrategy(
            min_trials=1, gs_provider=lambda: None, delegate=delegate)
        monkeypatch.setattr(gss, "collect", lambda exp: None)

        stop, msg = gss.should_stop_optimization(MagicMock())
        assert stop is True
        assert "no improvement" in msg

    def test_observe_only_tolerates_absent_verdict(self, monkeypatch):
        from unittest.mock import MagicMock

        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(
            min_trials=1, gs_provider=lambda: None, observe_only=True)
        monkeypatch.setattr(gss, "collect", lambda exp: None)

        stop, msg = gss.should_stop_optimization(MagicMock())
        assert stop is False
        assert "no verdict" in msg

    def test_low_fidelity_trial_does_not_stop(self):
        """A cheap trial must not advance the horizon or the history."""
        from unittest.mock import MagicMock

        from ax.core.trial_status import TrialStatus
        from axbo_extensions.stopping import ProbabilisticGlobalStoppingStrategy

        gss = ProbabilisticGlobalStoppingStrategy(min_trials=1, gs_provider=lambda: None)

        fid = MagicMock()
        fid.is_fidelity = True
        fid.target_value = 1.0
        arm = MagicMock()
        arm.parameters = {"fid": 0.25}
        trial = MagicMock()
        trial.arms = [arm]

        exp = MagicMock()
        exp.trial_indices_by_status = {TrialStatus.COMPLETED: {0}}
        exp.search_space.parameters = {"fid": fid}
        exp.trials = {0: trial}

        stop, msg = gss._should_stop_optimization(exp)
        assert stop is False
        assert "target fidelity" in msg
        assert gss.pi_history == {}, "a cheap trial must not advance the PI history"
