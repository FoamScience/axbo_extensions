"""Multi-fidelity wiring tests (ported from foamBO tests/unit/test_multifidelity.py).

Guards the cost chain:
  is_cost metric observed -> experiment.lookup_data().df has 'cost' rows
    -> update_cost_state reads them, writes botorch_acqf_options['cost_intercept',
       'fidelity_weights'] -> qMultiFidHVKG input constructor builds AffineFidelityCostModel
    -> cost(fid=0) != cost(fid=1) (acqf is cost-aware).

(foamBO's config-layer is_cost tracking/objective-split tests stay in foamBO; UPO covers
that split via its pydantic ObjectiveConfig.is_cost + objective-string exclusion.)
"""

from __future__ import annotations

import pytest
import torch


class TestAffineFidelityCostModel:
    """Sanity-check the downstream cost model qMultiFidHVKG builds internally."""

    @pytest.mark.parametrize(
        "intercept,weight,s0,s1,expected_ratio",
        [
            (1.0, 0.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0, 1.0, 2.0),
            (0.3, 0.7, 0.0, 1.0, 10 / 3),
            (0.7, 0.3, 0.0, 1.0, 10 / 7),
            (0.01, 0.99, 0.0, 1.0, 100.0),
        ],
    )
    def test_cost_ratio(self, intercept, weight, s0, s1, expected_ratio):
        from botorch.models.cost import AffineFidelityCostModel

        m = AffineFidelityCostModel(fidelity_weights={0: weight}, fixed_cost=intercept)
        x0 = torch.tensor([[s0, 0.5]], dtype=torch.double)
        x1 = torch.tensor([[s1, 0.5]], dtype=torch.double)
        c0 = float(m(x0).item())
        c1 = float(m(x1).item())
        assert c0 == pytest.approx(intercept + weight * s0, rel=1e-6)
        assert c1 == pytest.approx(intercept + weight * s1, rel=1e-6)
        assert (c1 / c0) == pytest.approx(expected_ratio, rel=1e-6)


class _FakeArm:
    def __init__(self, params):
        self.parameters = params


class _FakeTrial:
    def __init__(self, idx, fid, status):
        from ax.core.base_trial import TrialStatus

        self.index = idx
        self.status = status if isinstance(status, TrialStatus) else TrialStatus.COMPLETED
        self.arm = _FakeArm({"fidelity": fid})


class _FakeSearchSpace:
    def __init__(self, names):
        self._names = names

    @property
    def parameters(self):
        return {n: object() for n in self._names}


class _FakeExperiment:
    def __init__(self, df, trials, param_names):
        self._df = df
        self.trials = trials
        self.search_space = _FakeSearchSpace(param_names)

    def lookup_data(self):
        class _D:
            def __init__(self, d):
                self.df = d

        return _D(self._df)


class _FakeClient:
    def __init__(self, exp):
        self._experiment = exp


class TestUpdateCostState:
    """update_cost_state: read per-fidelity costs from df, write opts."""

    @pytest.fixture
    def scenario(self):
        import pandas as pd

        df = pd.DataFrame([
            {"trial_index": 0, "metric_name": "cost", "mean": 0.3},
            {"trial_index": 1, "metric_name": "cost", "mean": 0.3},
            {"trial_index": 2, "metric_name": "cost", "mean": 1.0},
            {"trial_index": 3, "metric_name": "cost", "mean": 1.0},
        ])
        trials = {
            0: _FakeTrial(0, 0, None),
            1: _FakeTrial(1, 0, None),
            2: _FakeTrial(2, 1, None),
            3: _FakeTrial(3, 1, None),
        }
        exp = _FakeExperiment(df, trials, param_names=["fidelity", "x"])
        client = _FakeClient(exp)
        opts: dict = {}
        multifid_cost = {
            "state": {"per_fidelity": {}},
            "metric": "cost",
            "fidelity_param": "fidelity",
            "acqf_opts_ref": opts,
        }
        return client, multifid_cost, opts

    def test_writes_intercept_and_weight(self, scenario):
        from axbo_extensions.multifidelity import update_cost_state

        client, multifid_cost, opts = scenario
        update_cost_state(client, multifid_cost)
        # cost_intercept = positivity-floored normalized cheap cost:
        #   max(0.3/max_cost, 0.5*weight_norm + 1e-3) = max(0.3, 0.5*0.7 + 1e-3) = 0.351.
        # foamBO's upstream test still asserts the pre-floor 0.3 and FAILS (stale); the
        # floor is load-bearing — see test_downstream_cost_model_reflects_update.
        assert opts["cost_intercept"] == pytest.approx(0.351, rel=1e-6)
        assert 0 in opts["fidelity_weights"]
        assert opts["fidelity_weights"][0] == pytest.approx(0.7, rel=1e-6)

    def test_no_update_when_df_empty(self, scenario):
        """If cost metric isn't tracked, df is empty -> opts untouched (not zeroed)."""
        import pandas as pd

        from axbo_extensions.multifidelity import update_cost_state

        client, multifid_cost, opts = scenario
        client._experiment._df = pd.DataFrame(columns=["trial_index", "metric_name", "mean"])
        update_cost_state(client, multifid_cost)
        assert "cost_intercept" not in opts
        assert "fidelity_weights" not in opts

    def test_downstream_cost_model_reflects_update(self, scenario):
        from botorch.models.cost import AffineFidelityCostModel

        from axbo_extensions.multifidelity import update_cost_state

        client, multifid_cost, opts = scenario
        update_cost_state(client, multifid_cost)
        m = AffineFidelityCostModel(
            fidelity_weights=opts["fidelity_weights"], fixed_cost=opts["cost_intercept"]
        )
        x_cheap = torch.tensor([[0.0, 0.5]], dtype=torch.double)
        x_exp = torch.tensor([[1.0, 0.5]], dtype=torch.double)
        x_extended = torch.tensor([[-0.5, 0.5]], dtype=torch.double)  # Ax INT-bound extension
        c_cheap = float(m(x_cheap).item())
        c_exp = float(m(x_exp).item())
        # floored intercept 0.351 + weight 0.7 -> cost(0)=0.351, cost(1)=1.051
        assert c_cheap == pytest.approx(0.351, rel=1e-6)
        assert c_exp == pytest.approx(1.051, rel=1e-6)
        assert c_exp > c_cheap  # acqf is cost-aware: expensive fidelity costs more
        # the floor's whole purpose: cost stays strictly > 0 across [-0.5, 1.5]
        assert float(m(x_extended).item()) > 0


class TestQMultiFidHVKGInputConstructor:
    """Lock the contract: qMultiFidHVKG's input_constructor accepts cost_intercept +
    fidelity_weights (Ax spreads botorch_acqf_options into the constructor kwargs)."""

    def test_accepts_cost_kwargs(self):
        import inspect

        from botorch.acquisition.input_constructors import get_acqf_input_constructor
        from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
            qMultiFidelityHypervolumeKnowledgeGradient,
        )

        ctor = get_acqf_input_constructor(qMultiFidelityHypervolumeKnowledgeGradient)
        target = getattr(ctor, "__wrapped__", ctor)
        try:
            names = set(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            names = set()
        if not {"cost_intercept", "fidelity_weights"} <= names:
            import botorch.acquisition.input_constructors as ic

            src = inspect.getsource(ic)
            assert "qMultiFidelityHypervolumeKnowledgeGradient" in src
            assert "cost_intercept" in src
            assert "fidelity_weights" in src
        else:
            assert "cost_intercept" in names
            assert "fidelity_weights" in names


class TestMultiFidHVKGAcquisitionImport:
    """The Ax Acquisition subclass imports cleanly on the pinned ax/botorch."""

    def test_import(self):
        from axbo_extensions.multifidelity import MultiFidHVKGAcquisition
        from ax.generators.torch.botorch_modular.acquisition import Acquisition

        assert issubclass(MultiFidHVKGAcquisition, Acquisition)


class TestAugmentSpecsForMultifidelity:
    """augment_specs_for_multifidelity injects MultiFid surrogate + qMultiFidHVKG + MultiFidHVKGAcquisition
    into a real Client's generation strategy and populates the cost-model state."""

    def _mf_client(self):
        from ax import Client, RangeParameterConfig
        from ax.adapter.registry import Generators
        from ax.generation_strategy.generation_node import GenerationNode
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        from ax.generation_strategy.generator_spec import GeneratorSpec
        from ax.generation_strategy.transition_criterion import MinTrials

        c = Client()
        c.configure_experiment(parameters=[
            RangeParameterConfig(name="fidelity", parameter_type="int", bounds=(0, 1)),
            RangeParameterConfig(name="x", parameter_type="float", bounds=(0.0, 1.0)),
        ])
        c.configure_optimization(objective="-f1, -f2")
        gs = GenerationStrategy(name="s", nodes=[
            GenerationNode(
                name="Sobol",
                generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                transition_criteria=[
                    MinTrials(threshold=4, transition_to="MBM", use_all_trials_in_exp=True)
                ],
            ),
            GenerationNode(
                name="MBM",
                generator_specs=[GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)],
            ),
        ])
        c.set_generation_strategy(gs)
        return c

    def test_wires_mf_surrogate_acqf_and_cost(self):
        from ax.adapter.registry import Generators
        from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
            qMultiFidelityHypervolumeKnowledgeGradient,
        )
        from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

        from axbo_extensions.multifidelity import (
            MultiFidHVKGAcquisition,
            apply_fidelity_parameters,
            augment_specs_for_multifidelity,
        )

        c = self._mf_client()
        apply_fidelity_parameters(c, {"fidelity": 1})
        assert getattr(c._experiment.search_space.parameters["fidelity"], "is_fidelity", False)

        multifid_cost = augment_specs_for_multifidelity(c, cost_metric_name="cost")

        gk = None
        for node in c._generation_strategy._nodes:
            for spec in node.generator_specs:
                if spec.generator_enum == Generators.BOTORCH_MODULAR:
                    gk = spec.generator_kwargs
        assert gk is not None
        assert gk["botorch_acqf_class"] is qMultiFidelityHypervolumeKnowledgeGradient
        assert gk["acquisition_class"] is MultiFidHVKGAcquisition
        assert gk["surrogate_spec"].model_configs[0].botorch_model_class is SingleTaskMultiFidelityGP
        assert gk["surrogate_spec"].allow_batched_models is False
        # UnitX-inclusive transforms forced so qMultiFidHVKG optimizes in [0,1] (it diverges on
        # raw multi-scale bounds; MBM drops UnitX by default). Regression for the pump-scale fix.
        from ax.adapter.transforms.unit_x import UnitX

        assert UnitX in gk["transforms"]
        # cost model populated
        assert multifid_cost["metric"] == "cost"
        assert multifid_cost["fidelity_param"] == "fidelity"
        assert "cost_intercept" in multifid_cost["acqf_opts_ref"]

    def test_no_fidelity_returns_none(self):
        from axbo_extensions.multifidelity import augment_specs_for_multifidelity

        c = self._mf_client()  # fidelity param present but never stamped is_fidelity
        assert augment_specs_for_multifidelity(c, cost_metric_name="cost") is None
