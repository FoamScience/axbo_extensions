"""Unit tests for axbo_extensions.robustness (ported from foamBO tests/unit/test_robustness.py).

Adapted to the lib's decoupled interface: context helpers take a ``params`` sequence
(duck-typed .name/.bounds/.parameter_type/.groups) instead of foamBO's ExperimentOptions.
The vestigial ``risk_measure`` field is dropped (§0.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch


@dataclass
class P:
    """Minimal parameter spec satisfying the lib's duck-typed interface."""

    name: str
    bounds: tuple
    parameter_type: str = "float"
    groups: Optional[list] = None


# ---------------------------------------------------------------------------
# SubstituteContextFeatures
# ---------------------------------------------------------------------------

class TestSubstituteContextFeatures:
    @pytest.fixture
    def feature_set(self):
        return torch.tensor([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]], dtype=torch.double)

    @pytest.fixture
    def transform(self, feature_set):
        from axbo_extensions.robustness import SubstituteContextFeatures

        t = SubstituteContextFeatures(context_indices=[2, 3], feature_set=feature_set)
        t.eval()
        return t

    def test_output_shape(self, transform):
        Y = transform(torch.rand(2, 4, dtype=torch.double))
        assert Y.shape == (6, 4)

    def test_context_dims_substituted(self, transform, feature_set):
        Y = transform(torch.rand(1, 4, dtype=torch.double))
        for w in range(3):
            assert torch.allclose(Y[w, 2], feature_set[w, 0])
            assert torch.allclose(Y[w, 3], feature_set[w, 1])

    def test_design_dims_preserved(self, transform):
        X = torch.rand(1, 4, dtype=torch.double)
        Y = transform(X)
        for w in range(3):
            assert torch.allclose(Y[w, :2], X[0, :2])

    def test_no_transform_in_train_mode(self, feature_set):
        from axbo_extensions.robustness import SubstituteContextFeatures

        t = SubstituteContextFeatures(context_indices=[2, 3], feature_set=feature_set)
        t.train()
        Y = t(torch.rand(2, 4, dtype=torch.double))
        assert Y.shape == (2, 4)

    def test_batch_input(self, transform):
        Y = transform(torch.rand(3, 2, 4, dtype=torch.double))
        assert Y.shape == (3, 6, 4)

    def test_n_w_property(self, transform):
        assert transform.n_w == 3

    def test_is_one_to_many(self, transform):
        assert transform.is_one_to_many is True

    def test_with_model_list_gp(self, feature_set):
        """Critical integration: must work with ModelListGP (AppendFeatures doesn't)."""
        from botorch.models import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.input import ChainedInputTransform, Normalize

        from axbo_extensions.robustness import SubstituteContextFeatures

        train_X = torch.rand(10, 4, dtype=torch.double)

        def make_model(Y):
            intf = ChainedInputTransform(**{
                "normalize": Normalize(d=4),
                "substitute": SubstituteContextFeatures(
                    context_indices=[2, 3], feature_set=feature_set
                ),
            })
            return SingleTaskGP(train_X, Y, input_transform=intf)

        ml = ModelListGP(
            make_model(torch.rand(10, 1, dtype=torch.double)),
            make_model(torch.rand(10, 1, dtype=torch.double)),
        )
        posterior = ml.posterior(torch.rand(2, 4, dtype=torch.double))
        assert posterior.mean.shape == (6, 2)


# ---------------------------------------------------------------------------
# RobustOptimizationConfig
# ---------------------------------------------------------------------------

class TestRobustOptimizationConfig:
    def test_basic_creation(self):
        from axbo_extensions.robustness import RobustOptimizationConfig

        cfg = RobustOptimizationConfig(context_groups=["operating_point"])
        assert cfg.context_groups == ["operating_point"]
        assert cfg.robustness == 0.5
        assert cfg.context_points is None
        assert cfg.context_samples == 10
        assert not hasattr(cfg, "risk_measure")  # vestigial field dropped (§0.1)

    def test_explicit_context_points(self):
        from axbo_extensions.robustness import RobustOptimizationConfig

        pts = [{"flowRate": 0.02, "rpm": 2900}, {"flowRate": 0.03, "rpm": 3100}]
        cfg = RobustOptimizationConfig(context_groups=["op"], context_points=pts)
        assert len(cfg.context_points) == 2

    def test_robustness_bounds(self):
        from axbo_extensions.robustness import RobustOptimizationConfig

        with pytest.raises(Exception):
            RobustOptimizationConfig(context_groups=["x"], robustness=1.5)
        with pytest.raises(Exception):
            RobustOptimizationConfig(context_groups=["x"], robustness=-0.1)

    def test_from_dict(self):
        from axbo_extensions.robustness import RobustOptimizationConfig

        cfg = RobustOptimizationConfig.model_validate(
            {"context_groups": ["env"], "robustness": 0.8, "context_samples": 4}
        )
        assert cfg.robustness == 0.8
        assert cfg.context_samples == 4

    def test_alpha_clamp(self):
        from axbo_extensions.robustness import alpha_from_robustness

        assert alpha_from_robustness(0.0) == 0.05
        assert alpha_from_robustness(0.8) == 0.8


# ---------------------------------------------------------------------------
# Context point resolution
# ---------------------------------------------------------------------------

class TestResolveContextPoints:
    def _params(self):
        return [
            P("x1", (0.0, 1.0)),
            P("x2", (0.0, 1.0)),
            P("w1", (0.0, 10.0), groups=["env"]),
            P("w2", (100.0, 200.0), groups=["env"]),
        ]

    def test_sobol_generation(self):
        from axbo_extensions.robustness import RobustOptimizationConfig, resolve_context_points

        cfg = RobustOptimizationConfig(context_groups=["env"], context_samples=5)
        resolve_context_points(cfg, self._params())
        assert cfg.context_points is not None
        assert len(cfg.context_points) == 5
        for pt in cfg.context_points:
            assert 0.0 <= pt["w1"] <= 10.0
            assert 100.0 <= pt["w2"] <= 200.0

    def test_explicit_points_not_overwritten(self):
        from axbo_extensions.robustness import RobustOptimizationConfig, resolve_context_points

        pts = [{"w1": 5.0, "w2": 150.0}]
        cfg = RobustOptimizationConfig(context_groups=["env"], context_points=pts)
        resolve_context_points(cfg, self._params())
        assert cfg.context_points == pts

    def test_no_matching_group_raises(self):
        from axbo_extensions.robustness import RobustOptimizationConfig, resolve_context_points

        cfg = RobustOptimizationConfig(context_groups=["nonexistent"], context_samples=3)
        with pytest.raises(ValueError, match="No parameters found"):
            resolve_context_points(cfg, self._params())


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

class TestContextHelpers:
    def _params(self):
        return [
            P("a", (0.0, 1.0)),
            P("b", (0.0, 1.0), groups=["ctx"]),
            P("c", (0.0, 1.0), groups=["ctx"]),
        ]

    def test_context_param_names(self):
        from axbo_extensions.robustness import RobustOptimizationConfig, context_param_names

        cfg = RobustOptimizationConfig(context_groups=["ctx"])
        assert context_param_names(cfg, self._params()) == ["b", "c"]

    def test_context_dim_indices(self):
        from axbo_extensions.robustness import RobustOptimizationConfig, context_dim_indices

        cfg = RobustOptimizationConfig(context_groups=["ctx"])
        assert context_dim_indices(cfg, self._params()) == [1, 2]

    def test_build_context_tensor(self):
        from axbo_extensions.robustness import (
            RobustOptimizationConfig,
            build_context_tensor,
            resolve_context_points,
        )

        cfg = RobustOptimizationConfig(context_groups=["ctx"], context_samples=4)
        resolve_context_points(cfg, self._params())
        tensor = build_context_tensor(cfg, self._params())
        assert tensor.shape == (4, 2)
        assert tensor.dtype == torch.float64
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


# ---------------------------------------------------------------------------
# RobustAcquisition
# ---------------------------------------------------------------------------

class TestRobustAcquisition:
    def test_import_and_subclass(self):
        from ax.generators.torch.botorch_modular.acquisition import Acquisition

        from axbo_extensions.robustness import RobustAcquisition

        assert issubclass(RobustAcquisition, Acquisition)


class TestAugmentGeneratorSpecs:
    """The wiring spine: augment_generator_specs + cycle_context on a real Client."""

    def _client_and_params(self):
        from ax import Client, RangeParameterConfig
        from ax.adapter.registry import Generators
        from ax.generation_strategy.generation_node import GenerationNode
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        from ax.generation_strategy.generator_spec import GeneratorSpec
        from ax.generation_strategy.transition_criterion import MinTrials

        c = Client()
        c.configure_experiment(parameters=[
            RangeParameterConfig(name="x1", parameter_type="float", bounds=(0.0, 1.0)),
            RangeParameterConfig(name="x2", parameter_type="float", bounds=(0.0, 1.0)),
            RangeParameterConfig(name="w1", parameter_type="float", bounds=(0.0, 10.0)),
            RangeParameterConfig(name="w2", parameter_type="float", bounds=(100.0, 200.0)),
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
        params = [
            P("x1", (0.0, 1.0)), P("x2", (0.0, 1.0)),
            P("w1", (0.0, 10.0), groups=["env"]), P("w2", (100.0, 200.0), groups=["env"]),
        ]
        return c, params

    def _mbm_spec(self, client):
        from ax.adapter.registry import Generators

        for node in client._generation_strategy._nodes:
            for spec in node.generator_specs:
                if spec.generator_enum == Generators.BOTORCH_MODULAR:
                    return spec
        return None

    def test_robust_wiring_and_cycle(self):
        from axbo_extensions.robustness import (
            RobustAcquisition,
            RobustOptimizationConfig,
            SubstituteContextFeatures,
            augment_generator_specs,
            cycle_context,
        )

        c, params = self._client_and_params()
        cfg = RobustOptimizationConfig(context_groups=["env"], context_samples=5, robustness=0.7)
        state = augment_generator_specs(c, params, cfg, multifidelity=False)

        assert state["n_w"] == 5
        assert state["is_moo"] is True  # 2 objectives -> MARS branch
        assert state["ctx_indices"] == [2, 3]
        assert state["risk_config"]["alpha"] == pytest.approx(0.7)

        gk = self._mbm_spec(c).generator_kwargs
        assert gk["acquisition_class"] is RobustAcquisition
        assert gk["acquisition_options"]["risk_config"]["is_moo"] is True
        mc = gk["surrogate_spec"].model_configs[0]
        assert SubstituteContextFeatures in mc.input_transform_classes

        # fixed_features pinned to context point 0, in RAW parameter space (w1 in [0,10],
        # w2 in [100,200]) — NOT the normalized feature_set (that would be out of bounds).
        fixed0 = self._mbm_spec(c).fixed_features
        assert set(fixed0.parameters) == {"w1", "w2"}
        assert 0.0 <= fixed0.parameters["w1"] <= 10.0
        assert 100.0 <= fixed0.parameters["w2"] <= 200.0

        # cycle to context point 1 -> fixed_features match raw_context[1]
        cycle_context(c, state, trial_index=1)
        fixed1 = self._mbm_spec(c).fixed_features
        expected = {
            state["ctx_names"][i]: float(state["raw_context"][1, i])
            for i in range(len(state["ctx_names"]))
        }
        assert fixed1.parameters == pytest.approx(expected)
        assert 100.0 <= fixed1.parameters["w2"] <= 200.0  # raw, not normalized [0,1]

    def _soo_client_and_params(self):
        from ax import Client, RangeParameterConfig
        from ax.adapter.registry import Generators
        from ax.generation_strategy.generation_node import GenerationNode
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        from ax.generation_strategy.generator_spec import GeneratorSpec
        from ax.generation_strategy.transition_criterion import MinTrials

        c = Client()
        c.configure_experiment(parameters=[
            RangeParameterConfig(name="x", parameter_type="float", bounds=(0.0, 1.0)),
            RangeParameterConfig(name="w", parameter_type="float", bounds=(0.0, 1.0)),
        ])
        c.configure_optimization(objective="-obj")  # single objective -> CVaR branch
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
        params = [P("x", (0.0, 1.0)), P("w", (0.0, 1.0), groups=["env"])]
        return c, params

    def test_robust_soo_generates_trial(self):
        """CVaR (SOO) end-to-end regression: the one-to-many context transform is
        incompatible with qLogEI's best_f-from-Y constructor and with Ax's in-sample
        best_point. Both must be handled (acqf -> qLogNEI; best_point -> NotImplemented),
        else MBM generation raises. Would crash before those fixes."""
        from axbo_extensions.robustness import (
            RobustAcquisition,
            RobustOptimizationConfig,
            augment_generator_specs,
        )

        c, params = self._soo_client_and_params()
        cfg = RobustOptimizationConfig(context_groups=["env"], context_samples=5, robustness=0.8)
        state = augment_generator_specs(c, params, cfg, multifidelity=False)
        assert state["is_moo"] is False  # 1 objective -> CVaR branch
        assert self._mbm_spec(c).generator_kwargs["acquisition_class"] is RobustAcquisition

        # Complete Sobol trials, then generate through the MBM (robust CVaR) node.
        mbm_generated = False
        for _ in range(8):
            for idx, p in c.get_next_trials(max_trials=2).items():
                c.complete_trial(idx, {"obj": (p["x"] - 0.5) ** 2 + p["w"] * p["x"]})
            df = c.summarize()
            if (df["generation_node"] == "MBM").any():
                mbm_generated = True
                break
        assert mbm_generated, "MBM (CVaR) node never generated a trial"

    def test_multifid_robust_branch_uses_qmultifidhvkg_no_scf(self):
        """MultiFid+robust composition (§6): qMultiFidHVKG + STMFGP, NO explicit input
        transforms (Ax MF defaults — a forced Normalize/SCF breaks the fidelity dim)."""
        from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
            qMultiFidelityHypervolumeKnowledgeGradient,
        )
        from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

        from axbo_extensions.multifidelity import MultiFidHVKGAcquisition
        from axbo_extensions.robustness import (
            RobustOptimizationConfig,
            SubstituteContextFeatures,
            augment_generator_specs,
        )

        c, params = self._client_and_params()
        cfg = RobustOptimizationConfig(context_groups=["env"], context_samples=3)
        augment_generator_specs(c, params, cfg, multifidelity=True)
        gk = self._mbm_spec(c).generator_kwargs
        assert gk["botorch_acqf_class"] is qMultiFidelityHypervolumeKnowledgeGradient
        assert gk["acquisition_class"] is MultiFidHVKGAcquisition
        mc = gk["surrogate_spec"].model_configs[0]
        assert mc.botorch_model_class is SingleTaskMultiFidelityGP
        # no explicit input transforms -> Ax MF defaults (a _DefaultType sentinel), so no SCF (§6)
        itc = mc.input_transform_classes
        assert not isinstance(itc, list) or SubstituteContextFeatures not in itc
        assert gk["surrogate_spec"].allow_batched_models is False
