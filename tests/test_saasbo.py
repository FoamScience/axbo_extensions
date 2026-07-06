"""SAASBO engagement predicate (#18)."""

from __future__ import annotations

from types import SimpleNamespace

from axbo_extensions.saasbo import saasbo_model_class, should_use_saasbo


def _cfg(enabled, min_dims):
    return SimpleNamespace(enabled=enabled, min_dims=min_dims)


def test_predicate():
    assert should_use_saasbo(20, _cfg(True, 15))
    assert not should_use_saasbo(10, _cfg(True, 15))   # below threshold
    assert not should_use_saasbo(20, _cfg(False, 15))  # disabled
    assert not should_use_saasbo(20, None)             # no config


def test_model_class_swap():
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    assert saasbo_model_class(20, _cfg(True, 15)) is SaasFullyBayesianSingleTaskGP
    assert saasbo_model_class(10, _cfg(True, 15)) is None
