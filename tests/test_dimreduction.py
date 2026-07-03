"""Dimensionality-reduction fix-selection (#17). The pure decision; the Sobol/CV/model
paths are Ax-private and validated on a real high-dim run."""

from __future__ import annotations

from axbo_extensions.dimreduction import select_params_to_fix


def test_fixes_below_threshold():
    imp = {"a": 0.5, "b": 0.01, "c": 0.4, "d": 0.02}
    # b, d are below 0.05; cap = min(4*0.5, 3) = 2
    assert set(select_params_to_fix(imp, 0.05, 0.5)) == {"b", "d"}


def test_cap_keeps_at_least_one_active():
    imp = {"a": 0.0, "b": 0.0, "c": 0.0}
    # all below threshold, but cap = min(3*1.0, 2) = 2 -> at least one stays active
    assert len(select_params_to_fix(imp, 0.05, 1.0)) == 2


def test_none_below_threshold():
    imp = {"a": 0.5, "b": 0.6}
    assert select_params_to_fix(imp, 0.05, 0.5) == []
