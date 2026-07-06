"""Trial registry rebuild (Feature B foundation, #14)."""

from __future__ import annotations


def _client_with_one_completed_trial():
    from ax import Client, RangeParameterConfig

    c = Client()
    c.configure_experiment(
        parameters=[RangeParameterConfig(name="x", parameter_type="float", bounds=(0.0, 1.0))]
    )
    c.configure_optimization(objective="-m")
    idx = c.attach_trial(parameters={"x": 0.5})
    c.complete_trial(trial_index=idx, raw_data={"m": 0.3})
    return c, idx


def test_rebuild_trial_registry():
    from axbo_extensions.trialdeps import rebuild_trial_registry

    client, idx = _client_with_one_completed_trial()
    reg = rebuild_trial_registry(client)
    assert reg[idx]["status"] == "COMPLETED"
    assert reg[idx]["parameters"]["x"] == 0.5
    # executor hasn't stamped an artifact_ref yet
    assert reg[idx]["artifact_ref"] is None


from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402


def _registry():
    return {
        0: {"status": "COMPLETED", "artifact_ref": "a0", "parameters": {"x": 0.0, "fid": 0.0}},
        1: {"status": "COMPLETED", "artifact_ref": "a1", "parameters": {"x": 0.9, "fid": 1.0}},
        2: {"status": "COMPLETED", "artifact_ref": "a2", "parameters": {"x": 0.5, "fid": 0.0}},
        3: {"status": "RUNNING", "artifact_ref": None, "parameters": {"x": 0.4, "fid": 0.0}},
    }


def _sel(**kw):
    kw.setdefault("fidelity_filter", "any")
    return SimpleNamespace(**kw)


def test_selectors():
    from axbo_extensions.trialdeps import resolve_source_trial

    reg = _registry()
    tgt = {"x": 0.45, "fid": 0.0}
    assert resolve_source_trial(_sel(strategy="latest"), tgt, reg) == (2, "a2")  # 3 is RUNNING
    assert resolve_source_trial(_sel(strategy="baseline"), tgt, reg) == (0, "a0")
    assert resolve_source_trial(_sel(strategy="by_index", index=1), tgt, reg) == (1, "a1")
    # nearest to x=0.45 is trial 2 (x=0.5)
    assert resolve_source_trial(_sel(strategy="nearest"), tgt, reg)[0] == 2


def test_best_needs_objectives_bug_fix():
    """foamBO silently degraded 'best' to lowest index; we skip loudly without objectives."""
    from axbo_extensions.trialdeps import resolve_source_trial

    reg = _registry()
    tgt = {"x": 0.45}
    assert resolve_source_trial(_sel(strategy="best"), tgt, reg) is None  # no objectives -> skip
    # with objectives, picks the minimum (trial 1)
    objs = {0: 5.0, 1: 1.0, 2: 3.0}
    assert resolve_source_trial(_sel(strategy="best"), tgt, reg, objectives=objs) == (1, "a1")


def test_fidelity_filter_same():
    from axbo_extensions.trialdeps import resolve_source_trial

    reg = _registry()
    tgt = {"x": 0.45, "fid": 0.0}
    # 'same' fidelity (fid=0.0) excludes trial 1 (fid=1.0) -> latest same-fid is trial 2
    sel = _sel(strategy="latest", fidelity_filter="same")
    assert resolve_source_trial(sel, tgt, reg, fidelity_param="fid") == (2, "a2")


def test_build_depends_on_and_fallback():
    from axbo_extensions.trialdeps import build_depends_on

    reg = _registry()
    tgt = {"x": 0.45, "fid": 0.0}
    dep = SimpleNamespace(name="warm", enabled=True, source=_sel(strategy="latest"))
    out = build_depends_on([dep], tgt, reg)
    assert out == [{"name": "warm", "resolved": True, "source_trial_index": 2, "source_artifact_ref": "a2"}]

    # unresolved by_index(99): skip records resolved False; error raises
    skip_dep = SimpleNamespace(name="d", enabled=True, source=_sel(strategy="by_index", index=99, fallback="skip"))
    assert build_depends_on([skip_dep], tgt, reg) == [{"name": "d", "resolved": False}]
    err_dep = SimpleNamespace(name="d", enabled=True, source=_sel(strategy="by_index", index=99, fallback="error"))
    with pytest.raises(RuntimeError):
        build_depends_on([err_dep], tgt, reg)
