"""Region-screening inactivity decision core (#19)."""

from __future__ import annotations

from axbo_extensions.regionscreening import next_streak, region_inactive


def test_inactive_absolute():
    r = region_inactive([1.0, 1.02, 0.99], min_delta=0.1)
    assert r["inactive"] and r["spread"] < 0.1


def test_active_absolute():
    assert not region_inactive([1.0, 5.0], min_delta=0.1)["inactive"]


def test_fractional_threshold():
    # spread 0.1, ref ~10, frac 0.05 -> threshold 0.5 -> inactive
    assert region_inactive([10.0, 10.1], min_delta_frac=0.05)["inactive"]


def test_streak():
    assert next_streak(2, True) == 3
    assert next_streak(2, False) == 0


def test_shrunk_bounds():
    from axbo_extensions.regionscreening import shrunk_bounds

    assert shrunk_bounds(0.0, 10.0, 5.0, 0.5) == (2.5, 7.5)   # centered
    assert shrunk_bounds(0.0, 10.0, 0.5, 0.5) == (0.0, 3.0)   # clamped at lower bound


def test_maybe_screen_regions_advise():
    from types import SimpleNamespace

    from axbo_extensions.regionscreening import maybe_screen_regions

    region = SimpleNamespace(group="g", min_delta=0.1, min_delta_frac=None)
    cfg = SimpleNamespace(
        enabled=True, mode="advise", after_trials=3, confirm_passes=1,
        regions=[region], shrink_factor=0.5, max_shrinkable=None,
    )
    scalars = {"g": {0: 1.0, 1: 1.02, 2: 0.99}}  # spread 0.03 < 0.1 -> inactive
    state: dict = {}
    maybe_screen_regions(None, cfg, scalars, {"a": ["g"], "b": ["g"]}, state)  # advise: client unused
    assert state["advisories"] == [{"group": "g", "params": ["a", "b"]}]
    assert "g" in state["shrunk_groups"]
