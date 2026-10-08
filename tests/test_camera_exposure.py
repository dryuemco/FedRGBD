"""src/data/camera_exposure.py: one fixed (exposure, gain, white balance) per camera
(prereg 13.1 draft).  White balance first (|R - B| in the neutral rectangle), then the
exposure (224 mean luma in [100, 130]), then a verification; fake cameras only.
"""

import json

import numpy as np
import pytest

from src.data import camera_capture_common as ccc
from src.data import camera_exposure as cex
from tests.test_camera_capture import NOFIRE_NOTES, FIRE_NOTES, FakeCamera, FakeClock, _capture

LAMPS = "lamps_on=tavan lambasi"
CAL_NOTES = NOFIRE_NOTES + "; " + LAMPS


class SceneCamera(FakeCamera):
    """Uniform frames: level = k * exposure; the white balance tilts R against B,
    neutral (R == B) at ``neutral_wb``."""

    def __init__(self, k=0.4, neutral_wb=4800.0, wb_dir=1.0, **kw):
        super().__init__(**kw)
        self.k, self.neutral_wb, self.wb_dir = k, neutral_wb, wb_dir
        self.setting = {"exposure": 0.0, "white_balance": neutral_wb}
        self.log = []

    def apply_lock(self, lock):
        self.setting = dict(lock)
        self.log.append(dict(lock))
        return super().apply_lock(lock)

    def grab(self):
        fr = super().grab()
        level = min(250.0, self.k * float(self.setting["exposure"])
                    * (1.0 + float(self.setting["gain"]) / 64.0) / 2.0)
        tilt = self.wb_dir * 0.5 * (float(self.setting["white_balance"]) - self.neutral_wb) / 1000.0
        fr.rgb[..., 0] = np.uint8(np.clip(level * (1 + tilt), 0, 255))
        fr.rgb[..., 1] = np.uint8(np.clip(level, 0, 255))
        fr.rgb[..., 2] = np.uint8(np.clip(level * (1 - tilt), 0, 255))
        return fr


def _cal(tmp_path, cam, node="node_b", **kw):
    args = dict(notes=CAL_NOTES, log=lambda m: None)
    args.update(kw)
    return ccc.run_exposure_calibration(cam, str(tmp_path), node, **args)


def test_white_balance_then_exposure_then_verification(tmp_path):
    cam = SceneCamera(k=0.8, clock=FakeClock())      # default 166 -> luma 66: too dark
    rec = _cal(tmp_path, cam)
    steps = [t["step"] for t in rec["trace"]]
    assert steps == sorted(steps, key=lambda s: s != "white_balance")   # WB first
    assert abs(rec["white_balance"] - 4800.0) <= 10.0                     # |R - B| minimal
    wb_steps = [t for t in rec["trace"] if t["step"] == "white_balance"]
    assert all(t["setting"]["exposure"] == 166.0 and t["setting"]["gain"] == 64.0
               for t in wb_steps)                       # default exposure and gain
    assert rec["gain"] == 64.0                          # the camera's default gain
    assert 100 <= rec["luma_224_median"] <= 130
    v = rec["verification"]
    assert v["setting"] == {"exposure": rec["exposure"], "gain": 64.0,
                            "white_balance": rec["white_balance"]}
    assert abs(v["r_minus_b"]) <= 2.0 and len(v["region_rgb_mean"]) == 3
    assert rec["lamps_on"] == "tavan lambasi" and rec["flame_frames_used"] is False
    assert rec["wb_region"] == {"x0": 0.03, "x1": 0.30, "y0": 0.10, "y1": 0.30}
    assert cam.closed
    stored = json.loads((tmp_path / "node_b" / "_exposure" / "exposure.json").read_text())
    assert stored["exposure"] == rec["exposure"] and stored["serial"] == "123"


def test_the_search_is_deterministic(tmp_path):
    a = _cal(tmp_path / "a", SceneCamera(clock=FakeClock()))
    b = _cal(tmp_path / "b", SceneCamera(clock=FakeClock()))
    key = lambda r: [(t["step"], t["setting"], t["luma_224_median"], t["r_minus_b"])
                     for t in r["trace"]]
    assert key(a) == key(b) and (a["exposure"], a["white_balance"]) == (b["exposure"],
                                                                        b["white_balance"])


def test_brighter_than_target_lowers_the_exposure(tmp_path):
    rec = _cal(tmp_path, SceneCamera(k=2.4, clock=FakeClock()))   # 166 -> luma 199
    assert rec["exposure"] < 166 and 100 <= rec["luma_224_median"] <= 130


@pytest.mark.parametrize("notes,match", [
    (FIRE_NOTES + "; " + LAMPS, "flame-free"),
    (NOFIRE_NOTES, "lamps_on"),
])
def test_calibration_scene_is_flame_free_and_names_its_lamps(tmp_path, notes, match):
    with pytest.raises(cex.ExposureError, match=match):
        _cal(tmp_path, SceneCamera(clock=FakeClock()), notes=notes)


def test_a_camera_without_a_neutral_rectangle_is_not_calibrated(tmp_path):
    # a region file in which this camera has no rectangle yet (independent of the
    # committed configs/camera_wb_region.json, which fills them in camera by camera)
    cfg = tmp_path / "wb.json"
    cfg.write_text(json.dumps({"regions": {"node_b": None}}))
    with pytest.raises(cex.ExposureError, match="no neutral white-balance rectangle"):
        cex.load_wb_region("node_b", str(cfg))
    with pytest.raises(cex.ExposureError, match="no neutral white-balance rectangle"):
        cex.load_wb_region("node_x", str(cfg))


def test_one_setting_per_camera_unless_recalibrated(tmp_path):
    _cal(tmp_path, SceneCamera(clock=FakeClock()))
    with pytest.raises(cex.ExposureError, match="one setting per camera"):
        _cal(tmp_path, SceneCamera(clock=FakeClock()))
    rec = _cal(tmp_path, SceneCamera(k=2.4, clock=FakeClock()), recalibrate=True)
    old = list((tmp_path / "node_b" / "_exposure" / "_superseded").iterdir())
    assert len(old) == 1 and rec["exposure"] < 166


def test_too_dark_at_the_exposure_cap_raises_the_gain(tmp_path):
    # k = 0.25: 333 at gain 64 gives luma 83, so the exposure stays at the cap (333) and
    # the gain goes up until the luma is in [100, 130]
    cam = SceneCamera(k=0.25, clock=FakeClock())
    rec = _cal(tmp_path, cam)
    assert rec["exposure"] == 333.0 and rec["gain"] > 64.0
    assert 100 <= rec["luma_224_median"] <= 130
    steps = [t["step"] for t in rec["trace"]]
    assert steps.index("gain") > max(i for i, s in enumerate(steps) if s == "exposure")
    assert all(t["setting"]["exposure"] == 333.0 for t in rec["trace"] if t["step"] == "gain")


def test_unreachable_target_is_an_error_and_writes_nothing(tmp_path):
    with pytest.raises(cex.ExposureError, match="no exposure"):
        _cal(tmp_path, SceneCamera(k=0.001, clock=FakeClock()))
    assert not (tmp_path / "node_b" / "_exposure" / "exposure.json").exists()


def test_a_lamp_change_refuses_the_capture(tmp_path):
    _cal(tmp_path, SceneCamera(clock=FakeClock()))
    base = dict(node="node_b", study_gates=True, exposure_lock=True)
    rec, cam = _capture(tmp_path, notes=NOFIRE_NOTES + "; " + LAMPS, **base)
    assert rec["status"] == "complete" and rec["exposure_lock"]["how"] == "fixed_camera"
    with pytest.raises(ccc.CaptureError, match="recalibrate"):
        _capture(tmp_path, notes=NOFIRE_NOTES + "; lamps_on=tavan lambasi, masa lambasi",
                 retake=True, **base)


def test_zed_gain_starts_at_0_and_white_balance_is_searched_on_its_grid(tmp_path):
    class ZedLike(SceneCamera):
        def exposure_defaults(self):
            return {"gain": 0.0,
                    "gain_range": {"min": 0.0, "max": 100.0, "step": 1.0, "default": 0.0},
                    "white_balance_range": {"min": 2800.0, "max": 6500.0, "step": 100.0,
                                            "default": None},
                    "exposure_range": {"min": 1.0, "max": 100.0, "step": 1.0,
                                       "default": None}}
    rec = _cal(tmp_path, ZedLike(k=8.0, clock=FakeClock()))
    assert rec["gain"] == 0.0 and rec["white_balance"] % 100 == 0
    assert abs(rec["white_balance"] - 4800.0) <= 100.0
    assert 100 <= rec["luma_224_median"] <= 130


def test_rectangle_and_224_luma_measure_what_they_say():
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    rgb[:, :420] = 255                                 # outside the 224 centre square only
    assert cex.mean_luma_224(rgb) < 1.0
    r = {"x0": 0.25, "x1": 0.50, "y0": 0.08, "y1": 0.40}
    assert cex.region_box(1920, 1080, r) == (480, 86, 960, 432)
    rgb[86:432, 480:960] = (200, 100, 50)
    assert cex.region_rgb(rgb, r) == (200.0, 100.0, 50.0)


def _check(tmp_path, cam, node="node_b", notes=CAL_NOTES):
    return ccc.run_session_check(cam, str(tmp_path), node, notes, log=lambda m: None)


def test_session_check_passes_within_10_percent_and_is_logged(tmp_path):
    rec = _cal(tmp_path, SceneCamera(k=0.8, clock=FakeClock()))
    chk = _check(tmp_path, SceneCamera(k=0.8, clock=FakeClock()))
    assert chk["session_check"] == "PASS"
    assert chk["setting"] == {"exposure": rec["exposure"], "gain": rec["gain"],
                              "white_balance": rec["white_balance"]}
    ref = rec["luma_224_median"]
    assert chk["band"] == [round(0.9 * ref, 3), round(1.1 * ref, 3)]
    lines = (tmp_path / "node_b" / "_exposure" / "session_checks.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["session_check"] == "PASS"


def test_session_check_fails_outside_the_band(tmp_path):
    _cal(tmp_path, SceneCamera(k=0.8, clock=FakeClock()))
    chk = _check(tmp_path, SceneCamera(k=0.6, clock=FakeClock()))   # 25 % darker scene
    assert chk["session_check"] == "FAIL -> recalibrate"
    assert len((tmp_path / "node_b" / "_exposure" / "session_checks.jsonl")
               .read_text().splitlines()) == 1


@pytest.mark.parametrize("notes,match", [
    (FIRE_NOTES + "; " + LAMPS, "flame-free"),
    (NOFIRE_NOTES + "; lamps_on=tavan lambasi, masa lambasi", "recalibrate"),
])
def test_session_check_refuses_flame_or_other_lamps(tmp_path, notes, match):
    _cal(tmp_path, SceneCamera(clock=FakeClock()))
    with pytest.raises(cex.ExposureError, match=match):
        _check(tmp_path, SceneCamera(clock=FakeClock()), notes=notes)


def test_session_check_same_measurement_for_zed_and_realsense(tmp_path):
    class ZedLike(SceneCamera):
        def exposure_defaults(self):
            return {"gain": 0.0,
                    "gain_range": {"min": 0.0, "max": 100.0, "step": 1.0, "default": 0.0},
                    "white_balance_range": {"min": 2800.0, "max": 6500.0, "step": 100.0,
                                            "default": None},
                    "exposure_range": {"min": 1.0, "max": 100.0, "step": 1.0,
                                       "default": None}}
    _cal(tmp_path, ZedLike(k=8.0, clock=FakeClock()), node="node_c")
    chk = _check(tmp_path, ZedLike(k=8.0, clock=FakeClock()), node="node_c")
    assert chk["session_check"] == "PASS" and chk["node"] == "node_c"


@pytest.mark.parametrize("wb_dir,neutral", [(1.0, 4800.0), (-1.0, 4800.0), (-1.0, 3500.0),
                                            (1.0, 6000.0)])
def test_white_balance_search_works_in_both_directions(tmp_path, wb_dir, neutral):
    # wb_dir -1: R - B falls with the setting, as on the ZED 2i (+47 at 2800, -38 at 6500)
    cam = SceneCamera(k=0.8, neutral_wb=neutral, wb_dir=wb_dir, clock=FakeClock())
    wb, trace = cex.search_white_balance(cam, 64.0, 166.0,
                                         {"min": 2800.0, "max": 6500.0, "step": 10.0},
                                         {"x0": 0.1, "x1": 0.4, "y0": 0.1, "y1": 0.3},
                                         log=lambda m: None)
    assert abs(wb - neutral) <= 20.0
    assert abs(trace[0]["r_minus_b"]) > 5 and abs(trace[1]["r_minus_b"]) > 5   # both ends


def test_white_balance_without_a_sign_change_takes_the_better_end(tmp_path):
    cam = SceneCamera(k=0.8, neutral_wb=2000.0, wb_dir=-1.0, clock=FakeClock())  # below range
    wb, _ = cex.search_white_balance(cam, 64.0, 166.0,
                                     {"min": 2800.0, "max": 6500.0, "step": 100.0},
                                     {"x0": 0.1, "x1": 0.4, "y0": 0.1, "y1": 0.3},
                                     log=lambda m: None)
    assert wb == 2800.0
