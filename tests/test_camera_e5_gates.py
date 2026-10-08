"""Study-capture gates of the camera prereg's section 13 draft (e5 code), fake cameras only.

Mandatory parsed --notes, the USB 3.x gate, the exposure lock (one fixed exposure per
camera, the same for fire and no-fire) with its auto-exposure gate, their checks in the
orchestrator, and the flame-height tool.  No camera SDK is needed.
"""

import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

from src.data import camera_capture_common as ccc
from tests.test_camera_capture import (
    FAKE_NODES, FIRE_NOTES, NOFIRE_NOTES, FakeCamera, FakeClock, _calibrate, _capture, _rec,
    _session,
)


def _study(tmp_path, label, camera=None, clock=None, **kw):
    clock = clock or FakeClock()
    notes = FIRE_NOTES if label == "fire" else NOFIRE_NOTES
    calibrated = kw.pop("calibrated", True)
    args = dict(label=label, notes=notes, study_gates=True, exposure_lock=True, clock=clock)
    args.update(kw)
    node = args.get("node", "node_a")
    if args["exposure_lock"] and calibrated:
        from src.data.camera_exposure import exposure_path
        if not os.path.isfile(exposure_path(str(tmp_path), node)):
            cam = args.get("camera") or camera
            _calibrate(tmp_path, node, serial=cam.info().get("serial") if cam else "123")
    return _capture(tmp_path, camera=camera or FakeCamera(clock=clock), **args)


# --------------------------------------------------------------------------- #
# --notes
# --------------------------------------------------------------------------- #
def test_parse_notes():
    assert ccc.parse_notes(" source=torch1 ;distance_measured_m=3.02; flame_height_cm=12 ; "
                           "lamp left on") == {"source": "torch1", "distance_measured_m": "3.02",
                                               "flame_height_cm": "12", "text": "lamp left on"}
    assert ccc.parse_notes(None) == {} and ccc.parse_notes("") == {}


@pytest.mark.parametrize("notes,label,match", [
    ("", "fire", "lacks source"),
    ("source=torch1; flame_height_cm=12", "fire", "lacks distance_measured_m"),
    ("source=torch1; distance_measured_m=3", "fire", "lacks flame_height_cm"),
    ("source=none; distance_measured_m=3; flame_height_cm=12", "fire", "source=none"),
    ("source=torch1; distance_measured_m=3; flame_height_cm=0", "fire", "flame_height_cm > 0"),
    ("source=torch1; distance_measured_m=3; flame_height_cm=0", "no_fire", "source=none"),
    ("source=none; distance_measured_m=3; flame_height_cm=5", "no_fire", "flame_height_cm=0"),
    ("source=t; distance_measured_m=three; flame_height_cm=5", "fire", "not a number"),
    ("source=t; distance_measured_m=30; flame_height_cm=5", "fire", "outside"),
    ("source=t; distance_measured_m=3; flame_height_cm=tall", "fire", "not a number"),
])
def test_notes_problems(notes, label, match):
    problems = ccc.notes_problems(notes, label)
    assert problems and any(match in p for p in problems), problems


def test_complete_notes_pass():
    assert ccc.notes_problems(FIRE_NOTES, "fire") == []
    assert ccc.notes_problems(NOFIRE_NOTES, "no_fire") == []


def test_study_capture_without_notes_is_refused_before_anything_is_written(tmp_path):
    cam = FakeCamera()
    with pytest.raises(ccc.CaptureError, match="refused before it started"):
        _study(tmp_path, "fire", camera=cam, notes="source=torch1", calibrated=False)
    assert not cam.opened
    assert not (tmp_path / "node_a").exists()


# --------------------------------------------------------------------------- #
# USB 3.x
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("info,ok", [
    ({"usb_type": "3.2"}, True), ({"usb_type": "3.1"}, True), ({"usb_type": "2.1"}, False),
    ({"usb_type": None}, False), ({}, False), ({"usb_type": "unknown"}, False),
    ({"usb_speed_mbps": 5000.0}, True), ({"usb_speed_mbps": 480.0}, False),
    ({"usb_speed_mbps": 480.0, "usb_type": "3.2"}, False),   # the measured speed wins
])
def test_usb_gate(info, ok):
    assert (ccc.usb_problems(info) == []) == ok


def test_usb2_camera_is_refused_and_recorded(tmp_path):
    clock = FakeClock()
    with pytest.raises(ccc.CaptureError, match="USB 3.x gate"):
        _study(tmp_path, "no_fire", camera=FakeCamera(clock=clock, usb_type="2.1"), clock=clock)
    rec = json.loads((tmp_path / "node_a" / "_captures" / "s03_no_fire_d200.json").read_text())
    assert rec["status"] == "incomplete" and rec["n_frames_written"] == 0
    assert rec["usb"]["problems"] == ["USB type 2.1, need 3.x"]


def test_zed_usb_devices_from_sysfs(tmp_path):
    def dev(name, vendor, product, speed, prod="ZED 2i"):
        d = tmp_path / name
        d.mkdir()
        for k, v in (("idVendor", vendor), ("idProduct", product), ("speed", speed),
                     ("product", prod)):
            (d / k).write_text(v + "\n")
    dev("2-1", "2b03", "f880", "5000")
    dev("1-2.1", "2b03", "f881", "12", prod="ZED 2i HID")
    dev("1-3", "8086", "0b3a", "5000", prod="Intel RealSense")
    devs = ccc.zed_usb_devices(str(tmp_path))
    assert [d["id_product"] for d in devs] == ["f881", "f880"]
    assert ccc.zed_usb_speed_mbps(devs) == 5000.0
    assert ccc.zed_usb_devices(str(tmp_path / "missing")) == []
    assert ccc.zed_usb_speed_mbps([]) is None


# --------------------------------------------------------------------------- #
# exposure lock
# --------------------------------------------------------------------------- #
def test_fire_and_no_fire_use_the_one_fixed_exposure_of_the_camera(tmp_path):
    _calibrate(tmp_path, "node_a", exposure=210, gain=64, white_balance=4600)
    rec_n, cam_n = _study(tmp_path, "no_fire")
    rec_f, cam_f = _study(tmp_path, "fire")
    for rec, cam in ((rec_n, cam_n), (rec_f, cam_f)):
        assert rec["status"] == "complete" and rec["ae_on_frames"] == 0
        assert rec["exposure_lock"]["how"] == "fixed_camera"
        assert cam.applied == {"exposure": 210, "gain": 64, "white_balance": 4600}
    meta = json.loads((tmp_path / "node_a" / "s03_fire_d200_0000_meta.json").read_text())
    assert meta["exposure_source"] == "manual_lock" and meta["auto_exposure"] is False
    assert meta["exposure_lock"] == {"exposure": 210, "gain": 64, "white_balance": 4600}
    # another distance and another scene: the same values, nothing determined per capture
    rec_o, cam_o = _study(tmp_path, "no_fire", scene="s07", distance_m=3.0)
    assert cam_o.applied == {"exposure": 210, "gain": 64, "white_balance": 4600}
    assert not (tmp_path / "node_a" / "_locks").exists()


@pytest.mark.parametrize("label", ["no_fire", "fire"])
def test_a_capture_without_a_calibrated_exposure_is_refused(tmp_path, label):
    with pytest.raises(ccc.CaptureError, match="no fixed exposure"):
        _study(tmp_path, label, calibrated=False)


def test_an_exposure_file_of_another_camera_is_refused(tmp_path):
    _calibrate(tmp_path, "node_a", serial="999")
    with pytest.raises(ccc.CaptureError, match="calibrated on camera 999"):
        _study(tmp_path, "no_fire")


def test_auto_exposure_on_in_a_locked_capture_fails_it(tmp_path):
    clock = FakeClock()
    rec, _ = _study(tmp_path, "no_fire", camera=FakeCamera(clock=clock, ae_stuck=True),
                    clock=clock)
    assert rec["status"] == "incomplete" and rec["ae_on_frames"] == 4
    assert "exposure lock gate: auto exposure on in 4" in rec["notes"]


def test_unreported_auto_exposure_flag_fails_a_locked_capture(tmp_path):
    class NoFlag(FakeCamera):
        def grab(self):
            fr = super().grab()
            fr.meta.pop("auto_exposure")
            return fr
    clock = FakeClock()
    rec, _ = _study(tmp_path, "no_fire", camera=NoFlag(clock=clock), clock=clock)
    assert rec["status"] == "incomplete" and rec["ae_unknown_frames"] == 4


def test_a_lock_the_camera_does_not_take_is_refused(tmp_path):
    clock = FakeClock()
    with pytest.raises(ccc.CaptureError, match="exposure lock not applied"):
        _study(tmp_path, "no_fire", camera=FakeCamera(clock=clock, lock_offset=50), clock=clock)


def test_unlocked_capture_records_no_lock(tmp_path):
    rec, _ = _study(tmp_path, "no_fire", exposure_lock=False)
    assert rec["exposure_lock"] is None and rec["ae_on_frames"] is None
    assert rec["status"] == "complete"


# --------------------------------------------------------------------------- #
# back-ends (fake SDK objects)
# --------------------------------------------------------------------------- #
def test_realsense_exposure_defaults_and_lock(monkeypatch):
    from src.data import realsense_capture as rsc

    opts = NS(enable_auto_exposure="ae", enable_auto_white_balance="awb", exposure="exp",
              gain="gain", white_balance="wb")
    monkeypatch.setattr(rsc, "rs", NS(option=opts))

    class Sensor:
        def __init__(self):
            self.v = {"ae": 1.0, "awb": 1.0, "exp": 156.0, "gain": 64.0, "wb": 4603.0}
            self.log = []

        def supports(self, a):
            return True

        def get_option(self, a):
            return self.v[a]

        def set_option(self, a, value):
            self.log.append((a, value))
            self.v[a] = value

        def get_option_range(self, a):
            return {"wb": NS(min=2800.0, max=6500.0, step=10.0, default=4600.0),
                    "gain": NS(min=0.0, max=128.0, step=1.0, default=64.0)}.get(
                a, NS(min=1.0, max=10000.0, step=1.0, default=166.0))

    b = rsc.RealSenseBackend()
    b.color_sensor = Sensor()
    d = b.exposure_defaults()
    assert d["gain"] == 64.0 and d["gain_range"]["max"] == 128.0
    assert d["white_balance_range"] == {"min": 2800.0, "max": 6500.0, "step": 10.0,
                                        "default": 4600.0}
    # 30 fps stream: exposure (100 us units) capped at one frame period
    assert d["exposure_range"] == {"min": 1.0, "max": 333.0, "step": 1.0, "default": 166.0}
    applied = b.apply_lock({"exposure": 210.0, "gain": 64.0, "white_balance": 4600.0})
    assert applied["exposure"] == 210.0 and applied["auto_exposure"] == 0.0
    assert applied["white_balance"] == 4600.0 and applied["auto_white_balance"] == 0.0


def test_zed_exposure_defaults_and_lock(monkeypatch):
    from src.data import zed_capture as zc

    class Zed:
        def __init__(self):
            self.v = {"AEC_AGC": 1, "WHITEBALANCE_AUTO": 1, "EXPOSURE": 37, "GAIN": 52,
                      "WHITEBALANCE_TEMPERATURE": 4620}

        def get_camera_settings(self, key):
            return ("SUCCESS", self.v[key])

        def set_camera_settings(self, key, value):
            self.v[key] = value
            return "SUCCESS"

    sl = NS(VIDEO_SETTINGS=NS(**{k: k for k in ("AEC_AGC", "WHITEBALANCE_AUTO", "EXPOSURE",
                                                  "GAIN", "WHITEBALANCE_TEMPERATURE")}),
            ERROR_CODE=NS(SUCCESS="SUCCESS"))
    b = zc.ZedBackend()
    b.sl, b.zed, b.runtime = sl, Zed(), None
    d = b.exposure_defaults()
    assert d["gain"] == 0.0 and d["gain_range"]["max"] == 100.0   # starts at 0
    assert d["white_balance_range"]["step"] == 100.0
    assert d["exposure_range"]["min"] == 1.0 and d["exposure_range"]["max"] == 100.0
    lock = {"exposure": 30, "gain": 40, "white_balance": 4600}
    applied = b.apply_lock(lock)
    assert applied == {"exposure": 30, "gain": 40, "white_balance": 4600, "auto_exposure": 0,
                       "auto_white_balance": 0}
    b.zed.set_camera_settings = lambda k, v: "FAILURE"
    with pytest.raises(ccc.CaptureError, match="set_camera_settings"):
        b.apply_lock(lock)


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def test_session_refuses_incomplete_notes_before_anything_starts(monkeypatch):
    ccs = _session(monkeypatch)       # Popen / subprocess.run raise if anything starts
    out = []
    rc = ccs.main(["--scene", "s04", "--label", "fire", "--distance_m", "2",
                   "--notes", "source=torch1"], log=out.append)
    assert rc == 2
    assert "nothing was started" in "\n".join(out)
    assert any("lacks flame_height_cm" in l for l in out)


def test_session_no_lock_is_passed_to_every_node(monkeypatch):
    ccs = _session(monkeypatch)
    out = []
    ccs.main(["--scene", "s04", "--label", "no_fire", "--distance_m", "3", "--dry_run",
              "--no_lock", "--notes", NOFIRE_NOTES], clock=lambda: 100.0, log=out.append)
    caps = [l for l in out if " capture: " in l]
    assert len(caps) == 3 and all("--no_lock" in l for l in caps)


def test_verification_checks_status_usb_and_lock():
    from scripts.camera_capture_session import EXPECTED_SERIAL, verify_records

    s = 1000.0
    good = {n: _rec(s, s + 0.05, serial=EXPECTED_SERIAL[n]) for n in FAKE_NODES}
    assert verify_records(good, s, 40)[0]
    cases = [
        ({"status": "incomplete"}, "status incomplete"),
        ({"usb": {"problems": ["USB type 2.1, need 3.x"]}}, "USB type 2.1"),
        ({"usb": None}, "USB type not recorded"),
        ({"exposure_lock": None}, "no exposure lock"),
        ({"ae_on_frames": 3}, "auto exposure on in 3"),
        ({"ae_unknown_frames": 2}, "unreported in 2"),
    ]
    for change, reason in cases:
        rec = dict(good["node_b"], **change)
        ok, lines = verify_records(dict(good, node_b=rec), s, 40)
        assert not ok and reason in lines["node_b"], (change, lines["node_b"])
    # --no_lock (e5 tests only): an unlocked capture passes the other checks
    unlocked = dict(good["node_b"], exposure_lock=None, ae_on_frames=None)
    assert verify_records(dict(good, node_b=unlocked), s, 40, require_lock=False)[0]


def test_check_applies_the_usb_gate_to_the_probe(monkeypatch):
    ccs = _session(monkeypatch)
    from scripts.camera_capture_session import EXPECTED_SERIAL
    monkeypatch.setattr(ccs, "open_masters", lambda nodes, log=print: True)
    monkeypatch.setattr(ccs, "measure_offset", lambda cfg: (0.0, 0.001))
    probes = {
        "node_a": {"pyrealsense2": {"import": True, "cameras": [
            {"serial": EXPECTED_SERIAL["node_a"], "usb_type": "2.1"}]}},
        "node_b": {"pyrealsense2": {"import": True, "cameras": [
            {"serial": EXPECTED_SERIAL["node_b"], "usb_type": "3.2"}]}},
        "node_c": {"pyzed": {"import": True, "cameras": [
            {"serial": int(EXPECTED_SERIAL["node_c"]), "usb_speed_mbps": 5000.0}]}},
    }
    calls = iter(["node_a", "node_b", "node_c"])
    monkeypatch.setattr(ccs, "_run", lambda argv, timeout: NS(
        returncode=0, stdout=json.dumps(probes[next(calls)])))
    out = []
    assert ccs.run_check(FAKE_NODES, log=out.append) == 1
    line = {l.split()[0]: l for l in out if l.startswith("node_")}
    assert "USB type 2.1, need 3.x" in line["node_a"]
    assert line["node_b"].endswith("OK") and line["node_c"].endswith("OK")


# --------------------------------------------------------------------------- #
# flame-height tool
# --------------------------------------------------------------------------- #
def _write_capture(root, node, scene, label, dist_m, fy, size_hw, notes, flame_rows=0,
                   n=5):
    h, w = size_hw
    cid = ccc.make_capture_id(scene, label, dist_m)
    d = os.path.join(root, node)
    os.makedirs(os.path.join(d, "_captures"), exist_ok=True)
    for i in range(n):
        img = np.full((h, w, 3), 90, np.uint8)
        rows_i = flame_rows[i] if isinstance(flame_rows, (list, tuple)) else flame_rows
        if rows_i:
            top = h // 2 - rows_i // 2
            img[top:top + rows_i, w // 2 - 6:w // 2 + 6] = (250, 200, 60)
        Image.fromarray(img).save(os.path.join(d, ccc.make_frame_id(cid, i) + "_rgb.png"))
    rec = {"capture_id": cid, "n_frames_written": n, "notes": notes,
           "notes_parsed": ccc.parse_notes(notes),
           "intrinsics": {"rgb": {"fy": fy, "width": w, "height": h}}}
    with open(os.path.join(d, "_captures", cid + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)


def test_flame_height_tool_measures_at_224_and_compares_with_the_ruler(tmp_path, capsys):
    from scripts import camera_flame_height as cfh

    root = str(tmp_path / "camera_pilot")
    # 1080 rows -> 224: scale 224/1080; a 108-row flame is 22.4 px at 224
    fy = 1050.0
    for label, notes, rows in (("no_fire", NOFIRE_NOTES, 0), ("fire", FIRE_NOTES, 108)):
        _write_capture(root, "node_c", "s01", label, 3.0, fy, (1080, 1920), notes, rows)
    row, frame, reg = cfh.measure_node(root, "node_c", "s01", 3.0)
    f224 = fy * 224 / 1080
    assert row["fy_224"] == pytest.approx(f224, abs=0.01)
    assert row["expected_px"] == pytest.approx(0.12 * f224 / 3.02, abs=0.01)
    assert abs(row["measured_px_median"] - 22.4) <= 1.5
    assert row["n_frames_detected"] == row["n_fire_frames"] == 5
    assert not row["touches_crop_border"] and frame.shape == (224, 224, 3)
    assert reg is not None and row["ratio_measured_expected"] > 0
    assert row["criterion_px"] == 5 and row["criterion"] == "INCOMPLETE"   # 5 frames, not 40

    out_png = str(tmp_path / "overlay.png")
    assert cfh.main(["--root", root, "--scene", "s01", "--distance_m", "3", "--nodes",
                     "node_c", "--overlay", out_png]) == 0
    assert os.path.isfile(out_png)
    assert capsys.readouterr().out.splitlines()[0].startswith("node,scene,distance_m")


def _criterion_row(tmp_path, node, flame_rows, n=40):
    from scripts import camera_flame_height as cfh

    root = str(tmp_path / "camera_pilot")
    _write_capture(root, node, "s02", "no_fire", 3.0, 1050.0, (1080, 1920), NOFIRE_NOTES, 0, n)
    _write_capture(root, node, "s02", "fire", 3.0, 1050.0, (1080, 1920), FIRE_NOTES,
                   flame_rows, n)
    return cfh.measure_node(root, node, "s02", 3.0)[0]


def test_flame_region_ignores_glow_and_reflection_and_takes_the_flame():
    from scripts import camera_flame_height as cfh

    ref = np.full((224, 224, 3), 150, np.uint8)            # white wall / floor at 150
    fire = ref.copy()
    fire[60:140, 40:200] = (215, 190, 200)                 # wide diffuse pink glow
    fire[90:110, 100:108] = (255, 230, 150)                # the flame: 20 rows
    fire[150:156, 98:110] = (250, 220, 140)                # floor reflection below
    top, left, bottom, right = cfh.flame_region(fire, ref, 40)
    assert (top, bottom) == (90, 109) and (left, right) == (100, 107)
    # the old rule (largest brighter region) would have taken the glow: 80 rows
    old = (fire.astype(np.int16) - ref.astype(np.int16)).max(axis=2) >= 40
    assert old[60:140].any(axis=1).sum() == 80


def test_flame_components_touching_the_crop_edge_are_dropped():
    from scripts import camera_flame_height as cfh

    ref = np.full((224, 224, 3), 120, np.uint8)
    fire = ref.copy()
    fire[0:40, 150:170] = (255, 255, 255)                  # brightest, but on the top edge
    assert cfh.flame_region(fire, ref, 40) is None
    fire[100:108, 60:64] = (250, 230, 150)                 # a flame inside the crop
    assert cfh.flame_region(fire, ref, 40) == (100, 60, 107, 63)


def test_decision_rule_a_b_c():
    from scripts import camera_flame_height as cfh

    assert not hasattr(cfh, "VALIDATION_BAND") and cfh.MIN_DETECTED_FRACTION == 0.9
    assert cfh.criterion(6.0, 40, 36, 0) == ("PASS", [])
    assert cfh.criterion(6.0, 40, 35, 0) == ("FAIL", ["b_detected_35_of_40"])
    assert cfh.criterion(6.0, 40, 40, 2) == ("FAIL", ["a_nofire_detected_2"])
    assert cfh.criterion(4.5, 40, 40, 0) == ("FAIL", ["c_median_4.5_px"])
    assert cfh.criterion(9.0, 39, 39, 0) == ("INCOMPLETE", [])


def test_flame_region_is_none_on_a_flame_free_frame():
    from scripts import camera_flame_height as cfh

    ref = np.full((224, 224, 3), 120, np.uint8)
    frame = ref.copy()
    frame[:, :112] = (170, 168, 172)    # a +50 brightness change, grey (e.g. someone's shadow
                                         # leaving, a lamp flicker): not flame-like
    assert cfh.flame_region(frame, ref, 40) is None
    assert cfh.flame_region(ref, ref, 40) is None


def test_lit_candle_body_below_the_flame_is_part_of_the_component():
    from scripts import camera_flame_height as cfh

    ref = np.full((224, 224, 3), 120, np.uint8)
    fire = ref.copy()
    fire[100:110, 110:114] = (255, 235, 170)               # flame, 10 rows
    fire[110:130, 108:116] = (235, 170, 110)               # lit wax, warm and bright
    top, _, bottom, _ = cfh.flame_region(fire, ref, 40)
    # documented limitation: a lit, warm candle body joins the flame (1 m in s03)
    assert (top, bottom) == (100, 129)


def test_flame_criterion_is_the_median_over_40_frames_at_5_px(tmp_path):
    from scripts import camera_flame_height as cfh

    assert not hasattr(cfh, "CRITERION_CM")
    assert cfh.CRITERION_PX == 5 and cfh.CRITERION_FRAMES == 40
    # 1080 -> 224: 30 rows are 6.2 px, 19 rows are 3.9 px
    ok = _criterion_row(tmp_path, "node_a", 30)
    assert ok["n_fire_frames"] == 40 and ok["measured_px_median"] >= 5
    assert ok["criterion"] == "PASS" and ok["criterion_failed"] == ""
    assert ok["n_nofire_detected"] == 0 and ok["detected_fraction"] == 1.0
    small = _criterion_row(tmp_path, "node_b", 19)
    assert small["measured_px_median"] < 5 and small["criterion"] == "FAIL"
    assert small["criterion_failed"].startswith("c_median_")
    # 21 of 40 frames without a visible flame: detected 19 of 40 and median 0 -> FAIL
    flicker = _criterion_row(tmp_path, "node_c", [108] * 19 + [0] * 21)
    assert flicker["n_frames_detected"] == 19 and flicker["measured_px_min"] == 0
    assert flicker["criterion"] == "FAIL"
    assert flicker["criterion_failed"] == "b_detected_19_of_40;c_median_0_px"
    assert cfh.criterion(5.0, 40)[0] == "PASS" and cfh.criterion(4.5, 40)[0] == "FAIL"
    assert cfh.criterion(9.0, 39)[0] == "INCOMPLETE"


def test_a_distance_stays_only_if_all_three_cameras_pass():
    from scripts import camera_flame_height as cfh

    def rows(**v):
        return [{"node": n, "distance_m": 3.0, "criterion": c} for n, c in v.items()]
    assert "PASS on all three" in cfh.distance_decision(
        rows(node_a="PASS", node_b="PASS", node_c="PASS"))
    assert "FAIL -- removed from the study's distance set" in cfh.distance_decision(
        rows(node_a="PASS", node_b="PASS", node_c="FAIL"))
    assert "NOT DECIDED" in cfh.distance_decision(rows(node_a="PASS", node_b="PASS"))
    assert "NOT DECIDED" in cfh.distance_decision(
        rows(node_a="PASS", node_b="INCOMPLETE", node_c="PASS"))


def test_flame_height_tool_refuses_the_study_footage():
    from scripts import camera_flame_height as cfh

    assert cfh.main(["--root", os.path.join("data", "raw", "camera"), "--scene", "s01",
                     "--distance_m", "3"]) == 2


def test_fy_224_follows_the_preprocessing_geometry():
    from scripts import camera_flame_height as cfh

    # landscape 1920x1080 and 1280x720: shorter side (height) to 224, crop keeps scale
    assert cfh.fy_224(1000.0, 1920, 1080) == pytest.approx(1000.0 * 224 / 1080)
    assert cfh.fy_224(640.0, 1280, 720) == pytest.approx(640.0 * 224 / 720)


# --------------------------------------------------------------------------- #
# ZED raw calibration and factory file (prereg sec. 13.1 draft)
# --------------------------------------------------------------------------- #
def _zed_like(clock, conf_text="[LEFT_CAM_FHD]", drop=None):
    class ZedLike(FakeCamera):
        def info(self):
            i = super().info()
            left = dict(i["intrinsics"]["rgb"])
            i.update(requires_factory_calibration=True, usb_type=None, usb_speed_mbps=5000.0,
                     serial=35201583,
                     calibration_raw={"left": left, "right": dict(left),
                                      "stereo_transform": {"rotation_vector": [0, 0, 0],
                                                           "translation_mm": [-120, 0, 0]}},
                     factory_conf=None if conf_text is None else {
                         "source": "/usr/local/zed/settings/SN35201583.conf",
                         "md5": "abc123", "bytes": len(conf_text), "text": conf_text})
            if drop:
                drop(i)
            return i
    return ZedLike(clock=clock, ir=False)


def test_zed_capture_keeps_raw_calibration_and_a_copy_of_the_factory_file(tmp_path):
    clock = FakeClock()
    rec, _ = _study(tmp_path, "no_fire", camera=_zed_like(clock), clock=clock, node="node_c")
    assert rec["status"] == "complete"
    assert rec["calibration_raw"]["left"]["fx"] == 50.0
    assert rec["calibration_raw"]["stereo_transform"]["translation_mm"] == [-120, 0, 0]
    fc = rec["factory_conf"]
    assert fc["md5"] == "abc123" and "text" not in fc
    copy = tmp_path / "node_c" / fc["copy"]
    assert copy.read_text() == "[LEFT_CAM_FHD]"
    assert "text" not in rec["backend_info"]["factory_conf"]
    # a second capture with the same file reuses the copy
    clock2 = FakeClock()
    rec2, _ = _study(tmp_path, "fire", camera=_zed_like(clock2), clock=clock2, node="node_c")
    assert rec2["factory_conf"]["copy"] == fc["copy"]
    assert len(list((tmp_path / "node_c" / "_calibration").iterdir())) == 1


@pytest.mark.parametrize("conf,drop,match", [
    (None, None, "factory calibration file SN35201583.conf not found"),
    ("x", lambda i: i.update(calibration_raw=None), "raw left calibration missing"),
    ("x", lambda i: i["calibration_raw"].pop("stereo_transform"), "raw stereo transform"),
])
def test_zed_capture_without_raw_calibration_is_refused(tmp_path, conf, drop, match):
    clock = FakeClock()
    with pytest.raises(ccc.CaptureError, match=match):
        _study(tmp_path, "no_fire", camera=_zed_like(clock, conf, drop), clock=clock,
               node="node_c")


def test_factory_conf_reader(tmp_path):
    from src.data import zed_capture as zc
    (tmp_path / "SN35201583.conf").write_bytes(b"[STEREO]\r\nBaseline=119.9\r\n")
    fc = zc.factory_conf(35201583, dirs=(str(tmp_path / "none"), str(tmp_path)))
    assert fc["bytes"] == 26 and fc["text"].encode("latin-1") == b"[STEREO]\r\nBaseline=119.9\r\n"
    assert len(fc["md5"]) == 32
    assert zc.factory_conf(1, dirs=(str(tmp_path),)) is None
    assert zc.factory_conf(None) is None
