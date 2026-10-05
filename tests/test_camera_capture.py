"""Camera-experiment capture side (docs/CAMERA_EXPERIMENT_PREREG.md), with fake cameras.

No camera SDK is needed: the RealSense / ZED back-ends are replaced by fake ones that
implement ``camera_capture_common.CameraBackend``; the orchestrator is exercised through
``--dry_run`` and its pure verification function.
"""

import json
import os
import shlex

import numpy as np
import pytest
from PIL import Image

from src.data import camera_capture_common as ccc
from src.data.camera_capture_common import (
    CAPTURE_RECORD_KEYS, FRAME_META_KEYS, CameraBackend, CaptureError, CaptureExistsError,
    Frame, FrameSampler, depth_to_mm_uint16, make_capture_id, make_frame_id, run_capture,
    wait_until,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeClock:
    def __init__(self, t=1_790_000_000.0):
        self.t = float(t)
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.slept.append(dt)
        self.t += dt


class FakeCamera(CameraBackend):
    """RealSense-like (with IR) or ZED-like (no IR, NaN depth) fake."""

    def __init__(self, clock=None, stream_fps=30.0, ir=True, size=(48, 64), fail_at=None,
                 model="Fake D435I", serial="123", depth_mode=None):
        self.clock = clock
        self.stream_fps = stream_fps
        self.ir = ir
        self.h, self.w = size
        self.fail_at = fail_at
        self.model, self.serial, self.depth_mode = model, serial, depth_mode
        self.n = 0
        self.opened = self.closed = False
        self.ts_ms = 1000.0

    def open(self):
        self.opened = True

    def info(self):
        intr = {"width": self.w, "height": self.h, "fx": 50.0, "fy": 50.0, "ppx": 32.0,
                "ppy": 24.0, "model": "fake brown-conrady", "coeffs": [0.0] * 5}
        return {"camera_model": self.model, "serial": self.serial, "firmware": "5.16.0.1",
                "sdk_version": "2.55.1", "depth_mode": self.depth_mode,
                "rgb_resolution": [self.w, self.h], "stream_fps": self.stream_fps,
                "intrinsics": {"rgb": intr, "depth": dict(intr)},
                "stereo_baseline_mm": 50.0, "depth_scale_mm": 1.0, "depth_range_mm": None}

    def grab(self):
        if self.fail_at is not None and self.n >= self.fail_at:
            raise RuntimeError("camera unplugged")
        self.n += 1
        dt = 1.0 / self.stream_fps
        self.ts_ms += dt * 1000.0
        if self.clock is not None:
            self.clock.t += dt
        rgb = np.full((self.h, self.w, 3), self.n % 256, np.uint8)
        rgb[0, 0] = (255, 0, 0)
        depth = np.full((self.h, self.w), 1000 + self.n, np.float32)
        if not self.ir:
            depth[0, :4] = [np.nan, np.inf, -np.inf, -5.0]
        return Frame(rgb=rgb, depth_mm=depth_to_mm_uint16(depth),
                     ir=np.full((self.h, self.w), 7, np.uint8) if self.ir else None,
                     device_ts_ms=self.ts_ms,
                     meta={"exposure": 166, "gain": 64, "white_balance": 4600,
                           "auto_exposure": True, "timestamp_domain": "fake"})

    def close(self):
        self.closed = True


def _capture(tmp_path, node="node_a", clock=None, **kw):
    clock = clock or FakeClock()
    cam = kw.pop("camera", None) or FakeCamera(clock=clock, ir=(node != "node_c"))
    args = dict(scene="s03", label="no_fire", distance_m=2.0, node=node,
                root=str(tmp_path), frames=4, fps=5.0, clock=clock, sleep=clock.sleep,
                log=lambda m: None)
    args.update(kw)
    return run_capture(cam, **args), cam


# --------------------------------------------------------------------------- #
# ids and validation
# --------------------------------------------------------------------------- #
def test_capture_and_frame_ids_follow_the_contract():
    assert make_capture_id("s03", "no_fire", 2.0) == "s03_no_fire_d200"
    assert make_capture_id("s01", "fire", 1) == "s01_fire_d100"
    assert make_capture_id("s12", "fire", "3.0") == "s12_fire_d300"
    assert make_frame_id("s03_no_fire_d200", 7) == "s03_no_fire_d200_0007"


@pytest.mark.parametrize("scene,label,dist", [
    ("s1", "fire", 1.0), ("S01", "fire", 1.0), ("s001", "fire", 1.0), ("x01", "fire", 1.0),
    ("s01", "Fire", 1.0), ("s01", "nofire", 1.0), ("s01", "no-fire", 1.0),
    ("s01", "fire", 1.5), ("s01", "fire", 4.0), ("s01", "fire", "far"),
])
def test_invalid_ids_are_rejected(scene, label, dist):
    with pytest.raises(ValueError):
        make_capture_id(scene, label, dist)


def test_depth_conversion_zeroes_invalid_values():
    d = depth_to_mm_uint16(np.array([[np.nan, np.inf, -np.inf, -3.0, 1234.4, 70000.0]]))
    assert d.dtype == np.uint16
    assert d.tolist() == [[0, 0, 0, 0, 1234, 65535]]


# --------------------------------------------------------------------------- #
# layout, metadata, capture record
# --------------------------------------------------------------------------- #
def test_realsense_like_capture_writes_exactly_the_contract_files(tmp_path):
    rec, cam = _capture(tmp_path, node="node_a")
    node_dir = tmp_path / "node_a"
    cid = "s03_no_fire_d200"
    expected = {"_captures"}
    for i in range(4):
        fid = "%s_%04d" % (cid, i)
        expected |= {fid + "_rgb.png", fid + "_depth.png", fid + "_ir.png", fid + "_meta.json"}
    assert set(os.listdir(node_dir)) == expected
    assert os.listdir(node_dir / "_captures") == [cid + ".json"]
    assert cam.opened and cam.closed

    fid0 = cid + "_0000"
    rgb = np.asarray(Image.open(node_dir / (fid0 + "_rgb.png")))
    assert rgb.dtype == np.uint8 and rgb.shape == (48, 64, 3)
    assert tuple(rgb[0, 0]) == (255, 0, 0)  # RGB order kept
    depth = np.asarray(Image.open(node_dir / (fid0 + "_depth.png")))
    # values, not the read-back dtype: Pillow 9.0.1 (nodes) opens a 16-bit PNG as mode "I"
    # (int32), Pillow >= 10 as "I;16" (uint16); the file is 16-bit either way
    assert Image.open(node_dir / (fid0 + "_depth.png")).mode in ("I;16", "I")
    assert depth.shape == (48, 64) and depth.min() >= 0 and depth.max() <= 65535
    # frame 0 is the 16th grab (15 warm-up grabs), whose fake depth is 1000 + 16 mm
    assert np.array_equal(depth.astype(np.int64), np.full((48, 64), 1016, np.int64))
    ir = np.asarray(Image.open(node_dir / (fid0 + "_ir.png")))
    assert ir.dtype == np.uint8 and ir.shape == (48, 64)

    meta = json.loads((node_dir / (fid0 + "_meta.json")).read_text())
    assert set(FRAME_META_KEYS) <= set(meta)
    assert meta["frame_id"] == fid0 and meta["capture_id"] == cid
    assert (meta["scene"], meta["label"], meta["distance_m"]) == ("s03", "no_fire", 2.0)
    assert meta["frame_index"] == 0 and meta["node"] == "node_a"
    assert meta["rgb_resolution"] == [64, 48] and meta["fps"] == 5.0
    assert meta["exposure"] == 166 and meta["auto_exposure"] is True
    assert meta["depth_mode"] is None and meta["timestamp_domain"] == "fake"

    assert set(CAPTURE_RECORD_KEYS) <= set(rec)
    on_disk = json.loads((node_dir / "_captures" / (cid + ".json")).read_text())
    assert on_disk["n_frames_written"] == on_disk["n_frames_requested"] == 4
    assert on_disk["status"] == "complete" and on_disk["capture_id"] == cid
    assert on_disk["start_scheduled_unix"] is None


def test_zed_like_capture_has_no_ir_and_invalid_depth_is_zero(tmp_path):
    cam = FakeCamera(ir=False, model="ZED2i", serial=4242, depth_mode="NEURAL")
    rec, _ = _capture(tmp_path, node="node_c", camera=cam)
    names = os.listdir(tmp_path / "node_c")
    assert not [n for n in names if n.endswith("_ir.png")]
    depth = np.asarray(Image.open(tmp_path / "node_c" / "s03_no_fire_d200_0000_depth.png"))
    assert depth[0, :4].tolist() == [0, 0, 0, 0]
    meta = json.loads((tmp_path / "node_c" / "s03_no_fire_d200_0000_meta.json").read_text())
    assert meta["depth_mode"] == "NEURAL" and rec["depth_mode"] == "NEURAL"


def test_30fps_stream_is_sampled_to_5fps_by_time(tmp_path):
    clock = FakeClock()
    rec, cam = _capture(tmp_path, clock=clock, frames=6,
                        camera=FakeCamera(clock=clock, stream_fps=30.0), warmup_frames=0)
    ts = [json.loads((tmp_path / "node_a" / ("s03_no_fire_d200_%04d_meta.json" % i))
                     .read_text())["timestamp_device"] for i in range(6)]
    assert np.allclose(np.diff(ts), 200.0, atol=1e-6)


def test_5fps_stream_keeps_every_frame():
    s = FrameSampler(5.0)
    jitter = [0, 3, -4, 2, -1, 5, 0, -3]
    assert all(s.accept(i * 200.0 + j) for i, j in enumerate(jitter))
    s = FrameSampler(5.0, stream_fps=5.0)
    assert all(s.accept(i * 200.0 + j * 15) for i, j in enumerate(jitter))
    for stream in (15.0, 30.0):
        s = FrameSampler(5.0, stream_fps=stream)
        kept = [t for t in np.arange(0, 2000, 1000 / stream) if s.accept(t)]
        assert len(kept) == 10 and np.allclose(np.diff(kept), 200.0)


def test_failure_mid_capture_still_writes_an_incomplete_record(tmp_path):
    clock = FakeClock()
    cam = FakeCamera(clock=clock, fail_at=20)  # 15 warm-up grabs + 5 stream grabs
    with pytest.raises(RuntimeError):
        _capture(tmp_path, clock=clock, camera=cam)
    rec = json.loads((tmp_path / "node_a" / "_captures" / "s03_no_fire_d200.json").read_text())
    assert rec["status"] == "incomplete" and rec["n_frames_written"] < 4
    assert "camera unplugged" in rec["notes"] and cam.closed


# --------------------------------------------------------------------------- #
# no overwrite, retake
# --------------------------------------------------------------------------- #
def test_existing_capture_is_never_overwritten(tmp_path):
    _capture(tmp_path)
    before = {n: os.path.getmtime(tmp_path / "node_a" / n)
              for n in os.listdir(tmp_path / "node_a") if n != "_captures"}
    with pytest.raises(CaptureExistsError):
        _capture(tmp_path)
    after = {n: os.path.getmtime(tmp_path / "node_a" / n)
             for n in os.listdir(tmp_path / "node_a") if n != "_captures"}
    assert before == after


def test_partial_capture_without_record_is_also_refused(tmp_path):
    node_dir = tmp_path / "node_a"
    node_dir.mkdir()
    (node_dir / "s03_no_fire_d200_0000_rgb.png").write_bytes(b"x")
    with pytest.raises(CaptureExistsError):
        _capture(tmp_path)


def test_retake_moves_old_capture_aside_and_never_deletes(tmp_path):
    rec1, _ = _capture(tmp_path, notes="first")
    _capture(tmp_path, label="fire")  # another capture of the scene: must not move
    rec2, _ = _capture(tmp_path, retake=True, notes="second")
    retakes = tmp_path / "node_a" / "_retakes"
    (stamp,) = os.listdir(retakes)
    moved = set(os.listdir(retakes / stamp))
    assert "s03_no_fire_d200.json" in moved
    assert len([n for n in moved if n.startswith("s03_no_fire_d200_")]) == 16
    assert not [n for n in moved if "_fire_d200" in n and "no_fire" not in n]
    old = json.loads((retakes / stamp / "s03_no_fire_d200.json").read_text())
    new = json.loads((tmp_path / "node_a" / "_captures" / "s03_no_fire_d200.json").read_text())
    assert old["notes"] == "first" and new["notes"] == "second"
    assert new["retake_moved_to"] == str(retakes / stamp)
    assert os.path.isfile(tmp_path / "node_a" / "_captures" / "s03_fire_d200.json")
    # a second retake in the same second gets its own directory
    _capture(tmp_path, retake=True)
    assert len(os.listdir(retakes)) == 2


def test_retake_of_a_missing_capture_moves_nothing(tmp_path):
    rec, _ = _capture(tmp_path, retake=True)
    assert rec["retake_moved_to"] is None and rec["status"] == "complete"
    assert not os.path.exists(tmp_path / "node_a" / "_retakes")
    with pytest.raises(CaptureError):  # the low-level mover still refuses an empty move
        ccc.move_to_retakes(str(tmp_path), "node_b", "s03_no_fire_d200")


# --------------------------------------------------------------------------- #
# scheduling
# --------------------------------------------------------------------------- #
def test_wait_until_sleeps_to_the_scheduled_time():
    clock = FakeClock(100.0)
    late = wait_until(102.5, clock=clock, sleep=clock.sleep)
    assert late == pytest.approx(0.0) and clock.t == pytest.approx(102.5)
    assert max(clock.slept) <= 0.5


def test_wait_until_reports_lateness_and_none_means_now():
    clock = FakeClock(100.0)
    assert wait_until(98.8, clock=clock, sleep=clock.sleep) == pytest.approx(1.2)
    assert clock.slept == []
    assert wait_until(None, clock=clock, sleep=clock.sleep) == 0.0


def test_wait_until_idles_the_camera_then_sleeps_the_last_stretch():
    clock = FakeClock(0.0)
    calls = []

    def idle():
        calls.append(clock.t)
        clock.t += 1 / 30.0

    late = wait_until(2.0, clock=clock, sleep=clock.sleep, idle=idle)
    assert late == pytest.approx(0.0, abs=1e-9)
    assert calls and max(calls) < 2.0 - 0.25 + 1e-9
    assert sum(clock.slept) <= 0.25 + 1e-9


def test_capture_starts_at_the_scheduled_time(tmp_path):
    clock = FakeClock(1000.0)
    rec, _ = _capture(tmp_path, clock=clock, start_at=1010.0)
    assert rec["start_scheduled_unix"] == 1010.0
    assert 1010.0 <= rec["start_actual_unix"] < 1010.0 + 0.1
    assert rec["start_wait_lateness_s"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the loader reads what the writer writes
# --------------------------------------------------------------------------- #
def test_custom_dataset_reads_a_written_capture_tree(tmp_path):
    from src.data.custom_dataset import filter_index, load_frame_index

    for node in ("node_a", "node_b", "node_c"):
        cam = FakeCamera(ir=(node != "node_c"))
        _capture(tmp_path, node=node, camera=cam, scene="s01", label="fire", distance_m=1.0)
        _capture(tmp_path, node=node, camera=FakeCamera(ir=(node != "node_c")),
                 scene="s01", label="no_fire", distance_m=1.0)
    index = load_frame_index(str(tmp_path))
    assert len(index) == 3 * 2 * 4
    assert {r["label_source"] for r in index} == {"meta_json"}
    assert {r["scene"] for r in index} == {"s01"}
    assert {(r["label_name"], r["label"]) for r in index} == {("fire", 1), ("no_fire", 0)}
    assert all(r["has_ir"] for r in index if r["node"] != "node_c")
    assert not any(r["has_ir"] for r in index if r["node"] == "node_c")
    assert len(filter_index(index, modality="rgb_d")) == 24
    assert {r["node"] for r in index} == {"node_a", "node_b", "node_c"}


# --------------------------------------------------------------------------- #
# camera CLIs (fake back-ends; no SDK on the desktop)
# --------------------------------------------------------------------------- #
def test_realsense_module_imports_without_the_sdk_and_captures_via_cli(tmp_path):
    from src.data import realsense_capture as rsc

    assert rsc.build_parser().parse_args([]).frames is None  # legacy default resolved later
    rc = rsc.main(["--scene", "s02", "--label", "fire", "--distance_m", "3", "--node",
                   "node_b", "--frames", "3", "--root", str(tmp_path)],
                  backend_factory=lambda: FakeCamera())
    assert rc == 0
    rec = json.loads((tmp_path / "node_b" / "_captures" / "s02_fire_d300.json").read_text())
    assert rec["n_frames_written"] == 3 and rec["fps"] == 5.0
    assert rsc.main(["--scene", "s02", "--label", "fire", "--distance_m", "3", "--node",
                     "node_b", "--frames", "3", "--root", str(tmp_path)],
                    backend_factory=lambda: FakeCamera()) == 1  # refuses to overwrite


def test_realsense_sdk_version_fallbacks():
    from src.data import realsense_capture as rsc

    class WithVersion:
        __version__ = "2.55.1"

    assert rsc.realsense_sdk_version(WithVersion()) == "2.55.1"
    v = rsc.realsense_sdk_version(object())  # no attribute: metadata / pkg-config / None
    assert v is None or isinstance(v, str)


def test_zed_cli_defaults_and_capture(tmp_path):
    from src.data import zed_capture as zc

    args = zc.build_parser().parse_args([])
    assert (args.depth_mode, args.resolution, args.node) == ("NEURAL", "HD1080", "node_c")
    assert (args.frames, args.fps, args.root) == (40, 5.0, ccc.DEFAULT_ROOT)
    rc = zc.main(["--scene", "s05", "--label", "no_fire", "--distance_m", "1", "--frames",
                  "2", "--root", str(tmp_path)],
                 backend_factory=lambda: FakeCamera(ir=False, depth_mode="NEURAL"))
    assert rc == 0
    assert os.path.isfile(tmp_path / "node_c" / "_captures" / "s05_no_fire_d100.json")
    assert zc.main(["--label", "fire"]) == 2


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
FAKE_NODES = {
    "node_a": {"host": "192.168.1.10", "user": "ua", "repo": "/home/ua/FedRGBD",
               "venv": "/home/ua/fedrgbd_venv", "local": True},
    "node_b": {"host": "192.168.1.7", "user": "ub", "repo": "/home/ub/FedRGBD",
               "venv": "/home/ub/fedrgbd_venv", "local": False},
    "node_c": {"host": "192.168.1.6", "user": "uc", "repo": "/home/uc/FedRGBD",
               "venv": "/home/uc/fedrgbd_venv", "local": False},
}


def _session(monkeypatch):
    from scripts import camera_capture_session as ccs

    monkeypatch.setattr(ccs, "load_testbed", lambda path: (FAKE_NODES, 8080))
    monkeypatch.setattr(ccs.subprocess, "run", _forbidden)
    monkeypatch.setattr(ccs.subprocess, "Popen", _forbidden)
    return ccs


def _forbidden(*a, **k):
    raise AssertionError("dry run must not execute anything")


def test_dry_run_builds_the_three_node_commands(monkeypatch):
    ccs = _session(monkeypatch)
    out = []
    rc = ccs.main(["--scene", "s04", "--label", "fire", "--distance_m", "2", "--dry_run"],
                  clock=lambda: 1_790_000_000.0, log=out.append)
    assert rc == 0
    caps = {l.split()[0]: l.split(" capture: ", 1)[1] for l in out if " capture: " in l}
    assert set(caps) == {"node_a", "node_b", "node_c"}

    a = shlex.split(caps["node_a"])
    assert a[:2] == ["bash", "-lc"]
    b, c = shlex.split(caps["node_b"]), shlex.split(caps["node_c"])
    for argv, user, host in ((b, "ub", "192.168.1.7"), (c, "uc", "192.168.1.6")):
        assert argv[0] == "ssh" and "%s@%s" % (user, host) in argv
        assert "ControlMaster=auto" in argv and "ControlPersist=600" in argv
        assert any(x.startswith("ControlPath=") for x in argv)
        assert "BatchMode=yes" in argv

    inner = {"node_a": a[2], "node_b": shlex.split(b[-1])[2], "node_c": shlex.split(c[-1])[2]}
    for node, cfg in FAKE_NODES.items():
        cmd = inner[node]
        assert cmd.startswith("cd %s && source %s/bin/activate && python "
                              % (cfg["repo"], cfg["venv"]))
        script = "src/data/zed_capture.py" if node == "node_c" else "src/data/realsense_capture.py"
        tail = shlex.split(cmd.split(" && ")[-1])
        assert tail[:2] == ["python", script]
        opts = dict(zip(tail[2::2], tail[3::2]))
        assert opts["--scene"] == "s04" and opts["--label"] == "fire"
        assert opts["--distance_m"] == "2.0" and opts["--frames"] == "40"
        assert opts["--fps"] == "5" and opts["--node"] == node
        assert opts["--root"] == "data/raw/camera"
        assert float(opts["--start_at"]) == 1_790_000_015.0  # now + default 15 s lead
    masters = [l for l in out if " master : " in l]
    assert len(masters) == 2 and not any(l.startswith("node_a") for l in masters)


def test_dry_run_passes_retake_and_lead(monkeypatch):
    ccs = _session(monkeypatch)
    out = []
    ccs.main(["--scene", "s04", "--label", "no_fire", "--distance_m", "3", "--dry_run",
              "--retake", "--lead", "30"], clock=lambda: 100.0, log=out.append)
    caps = [l for l in out if " capture: " in l]
    assert len(caps) == 3 and all("--retake" in l for l in caps)
    assert all("--start_at 130.000" in l for l in caps)


def test_check_dry_run_prints_probe_commands(monkeypatch):
    ccs = _session(monkeypatch)
    out = []
    assert ccs.main(["--check", "--dry_run"], log=out.append) == 0
    probes = [l for l in out if " probe  : " in l]
    assert len(probes) == 3 and all("camera_capture_common.py --probe" in l for l in probes)


def _rec(start, actual, written=40, **kw):
    r = {"start_scheduled_unix": start, "start_actual_unix": actual,
         "n_frames_written": written, "camera_model": "X", "serial": "1"}
    r.update(kw)
    return r


def test_verification_pass_and_fail(monkeypatch):
    from scripts.camera_capture_session import verify_records

    s = 1000.0
    good = {n: _rec(s, s + 0.05) for n in ("node_a", "node_b", "node_c")}
    ok, lines = verify_records(good, s, 40, returncodes={n: 0 for n in good})
    assert ok and all(l.endswith("PASS") for l in lines.values())

    cases = [
        ({"node_b": None}, "no capture record"),
        ({"node_c": _rec(s, s + 0.1, written=39)}, "39 of 40 frames"),
        ({"node_b": _rec(s, s + 1.2)}, "off schedule"),
        ({"node_a": _rec(s, s - 1.3)}, "off schedule"),
        ({"node_c": _rec(s - 60, s - 59.9)}, "not from this session"),
        ({"node_a": _rec(s, None)}, "no start_actual"),
    ]
    for change, reason in cases:
        recs = dict(good, **change)
        ok, lines = verify_records(recs, s, 40)
        (node,) = change
        assert not ok and reason in lines[node] and "FAIL" in lines[node]
        assert all(lines[n].endswith("PASS") for n in lines if n != node)
    ok, lines = verify_records(good, s, 40, returncodes={"node_a": 0, "node_b": 1, "node_c": 0})
    assert not ok and "exit code 1" in lines["node_b"]
    # exactly 1.0 s late is still within the declared tolerance
    assert verify_records(dict(good, node_b=_rec(s, s + 1.0)), s, 40)[0]


def test_clock_offset_uses_the_tightest_round_trip():
    from scripts.camera_capture_session import clock_offset

    samples = [(10.0, 10.9, 11.0), (20.0, 20.012, 20.004), (30.0, 30.5, 30.2)]
    off, rtt = clock_offset(samples)
    assert off == pytest.approx(0.010) and rtt == pytest.approx(0.004)
    with pytest.raises(ValueError):
        clock_offset([])


def test_clock_gate_uses_the_start_tolerance():
    from scripts.camera_capture_session import START_TOLERANCE_S, clock_gate

    assert clock_gate({"node_a": 0.0, "node_b": 0.010, "node_c": -0.9}) == []
    problems = clock_gate({"node_a": 0.0, "node_b": -1320.0, "node_c": None})
    assert len(problems) == 2
    assert problems[0].startswith("node_b: clock -1320.000 s off node_a")
    assert "node_c" in problems[1] and "could not be measured" in problems[1]
    assert clock_gate({"node_b": START_TOLERANCE_S + 0.001}) != []


def test_capture_refuses_to_start_with_unsynchronised_clocks(monkeypatch):
    """After a reboot a node can be minutes off until NTP syncs: nothing is scheduled."""
    ccs = _session(monkeypatch)       # Popen / subprocess.run raise if anything starts
    monkeypatch.setattr(ccs, "open_masters", lambda nodes, log=print: True)
    offsets = {"node_a": (0.0, 0.0), "node_b": (0.004, 0.002), "node_c": (-1320.0, 0.003)}
    monkeypatch.setattr(ccs, "measure_offset",
                        lambda cfg: offsets[[n for n, c in FAKE_NODES.items() if c is cfg][0]])
    out = []
    rc = ccs.main(["--scene", "s04", "--label", "fire", "--distance_m", "2"], log=out.append)
    assert rc == 1
    text = "\n".join(out)
    assert "node_c: clock -1320.000 s off node_a" in text
    assert "nothing was started" in text and "capture begins" not in text


def test_session_log_is_appended(tmp_path):
    from scripts.camera_capture_session import append_session_log

    p1 = append_session_log(str(tmp_path), {"capture_id": "a"})
    append_session_log(str(tmp_path), {"capture_id": "b"})
    rows = [json.loads(l) for l in open(p1, encoding="utf-8")]
    assert [r["capture_id"] for r in rows] == ["a", "b"]
    assert os.path.basename(p1) == "session_log.jsonl"


def test_scenes_template_header():
    path = os.path.join(_REPO, "data", "raw", "camera", "scenes_template.csv")
    if not os.path.isfile(path):  # data/raw is gitignored: present only if force-added
        pytest.skip("scenes_template.csv not in this checkout")
    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines[0] == "scene,location,background,distractors,fire_source,distances_m,notes"
    assert lines[1].startswith("# s01,")


# --------------------------------------------------------------------------- #
# calibration at every capture (prereg Amendment 3)
# --------------------------------------------------------------------------- #
def test_every_capture_record_carries_the_calibration(tmp_path):
    rec, _ = _capture(tmp_path, node="node_a")
    assert rec["status"] == "complete"
    assert rec["intrinsics"]["rgb"]["fx"] == 50.0 and rec["intrinsics"]["depth"]["coeffs"]
    assert (rec["stereo_baseline_mm"], rec["depth_scale_mm"]) == (50.0, 1.0)
    assert rec["sdk_version"] and rec["firmware"]


@pytest.mark.parametrize("drop,match", [
    (lambda i: i.pop("intrinsics"), "rgb intrinsics missing"),
    (lambda i: i["intrinsics"]["depth"].pop("coeffs"), "depth intrinsics missing coeffs"),
    (lambda i: i.update(stereo_baseline_mm=0.12), "stereo_baseline_mm"),   # metres: unit error
    (lambda i: i.update(stereo_baseline_mm=None), "stereo_baseline_mm"),
    (lambda i: i.update(depth_scale_mm=None), "depth_scale_mm"),
    (lambda i: i.update(firmware=""), "firmware"),
])
def test_a_capture_without_its_calibration_is_refused_and_recorded(tmp_path, drop, match):
    class Uncalibrated(FakeCamera):
        def info(self):
            i = super().info()
            drop(i)
            return i
    clock = FakeClock()
    with pytest.raises(ccc.CaptureError, match=match):
        _capture(tmp_path, clock=clock, camera=Uncalibrated(clock=clock))
    rec = json.loads((tmp_path / "node_a" / "_captures" / "s03_no_fire_d200.json").read_text())
    assert rec["status"] == "incomplete" and rec["n_frames_written"] == 0
    assert "Amendment 3" in rec["notes"]


class _FakeSl:
    """Just enough of pyzed.sl for ZedBackend.open()."""

    class ERROR_CODE:
        SUCCESS = "SUCCESS"

    class RESOLUTION:
        HD1080 = "HD1080"

    class DEPTH_MODE:
        NEURAL = "NEURAL"

    class UNIT:
        MILLIMETER = "MILLIMETER"

    class InitParameters:
        depth_minimum_distance = -1.0
        depth_maximum_distance = -1.0

    clamp_max = None

    class Camera:
        @staticmethod
        def get_sdk_version():
            return "5.2.3"

        def open(self, init):
            self.init = init
            return _FakeSl.ERROR_CODE.SUCCESS

        def get_init_parameters(self):
            if _FakeSl.clamp_max is not None:
                self.init.depth_maximum_distance = _FakeSl.clamp_max
            return self.init

        def get_camera_information(self):
            from types import SimpleNamespace as NS
            cam = lambda fx: NS(fx=fx, fy=fx, cx=960.5, cy=540.5, disto=[0.1, -0.02, 0, 0, 0.003])
            calib = NS(left_cam=cam(1066.0), right_cam=cam(1067.0),
                       get_camera_baseline=lambda: 120.0)
            conf = NS(firmware_version=1523, resolution=NS(width=1920, height=1080), fps=15,
                      calibration_parameters=calib)
            return NS(camera_model="ZED2i", serial_number=32608934, camera_configuration=conf,
                      sensors_configuration=NS(firmware_version=777))

    class RuntimeParameters:
        pass

    class Mat:
        pass


def test_zed_sets_the_v1_depth_range_and_records_its_calibration(monkeypatch):
    from src.data import zed_capture as zc
    monkeypatch.setattr(zc, "_import_sl", lambda: _FakeSl)
    _FakeSl.clamp_max = None
    b = zc.ZedBackend()
    b.open()
    i = b.info()
    assert (b.zed.init.depth_minimum_distance, b.zed.init.depth_maximum_distance) == (300.0, 20000.0)
    assert i["depth_range_mm"] == [300.0, 20000.0] and i["depth_scale_mm"] == 1.0
    assert i["stereo_baseline_mm"] == 120.0 and i["firmware"] == "1523" and i["sdk_version"] == "5.2.3"
    assert i["intrinsics"]["rgb"]["fx"] == 1066.0 and i["intrinsics"]["right"]["fx"] == 1067.0
    assert i["intrinsics"]["depth"] == i["intrinsics"]["rgb"]      # depth is in the left frame
    assert i["intrinsics"]["rgb"]["coeffs"] == [0.1, -0.02, 0, 0, 0.003]
    assert ccc.calibration_problems(i) == []
    _FakeSl.clamp_max = 15000.0                                      # SDK did not apply it
    with pytest.raises(ccc.CaptureError, match="depth range"):
        zc.ZedBackend().open()
    _FakeSl.clamp_max = None


def test_realsense_stereo_baseline_from_option_or_ir_extrinsics(monkeypatch):
    from types import SimpleNamespace as NS
    from src.data import realsense_capture as rsc
    fake_rs = NS(option=NS(stereo_baseline="stereo_baseline"), stream=NS(infrared="ir"))
    monkeypatch.setattr(rsc, "rs", fake_rs)

    class Sensor:
        def __init__(self, opt):
            self.opt = opt

        def supports(self, attr):
            return self.opt is not None

        def get_option(self, attr):
            return self.opt

        def get_stream_profiles(self):
            right = NS(stream_type=lambda: "ir", stream_index=lambda: 2)
            left = NS(stream_type=lambda: "ir", stream_index=lambda: 1,
                      get_extrinsics_to=lambda other: NS(translation=[-0.0501, 0.0, 0.0]))
            return [left, right]

    assert rsc.stereo_baseline_mm(Sensor(49.9)) == pytest.approx(49.9)
    assert rsc.stereo_baseline_mm(Sensor(None)) == pytest.approx(50.1)
