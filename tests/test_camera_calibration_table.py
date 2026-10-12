"""The calibration table of the revision cameras (scripts/camera_calibration_table.py):
from the study capture records only, one calibration per camera, never the pilot."""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import camera_calibration_table as cct  # noqa: E402
from src.data.camera_capture_common import run_capture  # noqa: E402
from tests.test_camera_capture import FakeCamera, FakeClock  # noqa: E402
from tests.test_export_latex_tables import assert_valid_latex  # noqa: E402

RAW = {"left": {"width": 1920, "height": 1080, "fx": 1051.0, "fy": 1050.5, "ppx": 960.1,
                "ppy": 540.2, "model": "zed raw", "coeffs": [-0.05, 0.02, 0.001, -0.002, 0.003]
                + [0.0] * 7},
       "right": {"width": 1920, "height": 1080, "fx": 1052.0, "fy": 1051.5, "ppx": 959.9,
                 "ppy": 539.8, "model": "zed raw", "coeffs": [-0.04, 0.01, 0.0, 0.0, 0.002]
                 + [0.0] * 7},
       "stereo_transform": {"rotation_vector": [0.0, 0.0, 0.0], "translation_mm": [-119.9, 0, 0]}}


def _study(tmp_path):
    root = tmp_path / "camera"
    clock = FakeClock()
    for node, cam in (("node_a", lambda: FakeCamera(clock=clock, serial="239722070442")),
                      ("node_b", lambda: FakeCamera(clock=clock, serial="405622076256")),
                      ("node_c", lambda: FakeCamera(clock=clock, ir=False, depth_mode="NEURAL",
                                                    serial="35201583"))):
        for label in ("no_fire", "fire"):
            run_capture(cam(), scene="s12", label=label, distance_m=2.0, node=node,
                        root=str(root), frames=2, fps=5.0, clock=clock, sleep=clock.sleep,
                        log=lambda m: None)
    for name in ("s12_no_fire_d200.json", "s12_fire_d200.json"):    # the ZED records' raw calibration
        p = root / "node_c" / "_captures" / name
        rec = json.loads(p.read_text())
        rec["calibration_raw"] = RAW
        p.write_text(json.dumps(rec))
    return root


def test_one_row_per_camera_and_stream_from_the_records(tmp_path):
    root = _study(tmp_path)
    rows = cct.calibration_rows(cct.load_records(str(root)))
    got = list(zip(rows["node"], rows["stream"]))
    assert got == [("node_a", "RGB"), ("node_a", "depth"), ("node_b", "RGB"), ("node_b", "depth"),
                   ("node_c", "RGB"), ("node_c", "depth"), ("node_c", "raw left"),
                   ("node_c", "raw right")]
    assert set(rows["n_captures"]) == {2}
    left = rows[(rows.node == "node_c") & (rows.stream == "raw left")].iloc[0]
    assert left["fx"] == 1051.0 and left["coeffs"].split()[0] == "-0.05"
    text = cct.tabular(rows)
    assert_valid_latex(text, tabular_only=True)
    assert "-0.0500, 0.0200, 0.0010, -0.0020, 0.0030" in text and "S/N 35201583" in text


def test_a_camera_whose_calibration_changes_stops_the_tool(tmp_path):
    root = _study(tmp_path)
    p = root / "node_b" / "_captures" / "s12_fire_d200.json"
    rec = json.loads(p.read_text())
    rec["firmware"] = "5.13.0.55"
    p.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="node_b.*firmware.*s12_fire_d200"):
        cct.calibration_rows(cct.load_records(str(root)))


def test_pilot_footage_is_refused(tmp_path):
    pilot = tmp_path / "camera_pilot"
    pilot.mkdir()
    with pytest.raises(SystemExit, match="pilot"):
        cct.main(["--data_dir", str(pilot), "--output_dir", str(tmp_path / "out")])
    root = _study(tmp_path)
    with pytest.raises(SystemExit, match="pilot"):
        cct.main(["--data_dir", str(root), "--output_dir", str(pilot / "analysis")])


def test_main_writes_csv_and_tabular(tmp_path):
    root = _study(tmp_path)
    out = tmp_path / "analysis"
    assert cct.main(["--data_dir", str(root), "--output_dir", str(out)]) == 0
    assert (out / "calibration.csv").is_file() and (out / "calibration_tabular.tex").is_file()
