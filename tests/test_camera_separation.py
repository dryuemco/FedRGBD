"""The camera experiment is separated from FLAME by dataset, not only by power configuration.

Camera runs live in results/camera/ (testbed: results/camera/<power_config>/, desktop:
results/camera/desktop/); scripts/analyze_results.py analyses FLAME only, so a camera run
anywhere under results/ -- in its own tree, misplaced in a FLAME namespace, or under a
FLAME-looking name -- must not change a single byte of any FLAME output, and a camera
gate marker must not block the FLAME analysis.  The camera analyses read only
results/camera/.
"""

import filecmp
import json
import os
import shutil
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts import analyze_results as ar  # noqa: E402
from scripts import print_revision_commands as prc  # noqa: E402
from scripts import run_matrix as rm  # noqa: E402
from tests.test_analyze_selection import schema3_payload, write_run  # noqa: E402

FLAME_RUNS = [
    ("rev_iid_fedavg_seed42", "fedavg", "iid", 42, [0.6, 0.2, 0.3], [0.80, 0.85, 0.90]),
    ("rev_iid_fedavg_seed123", "fedavg", "iid", 123, [0.5, 0.4, 0.3], [0.81, 0.83, 0.86]),
    ("rev_iid_fedprox_0.01_seed42", "fedprox", "iid", 42, [0.5, 0.3, 0.4], [0.79, 0.84, 0.88]),
    ("rev_noniid_fedavg_seed42", "fedavg", "non_iid_label", 42, [0.7, 0.5, 0.6],
     [0.70, 0.75, 0.74]),
    ("pc_maxn/rev_iid_fedavg_r10_seed42", "fedavg", "iid", 42, [0.6, 0.3, 0.2],
     [0.82, 0.86, 0.87]),
]


def _flame_tree(root):
    for name, strategy, dist, seed, losses, accs in FLAME_RUNS:
        write_run(root, name, schema3_payload(strategy, dist, seed, losses, accs))


def _camera_payload(fold, seed, strategy="fedavg"):
    # what the server writes for a camera run: client data dirs under camera_fold<f>
    return schema3_payload(strategy, "camera_fold%d" % fold, seed, [0.3, 0.1, 0.2],
                           [0.99, 0.99, 0.99])


def _add_camera_runs(root):
    # in its own tree, as the generator places it
    write_run(root, "camera/maxn/rev_camera_fold0_fedavg_r10_seed42", _camera_payload(0, 42))
    write_run(root, "camera/maxn/diag_camera_smoke", _camera_payload(0, 42))
    centralized = {"experiment": "centralized", "seed": 42, "epochs": 50,
                   "data_dirs": ["data/processed/camera_fold0/node_%s" % n for n in "abc"]}
    write_run(root, "camera/desktop/rev_camera_fold0_centralized_r10_seed42", centralized)
    # a camera gate failure must not block the FLAME analysis
    with open(os.path.join(root, "camera", "maxn", "diag_camera_smoke",
                           ar.IDENTITY_GATE_MARKER), "w") as f:
        f.write("{}")
    # misplaced into FLAME namespaces under camera names
    write_run(root, "pc_maxn/rev_camera_fold1_fedavg_r10_seed42", _camera_payload(1, 42))
    write_run(root, "rev_camera_fold2_fedbn_r10_seed42", _camera_payload(2, 42, "fedbn"))
    # misplaced under a FLAME-looking name: caught by its recorded data split
    write_run(root, "rev_iid_fedbn_seed999", _camera_payload(3, 999, "fedbn"))


def _analyse(results, out):
    rc = ar.main(["--results_dir", results, "--output_dir", out, "--no_plots",
                  "--bootstrap_B", "200"])
    assert rc in (0, None)


def _same_tree(a, b):
    cmp = filecmp.dircmp(a, b)
    stack, problems = [cmp], []
    while stack:
        c = stack.pop()
        problems += [os.path.join(c.left, n) for n in c.left_only + c.right_only + c.funny_files]
        _, mismatch, errors = filecmp.cmpfiles(c.left, c.right, c.common_files, shallow=False)
        problems += [os.path.join(c.left, n) for n in mismatch + errors]
        stack.extend(c.subdirs.values())
    return problems


def test_camera_runs_under_results_change_no_flame_output(tmp_path):
    results = str(tmp_path / "results")
    _flame_tree(results)
    _analyse(results, str(tmp_path / "before"))
    _add_camera_runs(results)
    _analyse(results, str(tmp_path / "after"))
    files = [os.path.join(d, f) for d, _, fs in os.walk(tmp_path / "before") for f in fs]
    assert files, "the FLAME analysis wrote nothing"
    assert _same_tree(str(tmp_path / "before"), str(tmp_path / "after")) == []


def test_collect_runs_never_yields_a_camera_run(tmp_path):
    results = str(tmp_path / "results")
    _flame_tree(results)
    _add_camera_runs(results)
    dirs = sorted(os.path.relpath(r["run_dir"], os.path.abspath(results)).replace("\\", "/")
                  for r in ar.collect_runs(results, warn=False))
    assert dirs == sorted(name for name, *_ in FLAME_RUNS)
    assert all(not ar.is_camera_run(os.path.join(results, n)) for n, *_ in FLAME_RUNS)


def test_generator_places_every_camera_run_in_the_camera_tree():
    cfg = prc.load_config(prc.DEFAULT_CONFIG)["revision"]
    fl = prc.expand_all(cfg, prc.CAMERA_BLOCK)[prc.CAMERA_BLOCK]
    smoke = prc.expand_all(cfg, prc.CAMERA_SMOKE_BLOCK)[prc.CAMERA_SMOKE_BLOCK]
    base = prc.expand_all(cfg, prc.CAMERA_BASELINE_BLOCK)[prc.CAMERA_BASELINE_BLOCK]
    assert len(fl) == 45 and len(smoke) == 2 and len(base) == 30
    assert all(r.output_dir.startswith("results/camera/maxn/rev_camera_fold") for r in fl)
    assert [r.output_dir for r in smoke] == list(cfg[prc.CAMERA_BLOCK]["determinism_gate"]["runs"])
    assert all(r.output_dir.startswith("results/camera/desktop/rev_camera_fold") for r in base)
    # and every FLAME block stays out of it
    for name, runs in prc.expand_all(prc.apply_all_seeds(cfg)).items():
        assert not any(r.output_dir.startswith("results/camera/") for r in runs), name
    # --baseline_root does not move the camera baselines
    by_block = {prc.CAMERA_BASELINE_BLOCK: base}
    prc.rebase_baselines(by_block, "results/rev_baselines_sel")
    assert all(r.output_dir.startswith("results/camera/desktop/") for r in base)


def test_run_matrix_namespaces_keep_camera_and_flame_apart():
    assert rm.CAMERA_TESTBED_BLOCKS == prc.CAMERA_TESTBED_BLOCKS
    assert rm.CAMERA_RESULTS == prc.CAMERA_RESULTS
    cam = [{"out_dir": "results/camera/maxn/rev_camera_fold0_fedavg_r10_seed42"}]
    flame = [{"out_dir": "results/pc_maxn/rev_iid_fedavg_r10_seed42"}]
    rm.check_namespace(cam, "maxn", "camera_sensor_skew")
    rm.check_namespace(flame, "maxn", "maxn_long_horizon")
    with pytest.raises(SystemExit):
        rm.check_namespace(flame, "maxn", "camera_sensor_skew")      # FLAME run in a camera block
    with pytest.raises(SystemExit):
        rm.check_namespace(cam, "maxn", "maxn_long_horizon")         # camera run in a FLAME block
    with pytest.raises(SystemExit):
        rm.check_namespace(cam, "maxn")                              # --script without --block
    with pytest.raises(SystemExit):
        rm.check_namespace([{"out_dir": "results/camera/x"}], "heterogeneous")
    assert rm.namespace_root("maxn", "camera_determinism_smoke") == "results/camera/maxn"
    assert rm.namespace_root("maxn", "maxn_long_horizon") == "results/pc_maxn"


def test_camera_fold_digests_come_from_the_materialised_manifest(tmp_path):
    path = tmp_path / "fl_materialised_manifest.csv"
    assert rm.camera_digests(str(path)) == {}
    rows = ["fold,node,split,class,n_images,n_scenes,fold_manifest_md5",
            "0,node_a,train,Fire,10,3," + "a" * 32, "0,node_b,test,No_Fire,4,1," + "a" * 32,
            "1,node_a,train,Fire,10,3," + "b" * 32]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert rm.camera_digests(str(path)) == {"camera_fold0": "a" * 32, "camera_fold1": "b" * 32}
    digests = rm.expected_digests(camera_path=str(path))
    assert digests["camera_fold1"] == "b" * 32 and "dirichlet_0.1" in digests   # FLAME ones untouched
    path.write_text("\n".join(rows + ["1,node_c,val,Fire,2,1," + "c" * 32]) + "\n",
                    encoding="utf-8")
    with pytest.raises(SystemExit):
        rm.camera_digests(str(path))


def test_camera_analyses_read_only_the_camera_tree():
    from scripts import camera_analysis_a as ca
    with pytest.raises(SystemExit):
        ca.check_camera_root(os.path.join(_REPO_ROOT, "results", "pc_maxn"))
    with pytest.raises(SystemExit):
        ca.check_camera_root(os.path.join(_REPO_ROOT, "results"))
    ca.check_camera_root(os.path.join(_REPO_ROOT, "results", "camera", "desktop", "loso"))
    from scripts import camera_loso as cl
    assert cl.build_parser().parse_args([]).output_root.replace("\\", "/") \
        .startswith("results/camera/")


# --------------------------------------------------------------------------- pilot footage
PILOT = os.path.join("data", "raw", "camera_pilot")


def test_pilot_paths_are_recognised():
    from scripts.camera_labels import is_pilot_path
    assert is_pilot_path(PILOT) and is_pilot_path(os.path.join(PILOT, "labels.csv"))
    assert is_pilot_path(PILOT.replace(os.sep, "/") + "/node_a")
    assert not is_pilot_path(os.path.join("data", "raw", "camera"))
    assert not is_pilot_path(os.path.join("data", "raw", "camera_pilots_x"))
    assert not is_pilot_path(None)


@pytest.mark.parametrize("module,argv", [
    ("camera_manifests", ["--labels_csv", os.path.join(PILOT, "labels.csv")]),
    ("camera_fl_prepare", ["--raw_dir", PILOT, "--verify"]),
    ("camera_fl_prepare", ["--labels_csv", os.path.join(PILOT, "labels.csv"), "--verify"]),
    ("camera_loso", ["--data_dir", PILOT]),
    ("camera_analysis_a", ["--labels_csv", os.path.join(PILOT, "labels.csv")]),
    ("camera_analysis_b", ["--labels_csv", os.path.join(PILOT, "labels.csv")]),
])
def test_study_scripts_refuse_pilot_footage(module, argv):
    import importlib
    mod = importlib.import_module("scripts." + module)
    with pytest.raises(SystemExit, match="pilot footage is outside the study"):
        mod.main(argv)


def test_pilot_frame_table_never_writes_into_the_study_splits(tmp_path):
    from scripts import camera_labels as cl
    with pytest.raises(SystemExit, match="--splits_dir must lie under camera_pilot"):
        cl.main(["--data_dir", str(tmp_path / "camera_pilot")])       # default study splits
    with pytest.raises(SystemExit, match="pilot footage is outside the study"):
        cl.main(["--data_dir", str(tmp_path / "camera"),
                 "--splits_dir", str(tmp_path / "camera_pilot" / "splits")])
