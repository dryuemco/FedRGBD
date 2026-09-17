"""End-to-end test of ``scripts/run_p0_leakage_and_split.py``.

Runs the whole P0 chain (audit -> decision -> group-safe split -> re-audit)
as a subprocess on the synthetic near-duplicate dataset from
``tests/test_leakage.py`` (12 triplets + 4 singletons, 40 JPEGs), with a
deliberately *leaky* image-level split pre-installed in ``processed/`` so the
pre-split (v1) audit has something to flag.  Then checks resumability
(second run skips every finished step), ``--dry_run`` (commands printed,
nothing executed), the missing-data exit code and the Kaggle-archive
flattening helper.  CPU only, a few seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from scripts import run_p0_leakage_and_split as p0
from tests.test_leakage import THRESHOLD, _build_raw_dataset, _copy_raw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_p0_leakage_and_split.py")
NODES = 2
ALPHA = 1.0
SUB = 0.5
EXPECTED_SPLITS = ["iid", "non_iid_label", "dirichlet_1",
                   "iid_sub0.5", "non_iid_label_sub0.5", "dirichlet_1_sub0.5"]


def _run(*extra, **kw):
    cmd = [sys.executable, SCRIPT, "--skip_download", "--workers", "1", "--threshold", str(THRESHOLD),
           "--nodes", str(NODES), "--dirichlet_alpha", str(ALPHA), "--dirichlet_min_size", "1",
           "--subsample_frac", str(SUB), "--link_mode", "copy", "--examples", "3",
           "--sweep", "4", "8", "12"] + list(extra)
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=kw.get("timeout", 240))


def _install_leaky_v1_split(raw_dir, processed_dir):
    """An image-level split: base0's copies sit in node_a/test AND node_b/train."""
    root = os.path.join(processed_dir, "iid")
    _copy_raw(raw_dir, "Fire/base1_orig.jpg", os.path.join(root, "node_a", "train", "Fire"))
    _copy_raw(raw_dir, "No_Fire/base9_orig.jpg", os.path.join(root, "node_a", "train", "No_Fire"))
    _copy_raw(raw_dir, "Fire/base0_shift.jpg", os.path.join(root, "node_a", "test", "Fire"))
    _copy_raw(raw_dir, "Fire/base3_orig.jpg", os.path.join(root, "node_a", "test", "Fire"))
    _copy_raw(raw_dir, "Fire/base0_orig.jpg", os.path.join(root, "node_b", "train", "Fire"))
    _copy_raw(raw_dir, "No_Fire/base10_orig.jpg", os.path.join(root, "node_b", "test", "No_Fire"))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("p0"))
    raw_dir, _ = _build_raw_dataset(root)
    processed = os.path.join(root, "processed")
    analysis = os.path.join(root, "analysis")
    _install_leaky_v1_split(raw_dir, processed)
    first = _run("--data_dir", raw_dir, "--processed_dir", processed, "--analysis_dir", analysis)
    return {"raw": raw_dir, "processed": processed, "analysis": analysis, "first": first}


# --------------------------------------------------------------------------- #
# full run
# --------------------------------------------------------------------------- #
def test_full_run_exits_zero_and_writes_summary(pipeline):
    proc = pipeline["first"]
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = os.path.join(pipeline["analysis"], "P0_SUMMARY.md")
    assert os.path.isfile(summary)
    text = open(summary, encoding="utf-8").read()
    assert "## Raw dataset near-duplicate statistics" in text
    assert "## Threshold sensitivity" in text
    assert "## Post-split audit" in text
    assert "**PASS**" in text
    assert "## Split digests" in text
    for name in EXPECTED_SPLITS:
        assert "{}/manifest.csv".format(name) in text
    # every step's command is echoed, so the log is reproducible
    assert proc.stdout.count("$ ") >= 4
    assert "analyze_flame_leakage.py" in proc.stdout and "data_splitter.py" in proc.stdout
    assert "--clean --verify" in proc.stdout


def test_v1_audit_flags_the_leaky_image_level_split(pipeline):
    v1 = json.load(open(os.path.join(pipeline["analysis"], "v1_audit", "leakage_report.json")))
    iid = v1["processed_splits"]["iid"]
    assert iid["groups_spanning_multiple_nodes"] >= 1
    assert iid["nodes"]["node_a"]["test"]["with_train_duplicate_any_node"] == 1
    assert "RE-SPLIT REQUIRED" in pipeline["first"].stdout
    assert "RE-SPLIT REQUIRED" in open(os.path.join(pipeline["analysis"], "P0_SUMMARY.md"), encoding="utf-8").read()


def test_raw_audit_has_diagnostics_and_group_file(pipeline):
    report = json.load(open(os.path.join(pipeline["analysis"], "leakage_report.json")))
    assert [r["threshold"] for r in report["threshold_sweep"]] == [4, 8, 12]
    assert "sequence_heuristic" in report
    assert "exact_duplicate_files_md5" in report["dataset"]
    assert os.path.isfile(os.path.join(pipeline["analysis"], "groups.json"))
    assert os.path.isfile(os.path.join(pipeline["analysis"], "example_groups.txt"))


def test_post_split_audit_reports_zero_leakage(pipeline):
    report = json.load(open(os.path.join(pipeline["analysis"], "post_split_audit", "leakage_report.json")))
    splits = report["processed_splits"]
    assert set(EXPECTED_SPLITS) <= set(splits)
    assert p0.check_zero_leakage(splits, EXPECTED_SPLITS) == []
    for rep in splits.values():
        assert rep["groups_spanning_multiple_nodes"] == 0
        assert rep["global_test_leak_rate_any_node"] == 0.0
        assert rep["global_val_leak_rate_any_node"] == 0.0
    stats = json.load(open(os.path.join(pipeline["processed"], "split_stats.json")))
    assert stats["_meta"]["verify_ok"] is True
    assert stats["_meta"]["nodes"] == NODES
    # the leaky v1 tree was replaced (--clean): base0 copies no longer split across nodes
    assert not os.path.exists(os.path.join(pipeline["processed"], "iid", "node_b", "train", "Fire", "base0_orig.jpg")) \
        or not os.path.exists(os.path.join(pipeline["processed"], "iid", "node_a", "test", "Fire", "base0_shift.jpg"))


# --------------------------------------------------------------------------- #
# resumability / dry run / failure paths
# --------------------------------------------------------------------------- #
def test_second_invocation_skips_finished_steps(pipeline):
    stats_path = os.path.join(pipeline["processed"], "split_stats.json")
    before = os.path.getmtime(stats_path)
    proc = _run("--data_dir", pipeline["raw"], "--processed_dir", pipeline["processed"],
                "--analysis_dir", pipeline["analysis"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for tag in ("SKIP 1a", "SKIP 1b", "SKIP 3", "SKIP 4"):
        assert tag in proc.stdout, proc.stdout
    assert "$ " not in proc.stdout  # no subprocess was launched
    assert os.path.getmtime(stats_path) == before
    # the summary is regenerated from the existing reports on every run
    text = open(os.path.join(pipeline["analysis"], "P0_SUMMARY.md"), encoding="utf-8").read()
    assert "**PASS**" in text and "## Split digests" in text


def test_dry_run_prints_commands_without_running(pipeline, tmp_path):
    analysis = str(tmp_path / "analysis_dry")
    processed = str(tmp_path / "processed_dry")
    proc = _run("--data_dir", pipeline["raw"], "--processed_dir", processed, "--analysis_dir", analysis,
                "--dry_run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY RUN" in proc.stdout
    assert "analyze_flame_leakage.py" in proc.stdout
    assert "data_splitter.py" in proc.stdout
    assert "--group_file" in proc.stdout and "--clean --verify" in proc.stdout
    assert "--sweep 4 8 12" in proc.stdout and "--sequence_heuristic" in proc.stdout
    assert not os.path.exists(analysis)
    assert not os.path.exists(processed)


def test_missing_dataset_with_skip_download_exits_2(tmp_path):
    proc = _run("--data_dir", str(tmp_path / "nowhere"), "--processed_dir", str(tmp_path / "p"),
                "--analysis_dir", str(tmp_path / "a"))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "kaggle datasets download" in proc.stdout
    assert p0.KAGGLE_DATASET in proc.stdout


def test_force_reruns_the_split(pipeline):
    stats_path = os.path.join(pipeline["processed"], "split_stats.json")
    before = json.load(open(stats_path))
    proc = _run("--data_dir", pipeline["raw"], "--processed_dir", pipeline["processed"],
                "--analysis_dir", pipeline["analysis"], "--force")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("data_splitter.py") >= 1 and "SKIP 3" not in proc.stdout
    after = json.load(open(stats_path))
    # same seed + same group file -> identical partition
    assert {k: v for k, v in before.items() if k != "_meta"} == {k: v for k, v in after.items() if k != "_meta"}


# --------------------------------------------------------------------------- #
# helpers used by step 0
# --------------------------------------------------------------------------- #
def test_flatten_dataset_dir_merges_nested_layouts(tmp_path):
    root = tmp_path / "flame"
    layout = {
        "Training/Fire": ["a.jpg", "b.jpg"],
        "Training/No_Fire": ["c.jpg"],
        "Test/Fire": ["a.jpg", "d.jpg"],          # a.jpg collides with Training/Fire/a.jpg
        "Test/nofire": ["e.png"],                 # lower-case alias
        "Test/Fire/notes": ["readme.txt"],        # non-image, left alone
    }
    for sub, files in layout.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_bytes(("%s/%s" % (sub, f)).encode())

    assert not p0.has_class_dirs(str(root))
    preview = p0.flatten_dataset_dir(str(root), dry_run=True)
    assert preview == {"Fire": 4, "No_Fire": 2, "renamed": 0}
    assert (root / "Training" / "Fire" / "a.jpg").exists()  # dry run moved nothing

    moved = p0.flatten_dataset_dir(str(root))
    assert moved["Fire"] == 4 and moved["No_Fire"] == 2 and moved["renamed"] == 1
    assert p0.has_class_dirs(str(root))
    fire = sorted(os.listdir(root / "Fire"))
    assert "a.jpg" in fire and "b.jpg" in fire and "d.jpg" in fire
    assert any(f.endswith("__a.jpg") for f in fire), fire   # collision kept, not deleted
    assert sorted(os.listdir(root / "No_Fire")) == ["c.jpg", "e.png"]
    # all 6 images survived, no directory with images left behind
    total = sum(1 for _, _, fs in os.walk(root) for f in fs if f.lower().endswith(p0.IMAGE_EXTS))
    assert total == 6
    assert (root / "Test" / "Fire" / "notes" / "readme.txt").exists()
    assert not (root / "Training").exists()


def test_processed_has_split_detection(tmp_path):
    assert not p0.processed_has_split(str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    assert not p0.processed_has_split(str(tmp_path / "empty"))
    (tmp_path / "empty" / "iid" / "node_a" / "train").mkdir(parents=True)
    assert p0.processed_has_split(str(tmp_path / "empty"))


def test_expected_split_names_match_splitter_convention():
    ns = p0.build_parser().parse_args(["--dirichlet_alpha", "0.1", "1.0", "--subsample_frac", "0.05"])
    assert p0.expected_split_names(ns) == ["iid", "non_iid_label", "dirichlet_0.1", "dirichlet_1",
                                           "iid_sub0.05", "non_iid_label_sub0.05",
                                           "dirichlet_0.1_sub0.05", "dirichlet_1_sub0.05"]
