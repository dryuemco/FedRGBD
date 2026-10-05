"""scripts/fetch_results.ps1: which run directories the hourly fetch considers.

Both filters are read from the script's own text -- the remote ``find -regex`` (POSIX
ERE, matched against the whole path) and ``$RunPathRegex`` (.NET, applied to every
line) -- and must agree: FLAME runs exactly as before, camera runs of every
configuration (POST_5B_CHECKLIST c3), nothing else.
"""

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEXT = open(os.path.join(_REPO, "scripts", "fetch_results.ps1"), encoding="utf-8").read()

FIND = re.search(r"find results -maxdepth (\d+) -name results\.json -mmin \+\{1\} "
                 r"-regextype posix-extended -regex '([^']+)'", _TEXT)
RUN_PATH = re.search(r"(?m)^\$RunPathRegex = '([^']+)'", _TEXT).group(1)

FETCHED = {
    "results/rev_iid_fedavg_seed42": "rev_iid_fedavg_seed42",
    "results/rev_noniid_fedbn_r10_seed123": "rev_noniid_fedbn_r10_seed123",
    "results/pc_maxn/rev_iid_fedavg_r10_seed42": "pc_maxn/rev_iid_fedavg_r10_seed42",
    "results/camera/maxn/rev_camera_fold0_fedavg_r10_seed42":
        "camera/maxn/rev_camera_fold0_fedavg_r10_seed42",
    "results/camera/maxn/diag_camera_smoke": "camera/maxn/diag_camera_smoke",
    "results/camera/maxn/diag_camera_smoke_r2": "camera/maxn/diag_camera_smoke_r2",
    "results/camera/desktop/rev_camera_loso_s01": "camera/desktop/rev_camera_loso_s01",
}
NOT_FETCHED = (
    "results/diag_smoke_maxn",                        # FLAME diagnostics never were
    "results/_interrupted/20261001_032913/pc_maxn/rev_noniid_fedprox_0.01_r10_seed789",
    "results/_gate_failed/pc_maxn/rev_iid_fedavg_r10_seed42",
    "results/camera/maxn/rev_iid_fedavg_seed1",       # camera namespace, not a camera run
    "results/camera/rev_camera_fold0",                # no configuration level
    "results/camera/maxn/sub/rev_camera_fold0",       # one level too deep
    "results/cameraX/maxn/rev_camera_fold0",
    "results/rev_iid_fedavg_seed42/predictions",
)


def _depth(path):
    return path.count("/")          # find results -maxdepth N counts levels below results


def test_both_filters_are_found_in_the_script():
    assert FIND and RUN_PATH


def test_flame_and_camera_runs_are_fetched_to_the_same_relative_path():
    maxdepth, ere = int(FIND.group(1)), FIND.group(2)
    for run, rel in FETCHED.items():
        path = run + "/results.json"
        assert _depth(path) <= maxdepth, path
        assert re.fullmatch(ere, path), path
        m = re.match(RUN_PATH, path)
        assert m and m.group(1) == rel, path


def test_nothing_else_is_fetched():
    maxdepth, ere = int(FIND.group(1)), FIND.group(2)
    for run in NOT_FETCHED:
        path = run + "/results.json"
        assert not (_depth(path) <= maxdepth and re.fullmatch(ere, path)), path
        assert not re.match(RUN_PATH, path), path
