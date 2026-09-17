#!/usr/bin/env python3
"""FedRGBD -- P0 pipeline: FLAME leakage audit -> group-safe re-split -> re-audit.

One resumable command for the hardware-independent part of the NCAA revision
plan (``docs/REVISION_PLAN_TR.md`` section 2.1-2.4).  It has to run on the
desktop **before** any Jetson experiment, because the protocol decision
(re-split or not) and the new ``data/processed`` come out of it.

Steps (each external step is a subprocess whose exact command is echoed, so
the log is reproducible; finished steps are skipped on a re-run unless
``--force``):

  0. data       ``<data_dir>/Fire`` and ``<data_dir>/No_Fire`` must exist.  If
                not, the Kaggle dataset is downloaded and flattened when the
                ``kaggle`` CLI and credentials are available; otherwise the
                download instructions are printed and the script exits with 2.
  1. audit      (a) leakage audit of the EXISTING ``data/processed`` (paper v1,
                image-level split) -> ``<analysis_dir>/v1_audit`` -- only when
                it contains a split;  (b) raw-data near-duplicate analysis ->
                ``<analysis_dir>`` (``groups.json``, ``leakage_report.json``,
                ``example_groups.txt``; ``--sweep 4 6 8 10 12
                --sequence_heuristic --examples 20``).
  2. decision   prints the numbers the plan asks for (test leak rate, groups
                spanning nodes, threshold sweep, cross-label groups, group size
                histogram, exact MD5 duplicates) and writes
                ``<analysis_dir>/P0_SUMMARY.md``.
  3. split      ``src/data/data_splitter.py --group_file <analysis_dir>/groups.json
                --dirichlet_alpha ... --subsample_frac ... --clean --verify``;
                a ``VERIFY FAIL`` aborts with exit 1.
  4. re-audit   leakage audit of the NEW tree -> ``<analysis_dir>/post_split_audit``;
                every val/test leak rate must be 0 and no group may span two
                nodes (exit 1 otherwise).  The md5 of every
                ``<processed_dir>/*/manifest.csv`` and of ``split_stats.json``
                is printed and appended to ``P0_SUMMARY.md`` so the three
                nodes can be compared (plan 2.3).

Usage
-----
    python3 scripts/run_p0_leakage_and_split.py                  # defaults = plan 2.1-2.4
    python3 scripts/run_p0_leakage_and_split.py --hash both --workers 8
    python3 scripts/run_p0_leakage_and_split.py --dry_run        # only print the commands
    python3 scripts/run_p0_leakage_and_split.py --skip_download  # never touch Kaggle

Kaggle download needs the ``kaggle`` package (``pip install kaggle``) and an API
token: ``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``.

Exit codes: 0 success, 1 a step failed (subprocess error, VERIFY FAIL or
non-zero post-split leakage), 2 dataset missing and not downloadable.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LEAKAGE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "analyze_flame_leakage.py")
SPLITTER_SCRIPT = os.path.join(REPO_ROOT, "src", "data", "data_splitter.py")

KAGGLE_DATASET = "smrutisanchitadas/flame-dataset-fire-classification"
KAGGLE_ZIP = "flame-dataset-fire-classification.zip"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
CLASS_DIRS = {"fire": "Fire", "no_fire": "No_Fire", "nofire": "No_Fire"}
SWEEP = [4, 6, 8, 10, 12]
EXAMPLES = 20
LEAK_TOLERANCE = 0.02  # plan 2.2: below this the v1 seeds could be reused
HASH_CACHE_GLOBS = ("hashes_*.npz", "md5s.npz")

EXIT_OK, EXIT_FAIL, EXIT_NO_DATA = 0, 1, 2


class StepError(RuntimeError):
    """A pipeline step failed; the message is the reason."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    log("\n" + "=" * 78)
    log("  " + title)
    log("=" * 78)


def fmt_cmd(cmd: Sequence[str]) -> str:
    """Shell-pasteable rendering of an argv list (POSIX quoting; Windows tolerates it)."""
    out = []
    for a in cmd:
        a = str(a)
        if not a or any(c in a for c in " \t\"'$&|;<>()"):
            out.append("'" + a.replace("'", "'\\''") + "'")
        else:
            out.append(a)
    return " ".join(out)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def fmt_g(x: float) -> str:
    return "{:g}".format(float(x))


def has_class_dirs(data_dir: str) -> bool:
    """``<data_dir>/Fire`` and ``<data_dir>/No_Fire`` exist and hold >= 1 image each."""
    for cls in ("Fire", "No_Fire"):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            return False
        if not any(f.lower().endswith(IMAGE_EXTS) for f in os.listdir(d)):
            return False
    return True


def processed_has_split(processed_dir: str) -> bool:
    """True when ``processed_dir`` holds at least one ``<split>/<node>/train`` tree."""
    if not os.path.isdir(processed_dir):
        return False
    for split in os.listdir(processed_dir):
        split_root = os.path.join(processed_dir, split)
        if not os.path.isdir(split_root):
            continue
        if os.path.isfile(os.path.join(split_root, "manifest.csv")):
            return True
        for node in os.listdir(split_root):
            if os.path.isdir(os.path.join(split_root, node, "train")):
                return True
    return False


def seed_hash_cache(dst_dir: str, candidates: Sequence[str]) -> List[str]:
    """Copy the leakage script's hash/MD5 caches into ``dst_dir`` so a second
    audit over the same raw images does not re-hash 48k files.  The script
    validates the cached path list itself, so a stale copy is harmless."""
    copied = []
    os.makedirs(dst_dir, exist_ok=True)
    for src_dir in candidates:
        if not os.path.isdir(src_dir) or os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            continue
        for pattern in HASH_CACHE_GLOBS:
            for src in glob.glob(os.path.join(src_dir, pattern)):
                dst = os.path.join(dst_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied.append(dst)
    return copied


# --------------------------------------------------------------------------- #
# step 0: dataset presence / Kaggle download
# --------------------------------------------------------------------------- #
def kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".kaggle")
    return os.path.isfile(os.path.join(cfg_dir, "kaggle.json"))


def kaggle_command(python: str) -> Optional[List[str]]:
    """argv prefix for the Kaggle CLI, or None when it is not installed."""
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    try:
        probe = subprocess.run([python, "-c", "import kaggle.cli"], capture_output=True)
    except OSError:
        return None
    if probe.returncode == 0:
        return [python, "-c",
                "import sys; from kaggle.cli import main; sys.argv = ['kaggle'] + sys.argv[1:]; main()"]
    return None


def print_download_instructions(data_dir: str) -> None:
    log("!" * 78)
    log("  FLAME veri seti bulunamadi: {}/Fire ve {}/No_Fire yok.".format(data_dir, data_dir))
    log("  Indirmek icin Kaggle hesabi + API anahtari gerekir (kaggle.com -> Settings ->")
    log("  'Create New Token' -> ~/.kaggle/kaggle.json, ya da KAGGLE_USERNAME/KAGGLE_KEY).")
    log("  Sonra:  pip install kaggle  ve bu komutu yeniden calistir (indirme otomatik),")
    log("  ya da elle:")
    log("    kaggle datasets download -d {} -p {}".format(KAGGLE_DATASET, os.path.dirname(data_dir) or "."))
    log("    unzip {} -d {}   # Fire/ ve No_Fire/ dogrudan {} altinda olmali".format(
        os.path.join(os.path.dirname(data_dir) or ".", KAGGLE_ZIP), data_dir, data_dir))
    log("  EN: dataset missing -- install `kaggle`, put your API token in ~/.kaggle/kaggle.json")
    log("      (or set KAGGLE_USERNAME/KAGGLE_KEY) and re-run, or unzip the Kaggle archive so")
    log("      that {}/Fire and {}/No_Fire exist.".format(data_dir, data_dir))
    log("!" * 78)


def flatten_dataset_dir(data_dir: str, dry_run: bool = False) -> Dict[str, int]:
    """Move every image found under a ``Fire``/``No_Fire``(``nofire``) folder anywhere
    below ``data_dir`` into ``<data_dir>/Fire`` / ``<data_dir>/No_Fire``.

    Handles the layouts the Kaggle archive may use (class folders at the top,
    under ``Training``/``Test``, or under a nested ``flame_dataset`` folder).
    Nothing is ever deleted: a basename that already exists in the target is
    renamed ``<subtree>__<name>`` (the MD5 pass of the audit reports it as an
    exact duplicate if it is one).  Emptied folders are removed afterwards.
    """
    data_dir = os.path.abspath(data_dir)
    moved: Dict[str, int] = {"Fire": 0, "No_Fire": 0, "renamed": 0}
    targets = {cls: os.path.join(data_dir, cls) for cls in ("Fire", "No_Fire")}
    for root, dirs, files in os.walk(data_dir, topdown=True):
        cls = CLASS_DIRS.get(os.path.basename(root).lower())
        if cls is None or os.path.abspath(root) == targets[cls]:
            continue
        rel_parent = os.path.relpath(os.path.dirname(root), data_dir).replace(os.sep, "_")
        prefix = "" if rel_parent in (".", "") else rel_parent + "__"
        for f in files:
            if not f.lower().endswith(IMAGE_EXTS):
                continue
            src = os.path.join(root, f)
            dst = os.path.join(targets[cls], f)
            if os.path.exists(dst):
                dst = os.path.join(targets[cls], prefix + f)
                n = 1
                while os.path.exists(dst):
                    stem, ext = os.path.splitext(prefix + f)
                    dst = os.path.join(targets[cls], "{}_{}{}".format(stem, n, ext))
                    n += 1
                moved["renamed"] += 1
            if not dry_run:
                os.makedirs(targets[cls], exist_ok=True)
                shutil.move(src, dst)
            moved[cls] += 1
    if not dry_run:
        # remove emptied directories bottom-up (never touches non-empty ones)
        for root, dirs, files in os.walk(data_dir, topdown=False):
            if os.path.abspath(root) == data_dir or root in targets.values():
                continue
            try:
                os.rmdir(root)
            except OSError:
                pass
    return moved


def ensure_dataset(args) -> None:
    banner("Step 0: dataset check ({})".format(args.data_dir))
    if has_class_dirs(args.data_dir):
        n_fire = sum(1 for f in os.listdir(os.path.join(args.data_dir, "Fire")) if f.lower().endswith(IMAGE_EXTS))
        n_nofire = sum(1 for f in os.listdir(os.path.join(args.data_dir, "No_Fire")) if f.lower().endswith(IMAGE_EXTS))
        log("  OK: Fire={} No_Fire={} images".format(n_fire, n_nofire))
        return
    # a nested layout (e.g. an unzipped archive) may already be there -> flatten it
    if os.path.isdir(args.data_dir):
        preview = flatten_dataset_dir(args.data_dir, dry_run=True)
        if preview["Fire"] and preview["No_Fire"]:
            log("  nested class folders found under {} -> flattening".format(args.data_dir))
            if not args.dry_run:
                moved = flatten_dataset_dir(args.data_dir)
                log("  moved Fire={} No_Fire={} (renamed {} name collisions)".format(
                    moved["Fire"], moved["No_Fire"], moved["renamed"]))
                if has_class_dirs(args.data_dir):
                    return
            else:
                return
    if args.skip_download:
        print_download_instructions(args.data_dir)
        raise SystemExit(EXIT_NO_DATA)
    kcmd = kaggle_command(args.python)
    if kcmd is None or not kaggle_credentials_present():
        if kcmd is None:
            log("  `kaggle` CLI not found (pip install kaggle).")
        else:
            log("  Kaggle credentials not found (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).")
        print_download_instructions(args.data_dir)
        raise SystemExit(EXIT_NO_DATA)

    download_dir = os.path.dirname(args.data_dir.rstrip("/\\")) or "."
    zip_path = os.path.join(download_dir, KAGGLE_ZIP)
    cmd = kcmd + ["datasets", "download", "-d", KAGGLE_DATASET, "-p", download_dir]
    if os.path.isfile(zip_path):
        log("  reusing existing archive {}".format(zip_path))
    else:
        run_cmd(cmd, args.dry_run, "kaggle download")
    if args.dry_run:
        log("  [dry_run] would unzip {} into {} and flatten Fire/ No_Fire/".format(zip_path, args.data_dir))
        return
    if not os.path.isfile(zip_path):
        # kaggle names the archive after the dataset slug; pick up whatever it wrote
        zips = glob.glob(os.path.join(download_dir, "*.zip"))
        if not zips:
            raise StepError("kaggle download finished but no .zip found in {}".format(download_dir))
        zip_path = max(zips, key=os.path.getmtime)
    log("  extracting {} -> {}".format(zip_path, args.data_dir))
    os.makedirs(args.data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.data_dir)
    moved = flatten_dataset_dir(args.data_dir)
    log("  flattened: Fire={} No_Fire={} (renamed {} name collisions)".format(
        moved["Fire"], moved["No_Fire"], moved["renamed"]))
    if not has_class_dirs(args.data_dir):
        raise StepError("archive extracted but {}/Fire and {}/No_Fire are still missing -- "
                        "inspect the layout manually".format(args.data_dir, args.data_dir))
    if not args.keep_zip:
        os.remove(zip_path)
        log("  removed {} (use --keep_zip to keep it)".format(zip_path))


# --------------------------------------------------------------------------- #
# subprocess runner
# --------------------------------------------------------------------------- #
def run_cmd(cmd: Sequence[str], dry_run: bool, what: str) -> None:
    log("\n$ " + fmt_cmd(cmd))
    if dry_run:
        log("  [dry_run] not executed")
        return
    t0 = time.perf_counter()
    proc = subprocess.run(list(cmd), cwd=REPO_ROOT)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise StepError("{} failed with exit code {} ({:.1f}s)".format(what, proc.returncode, dt))
    log("  [{} done in {:.1f}s]".format(what, dt))


def leakage_cmd(args, output_dir: str, processed_dir: Optional[str] = None,
                diagnostics: bool = False) -> List[str]:
    cmd = [args.python, LEAKAGE_SCRIPT, "--data_dir", args.data_dir, "--output_dir", output_dir,
           "--threshold", str(args.threshold), "--hash", args.hash, "--workers", str(args.workers)]
    if args.hash == "both":
        cmd += ["--phash_threshold", str(args.phash_threshold)]
    if processed_dir:
        cmd += ["--processed_dir", processed_dir]
    if diagnostics:
        cmd += ["--sweep"] + [str(t) for t in args.sweep]
        cmd += ["--sequence_heuristic", "--examples", str(args.examples)]
    return cmd


def splitter_cmd(args, group_file: str) -> List[str]:
    cmd = [args.python, SPLITTER_SCRIPT, "--data_dir", args.data_dir, "--output_dir", args.processed_dir,
           "--nodes", str(args.nodes), "--seed", str(args.seed), "--group_file", group_file]
    if args.dirichlet_alpha:
        cmd += ["--dirichlet_alpha"] + [fmt_g(a) for a in args.dirichlet_alpha]
        cmd += ["--dirichlet_min_size", str(args.dirichlet_min_size)]
    if args.subsample_frac:
        cmd += ["--subsample_frac"] + [fmt_g(f) for f in args.subsample_frac]
    if args.link_mode != "symlink":
        cmd += ["--link_mode", args.link_mode]
    cmd += ["--clean", "--verify"]
    return cmd


def expected_split_names(args) -> List[str]:
    base = ["iid", "non_iid_label"] + ["dirichlet_{}".format(fmt_g(a)) for a in (args.dirichlet_alpha or [])]
    subs = ["{}_sub{}".format(b, fmt_g(f)) for b in base for f in (args.subsample_frac or [])]
    return base + subs


# --------------------------------------------------------------------------- #
# step 1: audits
# --------------------------------------------------------------------------- #
def audit_is_done(report_path: str, need_sweep: bool, newer_than: Optional[str] = None) -> bool:
    report = load_json(report_path)
    if report is None:
        return False
    if need_sweep and ("threshold_sweep" not in report or "sequence_heuristic" not in report):
        return False
    if newer_than and os.path.isfile(newer_than) and os.path.getmtime(report_path) < os.path.getmtime(newer_than):
        return False
    return True


def step_audit(args) -> Dict[str, Any]:
    banner("Step 1: near-duplicate / leakage audit")
    v1_dir = os.path.join(args.analysis_dir, "v1_audit")
    v1_exists = processed_has_split(args.processed_dir)
    info: Dict[str, Any] = {"v1_audit": None, "raw_report": os.path.join(args.analysis_dir, "leakage_report.json")}

    # (a) audit of the existing (paper v1) split ------------------------- #
    if v1_exists:
        v1_report = os.path.join(v1_dir, "leakage_report.json")
        info["v1_audit"] = v1_report
        if audit_is_done(v1_report, need_sweep=False) and not args.force:
            log("  SKIP 1a: v1 audit exists ({})".format(v1_report))
        else:
            log("  1a: existing split found in {} -> auditing the image-level (v1) partition".format(args.processed_dir))
            if not args.dry_run:
                seed_hash_cache(v1_dir, [args.analysis_dir])
            run_cmd(leakage_cmd(args, v1_dir, processed_dir=args.processed_dir), args.dry_run, "v1 audit")
    else:
        log("  SKIP 1a: {} contains no split, nothing to audit before re-splitting".format(args.processed_dir))

    # (b) raw-data analysis -> groups.json ------------------------------- #
    raw_report = info["raw_report"]
    groups = os.path.join(args.analysis_dir, "groups.json")
    if audit_is_done(raw_report, need_sweep=True) and os.path.isfile(groups) and not args.force:
        log("  SKIP 1b: raw audit exists ({})".format(raw_report))
    else:
        if not args.dry_run:
            seed_hash_cache(args.analysis_dir, [v1_dir])
        run_cmd(leakage_cmd(args, args.analysis_dir,
                            processed_dir=args.processed_dir if v1_exists else None,
                            diagnostics=True), args.dry_run, "raw audit")
    return info


# --------------------------------------------------------------------------- #
# step 2: decision numbers -> P0_SUMMARY.md
# --------------------------------------------------------------------------- #
def histogram_summary(hist: Dict[str, int]) -> Dict[str, int]:
    out = {"size_1": 0, "size_2_5": 0, "size_6_20": 0, "size_gt_20": 0, "max_size": 0}
    for k, v in hist.items():
        size = int(k)
        out["max_size"] = max(out["max_size"], size)
        if size == 1:
            out["size_1"] += v
        elif size <= 5:
            out["size_2_5"] += v
        elif size <= 20:
            out["size_6_20"] += v
        else:
            out["size_gt_20"] += v
    return out


def split_audit_rows(processed_splits: Dict[str, Any]) -> List[List[str]]:
    """One row per split: name, groups spanning >1 node, global val/test leak, worst node leak."""
    rows = []
    for name in sorted(processed_splits):
        rep = processed_splits[name]
        worst = 0.0
        for nrep in rep.get("nodes", {}).values():
            for sp in ("val", "test"):
                if sp in nrep:
                    worst = max(worst, float(nrep[sp].get("leak_rate_any_node", 0.0)))
        rows.append([name, str(rep.get("groups_spanning_multiple_nodes", "?")),
                     "{:.4f}".format(rep.get("global_val_leak_rate_any_node", 0.0)),
                     "{:.4f}".format(rep.get("global_test_leak_rate_any_node", 0.0)),
                     "{:.4f}".format(worst), str(rep.get("unresolved_files", 0))])
    return rows


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join([line, sep] + body)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def step_decision(args, audit_info: Dict[str, Any]) -> Dict[str, Any]:
    banner("Step 2: decision numbers (plan 2.1 / 2.2)")
    summary_path = os.path.join(args.analysis_dir, "P0_SUMMARY.md")
    if args.dry_run:
        log("  [dry_run] would read {} and write {}".format(audit_info["raw_report"], summary_path))
        return {"summary_path": summary_path}
    report = load_json(audit_info["raw_report"])
    if report is None:
        raise StepError("missing or unreadable {}".format(audit_info["raw_report"]))
    ds = report.get("dataset", {})
    settings = report.get("settings", {})

    dataset_rows = [
        ["n_images", ds.get("n_images")],
        ["n_groups", ds.get("n_groups")],
        ["n_nontrivial_groups (size>1)", ds.get("n_nontrivial_groups")],
        ["images_in_nontrivial_groups", ds.get("images_in_nontrivial_groups")],
        ["fraction_images_with_near_duplicate", ds.get("fraction_images_with_near_duplicate")],
        ["largest_group", ds.get("largest_group")],
        ["cross_label_groups", ds.get("cross_label_groups")],
        ["exact_duplicate_images (identical hash)", ds.get("exact_duplicate_images")],
        ["exact_duplicate_files_md5 (byte-identical)", ds.get("exact_duplicate_files_md5", "n/a (--no_md5)")],
    ]
    hist = histogram_summary(ds.get("group_size_histogram", {}))
    hist_rows = [[k, str(v)] for k, v in hist.items()]

    sweep_rows = [[str(r["threshold"]), str(r["n_groups"]), str(r["n_nontrivial_groups"]),
                   str(r["images_in_nontrivial_groups"]), str(r["largest_group"]),
                   "{:.4f}".format(r["fraction_with_near_duplicate"])]
                  for r in report.get("threshold_sweep", [])]
    sweep_headers = ["threshold", "n_groups", "nontrivial", "imgs_in_nontrivial", "largest", "frac_near_dup"]

    seq = report.get("sequence_heuristic")
    seq_rows = []
    if seq:
        seq_rows = [["n_images_with_frame_number", str(seq["n_images_with_frame_number"])],
                    ["n_consecutive_pairs", str(seq["n_consecutive_pairs"])],
                    ["consecutive_pairs_same_group", str(seq["n_consecutive_pairs_same_group"])],
                    ["fraction_consecutive_pairs_same_group", str(seq["fraction_consecutive_pairs_same_group"])]]

    v1 = None
    if audit_info.get("v1_audit"):
        v1 = load_json(audit_info["v1_audit"])
    v1_splits = (v1 or {}).get("processed_splits") or report.get("processed_splits") or {}
    v1_rows = split_audit_rows(v1_splits)
    split_headers = ["split", "groups>1node", "val_leak_any", "test_leak_any", "worst_node", "unresolved"]

    worst_test = max([float(r.get("global_test_leak_rate_any_node", 0.0)) for r in v1_splits.values()] or [0.0])
    spanning = sum(int(r.get("groups_spanning_multiple_nodes", 0)) for r in v1_splits.values())
    if not v1_splits:
        decision = ("No previous data/processed split was found, so there is no v1 leakage number; "
                    "the group-safe split is produced regardless.")
    elif worst_test > LEAK_TOLERANCE or spanning > 0:
        decision = ("RE-SPLIT REQUIRED (plan 2.2): the image-level v1 split leaks {:.1%} of test "
                    "images (near-duplicate in some node's train) and {} near-duplicate groups span "
                    "several nodes. v1 results stay in a separate 'image-level split' table; all "
                    "revision runs use the group-safe split (run print_revision_commands.py --all_seeds)."
                    .format(worst_test, spanning))
    else:
        decision = ("Leakage is negligible ({:.1%} <= {:.0%}) and no group spans several nodes: the "
                    "v1 seeds could be reused (plan 2.2). The group-safe split is still written so "
                    "the reported protocol is leakage-free.".format(worst_test, LEAK_TOLERANCE))

    # ---- console ----------------------------------------------------- #
    log("  settings: method={} threshold={}{}".format(
        settings.get("method"), settings.get("threshold"),
        " phash_threshold={}".format(settings["phash_threshold"]) if "phash_threshold" in settings else ""))
    log("\n" + table(["dataset statistic", "value"], [[k, str(v)] for k, v in dataset_rows]))
    log("\n" + table(["group_size_histogram", "n_groups"], hist_rows))
    if sweep_rows:
        log("\n" + table(sweep_headers, sweep_rows))
    if seq_rows:
        log("\n" + table(["sequence heuristic", "value"], seq_rows))
    if v1_rows:
        log("\n" + table(split_headers, v1_rows))
    log("\n  DECISION: " + decision)

    # ---- P0_SUMMARY.md ----------------------------------------------- #
    lines = ["# P0 summary -- FLAME leakage audit and group-safe re-split", "",
             "Generated by `scripts/run_p0_leakage_and_split.py` on {}.".format(time.strftime("%Y-%m-%d %H:%M:%S")), "",
             "* data_dir: `{}`".format(args.data_dir),
             "* processed_dir: `{}`".format(args.processed_dir),
             "* hash: `{}`, threshold: `{}`{}".format(
                 settings.get("method"), settings.get("threshold"),
                 ", phash_threshold: `{}`".format(settings["phash_threshold"]) if "phash_threshold" in settings else ""),
             "* report: `{}`".format(audit_info["raw_report"]),
             "* group file: `{}`".format(report.get("group_file")), "",
             "## Raw dataset near-duplicate statistics", "",
             md_table(["statistic", "value"], [[k, str(v)] for k, v in dataset_rows]), "",
             "### Group size histogram (groups per size bucket)", "",
             md_table(["bucket", "n_groups"], hist_rows), ""]
    if sweep_rows:
        lines += ["## Threshold sensitivity (`--sweep`; groups.json uses threshold {})".format(settings.get("threshold")),
                  "", md_table(sweep_headers, sweep_rows), ""]
    if seq_rows:
        lines += ["## Filename frame-number heuristic (gap <= {})".format(seq["gap"]), "",
                  md_table(["statistic", "value"], seq_rows), ""]
    lines += ["## Pre-split audit of the previous `data/processed` (paper v1, image-level split)", ""]
    if v1_rows:
        lines += [md_table(split_headers, v1_rows), "",
                  "`val_leak_any` / `test_leak_any` = share of val/test images that have a near-duplicate "
                  "in the train split of *any* node (`global_*_leak_rate_any_node`); `worst_node` = max "
                  "per-node `leak_rate_any_node`.", ""]
    else:
        lines += ["No previous split found; nothing to audit.", ""]
    lines += ["## Decision (plan 2.2)", "", decision, ""]
    os.makedirs(args.analysis_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log("  summary written to {}".format(summary_path))
    return {"summary_path": summary_path, "decision": decision, "v1_worst_test_leak": worst_test}


# --------------------------------------------------------------------------- #
# step 3: group-safe split
# --------------------------------------------------------------------------- #
def split_is_done(args, group_file: str) -> bool:
    stats_path = os.path.join(args.processed_dir, "split_stats.json")
    stats = load_json(stats_path)
    if not stats:
        return False
    meta = stats.get("_meta", {})
    if meta.get("verify_ok") is not True:
        return False
    if not meta.get("group_file") or os.path.abspath(meta["group_file"]) != os.path.abspath(group_file):
        return False
    if int(meta.get("nodes", -1)) != args.nodes or int(meta.get("seed", -1)) != args.seed:
        return False
    if os.path.isfile(group_file) and os.path.getmtime(stats_path) < os.path.getmtime(group_file):
        return False
    for name in expected_split_names(args):
        if name not in stats or not os.path.isfile(os.path.join(args.processed_dir, name, "manifest.csv")):
            return False
    return True


def step_split(args) -> str:
    banner("Step 3: group-safe re-split ({} nodes, seed {})".format(args.nodes, args.seed))
    group_file = os.path.join(args.analysis_dir, "groups.json")
    if not args.dry_run and not os.path.isfile(group_file):
        raise StepError("{} missing -- step 1 did not produce a group file".format(group_file))
    if split_is_done(args, group_file) and not args.force:
        log("  SKIP 3: {}/split_stats.json already holds a VERIFY PASS split from this group file".format(
            args.processed_dir))
        return group_file
    run_cmd(splitter_cmd(args, group_file), args.dry_run, "data_splitter")
    if args.dry_run:
        return group_file
    stats = load_json(os.path.join(args.processed_dir, "split_stats.json"))
    if not stats or stats.get("_meta", {}).get("verify_ok") is not True:
        raise StepError("VERIFY FAIL: the splitter did not report verify_ok=true in {}/split_stats.json".format(
            args.processed_dir))
    missing = [n for n in expected_split_names(args) if n not in stats]
    if missing:
        raise StepError("split_stats.json lacks the expected splits: {}".format(", ".join(missing)))
    log("  VERIFY PASS confirmed in split_stats.json ({} splits)".format(
        len([k for k in stats if k != "_meta"])))
    return group_file


# --------------------------------------------------------------------------- #
# step 4: re-audit + md5 table
# --------------------------------------------------------------------------- #
def check_zero_leakage(processed_splits: Dict[str, Any], expected: Sequence[str]) -> List[str]:
    problems = []
    for name in expected:
        if name not in processed_splits:
            problems.append("split '{}' missing from the post-split audit".format(name))
    for name, rep in processed_splits.items():
        if int(rep.get("groups_spanning_multiple_nodes", 0)) != 0:
            problems.append("{}: groups_spanning_multiple_nodes = {}".format(name, rep["groups_spanning_multiple_nodes"]))
        for key in ("global_val_leak_rate_any_node", "global_test_leak_rate_any_node"):
            if float(rep.get(key, 0.0)) != 0.0:
                problems.append("{}: {} = {}".format(name, key, rep[key]))
        for node, nrep in rep.get("nodes", {}).items():
            for sp in ("val", "test"):
                if sp not in nrep:
                    continue
                for key in ("leak_rate_same_node", "leak_rate_any_node",
                            "with_train_duplicate_same_node", "with_train_duplicate_any_node"):
                    if float(nrep[sp].get(key, 0.0)) != 0.0:
                        problems.append("{}/{}/{}: {} = {}".format(name, node, sp, key, nrep[sp][key]))
        if int(rep.get("unresolved_files", 0)) != 0:
            problems.append("{}: {} processed files could not be mapped back to the raw images".format(
                name, rep["unresolved_files"]))
    return problems


def manifest_md5_rows(processed_dir: str) -> List[List[str]]:
    rows = []
    for split in sorted(os.listdir(processed_dir)):
        path = os.path.join(processed_dir, split, "manifest.csv")
        if os.path.isfile(path):
            with open(path, "r", newline="") as fh:
                n_rows = max(len(fh.read().splitlines()) - 1, 0)
            rows.append(["{}/manifest.csv".format(split), str(n_rows), md5_file(path)])
    stats_path = os.path.join(processed_dir, "split_stats.json")
    if os.path.isfile(stats_path):
        rows.append(["split_stats.json (raw bytes)", "", md5_file(stats_path)])
        stats = load_json(stats_path) or {}
        canon = json.dumps({k: v for k, v in stats.items() if k != "_meta"}, sort_keys=True,
                           separators=(",", ":"))
        rows.append(["split_stats.json (no _meta, canonical)", "", hashlib.md5(canon.encode("utf-8")).hexdigest()])
    return rows


def step_reaudit(args, summary_path: str) -> None:
    banner("Step 4: re-audit of the new split (must be zero leakage)")
    out_dir = os.path.join(args.analysis_dir, "post_split_audit")
    report_path = os.path.join(out_dir, "leakage_report.json")
    stats_path = os.path.join(args.processed_dir, "split_stats.json")
    if audit_is_done(report_path, need_sweep=False, newer_than=stats_path) and not args.force:
        log("  SKIP 4: post-split audit exists and is newer than split_stats.json ({})".format(report_path))
    else:
        if not args.dry_run:
            seed_hash_cache(out_dir, [args.analysis_dir, os.path.join(args.analysis_dir, "v1_audit")])
        run_cmd(leakage_cmd(args, out_dir, processed_dir=args.processed_dir), args.dry_run, "post-split audit")
    if args.dry_run:
        log("  [dry_run] would assert zero leakage in {} and print the manifest md5 table".format(report_path))
        return
    report = load_json(report_path)
    if report is None or "processed_splits" not in report:
        raise StepError("post-split audit produced no 'processed_splits' block in {}".format(report_path))
    splits = report["processed_splits"]
    rows = split_audit_rows(splits)
    headers = ["split", "groups>1node", "val_leak_any", "test_leak_any", "worst_node", "unresolved"]
    log("\n" + table(headers, rows))
    problems = check_zero_leakage(splits, expected_split_names(args))

    md5_rows = manifest_md5_rows(args.processed_dir)
    md5_headers = ["file", "rows", "md5"]
    log("\n  md5 of the split manifests (must be identical on every node, plan 2.3):")
    log(table(md5_headers, md5_rows))
    log("  note: split_stats.json raw bytes differ across machines (_meta.data_dir is absolute);")
    log("        compare the canonical digest and the manifest md5s.")

    lines = ["", "## Post-split audit of the new `{}` (group-safe split)".format(args.processed_dir), "",
             "Report: `{}`".format(report_path), "", md_table(headers, rows), ""]
    if problems:
        lines += ["**LEAKAGE CHECK FAILED**", ""] + ["* {}".format(p) for p in problems] + [""]
    else:
        lines += ["All val/test leak rates are 0 and no near-duplicate group spans two nodes: **PASS**.", ""]
    lines += ["## Split digests (compare across the three nodes, plan 2.3)", "",
              md_table(md5_headers, md5_rows), "",
              "`split_stats.json` raw bytes legitimately differ between machines (`_meta.data_dir` is "
              "absolute); the canonical digest and every `manifest.csv` md5 must match.", ""]
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log("  appended to {}".format(summary_path))
    if problems:
        for p in problems:
            log("  LEAK: " + p)
        raise StepError("post-split audit is not leakage-free ({} problem(s))".format(len(problems)))
    log("  PASS: zero leakage in every split")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/raw/flame_dataset")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--analysis_dir", default="analysis/leakage")
    p.add_argument("--threshold", type=int, default=8, help="dHash Hamming threshold (plan: 8)")
    p.add_argument("--hash", default="dhash", choices=["dhash", "phash", "ahash", "both"])
    p.add_argument("--phash_threshold", type=int, default=10, help="pHash threshold for --hash both")
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    p.add_argument("--sweep", type=int, nargs="+", default=SWEEP, metavar="T")
    p.add_argument("--examples", type=int, default=EXAMPLES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nodes", type=int, default=3, choices=[2, 3])
    p.add_argument("--dirichlet_alpha", type=float, nargs="*", default=[0.1, 0.5, 1.0])
    p.add_argument("--dirichlet_min_size", type=int, default=10)
    p.add_argument("--subsample_frac", type=float, nargs="*", default=[0.05, 0.01])
    p.add_argument("--link_mode", default="symlink", choices=["symlink", "hardlink", "copy"],
                   help="passed to the splitter (symlink falls back to copy automatically)")
    p.add_argument("--skip_download", action="store_true", help="never call Kaggle; exit 2 if data is missing")
    p.add_argument("--keep_zip", action="store_true", help="keep the Kaggle archive after extraction")
    p.add_argument("--force", action="store_true", help="re-run every step even if its outputs exist")
    p.add_argument("--dry_run", action="store_true", help="print the commands, execute nothing")
    p.add_argument("--python", default=sys.executable, help="interpreter for the sub-steps")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("data_dir", "processed_dir", "analysis_dir"):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    args.sweep = sorted({int(t) for t in args.sweep})
    banner("FedRGBD P0 pipeline: leakage audit -> group-safe split -> re-audit{}".format(
        "  [DRY RUN]" if args.dry_run else ""))
    log("  data_dir={}\n  processed_dir={}\n  analysis_dir={}\n  python={}".format(
        args.data_dir, args.processed_dir, args.analysis_dir, args.python))
    t0 = time.perf_counter()
    try:
        ensure_dataset(args)
        audit_info = step_audit(args)
        decision = step_decision(args, audit_info)
        step_split(args)
        step_reaudit(args, decision["summary_path"])
    except StepError as exc:
        log("\nP0 FAILED: {}".format(exc))
        return EXIT_FAIL
    banner("P0 {} in {:.0f}s -- summary: {}".format(
        "dry run finished" if args.dry_run else "COMPLETE", time.perf_counter() - t0,
        os.path.join(args.analysis_dir, "P0_SUMMARY.md")))
    if not args.dry_run:
        log("  next: copy {} (or its manifest.csv files) to the other nodes and compare the md5 table;".format(
            args.processed_dir))
        log("        then plan 2.5 (code on the Jetsons) and 2.6 (smoke test).")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
