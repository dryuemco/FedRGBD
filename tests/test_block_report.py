"""`scripts/block_report.py` must enumerate exactly the runs the testbed executes.

`scripts/run_matrix.py` launches a block with

    print_revision_commands.py --all_seeds --block <name> --format bash

so the set of runs the nodes actually execute is whatever that command emits.
`block_report.py` decides block completeness from its own enumeration, and if the
two ever disagree it will declare a block finished while runs are still missing
and fire the analysis pipeline over a partial block.

That is not hypothetical: the first version of `block_report.py` omitted
``--all_seeds`` and reported ``seed_extension`` as 8 runs rather than 20, so it
would have triggered with 12 of them missing.

These tests derive both sides from the same CLI and require equality.
"""

import os
import re
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.block_report import expected_runs  # noqa: E402
from scripts.print_revision_commands import BLOCK_ORDER  # noqa: E402

_OUTPUT_DIR_RE = re.compile(r"--output_dir\s+(\S+)")


def _runs_run_matrix_would_execute(block):
    """The run set of `print_revision_commands.py --all_seeds --block <block>`.

    ``--no_skip_existing`` so the answer is the block's full membership rather
    than only what is still outstanding on this machine.
    """
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "scripts", "print_revision_commands.py"),
         "--all_seeds", "--block", block, "--format", "bash", "--no_skip_existing"],
        cwd=_REPO, capture_output=True, text=True)
    assert proc.returncode == 0, "print_revision_commands failed:\n%s" % proc.stderr
    return set(_OUTPUT_DIR_RE.findall(proc.stdout))


@pytest.mark.parametrize("block", BLOCK_ORDER)
def test_block_report_enumerates_what_run_matrix_executes(block):
    reported = set(expected_runs(_REPO).get(block, []))
    actual = _runs_run_matrix_would_execute(block)

    assert actual, "the CLI emitted no runs for %s" % block
    missing = actual - reported
    extra = reported - actual
    assert not missing, (
        "block_report would miss %d run(s) of %s, so it could call the block "
        "complete while they are still running: %s"
        % (len(missing), block, sorted(missing)[:5]))
    assert not extra, (
        "block_report expects %d run(s) of %s that the testbed never executes, so "
        "the block could never be reported complete: %s"
        % (len(extra), block, sorted(extra)[:5]))


def test_seed_extension_spans_every_seed_not_only_the_new_ones():
    """The specific regression: --all_seeds must be applied.

    Without it seed_extension is the 8 new-seed runs; with it, all 20.
    """
    runs = expected_runs(_REPO)["seed_extension"]
    assert len(runs) == 20, "seed_extension should span all five seeds, got %d runs" % len(runs)
    for seed in (42, 123, 456, 789, 1011):
        assert any(r.endswith("_seed%d" % seed) for r in runs), \
            "seed %d missing from seed_extension" % seed


def test_apply_all_seeds_is_the_transformation_the_cli_performs():
    """The helper both paths share must match what the CLI's --all_seeds flag does."""
    from scripts.print_revision_commands import apply_all_seeds, expand_all, load_config

    cfg = load_config(os.path.join(_REPO, "configs", "experiment_matrix.yaml"))
    revision = cfg["revision"]

    plain = expand_all(revision)
    widened = expand_all(apply_all_seeds(revision))

    # it only ever widens, and only the two seed-limited blocks
    for block in BLOCK_ORDER:
        a = {r.output_dir for r in plain.get(block, [])}
        b = {r.output_dir for r in widened.get(block, [])}
        assert a <= b, "%s lost runs when --all_seeds was applied" % block
        if block not in ("seed_extension", "baselines_extension"):
            assert a == b, "%s should not depend on --all_seeds" % block
    assert len(widened["seed_extension"]) > len(plain["seed_extension"])


def test_no_caller_in_the_fetch_path_enumerates_without_all_seeds():
    """Nothing in the fetch/report path may expand the matrix without --all_seeds.

    A second caller that forgets the flag reintroduces exactly the same defect, so
    this greps the scripts the hourly job touches.
    """
    offenders = []
    for name in ("block_report.py", "fetch_results.ps1", "run_matrix.py"):
        path = os.path.join(_REPO, "scripts", name)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8").read()
        calls_expander = ("expand_all(" in text) or ("print_revision_commands" in text)
        if calls_expander and "all_seeds" not in text:
            offenders.append(name)
    assert not offenders, (
        "these enumerate the experiment matrix without --all_seeds: %s" % offenders)


def test_blocks_outside_the_chain_are_labelled_not_counted(capsys):
    """baselines_extension must not read as "0/62 missing".

    Its runs exist, under results/rev_baselines_sel/ rather than at the top level,
    so it can never be reported complete. Printing a bare count invites reading it
    as 62 outstanding runs.
    """
    from scripts.block_report import NOT_IN_CHAIN, main

    assert "baselines_extension" in NOT_IN_CHAIN
    main(["--repo", _REPO, "--status_only"])
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("baselines_extension"))
    assert NOT_IN_CHAIN["baselines_extension"] in line
    assert not re.search(r"\d+\s*/\s*\d+", line), \
        "a bare n/m count is still printed for a block outside the chain: %r" % line
    # the blocks that ARE in the chain keep their counts
    other = next(l for l in out.splitlines() if l.startswith("seed_extension"))
    assert re.search(r"\d+\s*/\s*\d+", other)


def test_a_block_outside_the_chain_never_fires_the_pipeline(tmp_path, capsys):
    """Even asked for by name with --force, it must not run the pipeline."""
    from scripts.block_report import main

    rc = main(["--repo", _REPO, "--block", "baselines_extension", "--force"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "running the pipeline" not in out
