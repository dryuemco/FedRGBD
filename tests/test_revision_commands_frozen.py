"""The generated commands of every existing FLAME block are frozen byte for byte.

``scripts/run_matrix.py`` (and ``resume_after_reboot.py`` through it) turns
``print_revision_commands.py --all_seeds --block <b> --format bash --power_config <pc>``
into the script the testbed executes, so any change to the printer or to
``configs/experiment_matrix.yaml`` that alters an existing block's commands changes what
a (resumed) block runs.  Additions -- such as the camera blocks, emitted only with
``--block`` -- must leave every existing output untouched.

``tests/fixtures/revision_commands_frozen.json`` holds, for a grid of CLI arguments over
the default (all-block) output and every block of ``FROZEN_BLOCKS``, the SHA-256 and
length of stdout and stderr and the exit code.  It was written from commit 2a565b3 (the
last commit before the camera blocks were added) by running this file as a script
against a worktree of that commit::

    git worktree add <tmp> 2a565b3
    python tests/test_revision_commands_frozen.py --root <tmp> --write \\
        tests/fixtures/revision_commands_frozen.json

Never regenerate it to make this test pass.  If an existing block has to change on
purpose, that is a change of the experiment: record it in docs/REVISION_CHANGES.md, then
regenerate from the new commit and say so in the commit message.  To see what differs,
render the same arguments at the fixture's commit and at HEAD (``--dump``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "revision_commands_frozen.json")

#: every block that existed before the camera experiment (BLOCK_ORDER at 2a565b3)
FROZEN_BLOCKS = [
    "seed_extension",
    "dirichlet_skew",
    "low_data",
    "long_horizon_fedbn",
    "mu_grid",
    "local_epochs",
    "learning_rate",
    "maxn_long_horizon",
    "baselines_extension",
]


def argument_grid():
    """(key, argv) for every frozen combination.  ``--no_skip_existing`` always: the
    default skips runs whose results exist, which depends on the machine, not the code."""
    grid = []
    for block, fmt, all_seeds, power in itertools.product(
            [None] + FROZEN_BLOCKS, ["text", "bash"], [True, False],
            [None, "heterogeneous", "maxn"]):
        argv = ["--no_skip_existing", "--format", fmt]
        if block:
            argv += ["--block", block]
        if all_seeds:
            argv.append("--all_seeds")
        if power:
            argv += ["--power_config", power]
        grid.append(argv)
    for block in (None, "baselines_extension"):
        argv = ["--no_skip_existing", "--format", "bash", "--all_seeds",
                "--baseline_root", "results/rev_baselines_sel"]
        grid.append(argv + (["--block", block] if block else []))
    return [(" ".join(argv), argv) for argv in grid]


def load_printer(root: str):
    """``scripts/print_revision_commands.py`` of ``root`` as a fresh module; its
    REPO_ROOT, and so its default config, is that tree's."""
    path = os.path.join(root, "scripts", "print_revision_commands.py")
    name = "_prc_frozen_" + hashlib.sha256(os.path.abspath(root).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses look their module up here
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def render(module, argv):
    """-> (exit code, stdout, stderr), in-process so no platform line-ending translation."""
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = module.main(list(argv))
            code = int(rc or 0)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if e.code is not None and not isinstance(e.code, int):
                err.write(str(e.code))
    return code, out.getvalue(), err.getvalue()


def fingerprint(module, argv):
    code, out, err = render(module, argv)
    digest = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    return {"exit": code, "stdout_sha256": digest(out), "stdout_len": len(out),
            "stderr_sha256": digest(err), "stderr_len": len(err)}


def _fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_covers_the_grid():
    fx = _fixture()
    assert sorted(fx["outputs"]) == sorted(key for key, _ in argument_grid())
    # every frozen block produced commands at the fixture's commit
    for block in FROZEN_BLOCKS:
        key = "--no_skip_existing --format bash --block %s --all_seeds" % block
        assert fx["outputs"][key]["exit"] == 0 and fx["outputs"][key]["stdout_len"] > 1000, block


def test_frozen_blocks_are_in_the_printer():
    prc = load_printer(REPO_ROOT)
    assert prc.BLOCK_ORDER == FROZEN_BLOCKS, \
        "BLOCK_ORDER changed: a new FLAME block needs its own entry in FROZEN_BLOCKS"


@pytest.mark.parametrize("key,argv", argument_grid(), ids=[k for k, _ in argument_grid()])
def test_existing_block_commands_are_byte_identical(key, argv):
    prc = load_printer(REPO_ROOT)
    expected = _fixture()["outputs"][key]
    got = fingerprint(prc, argv)
    assert got == expected, (
        "print_revision_commands.py %s no longer produces the output frozen at %s; "
        "compare with: python tests/test_revision_commands_frozen.py --dump %r"
        % (key, _fixture()["source_commit"], key))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default=REPO_ROOT, help="tree whose printer and yaml are used")
    p.add_argument("--write", metavar="JSON", help="write the fingerprints of --root here")
    p.add_argument("--source_commit", default="2a565b3")
    p.add_argument("--dump", metavar="KEY", help="print stdout of one grid key for --root")
    args = p.parse_args(argv)
    prc = load_printer(os.path.abspath(args.root))
    if args.dump:
        code, out, err = render(prc, dict(argument_grid())[args.dump])
        sys.stdout.write(out)
        sys.stderr.write(err + "\n[exit %d]\n" % code)
        return 0
    outputs = {key: fingerprint(prc, a) for key, a in argument_grid()}
    doc = {"source_commit": args.source_commit,
           "note": "written by tests/test_revision_commands_frozen.py; never regenerate "
                   "to make the test pass -- see its docstring",
           "outputs": outputs}
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if args.write:
        with open(args.write, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
