"""CPU tests for scripts/run_matrix.py (no SSH, no testbed).

The runner never builds commands itself: it parses the bash that
``print_revision_commands.py --all_seeds --format bash`` generates.  If the
generator's output format drifts, ``parse_block`` would silently mis-assign or
drop runs, so it is checked here against the real generated output of every
testbed block, cell by cell.
"""

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import print_revision_commands as prc  # noqa: E402
from scripts import run_matrix  # noqa: E402

FL_BLOCKS = [b for b in prc.BLOCK_ORDER if b != "baselines_extension"]


def _generated(block, tmp_path, capsys):
    """The exact bash the runner would parse (all seeds, nothing skipped)."""
    assert prc.main(["--all_seeds", "--block", block, "--format", "bash",
                     "--no_skip_existing"]) == 0
    path = tmp_path / "block_{}.sh".format(block)
    path.write_text(capsys.readouterr().out, encoding="utf-8")
    return str(path)


def _expected_runs(block):
    cfg = prc.load_config(prc.DEFAULT_CONFIG)["revision"]
    cfg = dict(cfg)
    for name in ("seed_extension", "baselines_extension"):
        cfg[name] = dict(cfg[name], all_seeds=True)
    runs, seen = [], set()
    for run in prc.expand_all(cfg, block)[block]:
        if run.output_dir not in seen:            # the generator prints [DUP] for repeats
            seen.add(run.output_dir)
            runs.append(run)
    return runs


def test_every_testbed_block_is_covered():
    assert set(FL_BLOCKS) == {"seed_extension", "dirichlet_skew", "low_data",
                              "long_horizon_fedbn", "mu_grid", "local_epochs",
                              "learning_rate"}


@pytest.mark.parametrize("block", FL_BLOCKS)
def test_parse_block_matches_the_generator_cell_by_cell(block, tmp_path, capsys):
    parsed = run_matrix.parse_block(_generated(block, tmp_path, capsys))
    expected = _expected_runs(block)
    assert expected, block
    assert [r["out_dir"] for r in parsed] == [r.output_dir for r in expected]
    for got, want in zip(parsed, expected):
        assert got["server"] == want.server_command()
        assert "--output_dir {} ".format(got["out_dir"]) in got["server"] + " "
        clients = want.client_commands()
        assert got["clients"] == {"node_a": clients[0], "node_b": clients[1],
                                  "node_c": clients[2]}
        # each node gets its own data dir, every client talks to node_a's server
        for node, cmd in got["clients"].items():
            assert "/{} ".format(node) in cmd + " "
            assert "--server 192.168.1.10:8080" in cmd
        # CLAUDE.md rule 5: new runs only ever go to results/rev_*
        assert got["out_dir"].startswith("results/rev_")


@pytest.mark.parametrize("block", FL_BLOCKS)
def test_every_split_a_block_reads_has_a_reference_md5(block, tmp_path, capsys):
    """Pre-flight compares node manifests with P0_SUMMARY; a split without a reference
    digest would make the check (and the block) fail on the testbed."""
    runs = run_matrix.parse_block(_generated(block, tmp_path, capsys))
    splits = run_matrix.block_splits(runs)
    assert splits
    digests = run_matrix.expected_digests(os.path.join(_REPO, run_matrix.P0_SUMMARY))
    assert set(splits) <= set(digests), set(splits) - set(digests)


def test_round_count_drives_the_timeout(tmp_path, capsys):
    runs = run_matrix.parse_block(_generated("long_horizon_fedbn", tmp_path, capsys))
    rounds = {int(run_matrix.RE_ROUNDS.search(r["server"]).group(1)) for r in runs}
    assert rounds == {10}


def test_baseline_block_is_refused(tmp_path, capsys):
    path = _generated("baselines_extension", tmp_path, capsys)
    with pytest.raises(SystemExit, match="desktop GPU"):
        run_matrix.parse_block(path)


def test_parse_block_survives_a_non_utf8_header(tmp_path, capsys):
    """On Windows the generator's em dash is written in cp1252; decoding must not fail."""
    text = open(_generated("mu_grid", tmp_path, capsys), encoding="utf-8").read()
    path = tmp_path / "cp1252.sh"
    path.write_bytes(text.encode("cp1252", errors="replace"))
    assert len(run_matrix.parse_block(str(path))) == len(_expected_runs("mu_grid"))


def test_incomplete_run_definition_is_a_parse_error(tmp_path):
    path = tmp_path / "broken.sh"
    path.write_text(
        "echo '--- results/rev_iid_fedavg_seed42 ---'\n"
        "echo '>>> START ON node_a (192.168.1.10):'\n"
        "echo '  python3 src/fl/client.py --server 192.168.1.10:8080 "
        "--data_dir data/processed/iid/node_a'\n"
        "python3 src/fl/server.py --strategy fedavg --output_dir results/rev_iid_fedavg_seed42\n",
        encoding="utf-8")
    with pytest.raises(SystemExit, match="incomplete run definition"):
        run_matrix.parse_block(str(path))


def test_check_testbed_flags_gui_and_partition_drift(monkeypatch):
    digests = run_matrix.expected_digests(os.path.join(_REPO, run_matrix.P0_SUMMARY))
    good = {"iid": digests["iid"]}
    facts = {
        "node_a": ("multi-user.target", dict(good)),
        "node_b": ("graphical.target", dict(good)),                 # GUI still on
        "node_c": ("multi-user.target", {"iid": "0" * 32}),         # re-derived split
    }
    monkeypatch.chdir(_REPO)
    monkeypatch.setattr(run_matrix, "NODES", {n: {} for n in facts})
    monkeypatch.setattr(run_matrix, "node_facts", lambda node, splits: facts[node])
    problems = run_matrix.check_testbed(["iid"])
    assert len(problems) == 2
    assert any("node_b" in p and "multi-user.target" in p for p in problems)
    assert any("node_c" in p and "md5" in p for p in problems)

    facts["node_b"] = ("multi-user.target", {"iid": None})           # not replayed at all
    facts["node_c"] = ("multi-user.target", dict(good))
    problems = run_matrix.check_testbed(["iid"])
    assert problems == ["node_b: data/processed/iid/manifest.csv missing -- replay the "
                        "partition with --from_manifest"]


# --------------------------------------------------------------------------- #
# testbed config: users / paths never live in the (public) repository
# --------------------------------------------------------------------------- #
EXAMPLE = os.path.join(_REPO, "configs", "testbed.example.yaml")


def _filled_example(tmp_path, **override):
    """The shipped example with its placeholders replaced (what an operator writes)."""
    text = open(EXAMPLE, encoding="utf-8").read()
    for node in ("a", "b", "c"):
        text = text.replace("<user-on-node-%s>" % node, "op%s" % node)
    for old, new in override.items():
        text = text.replace(old, new)
    path = tmp_path / "testbed.local.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_missing_local_config_tells_the_operator_what_to_do(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_matrix.load_testbed(str(tmp_path / "testbed.local.yaml"))
    msg = str(exc.value)
    assert "not found" in msg
    assert "cp configs" in msg and "testbed.example.yaml" in msg and "testbed.local.yaml" in msg


def test_unfilled_example_is_rejected():
    with pytest.raises(SystemExit, match="placeholder"):
        run_matrix.load_testbed(EXAMPLE)


def test_filled_config_loads(tmp_path):
    nodes, port = run_matrix.load_testbed(_filled_example(tmp_path))
    assert port == 8080
    assert set(nodes) == {"node_a", "node_b", "node_c"}
    assert {n: c["host"] for n, c in nodes.items()} == {
        "node_a": "192.168.1.10", "node_b": "192.168.1.7", "node_c": "192.168.1.6"}
    assert nodes["node_b"]["user"] == "opb"
    assert nodes["node_c"]["repo"] == "/home/opc/FedRGBD"
    assert [n for n, c in nodes.items() if c["local"]] == ["node_a"]


def test_config_must_run_on_node_a(tmp_path):
    path = _filled_example(tmp_path, **{"local: true            # the node": "local: false  # x"})
    with pytest.raises(SystemExit, match="node_a must have 'local: true'"):
        run_matrix.load_testbed(path)


def test_node_shell_uses_the_configured_paths(monkeypatch, tmp_path):
    nodes, _ = run_matrix.load_testbed(_filled_example(tmp_path))
    monkeypatch.setattr(run_matrix, "NODES", nodes)
    assert run_matrix.node_shell("node_b", "python3 -V") == (
        "cd /home/opb/FedRGBD && source /home/opb/fedrgbd_venv/bin/activate && python3 -V")


def test_repository_holds_no_node_usernames():
    """The example keeps the (private-LAN) IPs but only placeholder users; the runner
    itself contains no user or home-directory literals; the local file is ignored."""
    import re
    import subprocess

    example = open(EXAMPLE, encoding="utf-8").read()
    users = re.findall(r"^\s*user:\s*(\S+)", example, flags=re.M)
    assert users and all(u.startswith("<") and u.endswith(">") for u in users)
    for ip in ("192.168.1.10", "192.168.1.7", "192.168.1.6"):
        assert ip in example
    source = open(os.path.join(_REPO, "scripts", "run_matrix.py"), encoding="utf-8").read()
    assert "'user':" not in source and "/home/" not in source
    ignored = subprocess.run(["git", "check-ignore", "-q", "configs/testbed.local.yaml"],
                             cwd=_REPO)
    assert ignored.returncode == 0
