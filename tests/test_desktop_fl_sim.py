"""A5 desktop simulation sensitivity analysis (scripts/desktop_fl_sim.py): the cell
matrix, the unchanged server/client commands, the validation and stop-file guards, and
its separation from the FLAME testbed analysis.  No process is started."""

import os

import pytest

from scripts import desktop_fl_sim as sim


def test_eighteen_cells_in_declared_order():
    c = sim.cells()
    assert len(c) == 18 and len(set(c)) == 18
    assert c[0] == ("0.1", "ps42", "fedavg") and c[-1] == ("1", "ps456", "fedprox_0.01")
    assert {sim.partition(a, d) for a, d, _ in c} == {
        "dirichlet_0.1", "dirichlet_0.5", "dirichlet_1",
        "dirichlet_0.1_ps123", "dirichlet_0.5_ps123", "dirichlet_1_ps123",
        "dirichlet_0.1_ps456", "dirichlet_0.5_ps456", "dirichlet_1_ps456"}
    for a, d, s in c:
        out = sim.out_dir(a, d, s).replace(os.sep, "/")
        assert out.startswith("results/desktop_sim/sim_dirichlet") and out.endswith("_r10_seed42")


def test_commands_are_the_unchanged_server_and_client_with_the_declared_settings():
    srv = sim.server_argv("py", "0.1", "ps123", "fedprox_0.01", 8091, "OUT", 10)
    assert srv[1] == "src/fl/server.py"
    opts = dict(zip(srv[2::2], srv[3::2]))
    assert opts == {"--strategy": "fedprox_0.01", "--rounds": "10", "--seed": "42",
                    "--min_clients": "3", "--address": "127.0.0.1:8091", "--output_dir": "OUT",
                    "--tag": "dirichlet0.1_ps123"}
    cl = sim.client_argv("py", "ROOT", "dirichlet_0.1_ps123", "node_b", 8091, 5)
    assert cl[1] == "src/fl/client.py"
    opts = dict(zip(cl[2::2], cl[3::2]))
    assert opts["--data_dir"] == os.path.join("ROOT", "dirichlet_0.1_ps123", "node_b")
    assert (opts["--batch_size"], opts["--seed"], opts["--local_epochs"], opts["--lr"],
            opts["--node_name"], opts["--server"]) == ("8", "42", "5", "0.001", "node_b",
                                                       "127.0.0.1:8091")


def test_validation_never_writes_to_the_results_tree(monkeypatch):
    monkeypatch.setattr(sim, "run_cell", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="never writes"):
        sim.main(["run", "--data_root", "X", "--rounds", "1"])
    with pytest.raises(SystemExit, match="never writes"):
        sim.main(["run", "--data_root", "X", "--validation_split", "dirichlet_0.1_sub0.01"])


def test_stop_file_starts_no_cell(monkeypatch, tmp_path, capsys):
    stop = tmp_path / "STOP"
    stop.write_text("camera footage arrived")
    monkeypatch.setattr(sim, "digest_problems", lambda root, splits: [])
    monkeypatch.setattr(sim, "run_cell", lambda *a, **k: pytest.fail("must not run"))
    assert sim.main(["run", "--data_root", "X", "--stop_file", str(stop)]) == 0
    assert "stop file" in capsys.readouterr().out


def test_digest_failure_stops_before_any_cell(monkeypatch, tmp_path):
    monkeypatch.setattr(sim, "run_cell", lambda *a, **k: pytest.fail("must not run"))
    assert sim.main(["run", "--data_root", str(tmp_path), "--cells",
                     "dirichlet_0.1_ps123:fedavg", "--stop_file", str(tmp_path / "none")]) == 1


def test_unknown_cell_is_refused():
    with pytest.raises(SystemExit, match="unknown cell"):
        sim.parse_cells(["dirichlet_0.2:fedavg"])


def test_flame_analysis_skips_the_desktop_sim_tree(tmp_path):
    from scripts.analyze_results import iter_run_dirs
    for d in ("rev_dirichlet0.1_fedavg_seed42",
              os.path.join("desktop_sim", "sim_dirichlet0.1_ps123_fedavg_r10_seed42"),
              os.path.join("pc_maxn", "rev_iid_fedavg_r10_seed42")):
        p = tmp_path / d
        p.mkdir(parents=True)
        (p / "results.json").write_text("{}")
    got = sorted(os.path.relpath(r, tmp_path) for r in iter_run_dirs(str(tmp_path)))
    assert got == [os.path.join("pc_maxn", "rev_iid_fedavg_r10_seed42"),
                   "rev_dirichlet0.1_fedavg_seed42"]


def test_four_lanes_cover_every_cell_once_on_distinct_ports(tmp_path):
    plan = sim.lane_plan(sim.cells(), 4)
    assert sorted(c for lane in plan for c in lane) == sorted(sim.cells())
    assert [len(l) for l in plan] == [5, 5, 4, 4]
    seen = []
    res = sim.run_lanes(sim.cells(), 4, lambda a, d, s, port: seen.append((port, (a, d, s)))
                        or {"cell": s, "status": "ok"}, 9000, str(tmp_path / "none"))
    assert len(res) == 18 and sorted(c for _, c in seen) == sorted(sim.cells())
    ports = {c: p for p, c in seen}
    for i, lane in enumerate(plan):
        assert {ports[c] for c in lane} == {9000 + i}


def test_stop_file_halts_every_lane(tmp_path):
    stop = tmp_path / "STOP"
    calls = []

    def one(a, d, s, port):
        calls.append(s)
        stop.write_text("camera footage arrived")
        return {"cell": s, "status": "ok"}
    sim.run_lanes(sim.cells(), 2, one, 9000, str(stop))
    assert 1 <= len(calls) <= 2
