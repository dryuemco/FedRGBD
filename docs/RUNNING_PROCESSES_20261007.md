# Long-running processes, 2026-10-07 21:47 (before a Claude Code restart)

Written because a Claude Code update needs a restart. Which processes survive it:

| Process | Where | Tied to Claude Code? | Survives a restart |
|---|---|---|---|
| A5 testbed block (`run_matrix.py --block a5_dirichlet_draws --power_config maxn`) | node_a, `nohup`, started 09:42 | no | yes |
| FetchResults (hourly copy of node results to the desktop) | Windows scheduled task `FedRGBD-FetchResults` | no | yes |
| A5 desktop simulation (`scripts/desktop_fl_sim.py run --lanes 4`) | desktop, started 18:40 | **yes** (child of the Claude Code shell) | **no** |
| ssh session that launched the testbed block | desktop, PID 16536 | yes | no -- harmless, the block runs under `nohup` on node_a |
| desktop waiter for the three A5 testbed `results.json` | desktop bash loop | yes | no -- harmless, only a notification |

## A5 desktop simulation

* Process tree: `claude.exe` 22928 -> bash 27780 -> `desktop_fl_sim.py` 6148 (venv launcher)
  -> 4924 (interpreter) -> per lane one `src/fl/server.py` and three `src/fl/client.py`
  (ports 8091-8094).
* Logs: `logs/desktop_sim/run_4lanes.log` (driver), `logs/desktop_sim/sim_<cell>_<stamp>_{server,node_a,node_b,node_c}.log`
  (per cell). Results: `results/desktop_sim/sim_<partition>_<strategy>_r10_seed42/`.
* State at 21:47: 2 of 18 cells complete (`results.json` present); 4 running.
* Detaching it was tried (a one-off Windows scheduled task to relaunch it after a drain);
  the auto-mode permission check refused scheduled tasks, so the attempt was removed
  (task unregistered, script and STOP marker deleted; no lane had paused).

## Decision: the restart is deferred

The restart waits until the A5 testbed block has ended (expected around 04:30 on
2026-10-08) **and** the desktop simulation has finished all 18 cells. Restarting earlier
kills the four running cells; a killed cell has no `results.json` and is re-run from the
start on relaunch (finished cells are skipped), but its directory would hold files of the
aborted attempt -- if it happens, clear those cell directories before relaunching.

To restart earlier without losing work: create `logs/desktop_sim/STOP` (the running cells
finish, no new cell starts), wait until `desktop_fl_sim.py` has exited, restart, remove
`STOP`, relaunch:

    PY=~/venvs/fedrgbd-gpu/Scripts/python.exe
    $PY scripts/desktop_fl_sim.py run --data_root C:/Users/CORSAIR/fedrgbd_sim/processed --python $PY --lanes 4
