#!/usr/bin/env python3
"""FedRGBD -- which conclusions depend on how balanced accuracy is aggregated.

CLAUDE.md hard rule 8 declares two aggregations over the three nodes: pooled over
the union of the held-out images (primary) and the unweighted mean over the clients
(secondary).  The choice was fixed after the federated runs existed, so it is not
pre-registered; this script lists every comparison whose conclusion changes between
the two, so that no reported conclusion rests on the choice unseen.

**Paired comparisons (the flip list).**  Two configurations are compared when they
are evaluated on the same held-out images:

* *within a partition* (protocol ``group``, one power configuration): every pair of
  configurations -- references, federated strategies, every sweep variant;
* *across rho*: a partition and its subsampled versions (``iid``, ``iid_sub0.05``,
  ``iid_sub0.01``; likewise ``non_iid_label``) share their held-out splits node by
  node (checked), so the same method is compared across rho.

For each pair the balanced-accuracy difference A - B gets a joint stratified cluster
bootstrap under each aggregation, resampled the way the paper declares: for the
pooled figure one resample of the held-out *sequences across all nodes* (the
concatenated unit, stratified by class composition), for the client mean one
resample *within each client*; the resample is shared by every run of both
configurations.  The seeds of each configuration are resampled independently, with
the same seed draw for both aggregations (common random numbers), so a flip is not
resampling noise between the two columns.  B = 10000, repeated under three
independent RNG keys: a verdict that is not the same under all three keys is
flagged ``unstable`` and not counted as a flip.  A conclusion is ``A>B`` / ``A<B``
if the 95 % interval excludes 0, else ``n.s.``; a comparison flips if the verdict
or the sign of the point estimate differs between the aggregations.

The seed-paired t-test (the paper's statistics section) is also run on the seed
values of each pair under both aggregations; a flip there is a change of
significance at 0.05 or of direction.

**Cross-partition values.**  Partitions with different held-out sets (the three
Dirichlet draws, IID vs label skew) cannot be paired; claims across them (e.g. how
the references move with skew) are listed as per-partition values under both
aggregations, not tested.

Written to ``analysis/aggregation_flips.csv`` (every pair) and
``analysis/aggregation_flips.md`` (the flips, the unstable ones, and the
cross-partition values).  Every selection-rule configuration needs its per-image
prediction files; a missing one is an error, not a silent skip.

    python scripts/aggregation_sensitivity.py --results_dir results --output_dir analysis
"""

import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.analyze_results import (  # noqa: E402
    HEADLINE_SELECTED, attach_prediction_metrics, check_prediction_unions, collect_runs)
from src.evaluation.bootstrap import BASE_SEED, Unit, metrics_from_counts  # noqa: E402

AGGREGATIONS = ("pooled", "clientmean")
METRIC = "balanced_accuracy"
KEYS = (0, 1, 2)
RHO_SUFFIX = {"": "1.00", "_sub0.05": "0.05", "_sub0.01": "0.01"}


def _name(record):
    name = record.get("variant") or record.get("strategy_display") or record.get("strategy")
    # federated runs of another power configuration (e.g. maxn) share their variant name
    # with the heterogeneous matrix; tag them so every row of the flip list is unambiguous
    power = record.get("power_config")
    if record.get("kind") == "fl" and power not in (None, "heterogeneous", "unrecorded"):
        name = "%s [%s]" % (name, power)
    return name


def _verdict(lo, hi):
    return "A>B" if lo > 0 else ("A<B" if hi < 0 else "n.s.")


def _units(record, field, pooled):
    raw = record[field]
    if pooled:
        return [Unit(np.concatenate([u["label"] for u in raw]),
                     np.concatenate([u["margin"] for u in raw]),
                     np.concatenate([np.asarray(u["group"]).astype(str) for u in raw]))]
    return [Unit(u["label"], u["margin"], u["group"]) for u in raw]


def group_boot(configs, B, key, clean):
    """{config: {agg: (point per run (S,), replicate mean over drawn seeds (B,))}}.

    One resample of the held-out sequences per aggregation (across nodes for pooled,
    within each node for the client mean), shared by every run of every configuration
    of the group; one seed draw per configuration, shared by both aggregations."""
    field = "pred_units_clean" if clean else "pred_units"
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(("flips|" + key).encode("utf-8")))
    first = next(iter(configs.values()))[0]
    ref = {agg: _units(first, field, agg == "pooled") for agg in AGGREGATIONS}
    W = {agg: [u.weights(B, rng) for u in ref[agg]] for agg in AGGREGATIONS}
    out = {}
    for cfg, runs in configs.items():
        S = len(runs)
        pick = rng.integers(0, S, size=(B, S))                  # common random numbers
        col = np.broadcast_to(np.arange(B)[:, None], pick.shape)
        res = {}
        for agg in AGGREGATIONS:
            points, reps = [], []
            for r in runs:
                units = _units(r, field, agg == "pooled")
                if len(units) != len(ref[agg]) or not all(
                        u.G == v.G and np.array_equal(u.gid, v.gid) for u, v in zip(units, ref[agg])):
                    raise ValueError("%s is not evaluated on the group's held-out sequences"
                                     % r["run_name"])
                per = [(w @ u.counts, u.counts.sum(0)) for u, w in zip(units, W[agg])]
                b = np.stack([metrics_from_counts(*c.T)[METRIC] for c, _ in per])     # (U, B)
                p = np.array([float(metrics_from_counts(*f)[METRIC]) for _, f in per])
                points.append(float(p.mean()))
                reps.append(np.nanmean(b, axis=0))
            points, reps = np.array(points), np.stack(reps)                          # (S,), (S, B)
            res[agg] = (points, reps[pick, col].mean(axis=1))
        out[cfg] = res
    return out


def pair_rows(configs, names, kinds, seeds, B, key_base, clean, scope, pair_filter=None):
    """Every pair of the group's configurations, under every RNG key."""
    boots = {k: group_boot(configs, B, "%s|%d" % (key_base, k), clean) for k in KEYS}
    ids = sorted(configs, key=lambda c: names[c])
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if pair_filter and not pair_filter(a, b):
                continue
            row = dict(scope=scope, subset="clean" if clean else "full", a=names[a], b=names[b],
                       a_kind=kinds[a], b_kind=kinds[b], n_a=len(configs[a]), n_b=len(configs[b]))
            verdicts = {agg: [] for agg in AGGREGATIONS}
            for k in KEYS:
                for agg in AGGREGATIONS:
                    pa, ba = boots[k][a][agg]
                    pb, bb = boots[k][b][agg]
                    d = ba - bb
                    lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
                    verdicts[agg].append(_verdict(lo, hi))
                    if k == KEYS[0]:
                        row.update({"%s_diff" % agg: pa.mean() - pb.mean(), "%s_lo" % agg: lo,
                                    "%s_hi" % agg: hi})
            for agg in AGGREGATIONS:
                row["%s_verdict" % agg] = verdicts[agg][0]
                row["%s_stable" % agg] = len(set(verdicts[agg])) == 1
            row["stable"] = row["pooled_stable"] and row["clientmean_stable"]
            row["verdict_flip"] = row["stable"] and row["pooled_verdict"] != row["clientmean_verdict"]
            row["direction_flip"] = bool(np.sign(row["pooled_diff"]) != np.sign(row["clientmean_diff"]))
            # seed-paired t-test on the seeds both configurations have
            common = sorted(set(seeds[a]) & set(seeds[b]))
            for agg in AGGREGATIONS:
                p = np.nan
                if len(common) >= 2:
                    va = np.array([boots[KEYS[0]][a][agg][0][seeds[a].index(s)] for s in common])
                    vb = np.array([boots[KEYS[0]][b][agg][0][seeds[b].index(s)] for s in common])
                    if not np.allclose(va, vb):
                        p = float(stats.ttest_rel(va, vb).pvalue)
                row["ttest_p_%s" % agg] = p
            row["n_common_seeds"] = len(common)
            sig = [row["ttest_p_%s" % agg] < 0.05 if np.isfinite(row["ttest_p_%s" % agg]) else None
                   for agg in AGGREGATIONS]
            row["ttest_flip"] = (None not in sig) and (sig[0] != sig[1] or row["direction_flip"])
            rows.append(row)
    return rows


def _groups(runs):
    """{(protocol, power, dist): {config_id: [runs]}} with names, kinds and seeds."""
    groups, names, kinds, seeds = {}, {}, {}, {}
    for r in runs:
        key = (str(r.get("protocol")), str(r.get("power_config")), str(r.get("distribution")))
        cid = r["config_id"] + "|" + key[1]
        groups.setdefault(key, {}).setdefault(cid, []).append(r)
        names[cid] = _name(r)
        kinds[cid] = r.get("kind")
    for g in groups.values():
        for cid, rs in g.items():
            rs.sort(key=lambda r: (r.get("seed") is None, r.get("seed")))
            seeds[cid] = [r.get("seed") for r in rs]
    return groups, names, kinds, seeds


def analyse(runs, B=10000, clean=False):
    groups, names, kinds, seeds = _groups(runs)
    rows = []
    seen_ref_pairs = set()
    for (protocol, power, dist), configs in sorted(groups.items()):
        if len(configs) < 2:
            continue

        def keep(a, b, dist=dist):
            # reference-vs-reference pairs only once per partition, whatever the power groups
            if kinds[a] != "fl" and kinds[b] != "fl":
                k = (dist, names[a], names[b])
                if k in seen_ref_pairs:
                    return False
                seen_ref_pairs.add(k)
            return True
        for row in pair_rows(configs, names, kinds, seeds, B,
                             "|".join((protocol, power, dist, str(clean))), clean, "partition",
                             pair_filter=keep):
            row.update(protocol=protocol, power_config=power, distribution=dist)
            rows.append(row)
    # across rho: same method, same held-out set -- full set only: the clean subset of a
    # partition excludes near-duplicates of THAT partition's training images, so the clean
    # held-out sets of rho = 1, 0.05, 0.01 differ and cannot be paired
    for (protocol, power, dist) in ([] if clean else sorted(groups)):
        if "_sub" in dist:
            continue
        family = {}
        for suffix, rho in RHO_SUFFIX.items():
            g = groups.get((protocol, power, dist + suffix))
            if not g:
                continue
            for cid, rs in g.items():
                family[cid + "@" + rho] = rs
                names[cid + "@" + rho] = "%s @rho=%s" % (names[cid], rho)
                kinds[cid + "@" + rho] = kinds[cid]
                seeds[cid + "@" + rho] = seeds[cid]
        rhos = {cid: cid.rsplit("@", 1)[1] for cid in family}
        if len(set(rhos.values())) < 2:
            continue
        base = {cid: names[cid].rsplit(" @rho=", 1)[0] for cid in family}
        for row in pair_rows(family, names, kinds, seeds, B,
                             "|".join((protocol, power, dist, "rho", str(clean))), clean, "rho",
                             pair_filter=lambda a, b: base[a] == base[b] and rhos[a] != rhos[b]):
            row.update(protocol=protocol, power_config=power, distribution=dist + " (across rho)")
            rows.append(row)
    return pd.DataFrame(rows)


def cross_partition_values(runs):
    """Reference values per partition under both aggregations (descriptive, unpaired)."""
    rows = []
    for r in runs:
        if r.get("kind") == "fl":
            continue
        rows.append(dict(distribution=r.get("distribution"), method=_name(r),
                         pooled=(r.get("selected_test_metrics") or {}).get(METRIC),
                         clientmean=(r.get("selected_test_clientmean_metrics") or {}).get(METRIC)))
    df = pd.DataFrame(rows)
    return df.groupby(["method", "distribution"]).mean(numeric_only=True).reset_index()


def markdown(df, xpart):
    full = df[df.subset == "full"]
    out = ["# Conclusions that depend on the balanced-accuracy aggregation", "",
           "Pooled over all held-out images (primary; sequences resampled across nodes) vs "
           "unweighted mean over the clients (secondary; sequences resampled within each "
           "client). Balanced-accuracy difference A - B in points with the 95 % joint "
           "cluster-bootstrap interval (B = 10000; the same seed draw for both aggregations). "
           "`A>B` / `A<B`: interval excludes 0. A verdict counts only if it is the same under "
           "three independent RNG keys; unstable ones are listed separately. Scope: pairs "
           "within a partition and the same method across rho (both on the same held-out "
           "images; across rho on the full set only, because each partition's clean subset "
           "depends on its own training images); partitions with different held-out sets are "
           "not paired (last section). "
           "Generated by `scripts/aggregation_sensitivity.py`.", ""]
    for subset in ("full", "clean"):
        sub = df[df.subset == subset]
        out.append("- %s set: %d comparisons; %d verdict flips, %d direction-only flips, "
                   "%d unstable under the RNG keys; seed-paired t-test flips: %d."
                   % (subset, len(sub), int(sub.verdict_flip.sum()),
                      int((sub.direction_flip & ~sub.verdict_flip).sum()),
                      int((~sub.stable).sum()), int(sub.ttest_flip.fillna(False).astype(bool).sum())))
    del full

    def table(sub, title, what):
        if sub.empty:
            return []
        lines = ["", "## " + title, "",
                 "| partition | set | A | B | pooled (primary) | client mean (secondary) | %s |" % what,
                 "|---|---|---|---|---|---|---|"]
        for r in sub.sort_values(["distribution", "subset", "a", "b"]).itertuples():
            lines.append("| %s | %s | %s | %s | %+.1f [%+.1f, %+.1f] %s | %+.1f [%+.1f, %+.1f] %s | %s |" % (
                r.distribution, r.subset, r.a, r.b,
                100 * r.pooled_diff, 100 * r.pooled_lo, 100 * r.pooled_hi, r.pooled_verdict,
                100 * r.clientmean_diff, 100 * r.clientmean_lo, 100 * r.clientmean_hi,
                r.clientmean_verdict,
                what_value(r, what)))
        return lines

    def what_value(r, what):
        if what == "flip":
            return "verdict" if r.verdict_flip else "direction"
        if what == "t-test p (pooled / client mean)":
            return "%.3g / %.3g" % (r.ttest_p_pooled, r.ttest_p_clientmean)
        return "pooled %s, client mean %s" % ("stable" if r.pooled_stable else "UNSTABLE",
                                              "stable" if r.clientmean_stable else "UNSTABLE")

    out += table(df[df.verdict_flip], "Verdict flips (stable under all RNG keys)", "flip")
    out += table(df[df.direction_flip & ~df.verdict_flip],
                 "Direction-only flips (sign of the point estimate differs)", "flip")
    out += table(df[df.ttest_flip.fillna(False).astype(bool)],
                 "Seed-paired t-test flips (significance at 0.05 or direction)",
                 "t-test p (pooled / client mean)")
    out += table(df[~df.stable], "Unstable verdicts (differ between RNG keys; not counted)",
                 "stability")
    out += ["", "## Cross-partition reference values (not paired: different held-out sets)", "",
            "| method | partition | pooled BA (%) | client-mean BA (%) |", "|---|---|---|---|"]
    for r in xpart.sort_values(["method", "distribution"]).itertuples():
        out.append("| %s | %s | %.1f | %.1f |" % (r.method, r.distribution, 100 * r.pooled,
                                                   100 * r.clientmean))
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--output_dir", default=os.path.join(REPO, "analysis"))
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args(argv)
    runs = [r for r in collect_runs(args.results_dir, warn=False) if r.get("protocol") == "group"]
    attach_prediction_metrics(runs, warn=False)
    check_prediction_unions(runs)
    selected = [r for r in runs if r.get("headline_source") == HEADLINE_SELECTED]
    missing = [r["run_name"] for r in selected if not r.get("pred_units")]
    if missing:
        raise SystemExit("selection-rule runs without prediction files: %s" % ", ".join(missing[:10]))
    # the desktop baselines join every federated power group of their partition
    by_part = {}
    for r in selected:
        by_part.setdefault(str(r.get("distribution")), set()).add(str(r.get("power_config")))
    expanded = []
    for r in selected:
        if r.get("kind") == "fl":
            expanded.append(r)
            continue
        fl_powers = sorted(p for p in by_part[str(r.get("distribution"))]
                           if p != str(r.get("power_config")))
        for p in fl_powers or [str(r.get("power_config"))]:
            expanded.append(dict(r, power_config=p))
    df = pd.concat([analyse(expanded, args.B), analyse(expanded, args.B, clean=True)],
                   ignore_index=True)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "aggregation_flips.csv"), index=False)
    with open(os.path.join(args.output_dir, "aggregation_flips.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(markdown(df, cross_partition_values(selected)))
    full = df[df.subset == "full"]
    print("%d comparisons (full set); %d stable verdict flips, %d direction flips, %d unstable, "
          "%d t-test flips" % (len(full), int(full.verdict_flip.sum()), int(full.direction_flip.sum()),
                               int((~full.stable).sum()),
                               int(full.ttest_flip.fillna(False).astype(bool).sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
