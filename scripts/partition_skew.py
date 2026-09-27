#!/usr/bin/env python3
"""FedRGBD -- measured label skew and size imbalance of every partition.

The three Dirichlet partitions are single draws, one per concentration alpha, and
a single draw need not be ordered by alpha: which node receives which sequences
also changes the node sizes.  This script therefore describes each partition by
what it actually is, from the committed partition statistics
(``data/splits/split_stats.json``), and never by its alpha:

* per node: training images, training fire fraction, test images;
* per node: the Jensen-Shannon divergence (base 2, in [0, 1]) between the node's
  training label distribution and the partition's pooled training label
  distribution, and the total-variation distance |p_k(fire) - p(fire)| as a check;
* per partition: the node-size-weighted mean JSD -- the declared skew measure used
  to order partitions --, the maximum JSD, the size share of the largest node and
  the largest/smallest node size ratio.

The summary used to order partitions -- the node-size-weighted mean JSD -- was
chosen on 2026-09-27, AFTER the Dirichlet results existed; it is not
pre-registered.  It is not the only reasonable summary, and the order depends on
it: the unweighted mean and the maximum per-node JSD put Dirichlet 1.0 before
Dirichlet 0.5, because 0.5's low weighted value comes from one node holding 87 % of
the training images at the pooled ratio while its two small nodes are 97-98 %
fire.  The CSV therefore carries all of them (``jsd``, ``jsd_unweighted``,
``jsd_max``, ``tv``) and ``order_agrees`` says whether they order the tabulated
partitions the same way; the table note states the disagreement.

Writes ``analysis/partition_skew.csv`` (one row per node plus one ``ALL`` row per
partition) and ``analysis/partition_skew.tex`` (the Dirichlet partitions and the
two manual partitions, ordered by measured skew), which
``scripts/export_latex_tables.py`` copies to ``paper/tables/``.

    python scripts/partition_skew.py
"""

import argparse
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = ("node_a", "node_b", "node_c")
#: partitions tabulated in the paper; the subsampled ones inherit these proportions
MAIN = ("iid", "non_iid_label", "dirichlet_0.1", "dirichlet_0.5", "dirichlet_1")
LABELS = {"iid": "IID", "non_iid_label": "Manual label skew",
          "dirichlet_0.1": r"Dirichlet $\alpha{=}0.1$", "dirichlet_0.5": r"Dirichlet $\alpha{=}0.5$",
          "dirichlet_1": r"Dirichlet $\alpha{=}1.0$"}


def _h(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def jsd(p, q):
    """Jensen-Shannon divergence, base 2, of two discrete distributions."""
    m = [(a + b) / 2.0 for a, b in zip(p, q)]
    return _h(m) - (_h(p) + _h(q)) / 2.0


def describe(stats):
    """-> list of row dicts (per node + one ALL row per partition)."""
    rows = []
    for part, nodes in stats.items():
        if part.startswith("_") or not isinstance(nodes, dict):
            continue
        tr = {n: nodes[n]["train"] for n in NODES}
        n_tot = sum(t["total"] for t in tr.values())
        f_tot = sum(t["fire"] for t in tr.values())
        q = f_tot / float(n_tot)
        per = []
        for n in NODES:
            t = tr[n]
            p = t["fire"] / float(t["total"])
            row = dict(partition=part, node=n, n_train=t["total"], fire_frac_train=round(p, 4),
                       fire=t["fire"],
                       n_test=nodes[n]["test"]["total"],
                       fire_frac_test=round(nodes[n]["test"]["fire"] / float(nodes[n]["test"]["total"]), 4),
                       jsd=round(jsd([p, 1 - p], [q, 1 - q]), 5), tv=round(abs(p - q), 5),
                       size_share=round(t["total"] / float(n_tot), 4))
            per.append(row)
            rows.append(row)
        w = [r["n_train"] / float(n_tot) for r in per]
        sizes = [r["n_train"] for r in per]
        rows.append(dict(partition=part, node="ALL", n_train=n_tot, fire_frac_train=round(q, 4),
                         n_test=sum(r["n_test"] for r in per), fire_frac_test=None,
                         jsd=round(sum(wi * r["jsd"] for wi, r in zip(w, per)), 5),
                         jsd_unweighted=round(sum(r["jsd"] for r in per) / len(per), 5),
                         tv=round(sum(wi * r["tv"] for wi, r in zip(w, per)), 5),
                         size_share=round(max(sizes) / float(n_tot), 4),
                         jsd_max=round(max(r["jsd"] for r in per), 5),
                         size_ratio=round(max(sizes) / float(min(sizes)), 2)))
    return rows


def order(rows, parts, key="jsd"):
    allrow = {r["partition"]: r for r in rows if r["node"] == "ALL"}
    return sorted(parts, key=lambda p: allrow[p][key])


ORDER_NOTE = ""


def order_note(rows, parts):
    """One sentence on whether the alternative summaries give the same order."""
    names = {"jsd_unweighted": "the unweighted mean", "jsd_max": "the maximum per-node JSD",
             "tv": "the weighted total variation"}
    ref = order(rows, parts, "jsd")
    differ = [names[k] for k in ("jsd_unweighted", "jsd_max", "tv") if order(rows, parts, k) != ref]
    if not differ:
        return ("The unweighted mean, the maximum per-node JSD and the total variation give the "
                "same order.")
    return ("The order depends on the size weighting: %s order%s the partitions differently "
            "(Dirichlet $\\alpha{=}1.0$ before $\\alpha{=}0.5$), because the low weighted value "
            "of $\\alpha{=}0.5$ comes from one node holding most of the training images at "
            "the pooled class ratio. The weighted summary was chosen after the Dirichlet results "
            "existed." % (" and ".join(differ), "s" if len(differ) == 1 else ""))


def latex(rows):
    global ORDER_NOTE
    ORDER_NOTE = order_note(rows, order(rows, [p for p in MAIN if p in
                                               {r["partition"] for r in rows}]))
    allrow = {r["partition"]: r for r in rows if r["node"] == "ALL"}
    node = {(r["partition"], r["node"]): r for r in rows if r["node"] != "ALL"}
    parts = order(rows, [p for p in MAIN if p in allrow])
    lines = [
        "% generated by scripts/partition_skew.py -- do not edit by hand",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Measured Label Skew and Size Imbalance of the Partitions (Training Splits), "
        "Ordered by Skew}",
        "\\label{tab:partition_skew}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "Partition & Fire share A / B / C & Train images A / B / C & JSD & JSD, unweighted "
        "& max JSD & largest node \\\\",
        "\\midrule",
    ]
    for p in parts:
        a = allrow[p]
        # one decimal, from the exact counts (88.5 % must not print as 88)
        fire = " / ".join("%.1f" % (100.0 * node[(p, n)]["fire"] / node[(p, n)]["n_train"])
                          for n in NODES)
        size = " / ".join("{:,}".format(node[(p, n)]["n_train"]).replace(",", "{,}") for n in NODES)
        lines.append("%s & %s\\,\\%% & %s & %.3f & %.3f & %.3f & %.0f\\,\\%% \\\\"
                     % (LABELS.get(p, p), fire, size, a["jsd"], a["jsd_unweighted"], a["jsd_max"],
                        100 * a["size_share"]))
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\smallskip",
        "\\footnotesize JSD: Jensen--Shannon divergence (base 2) between a node's training label "
        "distribution and the partition's pooled one, averaged over nodes weighted by their "
        "training-set size (the ordering measure); unweighted: the plain mean over the three "
        "nodes; max JSD: the most skewed node; largest node: its share of all training images. "
        "Each Dirichlet partition is a single draw for its $\\alpha$, so the three are distinct "
        "instances ordered here by measured skew, not by $\\alpha$. " + ORDER_NOTE + " "
        "Produced from \\texttt{data/splits/split\\_stats.json}.",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats", default=os.path.join(REPO, "data", "splits", "split_stats.json"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "analysis"))
    args = ap.parse_args(argv)
    with open(args.stats, encoding="utf-8") as fh:
        rows = describe(json.load(fh))
    parts = [p for p in MAIN if any(r["partition"] == p for r in rows)]
    agrees = all(order(rows, parts, "jsd") == order(rows, parts, k)
                 for k in ("tv", "jsd_unweighted", "jsd_max"))
    for r in rows:
        if r["node"] == "ALL":
            r["order_agrees"] = agrees
    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["partition", "node", "n_train", "fire", "fire_frac_train", "n_test", "fire_frac_test",
            "jsd", "jsd_unweighted", "tv", "size_share", "jsd_max", "size_ratio", "order_agrees"]
    with open(os.path.join(args.out_dir, "partition_skew.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    with open(os.path.join(args.out_dir, "partition_skew.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(latex(rows))
    print("order by weighted-mean JSD:", " < ".join(order(rows, parts, "jsd")))
    for k in ("tv", "jsd_unweighted", "jsd_max"):
        print("order by %-15s" % k, " < ".join(order(rows, parts, k)))
    for p in order(rows, parts):
        a = [r for r in rows if r["partition"] == p and r["node"] == "ALL"][0]
        print("  %-15s JSD %.4f (max %.4f)  TV %.4f  largest node %.0f%%  size ratio %.1f"
              % (p, a["jsd"], a["jsd_max"], a["tv"], 100 * a["size_share"], a["size_ratio"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
