#!/usr/bin/env python3
"""FedRGBD -- the pre-registered clean-subset rule and its excluded-image lists.

Rule (fixed 2026-09-19, before any clean-subset metric was computed):

    For every partition P, a validation or test image x of P is EXCLUDED from the
    clean subset iff some training image t of P (training data of all nodes) has
        Hamming(dHash(x), dHash(t)) <= 12   or   Hamming(pHash(x), pHash(t)) <= 10.
    Headline metrics are reported on all held-out images and, recomputed, on the
    held-out images that are not excluded.

Hashes are those of the leakage audit (scripts/analyze_flame_leakage.py,
``hash_image``, hash_size 8): dHash is read from analysis/leakage/groups.csv
(``hash_hex``), pHash is computed from the raw images (or read from
analysis/leakage/phash.csv.gz, which this script writes).  The training set of
a low-data partition (``*_sub*``) is its reduced training split.

Outputs (analysis/leakage/clean_subset/):
    RULE.json                       the rule, inputs and their md5s, counts, list sha256s
    <partition>_excluded.csv.gz     one row per excluded image (node, split, path, distances)
    nearest_train_distance.csv      held-out images binned by distance to the nearest
                                    training image, per partition/split/hash
    nearest_train_distance.tex      the same for the five full partitions (paper table)

The rule and these lists must not change after revision results have been seen
(CLAUDE.md).  Re-running this script must reproduce the committed files exactly;
``--check`` does that without writing.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RULE = {
    "name": "clean_subset_v1",
    "fixed_on": "2026-09-19",
    "text": ("A validation or test image x of partition P is excluded iff some training image t "
             "of P (all nodes) has Hamming(dHash(x), dHash(t)) <= 12 or "
             "Hamming(pHash(x), pHash(t)) <= 10."),
    "dhash_max": 12,
    "phash_max": 10,
    "hash_size": 8,
    "hash_code": "scripts/analyze_flame_leakage.py::hash_image",
    "fixed_before": ("any clean-subset metric was computed; at that time the full-set "
                     "centralized/local-only results and one federated run already existed"),
}
BINS = [(0, 8, "<=8"), (9, 10, "9-10"), (11, 12, "11-12"), (13, 16, "13-16"), (17, 64, ">16")]
PARTITIONS = ["iid", "non_iid_label", "dirichlet_0.1", "dirichlet_0.5", "dirichlet_1"]
PARTITIONS += [p + s for p in PARTITIONS for s in ("_sub0.05", "_sub0.01")]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes(groups_csv, phash_csv, data_dir, workers):
    g = pd.read_csv(groups_csv)
    dh = pd.Series([np.uint64(int(h, 16)) for h in g.hash_hex], index=g.path)
    if os.path.isfile(phash_csv):
        p = pd.read_csv(phash_csv)
        ph = pd.Series([np.uint64(int(h, 16)) for h in p.phash_hex], index=p.path)
    else:
        from scripts.analyze_flame_leakage import compute_hash_columns
        paths = [os.path.join(data_dir, x) for x in g.path]
        cols, _md5, failures = compute_hash_columns(paths, ["phash"], hash_size=RULE["hash_size"],
                                                    workers=workers)
        if failures:
            raise SystemExit("could not hash %d images, e.g. %s" % (len(failures), failures[0]))
        ph = pd.Series(cols["phash"], index=g.path)
        os.makedirs(os.path.dirname(phash_csv), exist_ok=True)
        with open(phash_csv, "wb") as f:
            f.write(gz_csv_bytes(pd.DataFrame(
                {"path": g.path, "phash_hex": ["%016x" % int(v) for v in ph]})))
    return dh, ph.reindex(dh.index)


def popcount(x):
    x = x.astype(np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


def nearest(q, t, block=256):
    out = np.empty(len(q), dtype=np.uint8)
    for s in range(0, len(q), block):
        out[s:s + block] = popcount(q[s:s + block, None] ^ t[None, :]).min(axis=1)
    return out


def gz_csv_bytes(df):
    """Deterministic gzip (no timestamp, no name) so reruns are byte-identical."""
    raw = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as f:
        f.write(raw)
    return buf.getvalue()


def compute(splits_dir, dh, ph):
    files, rows = {}, []
    for part in PARTITIONS:
        m = pd.read_csv(os.path.join(splits_dir, part + ".csv.gz"), dtype={"group_id": str})
        tr = m[m.split == "train"]
        ho = m[m.split != "train"].reset_index(drop=True)
        d = nearest(dh[ho.path].values, dh[tr.path].values)
        p = nearest(ph[ho.path].values, ph[tr.path].values)
        excluded = (d <= RULE["dhash_max"]) | (p <= RULE["phash_max"])
        ex = ho[excluded].assign(min_dhash=d[excluded], min_phash=p[excluded])
        ex = ex.sort_values(["node", "split", "path"])[
            ["node", "split", "label", "path", "group_id", "min_dhash", "min_phash"]]
        files["%s_excluded.csv.gz" % part] = gz_csv_bytes(ex)
        for split in ("val", "test"):
            sel = (ho.split == split).values
            for name, dist in (("dhash", d), ("phash", p)):
                row = dict(partition=part, split=split, hash=name, n=int(sel.sum()))
                for lo, hi, lab in BINS:
                    row[lab] = int(((dist[sel] >= lo) & (dist[sel] <= hi)).sum())
                row["excluded_by_rule"] = int(excluded[sel].sum())
                rows.append(row)
    table = pd.DataFrame(rows)
    files["nearest_train_distance.csv"] = table.to_csv(index=False, lineterminator="\n").encode()
    files["nearest_train_distance.tex"] = latex(table).encode()
    return files, table


def latex(table):
    main = table[table.partition.isin(PARTITIONS[:5])]
    names = {"iid": "IID", "non_iid_label": "Label skew", "dirichlet_0.1": r"Dir.\ $\alpha{=}0.1$",
             "dirichlet_0.5": r"Dir.\ $\alpha{=}0.5$", "dirichlet_1": r"Dir.\ $\alpha{=}1$"}
    lines = [
        "% generated by scripts/clean_subset.py -- do not edit by hand",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Held-Out Images by Hamming Distance to the Nearest Training Image (all nodes), "
        r"dHash and pHash, 64-bit Hashes}",
        r"\label{tab:nn_distance}", r"\setlength{\tabcolsep}{2.5pt}", r"\small",
        r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
        r"\textbf{Partition} & \textbf{Split} & \multicolumn{3}{c}{\textbf{dHash}} & "
        r"\multicolumn{3}{c}{\textbf{pHash}} & \textbf{Excl.} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r" & & $\le$8 & 9--10 & 11--12 & $\le$8 & 9--10 & $n$ & \\", r"\midrule"]
    for part in PARTITIONS[:5]:
        for split in ("val", "test"):
            dd = main[(main.partition == part) & (main.split == split) & (main.hash == "dhash")].iloc[0]
            pp = main[(main.partition == part) & (main.split == split) & (main.hash == "phash")].iloc[0]
            lines.append("%s & %s & %d & %d & %d & %d & %d & %d & %d \\\\" % (
                names[part] if split == "val" else "", split, dd["<=8"], dd["9-10"], dd["11-12"],
                pp["<=8"], pp["9-10"], dd["n"], dd["excluded_by_rule"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\footnotesize The partition is built so that no held-out image is within dHash "
              r"distance 8 of a training image; that guarantee is relative to this definition. "
              r"Excl.: images removed by the pre-registered clean-subset rule (dHash $\le$ 12 or "
              r"pHash $\le$ 10). Produced by \texttt{scripts/clean\_subset.py}.",
              r"\end{table}", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_dir", default=os.path.join(REPO, "data", "splits"))
    ap.add_argument("--groups_csv", default=os.path.join(REPO, "analysis", "leakage", "groups.csv"))
    ap.add_argument("--phash_csv", default=os.path.join(REPO, "analysis", "leakage", "phash.csv.gz"))
    ap.add_argument("--data_dir", default=os.path.join(REPO, "data", "raw", "flame_dataset"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "analysis", "leakage", "clean_subset"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare with the committed files; write nothing")
    args = ap.parse_args(argv)

    dh, ph = load_hashes(args.groups_csv, args.phash_csv, args.data_dir, args.workers)
    files, table = compute(args.splits_dir, dh, ph)
    rule = dict(RULE)
    rule["inputs_md5"] = {
        "analysis/leakage/groups.csv": md5_file(args.groups_csv),
        "analysis/leakage/phash.csv.gz (decompressed)": hashlib.md5(
            gzip.open(args.phash_csv).read()).hexdigest(),
        **{"data/splits/%s.csv.gz (decompressed)" % p: hashlib.md5(
            gzip.open(os.path.join(args.splits_dir, p + ".csv.gz")).read()).hexdigest()
           for p in PARTITIONS}}
    ex = table[table.hash == "dhash"].groupby("partition")[["n", "excluded_by_rule"]].sum()
    rule["excluded_counts"] = {p: {"held_out": int(ex.loc[p, "n"]),
                                   "excluded": int(ex.loc[p, "excluded_by_rule"])} for p in PARTITIONS}
    rule["list_sha256"] = {name: hashlib.sha256(gzip.decompress(b)).hexdigest()
                           for name, b in files.items() if name.endswith(".csv.gz")}
    files["RULE.json"] = (json.dumps(rule, indent=2, sort_keys=True) + "\n").encode()

    if args.check:
        bad = []
        for name, content in files.items():
            path = os.path.join(args.out_dir, name)
            old = open(path, "rb").read() if os.path.isfile(path) else None
            if name.endswith(".gz"):
                same = old is not None and gzip.decompress(old) == gzip.decompress(content)
            else:   # a Windows checkout may have converted the text files to CRLF
                same = old is not None and old.replace(b"\r\n", b"\n") == content
            if not same:
                bad.append(name)
        print("clean-subset files: %s" % ("identical to the committed ones" if not bad
                                          else "DIFFER: " + ", ".join(bad)))
        return 1 if bad else 0

    os.makedirs(args.out_dir, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(args.out_dir, name), "wb") as f:
            f.write(content)
    for p in PARTITIONS[:5]:
        c = rule["excluded_counts"][p]
        print("%-14s excluded %4d of %5d held-out images (%.1f%%)"
              % (p, c["excluded"], c["held_out"], 100.0 * c["excluded"] / c["held_out"]))
    print("wrote %d files to %s" % (len(files), args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
