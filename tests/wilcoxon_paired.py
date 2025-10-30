import argparse, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon, rankdata

def method_dir(label: str) -> str:
    return "spd" if label == "spd_le" else label

def read_values(k: int, method: str, subjects, metric_cols):
    dname = method_dir(method)
    base = Path(f"results/k{k}/{dname}")
    vals = {m: [] for m in metric_cols}
    for s in subjects:
        cand = [base / f"result_{method}_k{k}_{s}.csv"]
        if s == "S1":
            cand.append(base / f"result_{method}_k{k}.csv")
        path = next((p for p in cand if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"missing: {method}, {s}, K={k}")
        df = pd.read_csv(path)
        for m in metric_cols:
            if m in df.columns:
                vals[m].append(pd.to_numeric(df[m], errors="coerce").mean())
            else:
                vals[m].append(np.nan)
    vals = {m: np.array(v, dtype=float) for m, v in vals.items() if not np.isnan(v).any()}
    return vals

def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    for i in range(n-1, -1, -1):
        adj[i] = min(prev, ranked[i] * n / (i+1))
        prev = adj[i]
    out = np.empty_like(adj)
    out[order] = adj
    return out

def rank_biserial(diffs):
    diffs = np.asarray(diffs, dtype=float)
    nz = diffs != 0
    if nz.sum() == 0:
        return 0.0
    r = rankdata(np.abs(diffs[nz]))
    pos = np.sum(r[diffs[nz] > 0])
    neg = np.sum(r[diffs[nz] < 0])
    return float((pos - neg) / (pos + neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--comparisons", type=str, required=True, help="e.g.: no_spd:spd_le,airm:hybrid_spd")
    ap.add_argument("--subjects", type=str, required=True)
    args = ap.parse_args()

    k = args.classes
    comps = [x.strip() for x in args.comparisons.split(",") if ":" in x]
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    first_left = comps[0].split(":")[0]
    possible = ["accuracy","precision","recall","iou","f1"]
 
    dname = method_dir(first_left)
    base = Path(f"results/k{k}/{dname}")
    ex = None
    for p in [base / f"result_{first_left}_k{k}_S1.csv", base / f"result_{first_left}_k{k}.csv"]:
        if p.exists():
            ex = pd.read_csv(p); break
    if ex is None:
        raise FileNotFoundError("Could not detect metrics.")
    metric_cols = [c for c in possible if c in ex.columns]
    if not metric_cols:
        metric_cols = [c for c in ex.columns if np.issubdtype(ex[c].dtype, np.number)]

    fam_p = {m: [] for m in metric_cols}
    fam_idx = {m: [] for m in metric_cols} 
    results = []  

    for pair in comps:
        left, right = pair.split(":")
        VL = read_values(k, left, subjects, metric_cols)
        VR = read_values(k, right, subjects, metric_cols)
        for metric in metric_cols:
            x = VL[metric]; y = VR[metric]
            stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
            r_rb = rank_biserial(x - y)
            n = int(np.isfinite(x - y).sum())
            fam_p[metric].append(p); fam_idx[metric].append(len(results))
            results.append((pair, metric, float(p), float(r_rb), n))

    results = list(results)
    results_df = pd.DataFrame(results, columns=["pair","metric","p_raw","r_rb","n"])
    results_df["p_fdr"] = np.nan
    for metric in metric_cols:
        if fam_p[metric]:
            adj = fdr_bh(np.array(fam_p[metric], dtype=float))
            idxs = fam_idx[metric]
            results_df.loc[idxs, "p_fdr"] = adj

    for pair in set(results_df["pair"]):
        left, right = pair.split(":")
        pair_df = results_df[results_df["pair"] == pair].copy()
        out = Path(f"results/wilcoxon_k{k}_{left}_vs_{right}.csv")
        pair_df[["metric","n","p_raw","p_fdr","r_rb"]].to_csv(out, index=False)

    out_all = Path(f"results/wilcoxon_k{k}_summary.csv")
    results_df.to_csv(out_all, index=False)

if __name__ == "__main__":
    sys.exit(main())
