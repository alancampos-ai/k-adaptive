import argparse, os, sys
import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path
from scipy.stats import t

def method_dir(label: str) -> str:
    return "spd" if label == "spd_le" else label

def read_csv_result(k: int, method: str, subject: str) -> pd.DataFrame:
    dname = method_dir(method)
    base = Path(f"results/k{k}/{dname}")
    cand = [
        base / f"result_{method}_k{k}_{subject}.csv",
        base / f"result_{method}_k{k}.csv" if subject == "S1" else None,
    ]
    for p in cand:
        if p is not None and p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"Missing CSV for {method}, K={k}, {subject}")

def ci95(v: np.ndarray):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n < 2:
        return (np.nan, np.nan, np.nan, n)
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1))
    se = s / sqrt(n)
    tcrit = float(t.ppf(0.975, df=n-1))
    lo, hi = m - tcrit*se, m + tcrit*se
    return (m, lo, hi, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--metrics", type=str, required=True, help="method name: no_spd, spd_le, airm, hybrid_spd, hybrid_no_spd")
    ap.add_argument("--subjects", type=str, required=True, help="list: S1,S2,...")
    args = ap.parse_args()

    k = args.classes
    method = args.metrics
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    ex = read_csv_result(k, method, subjects[0])
    possible = ["accuracy","precision","recall","iou","f1"]
    cols = [c for c in possible if c in ex.columns]
    if not cols:
        cols = [c for c in ex.columns if pd.api.types.is_numeric_dtype(ex[c])]
    if not cols:
        raise RuntimeError("No numeric metric found.")

    table = []
    for s in subjects:
        df = read_csv_result(k, method, s)
        row = {"subject": s}
        for c in cols:
            row[c] = pd.to_numeric(df[c], errors="coerce").mean()
        table.append(row)
    tab = pd.DataFrame(table).set_index("subject")

    summary = []
    for c in cols:
        m, lo, hi, n = ci95(tab[c].to_numpy())
        summary.append({"metric": c, "n": n, "mean": m, "ci95_low": lo, "ci95_high": hi, "std": float(np.std(tab[c], ddof=1))})
    res = pd.DataFrame(summary)

    out = Path(f"results/results_summary_k{k}_{method}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("# values per subject (means per CSV)\n")
        tab.reset_index().to_csv(f, index=False)
        f.write("\n# per-metric summary with 95% CI\n")
        res.to_csv(f, index=False)

if __name__ == "__main__":
    sys.exit(main())
