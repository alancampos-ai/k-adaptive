import argparse, numpy as np, csv
from pathlib import Path
from scipy.stats import wilcoxon

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    M = (a[:,None] > b[None,:]) - (a[:,None] < b[None,:])
    return float(M.mean())

def bootstrap_ci(x, it=10000, alpha=0.05, seed=123):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    est = np.median(x)
    reps = rng.integers(0, len(x), size=(it, len(x)))
    meds = np.median(x[reps], axis=1)
    lo, hi = np.quantile(meds, [alpha/2, 1-alpha/2])
    return float(est), float(lo), float(hi)

def read_metric(paths, metric):
    vals = []
    for p in paths:
        d = {}
        with open(p) as f:
            r = csv.reader(f)
            next(r)
            for k,v in r:
                d[k]=float(v)
        vals.append(d[metric])
    return np.array(vals, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", nargs="+", required=True)
    ap.add_argument("--right", nargs="+", required=True)
    ap.add_argument("--metric", default="iou")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x = read_metric(args.left, args.metric)
    y = read_metric(args.right, args.metric)
    assert len(x)==len(y)
    stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
    delta = cliffs_delta(x, y)
    med, lo, hi = bootstrap_ci(x - y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("n,metric,median_diff,ci_lo,ci_hi,wilcoxon_p,cliffs_delta\n")
        f.write(f"{len(x)},{args.metric},{med:.6f},{lo:.6f},{hi:.6f},{p:.6g},{delta:.6f}\n")

if __name__ == "__main__":
    main()
