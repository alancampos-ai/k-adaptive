from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 

here = Path(__file__).resolve()
root = here.parents[1]
data_path = root / "dataset" / "json"/"data.json"
outdir = root / "results" / "figs" / "figs_bars"
outdir.mkdir(parents=True, exist_ok=True)

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

alphas = [1.0, 1.25, 1.5, 1.75, 2.0]
alpha_labels = [r"$\alpha=1.0$", r"$\alpha=1.25$", r"$\alpha=1.5$", r"$\alpha=1.75$", r"$\alpha=2.0$"]

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.frameon": True,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
})

Ks = [2, 3, 4]
filters_order = list(data["Euclidean"]["2"].keys())

def to_label(s: str) -> str:
    return s.replace("+", " \u2192 ")

def series(metric: str, k: int, filt: str):
    vals = data[metric][str(k)][filt]
    return [v / 100.0 for v in vals]

all_vals = []
for metric in ["Euclidean", "Riemannian"]:
    for k in Ks:
        for f in filters_order:
            all_vals += series(metric, k, f)
HEADROOM = 0.04
YMIN = max(0.0, float(np.min(all_vals)) - 0.03)
YMAX = min(1.0, float(np.max(all_vals)) + HEADROOM)

E_COLOR = "tab:green"
R_COLOR = "tab:purple"

def _annotate_pair_outside(bars_left, bars_right, ax, pad_frac=0.012, stagger=0.006):
    pad = pad_frac * (YMAX - YMIN)
    cap = YMAX - pad
    for bl, br in zip(bars_left, bars_right):
        hl, hr = bl.get_height(), br.get_height()
        yL = min(hl + pad, cap)
        yR = min(hr + pad, cap)
        if abs(yL - yR) < 0.004:
            yL = min(yL + stagger, cap)
            yR = min(yR + 2 * stagger, cap)
        for b, y in ((bl, yL), (br, yR)):
            ax.annotate(f"{b.get_height()*100:.2f}%",
                        xy=(b.get_x() + b.get_width()/2, y),
                        ha="center", va="bottom", fontsize=5, clip_on=True)

def plot_grouped_perK(alpha_idx: int, k: int) -> Path:
    labels = [to_label(f) for f in filters_order]
    eu = np.array([series("Euclidean", k, f)[alpha_idx] for f in filters_order])
    ri = np.array([series("Riemannian", k, f)[alpha_idx] for f in filters_order])

    n = len(labels)
    x = np.arange(n, dtype=float)
    width = 0.38
    fig_w = max(10.0, 0.7 * n + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))

    b1 = ax.bar(x - width/2, eu, width, label="Euclidean", color=E_COLOR)
    b2 = ax.bar(x + width/2, ri, width, label="Riemannian", color=R_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(YMIN, YMAX)
    ax.set_ylabel("IoU (%)")
    ax.set_xlabel("Filter")
    ax.set_title(f"{alpha_labels[alpha_idx]} | k={k}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", which="major")
    ax.minorticks_on()
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.4)
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    _annotate_pair_outside(b1, b2, ax)
    fig.tight_layout()
    out = outdir / f"bars_exact_alpha_{str(alphas[alpha_idx]).replace('.','_')}_K{k}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

if __name__ == "__main__":
    outs = []
    for i in range(len(alphas)):
        for k in Ks:
            outs.append(plot_grouped_perK(i, k)) 
    print("Generated PNGs:")
    for p in outs:
        print(p.resolve())
