import os    
import json   
from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors, patches, patheffects

alphas = [1.00, 1.25, 1.50, 1.75, 2.00] 
ks = [2, 3, 4]   
 
here = Path(__file__).resolve()
root = here.parents[1]
data_path = root / "dataset" / "json"/"data.json"
outdir = root / "results" / "figs" / "figs_heatmaps"
outdir.mkdir(parents=True, exist_ok=True)

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
 
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12, 
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlepad": 6,
})

simple_filters = ["Unfiltered", "Average", "Median", "Hybrid"]
base = ["Average", "Median", "Hybrid"]
double_filters = [f"{a}+{b}" for a in base for b in base]

def to_label(f):
    return f.replace("+", " \u2192 ")

def M_from(d, metric, filt):
    M = np.zeros((len(ks), len(alphas)), dtype=float)
    for i, k in enumerate(ks):
        M[i, :] = np.asarray(d[metric][str(k)][filt], dtype=float)
    return M

def annotate(ax, M):
    r, c = M.shape
    for i in range(r):
        for j in range(c):
            t = ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
            t.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])
    for i in range(r):
        j = int(np.argmax(M[i, :]))
        ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                        linewidth=1.2, linestyle="--", edgecolor="white"))
    for j in range(c):
        i = int(np.argmax(M[:, j]))
        ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                        linewidth=1.4, linestyle="-", edgecolor="white"))

def panel(d, metric, filters, nrows, ncols, out_png, out_pdf, super_title):
    fig = plt.figure(figsize=(5.6*ncols, 5.0*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols,
                          left=0.07, right=0.87,
                          bottom=0.08, top=0.90,
                          wspace=0.18, hspace=0.30)
    axes = gs.subplots(sharex=False, sharey=False)
    norm = colors.Normalize(vmin=0, vmax=100)
    im = None
    for idx, filt in enumerate(filters):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        M = M_from(d, metric, filt)
        im = ax.imshow(M, norm=norm, aspect="auto")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.2f}" for a in alphas])
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([str(k) for k in ks])
        ax.set_xlabel("α")
        ax.set_ylabel("Clusters (k)")
        ax.set_title(to_label(filt), fontsize=11, pad=4)
        annotate(ax, M)
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("IoU (%)")
    fig.suptitle(super_title, fontsize=13, y=0.96)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

for metric in ["Euclidean", "Riemannian"]:
    panel(data, metric, simple_filters, 2, 2,
          outdir / f"heatmaps_{metric}_simple.png",
          outdir / f"heatmaps_{metric}_simple.pdf",
          f"IoU (%) | Metric: {metric} | Simple filters")

for metric in ["Euclidean", "Riemannian"]:
    panel(data, metric, double_filters, 3, 3,
          outdir / f"heatmaps_{metric}_double.png",
          outdir / f"heatmaps_{metric}_double.pdf",
          f"IoU (%) | Metric: {metric} | Double filtering (order matters)")

print("Saved to:", str(outdir))
