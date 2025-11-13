from pathlib import Path  
import json 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 
from matplotlib.patches import Patch, FancyArrowPatch

here = Path(__file__).resolve()
root = here.parents[1]
combined_json = root / "dataset" / "json" / "costs.json"
outdir = root / "results" / "figs" / "figs_costs"
outdir.mkdir(parents=True, exist_ok=True)
 
def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", ""))
        except Exception:
            return None
    return None 

def load_costs():
    if not combined_json.exists():
        raise SystemExit(f"Missing file: {combined_json}")
    j = _read_json(combined_json)
    filters = j.get("filters", [])
    clustering = j.get("clustering", [])

    def norm_filters(obj):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                items.append({"name": k, "operations": _to_float(v)})
        else:
            for d in obj:
                items.append({"name": d.get("name"), "operations": _to_float(d.get("operations"))})
        order = [
            "Unfiltered","Average","Median","Hybrid",
            "Hybrid+Hybrid","Hybrid+Average","Hybrid+Median",
            "Average+Hybrid","Average+Average","Average+Median",
            "Median+Hybrid","Median+Average","Median+Median"
        ]
        items = [x for x in items if x["name"] in order and x["operations"] is not None]
        items.sort(key=lambda d: order.index(d["name"]))
        return items

    def norm_clustering(obj):
        items = []
        if isinstance(obj, dict):
            for metric, dd in obj.items():
                for k, v in dd.items():
                    items.append({"metric": metric, "k": int(k), "operations": _to_float(v)})
        else:
            for d in obj:
                items.append({"metric": d.get("metric"), "k": int(d.get("k")), "operations": _to_float(d.get("operations"))})
        items = [x for x in items if x["metric"] in ("Euclidean","Riemannian") and x["k"] in (2,3,4) and x["operations"] is not None]
        items.sort(key=lambda d: (d["k"], 0 if d["metric"]=="Euclidean" else 1))
        return items

    return norm_filters(filters), norm_clustering(clustering)

def _safe_log_values(vals):
    arr = np.asarray(vals, float)
    eps = 1e-6
    arr_plot = np.where(arr <= 0, eps, arr)
    labels = [f"{v/1e6:.3f}" for v in arr]
    return arr_plot, labels

def plot_filters_cost(items, out_png):
    labels = [d["name"] for d in items]
    vals = [d["operations"] for d in items]
    vals_log, lbls = _safe_log_values(vals)
    y_vals = vals_log / 1e6
    fig_w = max(10.0, 0.45 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    bars = ax.bar(np.arange(len(labels)), y_vals, color="#2FB47C", edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Operations (×10$^6$ ops)")
    ax.set_title("Filter computational cost", fontsize=10)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.grid(axis="y", linestyle="--", linewidth=0.6)
    ax.margins(y=0.15)
    for b, txt in zip(bars, lbls):
        ax.annotate(txt, (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _brace(ax, x0, x1, y_axes, text, fs=9):
    trans = ax.get_xaxis_transform()
    brace = FancyArrowPatch((x0, y_axes), (x1, y_axes), transform=trans,
                            arrowstyle="]-[", mutation_scale=14, lw=1.0, color="black")
    ax.add_artist(brace)
    ax.text((x0+x1)/2, y_axes-0.06, text, transform=trans, ha="center", va="top", fontsize=fs)

def plot_cluster_cost(items, out_png):
    tick_labels = ["Euclidean","Riemannian","Euclidean","Riemannian","Euclidean","Riemannian"]
    vals = [d["operations"] for d in items]
    vals_log, lbls = _safe_log_values(vals)
    y_vals = vals_log / 1e6
    colors = ["tab:green","tab:purple","tab:green","tab:purple","tab:green","tab:purple"]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(tick_labels))
    bars = ax.bar(x, y_vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Operations (×10$^6$ ops)")
    ax.set_title("Clustering cost by metric and k", fontsize=10)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.grid(axis="y", linestyle="--", linewidth=0.6)
    leg = [Patch(facecolor="tab:green", edgecolor="black", label="Euclidean"),
           Patch(facecolor="tab:purple", edgecolor="black", label="Riemannian")]
    ax.legend(leg, ["Euclidean","Riemannian"], loc="upper left",
              fontsize=8, frameon=True, handlelength=1.2, handletextpad=0.5, borderpad=0.25)
    for b, txt in zip(bars, lbls):
        ax.annotate(txt, (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(bottom=0.28)
    _brace(ax, -0.3, 1.3, -0.08, "k = 2", fs=9)
    _brace(ax, 1.7, 3.3, -0.08, "k = 3", fs=9)
    _brace(ax, 3.7, 5.3, -0.08, "k = 4", fs=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

filters_items, cluster_items = load_costs()
plot_filters_cost(filters_items, outdir / "filters_cost_log.png")
plot_cluster_cost(cluster_items, outdir / "clustering_cost_log.png")
print(str(outdir))

