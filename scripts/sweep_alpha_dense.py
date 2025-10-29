import os, time, csv, argparse
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from dipy.io.image import load_nifti, save_nifti
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hybrid import make_border_interior
from segment_dti import segmentation
from metrics import (
    confusion_flat, iou_from_cm, dice_from_cm,
    precision_from_cm, recall_from_cm, f1_from_cm, accuracy_from_cm
)
from multirun import run_multi_seed
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from iteration_convergence import add_iteration_and_convergence_to_csv

def _cm_fg(y_true, y_pred, K_fg: int):
    t = y_true.reshape(-1); p = y_pred.reshape(-1)
    cm = np.zeros((K_fg, K_fg), dtype=np.int64)
    m = (t > 0) & (p > 0)
    for ti, pi in zip(t[m], p[m]):
        if 1 <= ti <= K_fg and 1 <= pi <= K_fg:
            cm[ti-1, pi-1] += 1
    return cm

def _relabel_by_hungarian(y_true, y_pred, K_fg: int):
    cm = _cm_fg(y_true, y_pred, K_fg)
    if cm.sum() == 0:
        return y_pred
    r, c = linear_sum_assignment(-cm)
    y_new = y_pred.copy()
    for ri, cj in zip(r, c):
        y_new[y_pred == (cj+1)] = (ri+1)
    return y_new

def _save_three_planes(y_ref, y_pred, out_dir: Path, method_name: str, seed: int, classes: int, fs=9):
    method_slug = method_name.lower().replace(" ", "_")
    folder_tag = f"k{classes}"
    X, Y, Z = y_ref.shape
    xs, ys, zs = X//2, Y//2, Z//2
    ref = np.rot90(y_ref[xs, :, :]); pred = np.rot90(y_pred[xs, :, :])
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow(ref);  plt.axis("off"); plt.title("reference", fontsize=fs)
    plt.subplot(1,2,2); plt.imshow(pred); plt.axis("off"); plt.title(f"sagittal • {method_name}", fontsize=fs)
    plt.tight_layout(); plt.savefig(out_dir/f"{folder_tag}_{method_slug}_seed{seed}_sagittal.png", dpi=150, bbox_inches="tight"); plt.close()
    ref = np.rot90(y_ref[:, ys, :]); pred = np.rot90(y_pred[:, ys, :])
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow(ref);  plt.axis("off"); plt.title("reference", fontsize=fs)
    plt.subplot(1,2,2); plt.imshow(pred); plt.axis("off"); plt.title(f"coronal • {method_name}", fontsize=fs)
    plt.tight_layout(); plt.savefig(out_dir/f"{folder_tag}_{method_slug}_seed{seed}_coronal.png", dpi=150, bbox_inches="tight"); plt.close()
    ref = np.rot90(y_ref[:, :, zs]); pred = np.rot90(y_pred[:, :, zs])
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow(ref);  plt.axis("off"); plt.title("reference", fontsize=fs)
    plt.subplot(1,2,2); plt.imshow(pred); plt.axis("off"); plt.title(f"axial • {method_name}", fontsize=fs)
    plt.tight_layout(); plt.savefig(out_dir/f"{folder_tag}_{method_slug}_seed{seed}_axial.png", dpi=150, bbox_inches="tight"); plt.close()

def _mean_std_ci(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(np.mean(x)) if n else float("nan")
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, ci, n

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ap = argparse.ArgumentParser(description="euclidean, riemannian, logeuclidean and hybrid; SPD aliases.")
    ap.add_argument("--metric",
        choices=[
            "riemannian","euclidean","logeuclidean","hybrid",
            "no_spd","spd_le","airm","hybrid_spd","hybrid_no_spd"
        ],
        default="riemannian"
    )
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--a-min", type=float, default=1.0)
    ap.add_argument("--a-max", type=float, default=2.0)
    ap.add_argument("--a-step", type=float, default=0.01)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument("--save-best", action="store_true")
    ap.add_argument("--data-dir", default="dataset/dataset1")
    ap.add_argument("--gt-pattern", default="stanford_hardi_denoised_segmentation_fa_{n}_classes.nii.gz")
    ap.add_argument("--dti-file", default="stanford_hardi_denoised_.nii.gz")
    ap.add_argument("--mask-file", default="stanford_hardi_denoised_mask.nii.gz")
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--iou-scheme", choices=["macro-fg","macro-all","micro-all-legacy"], default="macro-fg")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--border-metric", choices=["euclidean","riemannian","logeuclidean"], default="euclidean")
    ap.add_argument("--interior-metric", choices=["euclidean","riemannian","logeuclidean"], default="riemannian")
    ap.add_argument("--multi-seed", type=int, default=1)
    ap.add_argument("--seed-list", default="")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = (ROOT / args.data_dir).resolve()

    run_name_map = {
        "riemannian":"airm", "euclidean":"no_spd", "logeuclidean":"spd", "hybrid":"hybrid",
        "no_spd":"no_spd", "spd_le":"spd", "airm":"airm", "hybrid_spd":"hybrid_spd", "hybrid_no_spd":"hybrid_no_spd",
    }
    metric_label = args.metric
    run_name = run_name_map[metric_label]

    if metric_label == "no_spd":
        args.metric = "euclidean"
    elif metric_label == "spd_le":
        args.metric = "logeuclidean"
    elif metric_label == "airm":
        args.metric = "riemannian"
    elif metric_label == "hybrid_spd":
        args.metric = "hybrid"; args.border_metric = "euclidean"; args.interior_metric = "riemannian"
    elif metric_label == "hybrid_no_spd":
        args.metric = "hybrid"; args.border_metric = "riemannian"; args.interior_metric = "euclidean"

    OUT = (ROOT / "results" / f"k{args.classes}" / run_name).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Out").mkdir(parents=True, exist_ok=True)

    if args.seed_list:
        seeds_list = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
        for sd in seeds_list:
            run_multi_seed(
                script_path=Path(__file__).resolve(),
                metric=metric_label,
                classes=args.classes,
                a_min=args.a_min, a_max=args.a_max, a_step=args.a_step,
                max_iter=args.max_iter, restarts=args.restarts,
                data_dir=args.data_dir, dti_file=args.dti_file, mask_file=args.mask_file, gt_pattern=args.gt_pattern,
                seed_base=int(sd), seeds=1,
                iou_scheme=args.iou_scheme, radius=int(args.radius),
                border_metric=args.border_metric, interior_metric=args.interior_metric,
                save_best=bool(args.save_best)
            )

        base_csv = OUT / f"result_k{args.classes}_{run_name}.csv"

        if base_csv.exists():
            rows = []
            with open(base_csv, "r") as f:
                rd = csv.DictReader(f); rows.extend(rd)
            if rows:
                alphas = sorted(set(float(r["alpha"]) for r in rows))
                cols = ["precision_macro_fg","recall_macro_fg","f1_macro_fg","f1_micro_all","accuracy","iou_macro_fg","iou_macro_all","iou_micro_all","dice_macro_fg"]
                agg_path = OUT / f"result_k{args.classes}_{run_name}_aggregate_seeds.csv"
                with open(agg_path, "w", newline="") as fo:
                    wr = csv.writer(fo)
                    header = ["alpha"]
                    for c in cols:
                        header += [f"{c}_mean", f"{c}_std", f"{c}_ci95", f"{c}_n"]
                    wr.writerow(header)
                    for a in alphas:
                        sel = [r for r in rows if abs(float(r["alpha"])-a) < 1e-9]
                        stats = []
                        for c in cols:
                            arr = [float(r[c]) for r in sel]
                            m,s,ci,n = _mean_std_ci(arr)
                            stats.extend([f"{m:.6f}", f"{s:.6f}", f"{ci:.6f}", str(n)])
                        wr.writerow([f"{a:.6f}"] + stats)


                score_col = {"macro-fg":"iou_macro_fg","macro-all":"iou_macro_all","micro-all-legacy":"iou_micro_all"}[args.iou_scheme]
                best_by_seed = {}
                for r in rows:
                    sd = int(r["seed"])
                    sc = float(r[score_col])
                    a  = float(r["alpha"])
                    if (sd not in best_by_seed) or (sc > best_by_seed[sd]["score"]):
                        best_by_seed[sd] = {
                            "alpha": a, "score": sc,
                            "row": r
                        }
                best_csv = OUT / f"result_k{args.classes}_{run_name}_best_per_seed.csv"
                with open(best_csv, "w", newline="") as fb:
                    wrb = csv.writer(fb)
                    wrb.writerow([
                        "seed","alpha","score_metric",
                        "precision_macro_fg","recall_macro_fg","f1_macro_fg",
                        "f1_micro_all","accuracy","iou_macro_fg","iou_macro_all","iou_micro_all","dice_macro_fg"
                    ])
                    for sd in sorted(best_by_seed.keys()):
                        rr = best_by_seed[sd]["row"]
                        wrb.writerow([
                            str(sd), f"{best_by_seed[sd]['alpha']:.6f}", f"{best_by_seed[sd]['score']:.6f}",
                            rr["precision_macro_fg"], rr["recall_macro_fg"], rr["f1_macro_fg"],
                            rr["f1_micro_all"], rr["accuracy"],
                            rr["iou_macro_fg"], rr["iou_macro_all"], rr["iou_micro_all"],
                            rr["dice_macro_fg"]
                        ])

                alpha_star = float(np.median([v["alpha"] for v in best_by_seed.values()])) if best_by_seed else float("nan")

                for fname in [f"alpha_{metric_label}_k{args.classes}.txt", f"alpha_{run_name}_k{args.classes}.txt"]:
                    with open(OUT / fname, "w") as fa:
                        fa.write(f"{alpha_star:.6f}\n")
        return


    if args.multi_seed > 1:
        run_multi_seed(
            script_path=Path(__file__).resolve(),
            metric=metric_label,
            classes=args.classes,
            a_min=args.a_min, a_max=args.a_max, a_step=args.a_step,
            max_iter=args.max_iter, restarts=args.restarts,
            data_dir=args.data_dir, dti_file=args.dti_file, mask_file=args.mask_file, gt_pattern=args.gt_pattern,
            seed_base=int(args.seed), seeds=int(args.multi_seed),
            iou_scheme=args.iou_scheme, radius=int(args.radius),
            border_metric=args.border_metric, interior_metric=args.interior_metric,
            save_best=bool(args.save_best)
        )

        base_csv = OUT / f"result_k{args.classes}_{run_name}.csv"

        if base_csv.exists():
            rows = []
            with open(base_csv, "r") as f:
                rd = csv.DictReader(f); rows.extend(rd)
            if rows:
                alphas = sorted(set(float(r["alpha"]) for r in rows))
                cols = ["precision_macro_fg","recall_macro_fg","f1_macro_fg","f1_micro_all","accuracy","iou_macro_fg","iou_macro_all","iou_micro_all","dice_macro_fg"]
                agg_path = OUT / f"result_k{args.classes}_{run_name}_aggregate_seeds.csv"
                with open(agg_path, "w", newline="") as fo:
                    wr = csv.writer(fo)
                    header = ["alpha"]
                    for c in cols:
                        header += [f"{c}_mean", f"{c}_std", f"{c}_ci95", f"{c}_n"]
                    wr.writerow(header)
                    for a in alphas:
                        sel = [r for r in rows if abs(float(r["alpha"])-a) < 1e-9]
                        stats = []
                        for c in cols:
                            arr = [float(r[c]) for r in sel]
                            m,s,ci,n = _mean_std_ci(arr)
                            stats.extend([f"{m:.6f}", f"{s:.6f}", f"{ci:.6f}", str(n)])
                        wr.writerow([f"{a:.6f}"] + stats)

                score_col = {"macro-fg":"iou_macro_fg","macro-all":"iou_macro_all","micro-all-legacy":"iou_micro_all"}[args.iou_scheme]
                best_by_seed = {}
                for r in rows:
                    sd = int(r["seed"])
                    sc = float(r[score_col])
                    a  = float(r["alpha"])
                    if (sd not in best_by_seed) or (sc > best_by_seed[sd]["score"]):
                        best_by_seed[sd] = {
                            "alpha": a, "score": sc,
                            "row": r
                        }
                best_csv = OUT / f"result_k{args.classes}_{run_name}_best_per_seed.csv"
                with open(best_csv, "w", newline="") as fb:
                    wrb = csv.writer(fb)
                    wrb.writerow([
                        "seed","alpha","score_metric",
                        "precision_macro_fg","recall_macro_fg","f1_macro_fg",
                        "f1_micro_all","accuracy","iou_macro_fg","iou_macro_all","iou_micro_all","dice_macro_fg"
                    ])
                    for sd in sorted(best_by_seed.keys()):
                        rr = best_by_seed[sd]["row"]
                        wrb.writerow([
                            str(sd), f"{best_by_seed[sd]['alpha']:.6f}", f"{best_by_seed[sd]['score']:.6f}",
                            rr["precision_macro_fg"], rr["recall_macro_fg"], rr["f1_macro_fg"],
                            rr["f1_micro_all"], rr["accuracy"],
                            rr["iou_macro_fg"], rr["iou_macro_all"], rr["iou_micro_all"],
                            rr["dice_macro_fg"]
                        ])
                alpha_star = float(np.median([v["alpha"] for v in best_by_seed.values()])) if best_by_seed else float("nan")
                for fname in [f"alpha_{metric_label}_k{args.classes}.txt", f"alpha_{run_name}_k{args.classes}.txt"]:
                    with open(OUT / fname, "w") as fa:
                        fa.write(f"{alpha_star:.6f}\n")
        return


    dti_path = (DATA / args.dti_file)
    dti, affine = load_nifti(str(dti_path))
    mask, _   = load_nifti(str(DATA / args.mask_file)); mask = mask.astype(bool)
    y_true, _ = load_nifti(str(DATA / args.gt_pattern.format(n=args.classes)))
    if dti.ndim < 3 or mask.ndim != 3 or y_true.ndim != 3:
        raise ValueError("Invalid dimensionality.")
    if dti.shape[:3] != mask.shape or mask.shape != y_true.shape:
        raise ValueError("Shape mismatch among DTI, mask and ground truth.")
    if args.metric == "hybrid":
        border, interior = make_border_interior(mask, radius=args.radius)

    csv_path = OUT / f"result_k{args.classes}_{run_name}.csv"
    exists = csv_path.exists()

    alphas = np.arange(args.a_min, args.a_max + 1e-12, args.a_step)
    add_iteration_and_convergence_to_csv(OUT, alphas, dti, mask, y_true, args, _relabel_by_hungarian)

    with open(csv_path, "a", newline="") as f:
        wr = csv.writer(f)
        if not exists:
            wr.writerow([
                "alpha",
                "precision_macro_fg","recall_macro_fg","f1_macro_fg",
                "f1_micro_all","accuracy",
                "iou_macro_fg","iou_macro_all","iou_micro_all",
                "dice_macro_fg",
                "time_sec","seed","pred_path"
            ])

        per_restart_csv = OUT / f"result_k{args.classes}_{run_name}_per_restart.csv"
        per_exists = per_restart_csv.exists()
        per_f = open(per_restart_csv, "a", newline="")
        per_wr = csv.writer(per_f)
        if not per_exists:
            per_wr.writerow([
                "alpha","seed","restart",
                "precision_macro_fg","recall_macro_fg","f1_macro_fg",
                "f1_micro_all","accuracy",
                "iou_macro_fg","iou_macro_all","iou_micro_all",
                "dice_macro_fg","time_sec"
            ])

        agg_alpha_csv = OUT / f"result_k{args.classes}_{run_name}_aggregate_alpha.csv"
        agg_alpha_rows = []
        best_global = -1.0
        best_global_alpha = None
        best_global_pred = None
        best_global_seed = None

        alphas = np.arange(args.a_min, args.a_max + 1e-12, args.a_step)
        Kcm = args.classes + 1
        for a in alphas:
            t0 = time.time()
            best_local = -1.0
            best_seed  = None
            best_pred  = None
            best_metrics = None

            m_prec=[]; m_rec=[]; m_f1fg=[]; m_f1mi=[]; m_acc=[]; m_ioufg=[]; m_iouma=[]; m_ioumi=[]; m_dice=[]
            for r in range(args.restarts):
                np.random.seed(args.seed + r)
                r0 = time.time()

                if args.metric == "hybrid":
                    y_bor = segmentation(dti, n_claster=args.classes, mask=mask,
                                         metric_type=args.border_metric, expoent=float(a),
                                         max_iterations=args.max_iter, dim_point=3)
                    y_int = segmentation(dti, n_claster=args.classes, mask=mask,
                                         metric_type=args.interior_metric, expoent=float(a),
                                         max_iterations=args.max_iter, dim_point=3)
                    y_pred_raw = np.zeros_like(y_bor, dtype=np.int16)
                    if interior.any(): y_pred_raw[interior] = y_int[interior]
                    if border.any():   y_pred_raw[border]   = y_bor[border]
                else:
                    y_pred_raw = segmentation(dti, n_claster=args.classes, mask=mask,
                                              metric_type=args.metric, expoent=float(a),
                                              max_iterations=args.max_iter, dim_point=3)

                y_pred = _relabel_by_hungarian(y_true, y_pred_raw, args.classes)

                cm = confusion_flat(y_true.astype(np.int32), y_pred.astype(np.int32), n_classes=Kcm)
                iou_fg = iou_from_cm(cm, ignore_background=True,  background_class=0, mode="macro")
                iou_ma = iou_from_cm(cm, ignore_background=False, background_class=0, mode="macro")
                iou_mi = iou_from_cm(cm, ignore_background=False, background_class=0, mode="micro")
                score = {"macro-fg": iou_fg, "macro-all": iou_ma, "micro-all-legacy": iou_mi}[args.iou_scheme]

                prec_fg = precision_from_cm(cm, ignore_background=True,  background_class=0, mode="macro")
                rec_fg  = recall_from_cm(cm,    ignore_background=True,  background_class=0, mode="macro")
                f1_fg   = f1_from_cm(cm,        ignore_background=True,  background_class=0, mode="macro")
                f1_mi   = f1_from_cm(cm,        ignore_background=False, background_class=0, mode="micro")
                acc     = accuracy_from_cm(cm)
                dice_fg = dice_from_cm(cm, ignore_background=True, background_class=0, mode="macro")

                m_prec.append(prec_fg); m_rec.append(rec_fg); m_f1fg.append(f1_fg); m_f1mi.append(f1_mi)
                m_acc.append(acc); m_ioufg.append(iou_fg); m_iouma.append(iou_ma); m_ioumi.append(iou_mi); m_dice.append(dice_fg)

                dt_r = time.time() - r0
                per_wr.writerow([
                    f"{a:.6f}", str(args.seed + r), str(r),
                    f"{prec_fg:.6f}", f"{rec_fg:.6f}", f"{f1_fg:.6f}",
                    f"{f1_mi:.6f}", f"{acc:.6f}",
                    f"{iou_fg:.6f}", f"{iou_ma:.6f}", f"{iou_mi:.6f}",
                    f"{dice_fg:.6f}", f"{dt_r:.4f}"
                ])

                if score > best_local:
                    best_local = score
                    best_seed  = int(args.seed + r)
                    best_pred  = y_pred
                    best_metrics = (prec_fg, rec_fg, f1_fg, f1_mi, acc, iou_fg, iou_ma, iou_mi, dice_fg)

            dt = time.time() - t0
            pred_path = OUT/"Out"/f"gmm_k{args.classes}_{run_name}_alpha_{a:.3f}_seed{args.seed}.nii.gz"
            save_nifti(str(pred_path), best_pred.astype(np.int16), affine)
            wr.writerow([
                f"{a:.6f}",
                f"{best_metrics[0]:.6f}", f"{best_metrics[1]:.6f}", f"{best_metrics[2]:.6f}",
                f"{best_metrics[3]:.6f}", f"{best_metrics[4]:.6f}",
                f"{best_metrics[5]:.6f}", f"{best_metrics[6]:.6f}", f"{best_metrics[7]:.6f}",
                f"{best_metrics[8]:.6f}",
                f"{dt:.4f}", str(args.seed), str(pred_path)
            ])

            mp,sp,cp,np_ = _mean_std_ci(m_prec)
            mr,sr,cr,nr  = _mean_std_ci(m_rec)
            mfg,sfg,cfg,nfg = _mean_std_ci(m_f1fg)
            mmi,smi,cmi,nmi = _mean_std_ci(m_f1mi)
            ma,sa,ca,na = _mean_std_ci(m_acc)
            mif,sif,cif,nif = _mean_std_ci(m_ioufg)
            mia,sia,cia,nia = _mean_std_ci(m_iouma)
            mim,sim,cim,nim = _mean_std_ci(m_ioumi)
            md,sd,cd,nd = _mean_std_ci(m_dice)
            agg_alpha_rows.append([
                f"{a:.6f}",
                f"{mp:.6f}", f"{sp:.6f}", f"{cp:.6f}", str(np_),
                f"{mr:.6f}", f"{sr:.6f}", f"{cr:.6f}", str(nr),
                f"{mfg:.6f}", f"{sfg:.6f}", f"{cfg:.6f}", str(nfg),
                f"{mmi:.6f}", f"{smi:.6f}", f"{cmi:.6f}", str(nmi),
                f"{ma:.6f}", f"{sa:.6f}", f"{ca:.6f}", str(na),
                f"{mif:.6f}", f"{sif:.6f}", f"{cif:.6f}", str(nif),
                f"{mia:.6f}", f"{sia:.6f}", f"{cia:.6f}", str(nia),
                f"{mim:.6f}", f"{sim:.6f}", f"{cim:.6f}", str(nim),
                f"{md:.6f}", f"{sd:.6f}", f"{cd:.6f}", str(nd)
            ])

            if best_local > best_global:
                best_global = best_local
                best_global_alpha = float(a)
                best_global_pred = best_pred.copy()
                best_global_seed = int(best_seed)

        per_f.close()

    if 'best_pred' in locals() and best_pred is not None:
        _save_three_planes(
            y_true, best_pred, OUT,
            method_name=run_name, seed=int(args.seed), classes=int(args.classes), fs=9
        )

    if best_global_pred is not None and args.save_best:
        best_out = OUT/"Out"/f"gmm_k{args.classes}_{run_name}_best_alpha_{best_global_alpha:.3f}_seed{best_global_seed}.nii.gz"
        save_nifti(str(best_out), best_global_pred.astype(np.int16), affine)

        Kcm = args.classes + 1
        cm_best = confusion_flat(y_true.astype(np.int32), best_global_pred.astype(np.int32), n_classes=Kcm)
        prec_fg = precision_from_cm(cm_best, ignore_background=True,  background_class=0, mode="macro")
        rec_fg  = recall_from_cm(cm_best,    ignore_background=True,  background_class=0, mode="macro")
        f1_fg   = f1_from_cm(cm_best,        ignore_background=True,  background_class=0, mode="macro")
        f1_mi   = f1_from_cm(cm_best,        ignore_background=False, background_class=0, mode="micro")
        acc     = accuracy_from_cm(cm_best)
        iou_fg  = iou_from_cm(cm_best, ignore_background=True,  background_class=0, mode="macro")
        iou_ma  = iou_from_cm(cm_best, ignore_background=False, background_class=0, mode="macro")
        iou_mi  = iou_from_cm(cm_best, ignore_background=False, background_class=0, mode="micro")
        dice_fg = dice_from_cm(cm_best, ignore_background=True, background_class=0, mode="macro")

        with open(csv_path, "a", newline="") as f2:
            wr2 = csv.writer(f2)
            wr2.writerow([
                f"{best_global_alpha:.6f}",
                f"{prec_fg:.6f}", f"{rec_fg:.6f}", f"{f1_fg:.6f}",
                f"{f1_mi:.6f}", f"{acc:.6f}",
                f"{iou_fg:.6f}", f"{iou_ma:.6f}", f"{iou_mi:.6f}",
                f"{dice_fg:.6f}",
                f"{0.0000:.4f}", str(best_global_seed), str(best_out)
            ])

        _save_three_planes(
            y_true, best_global_pred, OUT,
            method_name=f"{run_name}_best", seed=int(best_global_seed), classes=int(args.classes), fs=9
        )

    agg_alpha_csv = OUT / f"result_k{args.classes}_{run_name}_aggregate_alpha.csv"
    if 'agg_alpha_rows' in locals() and agg_alpha_rows:
        with open(agg_alpha_csv, "w", newline="") as fa:
            w = csv.writer(fa)
            w.writerow([
                "alpha",
                "precision_macro_fg_mean","precision_macro_fg_std","precision_macro_fg_ci95","precision_macro_fg_n",
                "recall_macro_fg_mean","recall_macro_fg_std","recall_macro_fg_ci95","recall_macro_fg_n",
                "f1_macro_fg_mean","f1_macro_fg_std","f1_macro_fg_ci95","f1_macro_fg_n",
                "f1_micro_all_mean","f1_micro_all_std","f1_micro_all_ci95","f1_micro_all_n",
                "accuracy_mean","accuracy_std","accuracy_ci95","accuracy_n",
                "iou_macro_fg_mean","iou_macro_fg_std","iou_macro_fg_ci95","iou_macro_fg_n",
                "iou_macro_all_mean","iou_macro_all_std","iou_macro_all_ci95","iou_macro_all_n",
                "iou_micro_all_mean","iou_micro_all_std","iou_micro_all_ci95","iou_micro_all_n",
                "dice_macro_fg_mean","dice_macro_fg_std","dice_macro_fg_ci95","dice_macro_fg_n"
            ])
            for row in agg_alpha_rows:
                w.writerow(row)

    print(f"[OK] result: {run_name} | {OUT.name}")

if __name__ == "__main__":
    main()
