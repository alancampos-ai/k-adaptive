import csv   
import numpy as np
from segment_dti import segmentation
from metrics import confusion_flat, iou_from_cm
from hybrid import make_border_interior 

run_name_map = {
    "riemannian": "airm", "euclidean": "no_spd", "logeuclidean": "spd", "hybrid": "hybrid",
    "no_spd": "no_spd", "spd_le": "spd", "airm": "airm", "hybrid_spd": "hybrid_spd", "hybrid_no_spd": "hybrid_no_spd",
}

def _hybrid_run_tag(args):
    b = getattr(args, "border_metric", None)
    i = getattr(args, "interior_metric", None)
    if b == "euclidean" and i == "riemannian":
        return "hybrid_spd"
    if b == "riemannian" and i == "euclidean":
        return "hybrid_no_spd"
    return "hybrid"

def add_iteration_and_convergence_to_csv(OUT, alphas, dti, mask, y_true, args, relabel_function):
    if args.metric == "hybrid":
        run_tag = _hybrid_run_tag(args)
        print(f"[iteration_convergence] hybrid type: {run_tag} "
              f"(border={args.border_metric}, interior={args.interior_metric})")
    else:
        run_tag = run_name_map.get(args.metric, args.metric)

    iter_csv_path = OUT / f"result_k{args.classes}_{run_tag}_iteration_convergence.csv"
    iter_exists = iter_csv_path.exists()

    with open(iter_csv_path, "a", newline="") as iter_f:
        iter_writer = csv.writer(iter_f)
        if not iter_exists:
            iter_writer.writerow(["alpha", "iteration", "convergence"])

        for a in alphas:
            for r in range(args.restarts):
                np.random.seed(args.seed + r)

                if args.metric == "hybrid":
                    border, interior = make_border_interior(mask, radius=args.radius)

                    y_bor = segmentation(dti, n_claster=args.classes, mask=mask,
                                         metric_type=args.border_metric, expoent=float(a),
                                         max_iterations=args.max_iter, dim_point=3)
                    y_int = segmentation(dti, n_claster=args.classes, mask=mask,
                                         metric_type=args.interior_metric, expoent=float(a),
                                         max_iterations=args.max_iter, dim_point=3)

                    y_pred_raw = np.zeros_like(y_bor, dtype=np.int16)
                    if interior.any(): y_pred_raw[interior] = y_int[interior]
                    if border.any():   y_pred_raw[border]   = y_bor[border]

                elif args.metric == "no_spd":
                    y_pred_raw = segmentation(dti, n_claster=args.classes, mask=mask,
                                              metric_type="euclidean", expoent=float(a),
                                              max_iterations=args.max_iter, dim_point=3)

                elif args.metric == "spd_le":
                    y_pred_raw = segmentation(dti, n_claster=args.classes, mask=mask,
                                              metric_type="logeuclidean", expoent=float(a),
                                              max_iterations=args.max_iter, dim_point=3)

                elif args.metric == "airm":
                    y_pred_raw = segmentation(dti, n_claster=args.classes, mask=mask,
                                              metric_type="riemannian", expoent=float(a),
                                              max_iterations=args.max_iter, dim_point=3)

                else:
                    y_pred_raw = segmentation(dti, n_claster=args.classes, mask=mask,
                                              metric_type=args.metric, expoent=float(a),
                                              max_iterations=args.max_iter, dim_point=3)

                y_pred = relabel_function(y_true, y_pred_raw, args.classes)

                cm = confusion_flat(y_true.astype(np.int32), y_pred.astype(np.int32), n_classes=args.classes + 1)
                iou_fg = iou_from_cm(cm, ignore_background=True,  background_class=0, mode="macro")
                iou_ma = iou_from_cm(cm, ignore_background=False, background_class=0, mode="macro")
                iou_mi = iou_from_cm(cm, ignore_background=False, background_class=0, mode="micro")
                score = {"macro-fg": iou_fg, "macro-all": iou_ma, "micro-all-legacy": iou_mi}[args.iou_scheme]

                iteration = r + 1
                convergence = score
                iter_writer.writerow([f"{a:.6f}", iteration, convergence])
