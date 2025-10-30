import os, sys, argparse, re
import numpy as np
import pandas as pd
from typing import List, Optional
from io_utils import parse_list_int, parse_list_float, read_csv_auto, coerce_numeric_pair, load_pair_csvs
from spd import vec_to_spd
from eval_utils import (
    TYPE_NO_SPD_LE, TYPE_SPD_LE, TYPE_SPD_AIRM, TYPE_HYBRID_SPD, TYPE_HYBRID_NO_SPD,
    external_fit_predict, external_margin, silhouette_val, select_internal
)
def to_spd_stack_arrays(train_df: pd.DataFrame, val_df: pd.DataFrame, dim: int) -> tuple:
    m_tr = train_df.shape[1]
    m_va = val_df.shape[1]
    m = min(m_tr, m_va)
    d_eff = int((np.sqrt(8*m+1)-1)//2)
    while d_eff*(d_eff+1)//2 > m:
        d_eff -= 1
    if d_eff <= 1:
        raise RuntimeError("unable to infer SPD dimension from features")
    if d_eff < dim:
        dim = d_eff
    p = dim*(dim+1)//2
    Xtr = train_df.iloc[:, :p].values
    Xva = val_df.iloc[:, :p].values
    Str = vec_to_spd(Xtr, dim)
    Sva = vec_to_spd(Xva, dim)
    return Str, Sva, dim
def _normalize_type_name(s: str) -> str:
    z = re.sub(r"[^a-z0-9]+","_", s.strip().lower())
    if z == "no_spd":
        z = TYPE_NO_SPD_LE
    if z == "airm":
        z = TYPE_SPD_AIRM
    return z
def _method_display(t: str) -> str:
    if t == TYPE_NO_SPD_LE:
        return "no_spd"
    if t == TYPE_SPD_AIRM:
        return "airm"
    if t.startswith("hybrid_"):
        return "h_" + t.split("hybrid_",1)[1]
    return t
def _atomic_write(df: pd.DataFrame, out_path: str) -> int:
    if df is None or df.empty:
        return 2
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, out_path)
        return 0
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise
    finally:
        try:
            if os.path.exists(tmp) and os.path.getsize(tmp) == 0:
                os.remove(tmp)
        except Exception:
            pass
def _format_counts(labels: np.ndarray) -> tuple[int, str]:
    u, c = np.unique(labels, return_counts=True)
    return int(u.size), "|".join(f"{int(ui)}:{int(ci)}" for ui,ci in zip(u,c))
def _short_name(template: str, df: pd.DataFrame, split_key: Optional[str], split_val: Optional[str], k_override: Optional[int]) -> str:
    if split_key == "type" and split_val is not None:
        metodo = _method_display(split_val)
    elif "type" in df.columns and df["type"].nunique() == 1:
        metodo = _method_display(str(df["type"].iloc[0]))
    else:
        metodo = "mix"
    if split_key == "fold" and split_val is not None:
        f = str(split_val)
    else:
        f = str(int(df["fold"].nunique())) if "fold" in df.columns else "0"
    if k_override is not None:
        k = f"k{int(k_override)}"
    else:
        if "K_star" in df.columns:
            Ks = sorted(set(int(x) for x in df["K_star"].dropna().tolist()))
            k = f"k{Ks[0]}" if len(Ks)==1 else "kx"
        else:
            k = "kx"
    name = template.format(f=f, k=k, metodo=metodo)
    return name
def run_pair(types: List[str], K_list: List[int], alpha_list: List[float], seeds: List[int], boundary_q: float, spd_dim: Optional[int], train_csv: str, val_csv: str, label_col: Optional[str], subject_col: Optional[str]) -> pd.DataFrame:
    rows = []
    spd_types = [t for t in types if t in (TYPE_SPD_LE, TYPE_SPD_AIRM, TYPE_HYBRID_SPD)]
    nonspd_types = [t for t in types if t in (TYPE_NO_SPD_LE, TYPE_HYBRID_NO_SPD)]
    have_nonspd = False
    have_spd = False
    if nonspd_types:
        train_X_raw, val_X_raw, _, _, _, _ = load_pair_csvs(train_csv, val_csv, label_col, subject_col)
        have_nonspd = True
    if spd_types:
        df_tr = read_csv_auto(train_csv)
        df_va = read_csv_auto(val_csv)
        Xtr_raw, Xva_raw, _, _ = coerce_numeric_pair(df_tr, df_va, label_col, subject_col)
        if spd_dim is None:
            raise RuntimeError("spd_dim must be provided for SPD types")
        train_S, val_S, _ = to_spd_stack_arrays(Xtr_raw, Xva_raw, spd_dim)
        have_spd = True
    for seed in seeds:
        for t in types:
            if t in (TYPE_NO_SPD_LE, TYPE_HYBRID_NO_SPD) and not have_nonspd:
                continue
            if t in (TYPE_SPD_LE, TYPE_SPD_AIRM, TYPE_HYBRID_SPD) and not have_spd:
                continue
            if t in (TYPE_NO_SPD_LE, TYPE_HYBRID_NO_SPD):
                train_X = train_X_raw.values.astype(float)
                val_X = val_X_raw.values.astype(float)
            else:
                train_X = train_S
                val_X = val_S
            K_star, a_star = select_internal(t, train_X, K_list, alpha_list, seed=seed)
            _, lab_va, dist_va = external_fit_predict(t, train_X, val_X, K_star, a_star, seed=seed)
            sil = silhouette_val(t, train_X, val_X, lab_va, a_star)
            mar = external_margin(dist_va).mean() if dist_va is not None and dist_va.size else 0.0
            nc, cnts = _format_counts(lab_va)
            rows.append({
                "seed": int(seed),
                "type": t,
                "K_star": int(K_star),
                "alpha_star": (np.nan if a_star is None else float(a_star)),
                "n_clusters_val": int(nc),
                "counts_val": cnts,
                "silhouette": float(sil),
                "margin": float(mar),
                "train_csv": train_csv,
                "val_csv": val_csv
            })
    return pd.DataFrame(rows)
def run_from_folds(root: str, types: List[str], K_list: List[int], alpha_list: List[float], seeds: List[int], boundary_q: float, spd_dim: Optional[int], label_col: Optional[str], subject_col: Optional[str]) -> pd.DataFrame:
    folds = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    all_rows = []
    nonspd_types = [t for t in types if t in (TYPE_NO_SPD_LE, TYPE_HYBRID_NO_SPD)]
    spd_types    = [t for t in types if t in (TYPE_SPD_LE, TYPE_SPD_AIRM, TYPE_HYBRID_SPD)]
    for fold in folds:
        tr_csv = os.path.join(fold,"train.csv")
        va_csv = os.path.join(fold,"val.csv")
        tr_spd = os.path.join(fold,"train_spd.csv")
        va_spd = os.path.join(fold,"val_spd.csv")
        if nonspd_types and os.path.exists(tr_csv) and os.path.exists(va_csv):
            df = run_pair(nonspd_types, K_list, alpha_list, seeds, boundary_q, None, tr_csv, va_csv, label_col, subject_col)
            if not df.empty:
                df.insert(0,"fold",os.path.basename(fold))
                all_rows.append(df)
        if spd_types:
            if os.path.exists(tr_spd) and os.path.exists(va_spd):
                df = run_pair(spd_types, K_list, alpha_list, seeds, boundary_q, spd_dim, tr_spd, va_spd, label_col, subject_col)
            elif os.path.exists(tr_csv) and os.path.exists(va_csv):
                df = run_pair(spd_types, K_list, alpha_list, seeds, boundary_q, spd_dim, tr_csv, va_csv, label_col, subject_col)
            else:
                df = pd.DataFrame()
            if not df.empty:
                df.insert(0,"fold",os.path.basename(fold))
                all_rows.append(df)
    if not all_rows:
        return pd.DataFrame(columns=["fold","seed","type","K_star","alpha_star","n_clusters_val","counts_val","silhouette","margin","train_csv","val_csv"])
    return pd.concat(all_rows, ignore_index=True)
def main():
    ap = argparse.ArgumentParser(description="nested unsupervised k-means/k-medoids with LOSO internal selection")
    ap.add_argument("--root", type=str, default="dataset")
    ap.add_argument("--types", type=str, default=None)
    ap.add_argument("--metric", action="append")
    ap.add_argument("--K", type=str, required=True)
    ap.add_argument("--alpha", type=str, default="0.5")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--subject-col", type=str, default=None)
    ap.add_argument("--spd-dim", type=int, default=None)
    ap.add_argument("--boundary-q", type=float, default=0.3)
    ap.add_argument("--train-csv", type=str, default=None)
    ap.add_argument("--val-csv", type=str, default=None)
    ap.add_argument("--from-folds", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-template", type=str, default=None)
    ap.add_argument("--split-by", type=str, choices=["type","fold"], default=None)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()
    types: List[str] = []
    if args.types:
        types += [_normalize_type_name(t) for t in args.types.split(",") if t.strip()]
    if args.metric:
        types += [_normalize_type_name(t) for t in args.metric if t]
    types = [t for t in types if t]
    valid = {TYPE_NO_SPD_LE, TYPE_SPD_LE, TYPE_SPD_AIRM, TYPE_HYBRID_SPD, TYPE_HYBRID_NO_SPD}
    for t in types:
        if t not in valid:
            raise SystemExit(f"invalid type: {t}")
    if not types:
        raise SystemExit("provide --types or at least one --metric")
    K_list = parse_list_int(args.K)
    alpha_list = parse_list_float(args.alpha)
    seeds = parse_list_int(args.seeds)
    if args.from_folds:
        df = run_from_folds(args.root, types, K_list, alpha_list, seeds, args.boundary_q, args.spd_dim, args.label_col, args.subject_col)
    elif args.train_csv and args.val_csv:
        df = run_pair(types, K_list, alpha_list, seeds, args.boundary_q, args.spd_dim, args.train_csv, args.val_csv, args.label_col, args.subject_col)
    else:
        raise SystemExit("provide either --from-folds or --train-csv/--val-csv")
    if df is None or df.empty:
        sys.exit(2)
    if args.out_template:
        if args.split_by == "type":
            for t in sorted(df["type"].unique().tolist()):
                d = df[df["type"]==t].copy()
                out_path = _short_name(args.out_template, d, "type", t, args.k)
                code = _atomic_write(d, out_path)
                if code == 2:
                    sys.exit(2)
            return
        if args.split_by == "fold":
            for fval in sorted(df["fold"].unique().tolist()):
                d = df[df["fold"]==fval].copy()
                out_path = _short_name(args.out_template, d, "fold", fval, args.k)
                code = _atomic_write(d, out_path)
                if code == 2:
                    sys.exit(2)
            return
        out_path = _short_name(args.out_template, df, None, None, args.k)
        code = _atomic_write(df, out_path)
        if code == 2:
            sys.exit(2)
        return
    if args.out:
        code = _atomic_write(df, args.out)
        if code == 2:
            sys.exit(2)
        return
    df.to_csv(sys.stdout, index=False)
if __name__ == "__main__":
    main()
