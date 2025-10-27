#!/usr/bin/env python
import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

def spd_le(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.log(w)) @ V.T

def vec6(M):
    s2 = np.sqrt(2.0)
    return np.array([M[0,0], M[1,1], M[2,2], s2*M[0,1], s2*M[0,2], s2*M[1,2]], dtype=float)


def load_data(data_dir, K):
    d = Path(data_dir)
    dti = nib.load(str(d / "stanford_hardi_denoised_.nii.gz")).get_fdata()
    msk = nib.load(str(d / "stanford_hardi_denoised_mask.nii.gz")).get_fdata()
    ref = nib.load(str(d / f"stanford_hardi_denoised_segmentation_fa_{K}_classes.nii.gz")).get_fdata()
    assert dti.ndim == 5 and dti.shape[-2:] == (3, 3), "DTI must be (X,Y,Z,3,3)"
    assert msk.ndim == 3 and ref.ndim == 3, "Mask and GT must be 3D"
    assert dti.shape[:3] == msk.shape == ref.shape, "DTI/Mask/GT shapes must match"
    msk = msk > 0
    X, Y, Z = ref.shape
    dti = dti.reshape(X*Y*Z, 3, 3)
    mskv = msk.reshape(-1)
    refv = ref.reshape(-1).astype(int)
    idx = np.where(mskv & (refv > 0))[0]
    feats = np.stack([vec6(spd_le(dti[i])) for i in idx], axis=0)
    y = refv[idx]
    return feats, y

def confusion(y_true, y_pred, K):
    C = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 1 <= t <= K and 1 <= p <= K:
            C[t-1, p-1] += 1
    return C

def metrics_from_cm(C):
    tp = np.diag(C).astype(float)
    fp = C.sum(0) - tp
    fn = C.sum(1) - tp
    tn = C.sum() - (tp + fp + fn)
    prec = np.where(tp+fp>0, tp/(tp+fp), 0.0).mean()
    rec  = np.where(tp+fn>0, tp/(tp+fn), 0.0).mean()
    iou  = np.where(tp+fp+fn>0, tp/(tp+fp+fn), 0.0).mean()
    f1   = np.where(2*tp+fp+fn>0, 2*tp/(2*tp+fp+fn), 0.0).mean()
    acc  = np.trace(C)/C.sum() if C.sum()>0 else 0.0
    return acc, prec, rec, iou, f1

def hungarian_match(y_true, y_pred, K):
    C = confusion(y_true, y_pred, K)
    r, c = linear_sum_assignment(-C)
    m = {cj+1: ri+1 for ri, cj in zip(r, c)}
    return np.array([m.get(p, p) for p in y_pred], dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--algo", choices=["kmeans", "mbkmeans", "gmm"], default="kmeans")
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--covariance-type", choices=["full","tied","diag","spherical"], default="full")
    ap.add_argument("--reg-covar", type=float, default=1e-6)
    ap.add_argument("--standardize", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    K = args.classes
    X, y = load_data(args.data_dir, K)

    if args.standardize:
        X = StandardScaler().fit_transform(X)

    if args.algo == "kmeans":
        model = KMeans(n_clusters=K, random_state=args.seed, n_init=args.n_init, max_iter=args.max_iter)
        y_pred = model.fit_predict(X) + 1
    elif args.algo == "mbkmeans":
        model = MiniBatchKMeans(n_clusters=K, random_state=args.seed, batch_size=args.batch_size, max_iter=args.max_iter)
        y_pred = model.fit_predict(X) + 1
    else:
        model = GaussianMixture(
            n_components=K,
            covariance_type=args.covariance_type,
            random_state=args.seed,
            max_iter=args.max_iter,
            n_init=args.n_init,
            reg_covar=args.reg_covar
        )
        y_pred = model.fit_predict(X) + 1

    y_pred = hungarian_match(y, y_pred, K)
    C = confusion(y, y_pred, K)
    acc, prec, rec, iou, f1 = metrics_from_cm(C)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("metric,value\n")
        f.write(f"accuracy,{acc:.6f}\n")
        f.write(f"precision,{prec:.6f}\n")
        f.write(f"recall,{rec:.6f}\n")
        f.write(f"iou,{iou:.6f}\n")
        f.write(f"f1,{f1:.6f}\n")

if __name__ == "__main__":
    main()
