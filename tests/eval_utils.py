import numpy as np 
from typing import Optional, Tuple
from sklearn.mixture import GaussianMixture
from gmm_utils import StandardScalerSimple, euclid_pairwise, silhouette_from_dist
from spd import pairwise_logeuc, pairwise_airm, spd_logm
 
TYPE_NO_SPD_LE = "no_spd_le"
TYPE_SPD_LE = "spd_le"
TYPE_SPD_AIRM = "spd_airm"
TYPE_HYBRID_SPD = "hybrid_spd"
TYPE_HYBRID_NO_SPD = "hybrid_no_spd"

_MAX_TRAIN_SIL = 5000
_MAX_VAL_SIL = 6000

def _vecu(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(M.shape[-1])
    return M[..., idx[0], idx[1]]

def _to_features_nonspd(X_tr: np.ndarray, X_va: np.ndarray):
    scaler = StandardScalerSimple().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_va), scaler

def _to_features_spd_le(S_tr: np.ndarray, S_va: np.ndarray):
    L_tr = np.array([spd_logm(S) for S in S_tr])
    L_va = np.array([spd_logm(S) for S in S_va])
    V_tr = np.array([_vecu(S) for S in L_tr])
    V_va = np.array([_vecu(S) for S in L_va])
    scaler = StandardScalerSimple().fit(V_tr)
    return scaler.transform(V_tr), scaler.transform(V_va), scaler

def _kmedoids(D: np.ndarray, K: int, seed: int, iters: int = 100):
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    med = [rng.randint(0, n)]
    for _ in range(1, K):
        dist_to_nearest = np.min(D[:, med], axis=1)
        s = dist_to_nearest.sum()
        probs = np.ones(n)/n if s <= 0 else dist_to_nearest/s
        med.append(int(rng.choice(n, p=probs)))
    med = np.array(sorted(set(med)))[:K]
    if med.size < K:
        add = rng.choice([i for i in range(n) if i not in med], size=K-med.size, replace=False)
        med = np.concatenate([med, add])
    labels = np.argmin(D[:, med], axis=1)
    for _ in range(iters):
        improved = False
        for i in range(K):
            idx = np.where(labels == i)[0]
            if idx.size == 0:
                continue
            intra = D[np.ix_(idx, idx)]
            new_med_local = idx[np.argmin(intra.sum(axis=1))]
            if new_med_local != med[i]:
                med[i] = new_med_local
                labels = np.argmin(D[:, med], axis=1)
                improved = True
        if not improved:
            break
    return labels, med

def hybrid_pairwise_spd(S: np.ndarray, alpha: float) -> np.ndarray:
    D1 = pairwise_logeuc(S)
    L = np.array([spd_logm(A) for A in S])
    V = np.array([_vecu(A) for A in L])
    scaler = StandardScalerSimple().fit(V)
    Vn = scaler.transform(V)
    D2 = euclid_pairwise(Vn)
    return alpha*D1 + (1.0-alpha)*D2

def gmm_euclidean(X: np.ndarray, K: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    gm = GaussianMixture(n_components=K, covariance_type="full", random_state=seed, init_params="random")
    gm.fit(X)
    labels = gm.predict(X)
    centers = gm.means_
    return labels, centers

def external_fit_predict(model_type: str, train_X, val_X, K: int, alpha: Optional[float], seed: int):
    if model_type == TYPE_NO_SPD_LE or model_type == TYPE_HYBRID_NO_SPD:
        Xtr, Xva, _ = _to_features_nonspd(train_X, val_X)
        lab_tr, ctr = gmm_euclidean(Xtr, K, seed=seed)
        dist_va = ((Xva[:,None,:]-ctr[None,:,:])**2).sum(axis=2)
        lab_va = np.argmin(dist_va, axis=1)
        return lab_tr, lab_va, dist_va
    if model_type == TYPE_SPD_LE:
        Ftr, Fva, _ = _to_features_spd_le(train_X, val_X)
        lab_tr, ctr = gmm_euclidean(Ftr, K, seed=seed)
        dist_va = ((Fva[:,None,:]-ctr[None,:,:])**2).sum(axis=2)
        lab_va = np.argmin(dist_va, axis=1)
        return lab_tr, lab_va, dist_va
    if model_type == TYPE_SPD_AIRM:
        Dtr = pairwise_airm(train_X)
        lab_tr, med = _kmedoids(Dtr, K, seed=seed)
        med_S = train_X[med]
        nva = val_X.shape[0]
        dist_va = np.zeros((nva, K))
        for j in range(K):
            Sj = med_S[j]
            wS, VS = np.linalg.eigh(Sj)
            wS[wS < 1e-12] = 1e-12
            Sih = (VS * (1.0/np.sqrt(wS))) @ VS.T
            for i in range(nva):
                A = val_X[i]
                B = Sih @ A @ Sih
                w = np.linalg.eigvalsh(B)
                w[w < 1e-12] = 1e-12
                dist_va[i, j] = np.sqrt(np.sum(np.log(w)**2))
        lab_va = np.argmin(dist_va, axis=1)
        return lab_tr, lab_va, dist_va
    if model_type == TYPE_HYBRID_SPD:
        a = 0.5 if alpha is None else float(alpha)
        Dtr = hybrid_pairwise_spd(train_X, a)
        lab_tr, med = _kmedoids(Dtr, K, seed=seed)
        med_S = train_X[med]
        Lva = np.array([spd_logm(A) for A in val_X])
        Vva = np.array([_vecu(A) for A in Lva])
        Ltr = np.array([spd_logm(A) for A in train_X])
        Vtr = np.array([_vecu(A) for A in Ltr])
        scaler = StandardScalerSimple().fit(Vtr)
        Vva = scaler.transform(Vva)
        nva = Vva.shape[0]
        dist_va = np.zeros((nva, K))
        for j in range(K):
            Sj = med_S[j]
            wS, VS = np.linalg.eigh(Sj)
            wS[wS < 1e-12] = 1e-12
            Sih = (VS * (1.0/np.sqrt(wS))) @ VS.T
            for i in range(nva):
                A = val_X[i]
                B = Sih @ A @ Sih
                w = np.linalg.eigvalsh(B)
                w[w < 1e-12] = 1e-12
                d1 = np.sqrt(np.sum(np.log(w)**2))
                d2 = np.linalg.norm(Vva[i] - scaler.transform(np.array([_vecu(spd_logm(Sj))]))[0])
                dist_va[i, j] = a*d1 + (1.0-a)*d2
        lab_va = np.argmin(dist_va, axis=1)
        return lab_tr, lab_va, dist_va
    raise ValueError("invalid type")

def external_margin(dist_va: np.ndarray) -> np.ndarray:
    if dist_va is None or dist_va.ndim != 2 or dist_va.shape[1] < 2:
        return np.zeros((dist_va.shape[0] if dist_va is not None else 0,), dtype=float)
    part = np.partition(dist_va, kth=1, axis=1)
    d1 = part[:, 0]
    d2 = part[:, 1]
    denom = np.maximum(d1, d2)
    denom[denom <= 0] = 1.0
    return (d2 - d1) / denom

def _sample_idx(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=int)
    rng = np.random.RandomState(seed)
    return rng.choice(n, size=cap, replace=False)

def _silhouette_euclid_sampled(X: np.ndarray, labels: np.ndarray, cap: int, seed: int) -> float:
    if X.shape[0] < 2 or np.unique(labels).size < 2:
        return 0.0
    idx = _sample_idx(X.shape[0], cap, seed)
    D = euclid_pairwise(X[idx])
    return float(silhouette_from_dist(D, labels[idx]))

def silhouette_val(model_type: str, train_X, val_X, labels_val, alpha: Optional[float]) -> float:
    if labels_val.size < 2 or np.unique(labels_val).size < 2:
        return 0.0
    if model_type == TYPE_NO_SPD_LE or model_type == TYPE_HYBRID_NO_SPD:
        Xtr, Xva, _ = _to_features_nonspd(train_X, val_X)
        return _silhouette_euclid_sampled(Xva, labels_val, _MAX_VAL_SIL, seed=0)
    if model_type == TYPE_SPD_LE:
        Ftr, Fva, _ = _to_features_spd_le(train_X, val_X)
        return _silhouette_euclid_sampled(Fva, labels_val, _MAX_VAL_SIL, seed=0)
    if model_type == TYPE_SPD_AIRM:
        idx = _sample_idx(val_X.shape[0], _MAX_VAL_SIL, seed=0)
        D = pairwise_airm(val_X[idx])
        return float(silhouette_from_dist(D, labels_val[idx]))
    if model_type == TYPE_HYBRID_SPD:
        a = 0.5 if alpha is None else float(alpha)
        idx = _sample_idx(val_X.shape[0], _MAX_VAL_SIL, seed=0)
        D = hybrid_pairwise_spd(val_X[idx], a)
        return float(silhouette_from_dist(D, labels_val[idx]))
    return 0.0

def select_internal(model_type: str, train_X, K_list, alpha_list, seed: int):
    best = None
    if model_type in (TYPE_NO_SPD_LE, TYPE_HYBRID_NO_SPD):
        Xtr, _, _ = _to_features_nonspd(train_X, train_X)
        for K in K_list:
            lab, _ = gmm_euclidean(Xtr, K, seed=seed)
            s = _silhouette_euclid_sampled(Xtr, lab, _MAX_TRAIN_SIL, seed=seed)
            cand = (-s, K, None)
            if best is None or cand < best:
                best = cand
        return best[1], None
    if model_type == TYPE_SPD_LE:
        Ftr, _, _ = _to_features_spd_le(train_X, train_X)
        for K in K_list:
            lab, _ = gmm_euclidean(Ftr, K, seed=seed)
            s = _silhouette_euclid_sampled(Ftr, lab, _MAX_TRAIN_SIL, seed=seed)
            cand = (-s, K, None)
            if best is None or cand < best:
                best = cand
        return best[1], None
    if model_type == TYPE_SPD_AIRM:
        Dtr = pairwise_airm(train_X)
        for K in K_list:
            lab, _ = _kmedoids(Dtr, K, seed=seed)
            s = silhouette_from_dist(Dtr, lab)
            cand = (-s, K, None)
            if best is None or cand < best:
                best = cand
        return best[1], None
    if model_type == TYPE_HYBRID_SPD:
        for K in K_list:
            for a in alpha_list:
                Dtr = hybrid_pairwise_spd(train_X, float(a))
                lab, _ = _kmedoids(Dtr, K, seed=seed)
                s = silhouette_from_dist(Dtr, lab)
                cand = (-s, K, a)
                if best is None or cand < best:
                    best = cand
        return best[1], best[2]
    raise ValueError("invalid type")
