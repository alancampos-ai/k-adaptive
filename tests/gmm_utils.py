import numpy as np 

class StandardScalerSimple:
    def __init__(self):
        self.mean_ = None; self.scale_ = None
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0); s = X.std(axis=0, ddof=0); s[s==0] = 1.0; self.scale_ = s; return self
    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)

def euclid_pairwise(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    H = np.diag(G).reshape(-1,1)
    D2 = H + H.T - 2*G
    D2[D2<0] = 0.0
    return np.sqrt(D2)

def _gmm_pp_init(X: np.ndarray, K: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    n = X.shape[0]
    idx = np.random.randint(0, n)
    centers = [X[idx]]
    for _ in range(1, K):
        D = np.min(((X[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=2), axis=1)
        s = D.sum()
        probs = D / s if s > 0 else np.ones_like(D)/len(D)
        idx = np.random.choice(n, p=probs)
        centers.append(X[idx])
    return np.array(centers)

def _assign_labels(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    D = ((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
    return D.argmin(axis=1)

def _update_centers(X: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    centers = []
    for k in range(K):
        mask = labels==k
        if np.any(mask):
            centers.append(X[mask].mean(axis=0))
        else:
            centers.append(X[np.random.randint(0, X.shape[0])])
    return np.array(centers)

def gmm_euclidean(X: np.ndarray, K: int, seed: int, max_iter: int = 100):
    centers = _gmm_pp_init(X, K, seed)
    for _ in range(max_iter):
        labels = _assign_labels(X, centers)
        new_centers = _update_centers(X, labels, K)
        if np.allclose(new_centers, centers): break
        centers = new_centers
    return labels, centers

def silhouette_from_dist(D: np.ndarray, labels: np.ndarray) -> float:
    n = D.shape[0]
    a = np.zeros(n); b = np.zeros(n)
    for i in range(n):
        own = labels==labels[i]
        other = labels!=labels[i]
        if own.sum() > 1:
            a[i] = D[i, own].sum()/(own.sum()-1)
        else:
            a[i] = 0.0
        if other.sum() > 0:
            bs = []
            for k in np.unique(labels):
                if k==labels[i]: continue
                mk = labels==k
                if mk.any():
                    bs.append(D[i, mk].mean())
            b[i] = np.min(bs) if bs else 0.0
        else:
            b[i] = 0.0
    den = np.maximum(a, b)
    s = np.zeros_like(den)
    mask = den > 0
    s[mask] = (b[mask] - a[mask]) / den[mask]
    return float(np.mean(s))
