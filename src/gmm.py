import numpy as np

class Point:
    def __init__(self, value, id_point, name=""):
        self.id_point = id_point
        self.id_cluster = -1
        self.value = value
        self.name = name
        self.weight = {}
    def getID(self): return self.id_point
    def getCluster(self): return self.id_cluster
    def setCluster(self, cid): self.id_cluster = int(cid)
    def getValue(self): return self.value
    def getName(self): return self.name
    def setWeight(self, w):
        if isinstance(w, dict):
            self.weight = {int(k): float(v) for k, v in w.items()}
        else:
            self.weight = {i: float(w[i]) for i in range(len(w))}
    def Getweights(self): return dict(self.weight)

def spd_le(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    L = np.diag(np.log(w))
    return V @ L @ V.T

def vec6(M):
    return np.array([M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]], dtype=float)

def _features(M, metric_type, alpha):
    if metric_type in ("euclidean", "no_spd"):
        return vec6(M)
    if metric_type in ("logeuclidean", "spd", "riemannian", "airm"):
        return vec6(spd_le(M))
    if metric_type.startswith("hybrid"):
        a = 0.5 if alpha is None else float(alpha)
        v1 = vec6(spd_le(M))
        v2 = vec6(M)
        return np.concatenate([np.sqrt(a)*v1, np.sqrt(max(0.0,1.0-a))*v2])
    return vec6(M)

class GMM:
    def __init__(self, n_claster, total_points, metric_type='riemannian',
                 index_centers=None, max_iterations=1000, dim_point=3, tol=1e-3):
        self.n_claster = int(n_claster)
        self.total_points = int(total_points)
        self.max_iterations = int(max_iterations)
        self.index_centers = index_centers
        self.metric_type = metric_type
        self.dim = int(dim_point)
        self.tol = float(tol)
        self.classes = {}
        self.n_iter_ = 0
        self.converged_ = False

    def fit(self, points, expoent=1.0):
        X = []
        ids = []
        for p in points:
            M = np.array(p.getValue(), dtype=float)
            X.append(_features(M, self.metric_type, expoent))
            ids.append(p.getID())
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        mu = X.mean(axis=0)
        s0 = X.std(axis=0, ddof=0)
        s0[s0 == 0] = 1.0
        X = (X - mu) / s0
        K = self.n_claster
        rng = np.random
        if self.index_centers is not None and len(self.index_centers) >= K:
            means = X[np.array(self.index_centers[:K], dtype=int)]
        else:
            means = X[rng.choice(n, size=K, replace=False)]
        pi = np.full(K, 1.0 / K, dtype=float)
        Sigma = np.stack([np.cov(X.T) + 1e-6*np.eye(d) for _ in range(K)], axis=0)
        prev_ll = -np.inf
        for it in range(1, self.max_iterations + 1):
            log_prob = np.empty((n, K), dtype=float)
            for k in range(K):
                S = Sigma[k] + 1e-6*np.eye(d)
                L = np.linalg.cholesky(S)
                dx = X - means[k]
                y = np.linalg.solve(L, dx.T)
                quad = np.sum(y*y, axis=0)
                log_det = 2.0*np.sum(np.log(np.diag(L)))
                log_prob[:, k] = np.log(pi[k] + 1e-300) - 0.5*(d*np.log(2.0*np.pi) + log_det + quad)
            m = np.max(log_prob, axis=1, keepdims=True)
            prob = np.exp(log_prob - m)
            denom = np.sum(prob, axis=1, keepdims=True) + 1e-300
            R = prob / denom
            ll = float(np.sum(m[:, 0] + np.log(denom[:, 0])))
            Nk = np.sum(R, axis=0) + 1e-12
            pi = Nk / n
            means = (R.T @ X) / Nk[:, None]
            for k in range(K):
                dx = X - means[k]
                wk = R[:, k][:, None]
                S = (dx * wk).T @ dx / Nk[k]
                Sigma[k] = S + 1e-6*np.eye(d)
            self.n_iter_ = it            
            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                print(f"Converged by tol={self.tol:.1e} (loglikelihood delta={ll - prev_ll:.3e}) at iteration {it}")
                break
            prev_ll = ll
            if it >= self.max_iterations:
                print(f"Break in iteration {it} (max_iterations)")
                break
        y = np.argmax(R, axis=1)
        self.classes = {}
        for i, pid in enumerate(ids):
            points[i].setCluster(int(y[i]))
            self.classes[pid] = int(y[i]) + 1
        return self.classes
