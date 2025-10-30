import numpy as np

def infer_spd_dim_from_m(m: int) -> int:
    if m <= 0: return 0
    d = int((np.sqrt(8*m+1)-1)//2)
    while d*(d+1)//2 > m: d -= 1
    return d

def vec_to_spd(vec: np.ndarray, dim: int) -> np.ndarray:
    p = dim*(dim+1)//2
    if vec.shape[-1] < p: raise RuntimeError(f"not enough features for SPD dim={dim}")
    v = vec[..., :p]
    idx = np.triu_indices(dim)
    S = np.zeros(vec.shape[:-1] + (dim, dim))
    S[..., idx[0], idx[1]] = v
    S[..., idx[1], idx[0]] = v
    for i in range(S.shape[-1]):
        S[..., i, i] = np.abs(S[..., i, i]) + 1e-6
    return S

def spd_logm(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    w[w < 1e-12] = 1e-12
    logw = np.log(w)
    return (V * logw) @ V.T

def spd_invsqrtm(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    w[w < 1e-12] = 1e-12
    invsqw = 1.0 / np.sqrt(w)
    return (V * invsqw) @ V.T

def pairwise_logeuc(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    M = np.zeros((n, n))
    L = np.array([spd_logm(S) for S in X])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(L[i]-L[j], ord="fro")
            M[i,j] = M[j,i] = d
    return M

def pairwise_airm(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        Si = X[i]
        wS, VS = np.linalg.eigh(Si)
        wS[wS < 1e-12] = 1e-12
        Sih = (VS * (1.0/np.sqrt(wS))) @ VS.T
        for j in range(i+1, n):
            Sj = X[j]
            A = Sih @ Sj @ Sih
            w = np.linalg.eigvalsh(A)
            w[w < 1e-12] = 1e-12
            d = np.sqrt(np.sum(np.log(w)**2))
            M[i,j] = M[j,i] = d
    return M
