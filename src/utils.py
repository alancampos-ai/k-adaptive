from pymanopt.tools.multi import multilog, multiexp, multiprod, multitransp
from pymanopt.core.problem import Problem
from pymanopt.solvers import ConjugateGradient, SteepestDescent
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, norm, logm, inv, eig
from dipy.data import get_sphere
from dipy.viz import window, actor
from dipy.reconst.dti import fractional_anisotropy, color_fa
from pymanopt.solvers.linesearch import LineSearchAdaptive
class FilterDti:
    def __init__(self, manifold, tensorfield, tensormask=None, make_grad=True, typefilter='avg', s=3, eps=0.0001):
        self.manifold = manifold
        if tensorfield.ndim == 4:
            x, y, d1, d2 = tensorfield.shape
            self.tensorfield = tensorfield.reshape((x, y, 1, d1, d2))
        else:
            self.tensorfield = tensorfield
        if np.any(tensormask) == None:
            self.tensormask = np.ones_like(self.tensorfield[..., 0, 0], dtype=np.bool8)
        else:
            self.tensormask = tensormask
        self.make_grad = make_grad
        if s % 2 == 0:
            self.s = s + 1
        else:
            self.s = s
        self.eps = eps
        self.d = (self.s - 1) // 2
        self.typefilter = typefilter
        self.grad_espacial = np.zeros_like(self.tensormask)
    def get_grad_espacial(self):
        return self.grad_espacial
    def set_grad_espacial(self, grad):
        self.grad_espacial = grad
    def gradient_espatial(self):
        x, y, z, d1, d2 = self.tensorfield.shape
        tmpIM = np.zeros((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d, d1, d2))
        tmpIM[self.d:-self.d, self.d:-self.d, self.d:-self.d, ...] = self.tensorfield.copy()
        tmpmask = np.ones((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d), dtype=np.bool8)
        tmpmask[self.d:-self.d, self.d:-self.d, self.d:-self.d] = self.tensormask
        dims = tmpIM.shape
        gb = np.zeros((dims[0], dims[1], dims[2]))
        for x in range(self.d, dims[0] - self.d):
            for y in range(self.d, dims[1] - self.d):
                for z in range(self.d, dims[2] - self.d):
                    if tmpmask[x, y, z]:
                        S1 = tmpIM[x, y, z, ...].copy()
                        if S1.any():
                            Sx1 = tmpIM[x + 1, y, z, ...].copy()
                            Sx2 = tmpIM[x - 1, y, z, ...].copy()
                            Sy1 = tmpIM[x, y + 1, z, ...].copy()
                            Sy2 = tmpIM[x, y - 1, z, ...].copy()
                            Sz1 = tmpIM[x, y, z + 1, ...].copy()
                            Sz2 = tmpIM[x, y, z - 1, ...].copy()
                            l, v = eig(S1)
                            l = np.abs(l) + self.eps
                            S1 = np.real(v @ (np.diag(l) @ inv(v)))
                            if Sx1.any() and tmpmask[x + 1, y, z] == True:
                                l, v = eig(Sx1)
                                l = np.abs(l) + self.eps
                                Sx1 = np.real(v @ (np.diag(l) @ inv(v)))
                            else:
                                Sx1 = S1
                            if Sx2.any() and tmpmask[x - 1, y, z] == True:
                                l, v = eig(Sx2)
                                l = np.abs(l) + self.eps
                                Sx2 = np.real(v @ (np.diag(l) @ inv(v)))
                            else:
                                Sx2 = S1
                            if Sy1.any() and tmpmask[x, y + 1, z] == True:
                                l, v = eig(Sy1)
                                l = np.abs(l) + self.eps
                                Sy1 = np.real(v @ (np.diag(l) @ inv(v)))
                            else:
                                Sy1 = S1
                            if Sy2.any() and tmpmask[x, y - 1, z] == True:
                                l, v = eig(Sy2)
                                l = np.abs(l) + self.eps
                                Sy2 = np.real(v @ (np.diag(l) @ inv(v)))
                            else:
                                Sy2 = S1
                            if Sz1.any() and tmpmask[x, y, z + 1] == True:
                                l, v = eig(Sz1)
                                l = np.abs(l) + self.eps
                                Sz1 = np.real(v @ (np.diag(l) @ inv(v)))
                            else:
                                Sz1 = S1
                            if Sz2.any() and tmpmask[x + 1, y, z - 1] == True:
                                l, v = eig(Sz2)
                                l = np.abs(l) + self.eps
                                Sz2 = np.real(v @ (np.diag(l) @ inv(v)))
                            else:
                                Sz2 = S1
                            dx1 = S1 * logm(solve(Sx1, S1))
                            dx1 = 0.5 * (dx1 + dx1.T)
                            dx2 = S1 * logm(solve(Sx2, S1))
                            dx2 = 0.5 * (dx2 + dx2.T)
                            dy1 = S1 * logm(solve(Sy1, S1))
                            dy1 = 0.5 * (dy1 + dy1.T)
                            dy2 = S1 * logm(solve(Sy2, S1))
                            dy2 = 0.5 * (dy2 + dy2.T)
                            dz2 = S1 * logm(solve(Sz2, S1))
                            dz2 = 0.5 * (dz2 + dz2.T)
                            dz1 = S1 * logm(solve(Sz1, S1))
                            dz1 = 0.5 * (dz1 + dz1.T)
                            dxx = 0.5 * (dx1 - dx2)
                            dyy = 0.5 * (dy1 - dy2)
                            dzz = 0.5 * (dz1 - dz2)
                            norm2 = 0.5 * (np.trace(solve(S1, dxx) * solve(S1, dxx)) + np.trace(solve(S1, dyy) * solve(S1, dyy)) + np.trace(solve(S1, dzz) * solve(S1, dzz)))
                            gb[x, y, z] = norm2
                            print(x, y, z)
        quantil = np.percentile(gb[self.d:-self.d, self.d:-self.d, self.d:-self.d], 95)
        gb[self.d:-self.d, self.d:-self.d, self.d:-self.d] = np.where(gb[self.d:-self.d, self.d:-self.d, self.d:-self.d] > quantil, quantil, gb[self.d:-self.d, self.d:-self.d, self.d:-self.d])
        bgmin = gb[self.d:-self.d, self.d:-self.d, self.d:-self.d].min()
        bgmax = gb[self.d:-self.d, self.d:-self.d, self.d:-self.d].max()
        print('pmin_max: ({}, {})'.format(bgmin, bgmax))
        p = (gb[self.d:-self.d, self.d:-self.d, self.d:-self.d] - bgmin) / (bgmax - bgmin)
        return p
    def filtering_dti(self):
        x, y, z, d1, d2 = self.tensorfield.shape
        tmpIM = np.zeros((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d, d1, d2))
        tmpIM[self.d:-self.d, self.d:-self.d, self.d:-self.d, ...] = self.tensorfield.copy()
        tmpmask = np.ones((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d), dtype=np.bool8)
        tmpmask[self.d:-self.d, self.d:-self.d, self.d:-self.d] = self.tensormask.copy()
        d1 = np.zeros((self.s, self.s, self.s))
        filtered_arf = tmpIM.copy()
        filtered_avg = tmpIM.copy()
        filtered_med = tmpIM.copy()
        dims = tmpIM.shape
        if self.typefilter == 'arf' and self.make_grad == True:
            bg = np.zeros((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d))
            self.set_grad_espacial(self, self.gradient_espatial())
            bg[self.d:-self.d, self.d:-self.d, self.d:-self.d] = self.get_grad_espacial()
        elif self.typefilter == 'arf' and self.make_grad == False:
            bg = np.zeros((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d))
            bg[self.d:-self.d, self.d:-self.d, self.d:-self.d] = self.get_grad_espacial()
        for z in range(self.d, dims[2] - self.d):
            for x in range(self.d, dims[0] - self.d):
                for y in range(self.d, dims[1] - self.d):
                    if tmpmask[x, y, z]:
                        neighbours = tmpIM[x - self.d:x + self.d + 1, y - self.d:y + self.d + 1, z - self.d:z + self.d + 1, ...]
                        S1 = neighbours[self.d, self.d, self.d, ...]
                        if S1.any():
                            l, v = eig(S1)
                            l = np.abs(l) + self.eps
                            S1 = np.real(v @ (np.diag(l) @ inv(v)))
                            neighbours[self.d, self.d, self.d, ...] = S1.copy()
                            peso = []
                            count = 0
                            for j in range(self.s):
                                for k in range(self.s):
                                    for m in range(self.s):
                                        S2 = neighbours[j, k, m, ...].copy()
                                        if S2.any() and ~(j == self.d and k == self.d and (m == self.d)):
                                            l, v = eig(S2)
                                            l = np.abs(l) + self.eps
                                            S2 = np.real(v @ (np.diag(l) @ inv(v)))
                                            neighbours[j, k, m, ...] = S2.copy()
                                        else:
                                            neighbours[j, k, m, ...] = S1.copy()
                                        if ~S2.any():
                                            count += 1
                                        peso.append(1.0 / self.s ** 3)
                            print(f'count: {count}')
                            if count > self.s ** 2:
                                filtered_arf[x, y, z, ...] = S1.copy()
                                filtered_avg[x, y, z, ...] = S1.copy()
                                filtered_med[x, y, z, ...] = S1.copy()
                            else:
                                A = neighbours.reshape((neighbours.shape[0] * neighbours.shape[1] * neighbours.shape[2], neighbours.shape[3], neighbours.shape[4]))
                                weight = np.array([i * np.ones((neighbours.shape[3], neighbours.shape[4])) for i in peso])
                                p = 2.0 - bg[x, y, z]
                                filtered_arf[x, y, z, ...] = AvgRiemann(self.manifold, A, p, weight)
                                filtered_avg[x, y, z, ...] = AvgRiemann(self.manifold, A, 2.0, weight)
                                filtered_med[x, y, z, ...] = AvgRiemann(self.manifold, A, 1.0, weight)
                            print(x, y, z)
        return (filtered_arf[self.d:-self.d, self.d:-self.d, self.d:-self.d], filtered_avg[self.d:-self.d, self.d:-self.d, self.d:-self.d], filtered_med[self.d:-self.d, self.d:-self.d, self.d:-self.d])
class FilterDtiNovo:
    def __init__(self, manifold, tensorfield, tensormask=None, typefilter='avg', s=3):
        self.manifold = manifold
        if tensorfield.ndim == 4:
            x, y, d1, d2 = tensorfield.shape
            self.tensorfield = tensorfield.reshape((x, y, 1, d1, d2))
        else:
            self.tensorfield = tensorfield
        if np.any(tensormask) == None:
            self.tensormask = np.ones_like(self.tensorfield[..., 0, 0], dtype=np.bool8)
        else:
            self.tensormask = tensormask
        if s % 2 == 0:
            self.s = s + 1
        else:
            self.s = s
        self.d = (self.s - 1) // 2
        self.typefilter = typefilter
    def gradient_espatial(self):
        x, y, z, d1, d2 = self.tensorfield.shape
        tmpIM = np.zeros((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d, d1, d2))
        tmpIM[self.d:-self.d, self.d:-self.d, self.d:-self.d, ...] = self.tensorfield.copy()
        tmpmask = np.ones((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d), dtype=np.bool8)
        tmpmask[self.d:-self.d, self.d:-self.d, self.d:-self.d] = self.tensormask
        dims = tmpIM.shape
        gb = np.zeros((dims[0], dims[1], dims[2]))
        for x in range(self.d, dims[0] - self.d):
            for y in range(self.d, dims[1] - self.d):
                for z in range(self.d, dims[2] - self.d):
                    if tmpmask[x, y, z]:
                        S1 = tmpIM[x, y, z, ...].copy()
                        Sx1 = tmpIM[x + 1, y, z, ...].copy()
                        Sx2 = tmpIM[x - 1, y, z, ...].copy()
                        Sy1 = tmpIM[x, y + 1, z, ...].copy()
                        Sy2 = tmpIM[x, y - 1, z, ...].copy()
                        Sz1 = tmpIM[x, y, z + 1, ...].copy()
                        Sz2 = tmpIM[x, y, z - 1, ...].copy()
                        if np.any(S1):
                            if ~np.any(Sx1):
                                Sx1 = S1
                            if ~np.any(Sx1):
                                Sx2 = S1
                            if ~np.any(Sx1):
                                Sy1 = S1
                            if ~np.any(Sx1):
                                Sy2 = S1
                            if ~np.any(Sx1):
                                Sz1 = S1
                            if ~np.any(Sx1):
                                Sz2 = S1
                            dx1 = S1 * logm(solve(Sx1, S1))
                            dx1 = 0.5 * (dx1 + dx1.T)
                            dx2 = S1 * logm(solve(Sx2, S1))
                            dx2 = 0.5 * (dx2 + dx2.T)
                            dy1 = S1 * logm(solve(Sy1, S1))
                            dy1 = 0.5 * (dy1 + dy1.T)
                            dy2 = S1 * logm(solve(Sy2, S1))
                            dy2 = 0.5 * (dy2 + dy2.T)
                            dz2 = S1 * logm(solve(Sz2, S1))
                            dz2 = 0.5 * (dz2 + dz2.T)
                            dz1 = S1 * logm(solve(Sz1, S1))
                            dz1 = 0.5 * (dz1 + dz1.T)
                            dxx = 0.5 * (dx1 - dx2)
                            dyy = 0.5 * (dy1 - dy2)
                            dzz = 0.5 * (dz1 - dz2)
                            norm2 = 0.5 * (np.trace(solve(S1, dxx) * solve(S1, dxx)) + np.trace(solve(S1, dyy) * solve(S1, dyy)) + np.trace(solve(S1, dzz) * solve(S1, dzz)))
                            gb[x, y, z] = norm2
        quantil = np.quantile(gb[self.d:-self.d, self.d:-self.d, self.d:-self.d], 95)
        np.where(gb[self.d:-self.d, self.d:-self.d, self.d:-self.d] > quantil, quantil, gb[self.d:-self.d, self.d:-self.d, self.d:-self.d])
        bgmin = gb[self.d:-self.d, self.d:-self.d, self.d:-self.d].min()
        bgmax = gb[self.d:-self.d, self.d:-self.d, self.d:-self.d].max()
        print('pmin_max: ({}, {})'.format(bgmin, bgmax))
        p = (gb[self.d:-self.d, self.d:-self.d, self.d:-self.d] - bgmin) / (bgmax - bgmin)
        axial_middle = p.shape[2] // 2
        plt.figure('Showing the edge DTI')
        plt.subplot(1, 1, 1).set_axis_off()
        plt.imshow(p[:, :, axial_middle].T, cmap='viridis', origin='lower', interpolation=None)
        plt.show()
        return p
    def filtering_dti(self):
        x, y, z, d1, d2 = self.tensorfield.shape
        tmpIM = np.zeros((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d, d1, d2))
        tmpIM[self.d:-self.d, self.d:-self.d, self.d:-self.d, ...] = self.tensorfield.copy()
        tmpmask = np.ones((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d), dtype=np.bool8)
        tmpmask[self.d:-self.d, self.d:-self.d, self.d:-self.d] = self.tensormask.copy()
        d1 = np.zeros((self.s, self.s, self.s))
        filtered = tmpIM.copy()
        dims = tmpIM.shape
        if self.typefilter == 'arf':
            bg = np.zeros((x + 2 * self.d, y + 2 * self.d, z + 2 * self.d))
            bg[self.d:-self.d, self.d:-self.d, self.d:-self.d] = self.gradient_espatial()
        for z in range(self.d, dims[2] - self.d):
            for x in range(self.d, dims[0] - self.d):
                for y in range(self.d, dims[1] - self.d):
                    if tmpmask[x, y, z]:
                        neighbours = tmpIM[x - self.d:x + self.d + 1, y - self.d:y + self.d + 1, z - self.d:z + self.d + 1, ...]
                        S1 = neighbours[self.d, self.d, self.d, ...]
                        if np.any(S1):
                            neighbours[self.d, self.d, self.d, ...] = S1.copy()
                            for j in range(self.s):
                                for k in range(self.s):
                                    for m in range(self.s):
                                        S2 = neighbours[j, k, m, ...].copy()
                                        if np.any(S2) and ~(j == self.d and k == self.d and (m == self.d)):
                                            neighbours[j, k, m, ...] = S2.copy()
                                            dist1 = self.manifold.dist(S1, S2)
                                            d1[j, k, m] = np.exp(-0.5 * dist1 ** 2)
                            A = neighbours.reshape((neighbours.shape[0] * neighbours.shape[1] * neighbours.shape[2], neighbours.shape[3], neighbours.shape[4]))
                            peso = np.reshape(d1, d1.shape[0] * d1.shape[1] * d1.shape[2]) / np.sum(d1)
                            weight = np.array([i * np.ones((neighbours.shape[3], neighbours.shape[4])) for i in peso])
                            if self.typefilter == 'arf':
                                p = 2 - bg[x, y, z]
                            elif self.typefilter == 'avg':
                                p = 2
                            elif self.typefilter == 'med':
                                p = 1
                            print('exponent: ', p)
                            filtered[x, y, z, ...] = AvgRiemann(self.manifold, A, p, weight)
                            print(x, y, z)
        return filtered[self.d:-self.d, self.d:-self.d, self.d:-self.d]
def logeuc_distance(T1, T2):
    if T1.ndim == 2:
        if T1.any() and T2.any():
            d = norm(multilog(T1, pos_def=True) - multilog(T2, pos_def=True), ord='fro')
    elif T1.ndim == 3:
        if T1.any() and T2.any():
            d = norm(multilog(T1, pos_def=True) - multilog(T2, pos_def=True), ord='fro', axis=(1, 2))
    return d
def avg_log_euclidean(T, W):
    mean = multiexp(np.sum(multilog(T, pos_def=True) * W, axis=0), sym=True)
    return mean
def euc_distance(T1, T2):
    if T1.ndim == 2:
        if T1.any() and T2.any():
            d = norm(T1 - T2, ord='fro')
    elif T1.ndim == 3:
        if T1.any() and T2.any():
            d = norm(T1 - T2, ord='fro', axis=(1, 2))
    return d
def avg_euclidean(T, W):
    return np.sum(T * W, axis=0)
def riemannian_distance(T1, T2):
    if T1.any() and T2.any():
        c = np.linalg.cholesky(T1)
        c_inv = np.linalg.inv(c)
        logm = multilog(multiprod(multiprod(c_inv, T2), multitransp(c_inv)), pos_def=True)
        d = np.linalg.norm(logm, ord='fro', axis=(1, 2))
    return d
def AvgRiemann(manifold, vizinhaca, pp, peso):
    def distance(x):
        xx = np.tile(x, (vizinhaca.shape[0], 1, 1))
        c = np.linalg.cholesky(xx)
        c_inv = np.linalg.inv(c)
        logm = multilog(multiprod(multiprod(c_inv, vizinhaca), multitransp(c_inv)), pos_def=True)
        d = np.linalg.norm(logm, ord='fro', axis=(1, 2))
        return d
    def cost(x):
        d = distance(x)
        N = np.array([(nn + 1e-06) ** pp * np.ones((vizinhaca.shape[1], vizinhaca.shape[2])) for nn in d])
        f = np.sum((peso * N)[:, 0, 0]) / pp
        return f
    def grad(x):
        xx = np.tile(x, (vizinhaca.shape[0], 1, 1))
        d = distance(x)
        N = np.array([(nn + 1e-06) ** (pp - 2) * np.ones((vizinhaca.shape[1], vizinhaca.shape[2])) for nn in d])
        g = -np.sum(peso * manifold.log(xx, vizinhaca) * N, axis=0)
        return g
    problem = Problem(manifold=manifold, cost=cost, grad=grad, hess=None)
    x = manifold.rand()
    solver = ConjugateGradient(linesearch=LineSearchAdaptive(), maxiter=50)
    Xopt = solver.solve(problem, x=x)
    return Xopt
def RiemannianCentroid(manifold, vizinhaca, pp, peso):
    def distance(x):
        xx = np.tile(x, (vizinhaca.shape[0], 1, 1))
        c = np.linalg.cholesky(xx)
        c_inv = np.linalg.inv(c)
        logm = multilog(multiprod(multiprod(c_inv, vizinhaca), multitransp(c_inv)), pos_def=True)
        d = np.linalg.norm(logm, axis=(1, 2))
        return d
    def cost(x):
        d = distance(x)
        N = np.array([(nn + 1e-06) ** pp * np.ones((vizinhaca.shape[1], vizinhaca.shape[2])) for nn in d])
        f = np.sum((peso * N)[:, 0, 0]) / pp
        return f
    def grad(x):
        xx = np.tile(x, (vizinhaca.shape[0], 1, 1))
        d = distance(x)
        N = np.array([(nn + 1e-06) ** (pp - 2) * np.ones((vizinhaca.shape[1], vizinhaca.shape[2])) for nn in d])
        g = np.sum(peso * manifold.log(xx, vizinhaca) * N, axis=0)
        return g
    x = 0.001 * manifold.rand()
    fx0 = cost(x)
    k = 0
    t = 1.0
    while True:
        x0 = x
        g = grad(x0)
        x = manifold.exp(x0, t * g)
        fx1 = cost(x)
        j = 0
        while fx0 < fx1:
            t *= 0.5
            x = manifold.exp(x0, t * g)
            fx1 = cost(x)
            if j > 10:
                break
            j += 1
        ng = np.linalg.norm(g)
        if (k > 100) | (ng < 1e-15):
            print(f'norma Grad: {ng}   iteration (main): {k}')
            break
        k += 1
        t = 1.0
        fx0 = fx1
    return x
def EuclideanCentroid(manifold, vizinhaca, pp, peso):
    def distance(x):
        xx = np.tile(x, (vizinhaca.shape[0], 1, 1))
        d = norm(xx - vizinhaca, ord='fro', axis=(1, 2))
        return d
    def cost(x):
        d = distance(x)
        N = np.array([(nn + 1e-06) ** pp * np.ones((vizinhaca.shape[1], vizinhaca.shape[2])) for nn in d])
        f = np.sum((peso * N)[:, 0, 0]) / pp
        return f
    def grad(x):
        xx = np.tile(x, (vizinhaca.shape[0], 1, 1))
        d = distance(x)
        N = np.array([(nn + 1e-06) ** (pp - 2) * np.ones((vizinhaca.shape[1], vizinhaca.shape[2])) for nn in d])
        g = -np.sum(peso * manifold.log(xx, vizinhaca) * N, axis=0)
        return g
    problem = Problem(manifold=manifold, cost=cost, grad=grad, hess=None)
    x = manifold.rand()
    solver = SteepestDescent()
    Xopt = solver.solve(problem, x=x)
    return Xopt
def plot_DTI(evecs, evals, dims=(300, 300), affine=None, mask=None, scale=0.7, interactive=False, n_frames=100, out_path='out.png'):
    FA = fractional_anisotropy(evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, evecs)
    sphere = get_sphere('repulsion724')
    ren = window.Scene()
    '\n    We can color the ellipsoids using the ``color_fa`` values that we calculated\n    above. In this example we additionally normalize the values to increase the\n    contrast.\n    '
    RGB /= RGB.max()
    ren.add(actor.tensor_slicer(evals, evecs, mask=mask, affine=affine, scalar_colors=RGB, sphere=sphere, scale=scale))
    if not out_path == None:
        print('Saving illustration as tensor_ellipsoids ...')
        window.record(ren, az_ang=10, n_frames=n_frames, cam_view=None, path_numbering=True, out_path=out_path, screen_clip=True, size=dims)
    if interactive:
        window.show(ren)
def plot_DTI_test(evecs, evals, dims=(300, 300), affine=None, mask=None, scale=0.7, interactive=True, n_frames=100, out_path='out.png'):
    FA = fractional_anisotropy(evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, evecs)
    sphere = get_sphere('repulsion724')
    ren = window.Scene()
    '\n    We can color the ellipsoids using the ``color_fa`` values that we calculated\n    above. In this example we additionally normalize the values to increase the\n    contrast.\n    '
    RGB /= RGB.max()
    ren.add(actor.tensor_slicer(evals, evecs, mask=mask, affine=affine, scalar_colors=RGB, sphere=sphere, scale=scale))
    if not out_path == None:
        print('Saving illustration as tensor_ellipsoids ...')
        window.record(ren, az_ang=45, n_frames=n_frames, cam_view=None, path_numbering=True, out_path=out_path, screen_clip=True, size=dims)
    if interactive:
        window.show(ren)
