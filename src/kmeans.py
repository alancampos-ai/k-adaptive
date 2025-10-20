import numpy as np
from pymanopt.manifolds import PositiveDefinite, Euclidean
from .utils import AIRMMean, airm_distance, NoSPDCentroid, no_spd_distance

class Point:
    def __init__(self, value, id_point, name=''):
        self.id_point = id_point
        self.id_cluster = -1
        self.value = value
        self.name = name
        self.weight = {}

    def getID(self):
        return self.id_point

    def Setweights(self, W):
        for i in range(len(W)):
            self.weight[i] = W[i]

    def Getweights(self):
        return self.weight

    def setCluster(self, id_cluster):
        self.id_cluster = id_cluster

    def getCluster(self):
        return self.id_cluster

    def getValue(self):
        return self.value

    def addValue(self, value):
        self.value = value

    def getName(self):
        return self.name


class Cluster:
    def __init__(self, id_cluster, point):
        self.id_cluster = id_cluster
        self.central_value = point.getValue()
        self.points = [point]
        self.mudou = False

    def addPoint(self, point):
        self.points.append(point)

    def removePoint(self, id_point):
        for p in self.points:
            if p.getID() == id_point:
                self.points.remove(p)
                return True
        return False

    def setMudou(self, mudou):
        self.mudou = mudou

    def getMudou(self):
        return self.mudou

    def getCentralValue(self):
        return self.central_value

    def setCentralValue(self, value):
        self.central_value = value

    def getPoint(self, index):
        return self.points[index]

    def getPoints(self):
        return self.points

    def getTotalPoints(self):
        return len(self.points)

    def getID(self):
        return self.id_cluster


class KMeans:
    def __init__(self, n_claster, total_points, metric_type='airm',
                 index_centers=None, max_iterations=1000, dim_point=3):
        self.n_claster = n_claster
        self.total_points = total_points
        self.max_iterations = max_iterations
        self.index_centers = index_centers
        self.clusters = []
        self.classes = {}
        self.metric_type = metric_type
        if self.metric_type == 'no_spd':
            self.manifold = Euclidean(dim_point, dim_point)
        else:
            self.manifold = PositiveDefinite(dim_point)
        self.dim = dim_point

    def fit(self, points, expoent=1):
        if self.n_claster > self.total_points:
            return
        prohibited_indexes = []
        if self.index_centers is None:
            for i in range(self.n_claster):
                while True:
                    index_point = np.random.randint(self.total_points)
                    if index_point not in prohibited_indexes:
                        prohibited_indexes.append(index_point)
                        points[index_point].setCluster(i)
                        self.classes[points[index_point].getID()] = i + 1
                        self.clusters.append(Cluster(i, points[index_point]))
                        break
        else:
            for i in range(self.n_claster):
                index_point = self.index_centers[i]
                points[index_point].setCluster(i)
                self.classes[points[index_point].getID()] = i + 1
                self.clusters.append(Cluster(i, points[index_point]))

        iteration = 0
        while True:
            iteration += 1
            done = True
            count_mudou = 0
            for p in points:
                id_old_cluster = p.getCluster()
                id_nearest_center, probs = self._getIDNearestCenter(p)
                p.Setweights(probs)
                if id_old_cluster != id_nearest_center:
                    if id_old_cluster != -1:
                        self.clusters[id_old_cluster].removePoint(p.getID())
                        self.clusters[id_old_cluster].setMudou(True)
                    p.setCluster(id_nearest_center)
                    self.classes[p.getID()] = id_nearest_center + 1
                    self.clusters[id_nearest_center].addPoint(p)
                    self.clusters[id_nearest_center].setMudou(True)
                    count_mudou += 1
                    done = False
            if count_mudou == 0:
                if self.clusters:
                    self.clusters[0].setMudou(False)
            if done or iteration >= self.max_iterations:
                break
            for cl in self.clusters:
                samples = []
                pesos = []
                if (cl.getTotalPoints() > 0) and cl.getMudou():
                    for p in cl.getPoints():
                        samples.append(p.getValue())
                        pesos.append(1.0 / cl.getTotalPoints() * np.ones((self.dim, self.dim)))
                    samples = np.array(samples)
                    pesos = np.array(pesos)
                    if self.metric_type == 'airm':
                        mean_sample = AIRMMean(self.manifold, samples, expoent, pesos)
                        cl.setCentralValue(mean_sample)
                    elif self.metric_type == 'no_spd':
                        mean_sample = NoSPDCentroid(self.manifold, samples, expoent, pesos)
                        cl.setCentralValue(mean_sample)
                    else:
                        return
        return self.classes

    def _getIDNearestCenter(self, point):
        id_cluster_center = 0
        dists = {}
        if self.metric_type == 'airm':
            min_dist = self.manifold.dist(self.clusters[0].getCentralValue(), point.getValue())
        elif self.metric_type == 'no_spd':
            min_dist = no_spd_distance(self.clusters[0].getCentralValue(), point.getValue())
        else:
            return
        exp_dist = 1.0 / (min_dist + 1e-06) ** 2
        s = exp_dist
        dists[0] = exp_dist
        for i in range(1, self.n_claster):
            if self.metric_type == 'airm':
                dist = self.manifold.dist(self.clusters[i].getCentralValue(), point.getValue())
            elif self.metric_type == 'no_spd':
                dist = no_spd_distance(self.clusters[i].getCentralValue(), point.getValue())
            else:
                return
            exp_dist = 1.0 / (dist + 1e-06) ** 2
            s += exp_dist
            dists[i] = exp_dist
            if dist < min_dist:
                min_dist = dist
                id_cluster_center = i
        for j in range(self.n_claster):
            dists[j] = dists[j] / s
        return (id_cluster_center, dists)

    def _getIDNearestCuster(self, values):
        dists = []
        for i in range(self.n_claster):
            centers = np.tile(self.clusters[i].getCentralValue(), (values.shape[0], 1, 1))
            if self.metric_type == 'airm':
                dist = airm_distance(centers, values)
                dists.append(dist)
            elif self.metric_type == 'no_spd':
                dist = no_spd_distance(centers, values)
                dists.append(dist)
            else:
                return
        dists = np.array(dists)
        id_cluster_center = np.argmin(dists, axis=0)
        return id_cluster_center
