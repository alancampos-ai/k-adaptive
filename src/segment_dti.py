import numpy as np
from .kmeans import KMeans, Point

def segmentation(dti, n_claster, mask=None, metric_type='airm', expoent=1,
                 index_centers=None, max_iterations=1000, dim_point=3):
    if mask is None:
        mask = np.ones(dti.shape[0:3], dtype=np.bool_)
    images = np.zeros(mask.shape, dtype=np.int16)
    points = []
    total_points = np.prod(dti.shape[0:3])
    mask_seg = np.reshape(mask, (total_points,))
    DTI = np.reshape(dti, (total_points, 3, 3))
    imagem = np.zeros((total_points,), dtype=np.int16)
    for i, value in enumerate(DTI):
        if mask_seg[i]:
            if value.any():
                point = Point(value, i)
                points.append(point)
    total_points_mask = len(points)
    kmeans = KMeans(n_claster=n_claster, metric_type=metric_type,
                    index_centers=index_centers, total_points=total_points_mask,
                    max_iterations=max_iterations, dim_point=dim_point)
    classes = kmeans.fit(points, expoent=expoent)
    imagem_idx = np.array([i for i in sorted(classes)])
    imagem_ = np.array([classes[i] for i in sorted(classes)], dtype=np.int16)
    imagem[imagem_idx] = imagem_
    images = np.reshape(imagem, (images.shape[0], images.shape[1], images.shape[2]))
    return images
