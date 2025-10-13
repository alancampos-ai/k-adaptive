import numpy as np
from kmeans import KMeans, Point
def segmentation(dti, n_claster, mask=None, metric_type='riemannian', expoent=1, index_centers=None, max_iterations=1000, dim_point=3):
    if None in mask:
        mask = np.ones(dti.shape[0:3], dtype=np.bool8)
    images = np.zeros(mask.shape)
    print('Initial segmentation process ...')
    points = []
    total_points = np.prod(dti.shape[0:3])
    mask_seg = np.reshape(mask, (total_points,))
    DTI = np.reshape(dti, (total_points, 3, 3))
    imagem = np.zeros((total_points,))
    for i, value in enumerate(DTI):
        if mask_seg[i]:
            if value.any():
                point = Point(value, i)
                points.append(point)
    total_points_mask = len(points)
    print('Total points in the mask is {}'.format(total_points_mask))
    kmeans = KMeans(n_claster=n_claster, metric_type=metric_type, index_centers=index_centers, total_points=total_points_mask, max_iterations=max_iterations, dim_point=dim_point)
    classes = kmeans.fit(points, expoent=expoent)
    imagem_idx = np.array([i for i in sorted(classes)])
    imagem_ = np.array([classes[i] for i in sorted(classes)])
    imagem[imagem_idx] = imagem_
    images = np.reshape(imagem, (images.shape[0], images.shape[1], images.shape[2]))
    return images
