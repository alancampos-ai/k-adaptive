import numpy as np 
from gmm import GMM, Point

def segmentation(dti, n_claster, mask=None, metric_type='riemannian',
                 expoent=1.0, index_centers=None, max_iterations=1000, dim_point=3):
    if mask is None:
        mask = np.ones(dti.shape[:3], dtype=np.bool_)
    else:
        mask = mask.astype(np.bool_)

    total = int(np.prod(dti.shape[:3]))
    mask_flat = mask.reshape(-1)
    dti_flat = dti.reshape(total, 3, 3)

    points = []
    for i, M in enumerate(dti_flat):
        if mask_flat[i] and np.any(M):
            points.append(Point(M, i))

    gmm = GMM(n_claster=n_claster, metric_type=metric_type,
                    index_centers=index_centers, total_points=len(points),
                    max_iterations=max_iterations, dim_point=dim_point)
    classes = gmm.fit(points, expoent=expoent)  

    y = np.zeros(total, dtype=np.int32)
    if classes:
        idx = np.fromiter(sorted(classes.keys()), dtype=np.int64)
        lbl = np.fromiter((classes[i] for i in idx), dtype=np.int32)
        y[idx] = lbl
    return y.reshape(mask.shape)
