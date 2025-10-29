import numpy as np
from scipy.ndimage import binary_erosion

def make_border_interior(mask: np.ndarray, radius: int = 2):
    m = mask.astype(bool)
    er = binary_erosion(m, iterations=max(1, int(radius)), border_value=0)
    interior = m & er
    border = m & (~er)
    return border, interior
