import os 
import os
import numpy as np
import nibabel as nib
from scipy.linalg import lstsq
from dipy.data import get_fnames
from dipy.viz import window, actor
from dipy.reconst.dti import fractional_anisotropy, color_fa
from dipy.data import get_sphere
def build_tensor_from_coefficients(D):
    tensor = np.zeros(D.shape[:-1] + (3, 3))
    tensor[..., 0, 0] = D[..., 0]
    tensor[..., 0, 1] = tensor[..., 1, 0] = D[..., 1]
    tensor[..., 0, 2] = tensor[..., 2, 0] = D[..., 2]
    tensor[..., 1, 1] = D[..., 3]
    tensor[..., 1, 2] = tensor[..., 2, 1] = D[..., 4]
    tensor[..., 2, 2] = D[..., 5]
    return tensor
def visualize_tensor(D):
    tensor = build_tensor_from_coefficients(D)
    evals, evecs = np.linalg.eigh(tensor)
    fa = fractional_anisotropy(evals)
    fa = np.clip(fa, 0, 1)
    cfa = color_fa(fa, evecs)
    ren = window.Scene()
    ren.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=get_sphere('repulsion724'), scale=0.3))
    window.show(ren)
def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return (data, img.affine)
def calculate_diffusion_tensor(dwi_data, bvals, gradients, eps=50):
    num_volumes = dwi_data.shape[-1]
    bvals = np.array(bvals)
    gradients = np.array(gradients)
    G = np.zeros((num_volumes, 7))
    G[:, 0] = -bvals * gradients[:, 0] ** 2
    G[:, 1] = -2 * bvals * gradients[:, 0] * gradients[:, 1]
    G[:, 2] = -2 * bvals * gradients[:, 0] * gradients[:, 2]
    G[:, 3] = -bvals * gradients[:, 1] ** 2
    G[:, 4] = -2 * bvals * gradients[:, 1] * gradients[:, 2]
    G[:, 5] = -bvals * gradients[:, 2] ** 2
    G[:, 6] = 1
    shape = dwi_data.shape[:-1]
    dwi_data_flat = dwi_data.reshape(-1, num_volumes)
    dwi_data_flat[dwi_data_flat <= eps] = eps
    log_S = np.log(dwi_data_flat)
    D_flat = np.zeros((dwi_data_flat.shape[0], 6))
    for i in range(dwi_data_flat.shape[0]):
        y = log_S[i]
        D, _, _, _ = lstsq(G, y)
        D_flat[i] = D[:6]
    D = D_flat.reshape(*shape, 6)
    return D
dwi_file = 'path_to_your_dwi_image.nii'
bvals_file = 'path_to_your_bvals.txt'
gradients_file = 'path_to_your_gradients.txt'
namebase = 'stanford_hardi_'
inputpath = '/home/stark/Documentos/Z_Artigos/artigos_base5/dataset/dataset1'
dwi_file, bvals_file, gradients_file = get_fnames('stanford_hardi')
dwi_data, affine = load_nifti(dwi_file)
bvals = np.loadtxt(bvals_file)
gradients = np.loadtxt(gradients_file).T
D = calculate_diffusion_tensor(dwi_data[..., 9:], bvals[9:], gradients[9:], eps=100)
visualize_tensor(D)
D_img = nib.Nifti1Image(D, affine)
nib.save(D_img, os.path.join('/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'diffusion_tensor.nii'))
