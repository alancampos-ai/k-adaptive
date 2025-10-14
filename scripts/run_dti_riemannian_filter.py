import os
import numpy as np 
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.dti import decompose_tensor
from pymanopt.manifolds import PositiveDefinite
from utils import FilterDti
from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import decompose_tensor
np.random.seed(seed=50)
typefilter = 'arf'
dim_point = 3
make_grad = True
namebase = 'stanford_hardi_'
inputpath = '/home/stark/Documentos/Z_Artigos/artigos_base5/dataset/dataset1'
fdti = os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/dataset/dataset1','stanford_hardi_denoised_dti.nii.gz'])
fmask = os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/dataset/dataset1','stanford_hardi_denoised_mask.nii.gz'])
volume_seg = os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_dti_grad_espacial.nii.gz'])
dti, affine = load_nifti(fdti)
mask, _ = load_nifti(fmask)
mask = mask.astype(np.bool8)
manifold = PositiveDefinite(dim_point)
if make_grad:
    obj_dti = FilterDti(manifold, dti, tensormask=mask, make_grad=make_grad, typefilter=typefilter, s=3)
    grad_espacial = obj_dti.gradient_espatial()
    save_nifti(volume_seg, grad_espacial.astype(np.float32), affine)
else:
    obj_dti = FilterDti(manifold, dti, tensormask=mask, make_grad=make_grad, typefilter=typefilter, s=3)
    grad_espacial, _ = load_nifti(volume_seg)
    obj_dti.set_grad_espacial(grad_espacial)
dti_arf, dti_avg, dti_med = obj_dti.filtering_dti()
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_dti_smooth_hibrido.nii.gz']), dti_arf.astype(np.float32), affine)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_dti_smooth_media.nii.gz']), dti_avg.astype(np.float32), affine)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_dti_smooth_mediana.nii.gz']), dti_med.astype(np.float32), affine)
evals_avg, evecs_avg = decompose_tensor(dti_avg)
FA_avg = fractional_anisotropy(evals_avg)
FA_avg[np.isnan(FA_avg)] = 0
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_dti_smooth_media_FA.nii.gz']), FA_avg.astype(np.float32), affine)
evals_med, evecs_med = decompose_tensor(dti_med)
FA_med = fractional_anisotropy(evals_med)
FA_med[np.isnan(FA_med)] = 0
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_dti_smooth_mediana_FA.nii.gz']), FA_med.astype(np.float32), affine)
