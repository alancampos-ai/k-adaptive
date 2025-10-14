import numpy as np
import matplotlib.pyplot as plt
from dipy.data import get_fnames
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa
from dipy.denoise.patch2self import patch2self
import xarray as xr
import os
from dipy.data import get_sphere
from dipy.viz import window, actor

namebase = 'stanford_hardi_'
inputpath = '/home/stark/Documentos/Z_Artigos/artigos_base5/dataset/dataset1'
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
denoised_arr = patch2self(data, bvals, model='ols', shift_intensity=False, clip_negative_vals=True, b0_threshold=50)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_patch2self.nii.gz']), denoised_arr, affine)
sli = data.shape[2] // 2
gra = 60
orig = data[:, :, sli, gra]
den = denoised_arr[:, :, sli, gra]
rms_diff = np.sqrt((orig - den) ** 2)
fig1, ax = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(orig.T, cmap='gray', interpolation='none', origin='lower')
ax.flat[0].set_title('Original')
ax.flat[1].imshow(den.T, cmap='gray', interpolation='none', origin='lower')
ax.flat[1].set_title('Denoised Output')
ax.flat[2].imshow(rms_diff.T, cmap='gray', interpolation='none', origin='lower')
ax.flat[2].set_title('Residuals')
plt.show()
gtab = gradient_table(bvals, bvecs)
maskdata, mask = median_otsu(denoised_arr, vol_idx=range(0, 9), median_radius=4, numpass=4, autocrop=True, dilate=None)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_maskdata.nii.gz']), maskdata, affine)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_mask.nii.gz']), mask.astype(np.int16), affine)
mask_xr = xr.DataArray(mask, name='Mask', coords=[np.arange(mask.shape[0]), np.arange(mask.shape[1]), np.arange(mask.shape[2])], dims=['sagittal', 'coronal', 'axial'])
maskdata_xr = xr.DataArray(maskdata.astype(np.float32), name='data', coords=[np.arange(maskdata.shape[0]), np.arange(maskdata.shape[1]), np.arange(maskdata.shape[2]), np.arange(maskdata.shape[3])], dims=['sagittal', 'coronal', 'axial', 'direction'])
for ax, d in zip([35, 35], [0, 50]):
    maskdata_xr.sel(axial=ax, direction=d).transpose().plot(cmap='gray')
    plt.savefig(f'axial_{ax}_direction_{d}.png', bbox_inches='tight', dpi=300)
    plt.close()
    maskdata_xr.sel(coronal=ax, direction=d).transpose().plot(cmap='gray')
    plt.savefig(f'coronal_{ax}_direction_{d}.png', bbox_inches='tight', dpi=300)
    plt.close()
    maskdata_xr.sel(sagittal=ax, direction=d).transpose().plot(cmap='gray')
    plt.savefig(f'sagittal_{ax}_direction_{d}.png', bbox_inches='tight', dpi=300)
    plt.close()
plt.show()
data = maskdata
q = np.percentile(data.mean(axis=-1), 99)
mask = data[..., 0] > q
fig = plt.figure()
a = fig.add_subplot(1, 3, 1)
img_ax = np.rot90(data[:, :, 40, 0])
imgplot = plt.imshow(img_ax, cmap='gray')
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 3, 2)
img_cor = np.rot90(data[:, 40, :, 0])
imgplot = plt.imshow(img_cor, cmap='gray')
a.axis('off')
a.set_title('Coronal')
a = fig.add_subplot(1, 3, 3)
img_cor = np.rot90(mask[:, :, 40])
imgplot = plt.imshow(img_cor, cmap='gray')
a.axis('off')
a.set_title('Mask axial')
plt.show()
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(maskdata, mask=mask)
FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
save_nifti(os.path.sep.join(['/h/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_tensor_fa.nii.gz']), FA.astype(np.float32), affine)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_dti.nii.gz']), tenfit.quadratic_form.astype(np.float32), affine)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'tensor_fa.nii.gz']), FA.astype(np.float32), affine)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'tensor_evecs.nii.gz']), tenfit.evecs.astype(np.float32), affine)
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
save_nifti(os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'tensor_rgb.nii.gz']), np.array(255 * RGB, 'uint8'), affine)
'\n    This block is responsable by generate the ground truth \n    segmentation images using a TissueClassifierHMRF function applied to FA.\n'
nclass = [2, 3, 4]
beta = 0.2
for nc in nclass:
    print(f'Segmentation to {nc} classes ...')
    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(FA, nc, beta, max_iter=100)
    save_nifti(f'stanford_hardi_denoised_segmentation_fa_{nc}_classes.nii.gz', final_segmentation, affine)
print('Computing tensor ellipsoids in a part of the splenium of the CC')
sphere = get_sphere('repulsion724')
evals = tenfit.evals
evecs = tenfit.evecs
cfa = RGB
cfa /= cfa.max()
for eixo, indice_fatia in zip(['sagittal', 'coronal', 'axial'], [35, 35, 35]):
    scene = window.Scene()
    if eixo == 'sagittal':
        evals_fatia = np.moveaxis(evals[indice_fatia:indice_fatia + 1, :, :], 0, 2)
        evecs_fatia = np.moveaxis(evecs[indice_fatia:indice_fatia + 1, :, :], 0, 2)
        cfa_fatia = np.moveaxis(cfa[indice_fatia:indice_fatia + 1, :, :], 0, 2)
    elif eixo == 'coronal':
        evals_fatia = np.moveaxis(evals[:, indice_fatia:indice_fatia + 1, :], 1, 2)
        evecs_fatia = np.moveaxis(evecs[:, indice_fatia:indice_fatia + 1, :], 1, 2)
        cfa_fatia = np.moveaxis(cfa[:, indice_fatia:indice_fatia + 1, :], 1, 2)
    elif eixo == 'axial':
        evals_fatia = evals[:, :, indice_fatia:indice_fatia + 1]
        evecs_fatia = evecs[:, :, indice_fatia:indice_fatia + 1]
        cfa_fatia = cfa[:, :, indice_fatia:indice_fatia + 1]
    tensor_actor = actor.tensor_slicer(evals_fatia, evecs_fatia, scalar_colors=cfa_fatia, sphere=sphere, scale=0.8, opacity=1.0)
    if tensor_actor is not None:
        scene.add(tensor_actor)
        scene.background((0, 0, 0))
        scene.reset_camera()
        scene.zoom(1.5)
        print('Saving illustration as tensor_ellipsoids.png')
        window.record(scene, n_frames=1, out_path=f'tensor_ellipsoids_{eixo}_slice_{indice_fatia}.png', size=(800, 800), magnification=1)
    else:
        print('Error: tensor_actor is None.')
    camera_position, focal_point, view_up = scene.get_camera()
    print(f'Camera position: {camera_position}')
