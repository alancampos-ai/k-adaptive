import os 
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti
'\n    This funtion it is responsable by generate the ground truth \n    segmentation images using a manual segmentation process based on \n    histogram of the DWI images.\n'
namebase = 'stanford_hardi_'
inputpath = '/home/stark/Documentos/Z_Artigos/artigos_base5/dataset/dataset6'
f_dwi = os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/dataset/dataset6','stanford_hardi_denoised_maskdata.nii.gz'])
dwi, affine = load_nifti(f_dwi)
dwi_mean = np.mean(dwi[:, :, :, :10], axis=-1)
classes_ref = np.zeros_like(dwi_mean)
plt.hist(dwi_mean[dwi_mean > 0])
idx_c1 = np.where((dwi_mean > 0) & (dwi_mean <= 950))
idx_c2 = np.where((dwi_mean > 950) & (dwi_mean <= 1830))
idx_c3 = np.where(dwi_mean > 1830)
classes_ref[idx_c1] = 1
classes_ref[idx_c2] = 2
classes_ref[idx_c3] = 3
fig = plt.figure()
a = fig.add_subplot(1, 1, 1)
plt.imshow(np.rot90(classes_ref[..., 50]))
a.axis('off')
a.set_title('Reference Image')
volume_seg = os.path.sep.join(['/home/stark/Documentos/Z_Artigos/artigos_base5/results', 'stanford_hardi_denoised_b0_segmented_manual.nii.gz'])
save_nifti(volume_seg, np.array(classes_ref, 'int16'), affine)

