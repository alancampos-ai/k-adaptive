import os 
import numpy as np
from pathlib import Path
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.dti import TensorModel, fractional_anisotropy

np.random.seed(50)
NAMEBASE = "stanford_hardi"
HMRF_CLASSES = [2, 3, 4]
HMRF_BETA = 0.2
B0_COUNT = 10

def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATASET_DIR = (ROOT / "dataset" / "dataset1").resolve()
    os.makedirs(DATASET_DIR, exist_ok=True)
    f_mask = DATASET_DIR / f"{NAMEBASE}_denoised_mask.nii.gz"
    f_tensor_dti = DATASET_DIR / f"{NAMEBASE}_denoised_.nii.gz"

    hardi_fname, bval_fname, bvec_fname = get_fnames(name="stanford_hardi")
    data, affine = load_nifti(hardi_fname)
    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    den = data.astype(np.float32)
    masked, mask = median_otsu(den, vol_idx=range(0, min(B0_COUNT, den.shape[-1])), median_radius=4, numpass=4, autocrop=True, dilate=None)
    save_nifti(str(f_mask), mask.astype(np.int16), affine)

    q = np.percentile(masked.mean(axis=-1), 99)
    mask_fit = masked[..., 0] > q
    tenfit = TensorModel(gtab).fit(masked, mask=mask_fit)
    save_nifti(str(f_tensor_dti), tenfit.quadratic_form.astype(np.float32), affine)

    FA = fractional_anisotropy(tenfit.evals).astype(np.float32)
    FA[np.isnan(FA)] = 0.0
    FA_clip = np.clip(FA, 0, 1)

    for nc in HMRF_CLASSES:
        hmrf = TissueClassifierHMRF()
        _, seg, _ = hmrf.classify(FA_clip, nc, HMRF_BETA, max_iter=100)
        save_nifti(str(DATASET_DIR / f"{NAMEBASE}_denoised_segmentation_fa_{nc}_classes.nii.gz"), seg.astype(np.int16), affine)

if __name__ == "__main__":
    main()
