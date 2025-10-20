import os
import argparse
from typing import List, Tuple, Union
from pathlib import Path
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti
from src.segment_dti import segmentation

DEFAULT_ALPHAS = [1.0, 1.25, 1.5, 1.75, 2.0]

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def choose_dataset(dataset_root: Path) -> Path:
    primary = dataset_root / 'dataset3'
    fallback = dataset_root / 'dataset1'
    return primary if primary.is_dir() else fallback

def mean_iou(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    ious = []
    for c in range(n_classes):
        t = y_true == c
        p = y_pred == c
        inter = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        if union > 0:
            ious.append(inter / union)
    if not ious:
        return float('nan')
    return float(np.mean(ious))

def _palette_rgba() -> np.ndarray:
    table = np.zeros((5, 4), dtype=np.uint8)
    table[0] = [0, 0, 0, 255]
    table[1] = [255, 0, 0, 255]
    table[2] = [0, 255, 0, 255]
    table[3] = [0, 0, 255, 255]
    table[4] = [255, 255, 0, 255]
    return table

def labels_to_rgb(label_slice: np.ndarray, k: int) -> np.ndarray:
    lut = _palette_rgba()
    max_idx = min(k, 4)
    out = np.zeros((label_slice.shape[0], label_slice.shape[1], 3), dtype=np.uint8)
    for c in range(max_idx + 1):
        mask = (label_slice == c)
        out[mask] = lut[c, :3]
    return out

def save_slice_png(arr, affine, plane, index, k, out_file, scale=4):
    import numpy as np, nibabel as nib
    from PIL import Image
    def to_ras(a, aff):
        return nib.as_closest_canonical(nib.Nifti1Image(a, aff)).get_fdata().astype(a.dtype)
    def palette(k):
        import numpy as np
        lut = np.zeros((5,3), dtype=np.uint8)
        lut[0]=[0,0,0]; lut[1]=[255,0,0]; lut[2]=[0,255,0]; lut[3]=[0,0,255]; lut[4]=[255,255,0]
        return lut
    A = to_ras(arr, affine)
    if plane=='axial': axis=2
    elif plane=='coronal': axis=1
    elif plane=='sagittal': axis=0
    else: raise ValueError('invalid plane')
    idx = A.shape[axis]//2 if (isinstance(index,str) and index=='mid') else max(0,min(int(index),A.shape[axis]-1))
    sl = A[:,:,idx] if axis==2 else (A[:,idx,:] if axis==1 else A[idx,:,:])
    sl = np.rot90(sl, 1)
    sl = np.fliplr(sl)
    lut = palette(k if k<=4 else 4)
    rgb = lut[(sl.clip(0,4)).astype(int)]
    im = Image.fromarray(rgb, mode='RGB')
    if scale and scale>1:
        im = im.resize((im.width*scale, im.height*scale), resample=Image.NEAREST)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_file.as_posix(), format='PNG')

def run_one(method: str, k: int, alpha: float, max_iterations: int,
            dataset_root: Path, results_root: Path, slice_axial: Union[int, str],
            slice_coronal: Union[int, str], slice_sagittal: Union[int, str],
            seed: int = 50):
    np.random.seed(seed)
    ds = choose_dataset(dataset_root)
    fdti = ds / 'stanford_hardi_denoised_dti_smooth_mediana.nii.gz'
    fmask = ds / 'stanford_hardi_denoised_mask.nii.gz'
    if not fdti.is_file():
        raise FileNotFoundError(str(fdti))
    if not fmask.is_file():
        raise FileNotFoundError(str(fmask))
    dti, affine = load_nifti(str(fdti))
    mask, _ = load_nifti(str(fmask))
    mask = mask.astype(np.bool_)
    seg = segmentation(dti, n_claster=k, mask=mask, metric_type=method,
                       expoent=alpha, index_centers=None, max_iterations=max_iterations, dim_point=3)
    alpha_tag = f"{alpha:.2f}"
    name = f"{method}_{alpha_tag}_k_{k}"
    seg_dir = results_root / name / 'out'
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_file = seg_dir / 'seg_DTI_3D.nii.gz'
    nib.save(nib.Nifti1Image(seg.astype(np.int16), affine), str(seg_file))
    ref_candidates = [
        ds / f"stanford_hardi_denoised_segmentation_fa_{k}_classes.nii.gz",
        ds / "stanford_hardi_denoised_segmentation_fa.nii.gz"
    ]
    ref = next((p for p in ref_candidates if p.is_file()), None)
    if ref is None:
        raise FileNotFoundError('reference segmentation not found')
    ref_vol, ref_aff = load_nifti(str(ref))
    iou = mean_iou(ref_vol.astype(np.int32).ravel(), seg.astype(np.int32).ravel(), k + 1)
    metrics_dir = results_root / name / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / 'metrics.csv'
    header_needed = not metrics_csv.is_file()
    with open(metrics_csv, 'a') as f:
        if header_needed:
            f.write('method,alpha,k,iou,seg_path,ref_path\n')
        f.write(f'{method},{alpha_tag},{k},{iou:.6f},{seg_file},{ref}\n')
    master_csv = results_root / 'master_metrics.csv'
    master_header = not master_csv.is_file()
    with open(master_csv, 'a') as f:
        if master_header:
            f.write('method,alpha,k,iou,seg_path,ref_path\n')
        f.write(f'{method},{alpha_tag},{k},{iou:.6f},{seg_file},{ref}\n')
    plots_dir = results_root / name / 'img'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_slice_png(ref_vol, ref_aff, 'axial', slice_axial, k, plots_dir / f'alpha{alpha_tag}_k{k}_{slice_axial}_axial.png')
    save_slice_png(ref_vol, ref_aff, 'coronal', slice_coronal, k, plots_dir / f'alpha{alpha_tag}_k{k}_{slice_coronal}_coronal.png')
    save_slice_png(ref_vol, ref_aff, 'sagittal', slice_sagittal, k, plots_dir / f'alpha{alpha_tag}_k{k}_{slice_sagittal}_sagittal.png')
    print(f'IOU,{method},{alpha_tag},{k},{iou:.6f}')

def parse_clusters(arg: str):
    allowed = {2, 3, 4}
    vals = set()
    for token in arg.split(','):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k not in allowed:
            raise argparse.ArgumentTypeError('clusters must be in {2,3,4}')
        vals.add(k)
    out = sorted(vals)
    if not out:
        raise argparse.ArgumentTypeError('at least one k in {2,3,4}')
    return out

def parse_alphas(arg: str):
    vals = []
    for token in arg.split(','):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise argparse.ArgumentTypeError('provide at least one alpha')
    return vals

def parse_slice(arg: str) -> Union[int, str]:
    if arg == 'mid':
        return 'mid'
    v = int(arg)
    if v < 0:
        raise argparse.ArgumentTypeError('slice must be >=0 or "mid"')
    return v

def main():
    root = project_root()
    parser = argparse.ArgumentParser(description='DTI KMeans with AIRM and Euclidean')
    parser.add_argument('-k', '--clusters', type=parse_clusters, default=[2,3,4])
    parser.add_argument('-m', '--method', choices=['airm', 'no_spd', 'both'], default='both')
    parser.add_argument('--alphas', type=parse_alphas, default=DEFAULT_ALPHAS)
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--dataset-root', type=str, default=str(root / 'dataset'))
    parser.add_argument('--results-root', type=str, default=str(root / 'results'))
    parser.add_argument('--slice-axial', type=parse_slice, default='mid')
    parser.add_argument('--slice-coronal', type=parse_slice, default='mid')
    parser.add_argument('--slice-sagittal', type=parse_slice, default='mid')
    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if results_root.name != 'out':
        results_root = results_root / 'out'
    methods = ['airm', 'no_spd'] if args.method == 'both' else [args.method]
    dataset_root = Path(args.dataset_root)
    for m in methods:
        for k in args.clusters:
            for a in args.alphas:
                run_one(m, k, a, args.max_iterations, dataset_root, results_root,
                        args.slice_axial, args.slice_coronal, args.slice_sagittal, args.seed)

if __name__ == '__main__':
    main()
