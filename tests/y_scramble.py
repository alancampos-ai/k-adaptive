import argparse, os, sys, shutil, subprocess, numpy as np
from pathlib import Path
import nibabel as nib

METHODS_ALL = ["no_spd","spd_le","airm","hybrid_spd","hybrid_no_spd"]

def method_dir(m: str) -> str:
    return "spd" if m == "spd_le" else m

def read_alpha_fixed(k: int, method: str) -> float:
    dname = method_dir(method)
    base = Path(f"results/k{k}/{dname}")
    for p in [base / f"alpha_{method}_k{k}.txt", base / f"alpha_{method}.txt", base / "alpha.txt"]:
        if p.exists():
            return float(p.read_text().strip())
    raise FileNotFoundError(f"alpha* not found for {method}, K={k}")

def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

def prepare_subject_symlinks(S: Path):
    dst = Path("dataset/dataset1")
    dst.mkdir(parents=True, exist_ok=True)
    m = {
        "stanford_hardi_denoised_.nii.gz": "*denoised_.nii.gz",
        "stanford_hardi_denoised_mask.nii.gz": "*mask*.nii.gz",
        "stanford_hardi_denoised_segmentation_fa_2_classes.nii.gz": "*2_classes*.nii.gz",
        "stanford_hardi_denoised_segmentation_fa_3_classes.nii.gz": "*3_classes*.nii.gz",
        "stanford_hardi_denoised_segmentation_fa_4_classes.nii.gz": "*4_classes*.nii.gz",
    }
    for out, patt in m.items():
        srcs = sorted(S.glob(patt))
        if not srcs:
            raise FileNotFoundError(f"missing file in {S} for pattern {patt}")
        safe_symlink(srcs[0], dst / out)

def scramble_labels_inplace(k: int, seed: int):
    lab_path = Path(f"dataset/dataset1/stanford_hardi_denoised_segmentation_fa_{k}_classes.nii.gz")
    msk_path = Path("dataset/dataset1/stanford_hardi_denoised_mask.nii.gz")
    if not lab_path.exists() or not msk_path.exists():
        raise FileNotFoundError("missing label or mask in dataset/dataset1")
    bak = lab_path.with_suffix(lab_path.suffix + ".bak")
    if bak.exists():
        bak.unlink()
    shutil.copy2(lab_path, bak)

    img = nib.load(str(lab_path)); lab = img.get_fdata().astype(np.int32)
    msk = nib.load(str(msk_path)).get_fdata().astype(bool)
    rng = np.random.default_rng(seed)
    flat = lab[msk].copy()
    idx = np.arange(flat.size); rng.shuffle(idx)
    lab_scr = lab.copy()
    lab_scr[msk] = flat[idx]
    nib.save(nib.Nifti1Image(lab_scr.astype(np.int16), img.affine, img.header), str(lab_path))
    return bak

def restore_label(bak_path: Path, k: int):
    lab_path = Path(f"dataset/dataset1/stanford_hardi_denoised_segmentation_fa_{k}_classes.nii.gz")
    if bak_path.exists():
        shutil.copy2(bak_path, lab_path)

def run_test_once(k: int, method: str, alpha: float, seed: int, radius: int, seed_list: str, restarts: int):
    dname = method_dir(method)
    Path(f"results/k{k}/{dname}").mkdir(parents=True, exist_ok=True)
    extra = []
    if method.startswith("hybrid_"):
        extra += ["--radius", str(radius)]
    cmd = [
        sys.executable, "scripts/sweep_alpha_dense.py",
        "--metric", method,
        "--classes", str(k),
        "--a-min", f"{alpha:.6f}",
        "--a-max", f"{alpha:.6f}",
        "--a-step", "0.001",
        "--max-iter", "300",
        "--restarts", str(restarts),
        "--seed", str(seed),
        "--seed-list", seed_list,
        "--iou-scheme", "macro-fg",
        "--data-dir", "dataset/dataset1",
    ] + extra
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, required=True, choices=[2,3,4])
    for s in range(1,8):
        ap.add_argument(f"--S{s}", type=str, required=True)
    ap.add_argument("--methods", type=str, default=",".join(METHODS_ALL))
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--seed-list", type=str, default="78")
    ap.add_argument("--restarts", type=int, default=1)
    args = ap.parse_args()

    K = args.classes
    subs = [(f"S{s}", Path(getattr(args, f"S{s}")).expanduser().resolve()) for s in range(1,8)]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    alpha_fixed = {m: read_alpha_fixed(K, m) for m in methods}

    for tag, S in subs:
        prepare_subject_symlinks(S)
        for p in range(args.permutations):
            seed_p = args.seed + p
            bak = scramble_labels_inplace(K, seed_p)
            try:
                for m in methods:
                    run_test_once(K, m, alpha_fixed[m], seed_p, args.radius, args.seed_list, args.restarts)
                    dname = method_dir(m)
                    base = Path(f"results/k{K}/{dname}")
                    csv_std = base / f"result_{m}_k{K}.csv"
                    if csv_std.exists():
                        (base / f"result_{m}_k{K}_{tag}_YS{p}.csv").write_bytes(csv_std.read_bytes())
            finally:
                restore_label(bak, K)
    print("y-scramble done")

if __name__ == "__main__":
    sys.exit(main())
