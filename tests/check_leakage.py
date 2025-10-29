import argparse, hashlib, os, sys
from pathlib import Path

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def pick_one(base: Path, patt: str):
    L = sorted(base.glob(patt))
    return L[0] if L else None

def collect_hashes(base: Path):
    out = {}
    patt = {
        "dti": "*denoised_.nii.gz",
        "mask": "*mask*.nii.gz",
        "lab2": "*2_classes*.nii.gz",
        "lab3": "*3_classes*.nii.gz",
        "lab4": "*4_classes*.nii.gz",
    }
    for k,p in patt.items():
        fp = pick_one(base, p)
        if fp and fp.is_file():
            out[k] = (str(fp), sha256sum(fp))
        else:
            out[k] = (None, None)
    return out

def main():
    ap = argparse.ArgumentParser()
    for s in range(1,8):
        ap.add_argument(f"--S{s}", type=str, default=None)
    ap.add_argument("--output", type=str, default="results/leakage_report.txt")
    args = ap.parse_args()

    subs = []
    for s in range(1,8):
        val = getattr(args, f"S{s}")
        if val:
            subs.append((f"S{s}", Path(val).expanduser().resolve()))
    if len(subs) < 2:
        print("ERROR: provide at least two subjects via --S1 .. --S7", file=sys.stderr)
        sys.exit(2)

    os.makedirs("results", exist_ok=True)
    reg = {}
    lines = ["# Leakage report (SHA-256 hashes)", ""]
    for tag, base in subs:
        m = collect_hashes(base)
        reg[tag] = m
        lines.append(f"## {tag} = {base}")
        for k,(pth,hs) in m.items():
            lines.append(f"{k}: {pth if pth else 'N/A'}  hash={hs if hs else 'N/A'}")
        lines.append("")

    alerts = []
    roles = ["dti","mask","lab2","lab3","lab4"]
    for i in range(len(subs)):
        for j in range(i+1, len(subs)):
            ti,_ = subs[i]; tj,_ = subs[j]
            for r in roles:
                pi, hi = reg[ti][r]
                pj, hj = reg[tj][r]
                if hi and hj and hi == hj:
                    alerts.append((ti,tj,r,pi,pj,hi))

    if not alerts:
        lines.append("OK: no bitwise duplicates detected across subjects.")
    else:
        lines.append("ALERT: potential duplicates/leakage")
        for (ti,tj,r,pi,pj,hs) in alerts:
            lines.append(f"- {ti} vs {tj} | type={r} | hash={hs}")
            lines.append(f"  {pi}")
            lines.append(f"  {pj}")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {args.output}")

if __name__ == "__main__":
    sys.exit(main())
