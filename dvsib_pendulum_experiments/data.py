"""
Download, extract, and preprocess the single-pendulum dataset for DVSIB experiments.

Raw data source (Zenodo):
    https://zenodo.org/records/6653856/files/single_pendulum.zip
    (Chen et al. "Automated discovery of fundamental variables hidden in
    experimental data", Nature Computational Science, 2022)

Preprocessed output (saved to --root, default: data/):
    single_pendulum_all   torch tensor  (1200, 60, 784) float32, pixels ∈ [0,1]
    phys_vars.npy         numpy dict with physical labels:
                              theta            (1200, 60) float32  — pendulum angle
                              vel_theta        (1200, 60) float32  — angular velocity
                              kinetic energy   (1200, 60) float32
                              potential energy (1200, 60) float32
                              total energy     (1200, 60) float32
                              reject           (1200, 60) bool

Usage:
    python data.py --all               # download → extract → preprocess (default root: data/)
    python data.py --all --root /path  # write to a custom directory

After running, point run_sweep.py to the same directory:
    export DATA_ROOT=data/             # or whatever --root you used
    python run_sweep.py --root_dir $DATA_ROOT ...
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ZENODO_URL   = "https://zenodo.org/records/6653856/files/single_pendulum.zip?download=1"
DEFAULT_ROOT = Path("data")
ZIP_NAME     = "single_pendulum.zip"
RAW_SUBDIR   = "single_pendulum"
TENSOR_NAME  = "single_pendulum_all"
PHYS_NAME    = "phys_vars.npy"

N_TRIALS = 1200
N_FRAMES = 60


def _urlopen_verified(url):
    """
    Open `url`, verifying TLS against certifi's CA bundle explicitly.

    Some conda environments (common on HPC clusters) ship a Python whose
    default OpenSSL CA path is stale or missing, causing a spurious
    CERTIFICATE_VERIFY_FAILED even though the system otherwise has working
    internet access. certifi bundles its own up-to-date CA store, so using
    it directly sidesteps that misconfiguration regardless of environment.
    """
    import ssl
    import urllib.error
    import urllib.request

    try:
        return urllib.request.urlopen(url)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, context=ctx)


def download(root=DEFAULT_ROOT, force=False):
    """Fetch single_pendulum.zip from Zenodo into `root`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / ZIP_NAME
    if zip_path.exists() and not force:
        print(f"[data] Already downloaded: {zip_path}")
        return zip_path

    print(f"[data] Downloading {ZENODO_URL}")
    with _urlopen_verified(ZENODO_URL) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="download"
        ) as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"[data] Saved: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def extract(zip_path, root=DEFAULT_ROOT, force=False):
    """Unzip the archive into `root/single_pendulum/`."""
    root = Path(root)
    raw_dir = root / RAW_SUBDIR
    sentinel = raw_dir / "phys_vars.npy"
    if sentinel.exists() and not force:
        print(f"[data] Already extracted: {raw_dir}")
        return raw_dir

    print(f"[data] Extracting {zip_path} → {root}/")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not sentinel.exists():
        raise RuntimeError(
            f"Extraction completed but {sentinel} not found — check archive layout."
        )
    return raw_dir


def preprocess(raw_dir, root=DEFAULT_ROOT, size=28, force=False):
    """Downsample frames and write the two files consumed by run_sweep.py.

    Writes to `root`:
        single_pendulum_all   torch.Tensor (1200, 60, size*size) float32, pixels in [0,1]
        phys_vars.npy         copy of phys_vars.npy from the raw archive (original keys)

    Args:
        raw_dir : path to the extracted single_pendulum/ directory
        root    : output directory
        size    : downsampled spatial resolution (28 → 28×28 = 784 pixels)
        force   : overwrite existing outputs
    """
    root        = Path(root)
    raw_dir     = Path(raw_dir)
    tensor_path = root / TENSOR_NAME
    phys_path   = root / PHYS_NAME

    if tensor_path.exists() and phys_path.exists() and not force:
        print(f"[data] Already preprocessed: {tensor_path}, {phys_path}")
        return tensor_path, phys_path

    D = size * size
    frames = np.empty((N_TRIALS, N_FRAMES, D), dtype=np.float32)

    print(f"[data] Downsampling {N_TRIALS}×{N_FRAMES} frames "
          f"(128×128 RGB → {size}×{size} grayscale, bilinear, /255 normalisation)")
    for trial in tqdm(range(N_TRIALS), desc="trials"):
        tdir = raw_dir / str(trial)
        for t in range(N_FRAMES):
            img_rgb = np.asarray(Image.open(tdir / f"{t}.png").convert("RGB"))
            gray    = np.asarray(
                Image.fromarray(img_rgb).convert("L").resize((size, size), Image.BILINEAR),
                dtype=np.float32,
            )
            frames[trial, t] = gray.ravel() / 255.0

    tensor = torch.from_numpy(frames)
    torch.save(tensor, tensor_path)
    print(f"[data] Saved tensor: {tensor_path}  shape={tuple(tensor.shape)}, "
          f"min={tensor.min():.3f}, max={tensor.max():.3f}")

    # Copy phys_vars.npy from the archive alongside the tensor so that
    # run_sweep.py --root_dir finds both files in the same directory.
    shutil.copy2(raw_dir / "phys_vars.npy", phys_path)
    print(f"[data] Copied: {phys_path}")

    return tensor_path, phys_path


def _main():
    p = argparse.ArgumentParser(
        description="Download and preprocess the pendulum dataset for DVSIB experiments."
    )
    p.add_argument("--root",       type=str, default=str(DEFAULT_ROOT),
                   help="Output directory for downloads and preprocessed files (default: data/)")
    p.add_argument("--download",   action="store_true", help="Download the archive from Zenodo")
    p.add_argument("--extract",    action="store_true", help="Extract the downloaded archive")
    p.add_argument("--preprocess", action="store_true", help="Downsample frames and save tensors")
    p.add_argument("--all",        action="store_true", help="Run all three steps")
    p.add_argument("--force",      action="store_true", help="Overwrite existing cached outputs")
    p.add_argument("--size",       type=int, default=28,
                   help="Downsampled spatial resolution (default: 28, giving 28×28 = 784 pixels)")
    args = p.parse_args()

    if not any([args.all, args.download, args.extract, args.preprocess]):
        p.print_help()
        return

    root = Path(args.root)

    if args.all or args.download:
        zip_path = download(root=root, force=args.force)
    else:
        zip_path = root / ZIP_NAME

    if args.all or args.extract:
        raw_dir = extract(zip_path, root=root, force=args.force)
    else:
        raw_dir = root / RAW_SUBDIR

    if args.all or args.preprocess:
        preprocess(raw_dir, root=root, size=args.size, force=args.force)


if __name__ == "__main__":
    _main()
