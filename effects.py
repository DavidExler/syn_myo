import numpy as np
from scipy import ndimage as ndi
from pathlib import Path
import tifffile

import argparse
import os

# ============================================================
# helpers
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic myocyte volumes")
    parser.add_argument("--output-folder", default="syn", help="Target output folder")
    parser.add_argument("--output-stem", default="syn_effects_", help="Filename stem for generated images")
    parser.add_argument("--input-folder", default="annotations", help="Annotation input folder")
    parser.add_argument("--input-stem", default="syn_", help="Filename stem for input annotations")
    return parser.parse_args()

def ensure_float01(img):
    """Convert image to float32. Does not rescale automatically."""
    return np.asarray(img, dtype=np.float32)


def get_binary_mask(mask):
    """Accept binary or instance mask, return boolean foreground mask."""
    return np.asarray(mask) > 0


def get_instance_ids(mask):
    """Return sorted positive instance ids from an instance mask."""
    mask = np.asarray(mask)
    ids = np.unique(mask)
    return ids[ids > 0]


def normalize_percentile(img, pmin=0.5, pmax=99.5, eps=1e-8):
    """Robust normalization to [0, 1]."""
    lo = np.percentile(img, pmin)
    hi = np.percentile(img, pmax)
    img = (img - lo) / (hi - lo + eps)
    return np.clip(img, 0.0, 1.0)


# ============================================================
# 1) background: nearly black, tiny baseline shifts only
# ============================================================

def generate_dark_background(
    shape_xyz,
    base_range=(0.0, 0.02),
    gaussian_sigma=0.002,
    rng=None,
):
    """
    Very dark confocal-like background with slight baseline shift and weak noise.

    Parameters
    ----------
    shape_xyz : tuple
        (X, Y, Z)
    base_range : tuple
        Random baseline offset.
    gaussian_sigma : float
        Very weak additive background grain.
    """
    if rng is None:
        rng = np.random.default_rng()

    base = rng.uniform(*base_range)
    bg = np.full(shape_xyz, base, dtype=np.float32)

    if gaussian_sigma > 0:
        bg += rng.normal(0.0, gaussian_sigma, size=shape_xyz).astype(np.float32)

    return np.clip(bg, 0.0, 1.0)


# ============================================================
# 2) debris blobs
# ============================================================

def add_blob_debris(
    img,
    mask,
    n_blobs=(10, 40),
    radius_range=(2.0, 12.0),
    intensity_range=(0.3, 1.0),
    blur_sigma_range=(0.4, 1.5),
    avoid_objects=False,
    avoid_margin=2,
    rng=None,
):
    """
    Add bright debris blobs to the image.

    Blobs are ellipsoidal Gaussians with random size/intensity.
    Since your images are (x, y, z), all coordinates follow that order.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = img.copy().astype(np.float32)
    fg = get_binary_mask(mask)
    shape = img.shape

    if avoid_objects:
        forbidden = ndi.binary_dilation(fg, iterations=avoid_margin)
    else:
        forbidden = None

    nb = rng.integers(n_blobs[0], n_blobs[1] + 1)

    xs = np.arange(shape[0], dtype=np.float32)[:, None, None]
    ys = np.arange(shape[1], dtype=np.float32)[None, :, None]
    zs = np.arange(shape[2], dtype=np.float32)[None, None, :]

    for _ in range(nb):
        # choose center
        for _try in range(50):
            cx = rng.integers(0, shape[0])
            cy = rng.integers(0, shape[1])
            cz = rng.integers(0, shape[2])
            if forbidden is None or not forbidden[cx, cy, cz]:
                break

        rxy = rng.uniform(*radius_range)
        rz = max(1.0, rxy * rng.uniform(0.4, 1.0))
        amp = rng.uniform(*intensity_range)

        # local box for efficiency
        rx = int(np.ceil(3 * rxy))
        ry = int(np.ceil(3 * rxy))
        rz_box = int(np.ceil(3 * rz))

        x0, x1 = max(0, cx - rx), min(shape[0], cx + rx + 1)
        y0, y1 = max(0, cy - ry), min(shape[1], cy + ry + 1)
        z0, z1 = max(0, cz - rz_box), min(shape[2], cz + rz_box + 1)

        xl = xs[x0:x1] - cx
        yl = ys[:, y0:y1] - cy
        zl = zs[:, :, z0:z1] - cz

        blob = np.exp(-0.5 * ((xl / rxy) ** 2 + (yl / rxy) ** 2 + (zl / rz) ** 2))
        blob = amp * blob.astype(np.float32)

        local = np.zeros_like(blob, dtype=np.float32)
        local += blob

        blur_sigma = rng.uniform(*blur_sigma_range)
        if blur_sigma > 0:
            local = ndi.gaussian_filter(local, sigma=(blur_sigma, blur_sigma, blur_sigma))

        img[x0:x1, y0:y1, z0:z1] += local

    return img


# ============================================================
# 3) dense salt-and-pepper / granular texture INSIDE objects
# ============================================================

def add_dense_object_salt_pepper(
    img,
    mask,
    amount=0.08,
    salt_vs_pepper=0.6,
    salt_intensity=(0.08, 0.25),
    pepper_intensity=(0.03, 0.12),
    dilate_prob=0.5,
    dilate_iters=(1, 2),
    per_instance_variation=False,
    rng=None,
):
    """
    Dense salt/pepper-like texture inside objects only.

    This is intentionally not classic binary salt-and-pepper corruption.
    It creates many short bright/dim marks inside object voxels.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = img.copy().astype(np.float32)
    mask = np.asarray(mask)
    fg = mask > 0

    if not per_instance_variation:
        idx = np.argwhere(fg)
        if len(idx) == 0:
            return img

        n = int(amount * len(idx))
        if n <= 0:
            return img

        chosen = idx[rng.choice(len(idx), size=n, replace=False)]
        salt_n = int(salt_vs_pepper * n)

        salt_pts = chosen[:salt_n]
        pepper_pts = chosen[salt_n:]

        salt_mask = np.zeros_like(fg, dtype=bool)
        pepper_mask = np.zeros_like(fg, dtype=bool)

        salt_mask[salt_pts[:, 0], salt_pts[:, 1], salt_pts[:, 2]] = True
        pepper_mask[pepper_pts[:, 0], pepper_pts[:, 1], pepper_pts[:, 2]] = True

        # elongate some points into short "specks/stripes"
        if rng.random() < dilate_prob:
            it = rng.integers(dilate_iters[0], dilate_iters[1] + 1)
            salt_mask = ndi.binary_dilation(salt_mask, iterations=it) & fg
        if rng.random() < dilate_prob:
            it = rng.integers(dilate_iters[0], dilate_iters[1] + 1)
            pepper_mask = ndi.binary_dilation(pepper_mask, iterations=it) & fg

        salt_amp = rng.uniform(*salt_intensity)
        pepper_amp = rng.uniform(*pepper_intensity)

        img[salt_mask] += salt_amp
        img[pepper_mask] -= pepper_amp
        return img

    # per-instance mode
    out = img.copy()
    ids = get_instance_ids(mask)
    for obj_id in ids:
        obj = mask == obj_id
        idx = np.argwhere(obj)
        if len(idx) == 0:
            continue

        obj_amount = amount * rng.uniform(0.7, 1.3)
        n = int(obj_amount * len(idx))
        if n <= 0:
            continue

        chosen = idx[rng.choice(len(idx), size=min(n, len(idx)), replace=False)]
        salt_n = int(salt_vs_pepper * len(chosen))
        salt_pts = chosen[:salt_n]
        pepper_pts = chosen[salt_n:]

        salt_mask = np.zeros_like(obj, dtype=bool)
        pepper_mask = np.zeros_like(obj, dtype=bool)

        if len(salt_pts) > 0:
            salt_mask[salt_pts[:, 0], salt_pts[:, 1], salt_pts[:, 2]] = True
        if len(pepper_pts) > 0:
            pepper_mask[pepper_pts[:, 0], pepper_pts[:, 1], pepper_pts[:, 2]] = True

        if rng.random() < dilate_prob:
            salt_mask = ndi.binary_dilation(
                salt_mask, iterations=rng.integers(dilate_iters[0], dilate_iters[1] + 1)
            ) & obj
        if rng.random() < dilate_prob:
            pepper_mask = ndi.binary_dilation(
                pepper_mask, iterations=rng.integers(dilate_iters[0], dilate_iters[1] + 1)
            ) & obj

        out[salt_mask] += rng.uniform(*salt_intensity)
        out[pepper_mask] -= rng.uniform(*pepper_intensity)

    return out


# ============================================================
# 4) slight xy halo + mainly downward z halo
# ============================================================

def add_directional_halo(
    img,
    mask,
    xy_sigma=1.0,
    z_decay=4.0,
    xy_strength=0.04,
    z_strength=0.12,
    max_down_z=12,
):
    """
    Adds a weak glow around objects:
    - slight symmetric xy halo
    - stronger downward-only z halo

    Assumes +z is the downward direction in your data ordering.
    If your "downward" is -z, flip the z kernel.
    """
    img = img.copy().astype(np.float32)
    fg = get_binary_mask(mask).astype(np.float32)

    # weak symmetric xy halo
    if xy_strength > 0:
        xy_blur = ndi.gaussian_filter(fg, sigma=(xy_sigma, xy_sigma, 0.0))
        xy_halo = np.clip(xy_blur - fg, 0.0, None)
        img += xy_strength * xy_halo

    # stronger downward-only z tail
    if z_strength > 0 and max_down_z > 0:
        kernel = np.exp(-np.arange(max_down_z + 1, dtype=np.float32) / max(z_decay, 1e-6))
        kernel[0] = 0.0  # no self-term here; only below
        kernel = kernel / (kernel.sum() + 1e-8)

        z_halo = ndi.convolve1d(fg, weights=kernel, axis=2, mode='constant', cval=0.0)
        # because convolve1d is centered, shift to make it one-sided "downward"
        shifted = np.zeros_like(z_halo)
        for dz in range(1, max_down_z + 1):
            w = kernel[dz]
            if w <= 0:
                continue
            shifted[:, :, dz:] += w * fg[:, :, :-dz]

        img += z_strength * shifted

    return img


# ============================================================
# 5) weak local edge blur only
# ============================================================

def apply_soft_edge_blending(
    img,
    mask,
    edge_sigma=(0.8, 0.8, 0.6),
    edge_width=1,
    blend_strength=0.5,
):
    """
    Blur only a thin band around object boundaries.
    Keeps object interiors sharper than a full blur.
    """
    img = img.copy().astype(np.float32)
    fg = get_binary_mask(mask)

    dil = ndi.binary_dilation(fg, iterations=edge_width)
    ero = ndi.binary_erosion(fg, iterations=edge_width)
    edge_band = dil ^ ero

    blurred = ndi.gaussian_filter(img, sigma=edge_sigma)
    img[edge_band] = (
        (1.0 - blend_strength) * img[edge_band]
        + blend_strength * blurred[edge_band]
    )
    return img


# ============================================================
# 6) weak additive Gaussian read noise
# ============================================================

def add_gaussian_read_noise(
    img,
    sigma=0.01,
    mean=0.0,
    rng=None,
):
    """Weak global additive Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()

    img = img.astype(np.float32, copy=True)
    img += rng.normal(mean, sigma, size=img.shape).astype(np.float32)
    return img


# ============================================================
# 7) anisotropic 3D PSF approximation
# ============================================================

def apply_anisotropic_psf(
    img,
    sigma_xyz=(1.2, 1.2, 3.5),
):
    """
    Naive confocal PSF approximation with anisotropic Gaussian blur.

    IMPORTANT:
    your arrays are (x, y, z), so sigma_xyz must also be (sx, sy, sz).
    """
    img = img.astype(np.float32, copy=True)
    return ndi.gaussian_filter(img, sigma=sigma_xyz)


# ============================================================
# 8) Poisson noise
# ============================================================

def add_poisson_noise(
    img,
    peak_photons=40.0,
    rng=None,
):
    """
    Apply Poisson shot noise.

    peak_photons controls noise strength:
    - lower values => noisier
    - higher values => cleaner

    img is assumed roughly in [0, 1] before this step.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    photons = img * peak_photons
    noisy = rng.poisson(photons).astype(np.float32) / max(peak_photons, 1e-8)
    return noisy


# ============================================================
# 9) clipping and output conversion
# ============================================================

def final_clip_and_uint8(img):
    """Clip to [0,1] and convert to uint8 [0,255]."""
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


# ============================================================
# optional: object filling helper if your synth object image
# is only geometry/mask and you need a base signal inside objects
# ============================================================

def render_base_signal_from_mask(
    mask,
    object_intensity=(0.18, 0.45),
    per_instance=True,
    rng=None,
):
    """
    Create a simple base image from a binary or instance mask.

    Useful if your synthetic input is only a mask and you want an initial signal
    before adding texture/halo/PSF/noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.asarray(mask)
    fg = mask > 0
    img = np.zeros(mask.shape, dtype=np.float32)

    if per_instance and mask.max() > 1:
        for obj_id in get_instance_ids(mask):
            obj = mask == obj_id
            img[obj] = rng.uniform(*object_intensity)
    else:
        img[fg] = rng.uniform(*object_intensity)

    return img


# ============================================================
# optional: small per-object intensity variation
# ============================================================

def modulate_per_instance_intensity(
    img,
    mask,
    factor_range=(0.85, 1.15),
    rng=None,
):
    """Randomly scale each instance slightly."""
    if rng is None:
        rng = np.random.default_rng()

    img = img.copy().astype(np.float32)
    mask = np.asarray(mask)

    for obj_id in get_instance_ids(mask):
        obj = mask == obj_id
        img[obj] *= rng.uniform(*factor_range)

    return img


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    path = Path(args.input_folder)
    images = [p for p in path.iterdir() if p.suffix == ".tif"]
    for i in range(len(images)):
        mask = tifffile.imread(path / f"{args.input_stem}{i}.tif")
        # 1) start with dark background
        img = generate_dark_background(mask.shape, base_range=(0.0, 0.01), gaussian_sigma=0.001)

        # 2) add base object signal
        img += render_base_signal_from_mask(mask, object_intensity=(0.15, 0.35), per_instance=True)

        # 3) add dense object-internal salt/pepper texture
        img = add_dense_object_salt_pepper(
            img, mask,
            amount=0.06,
            salt_vs_pepper=0.65,
            salt_intensity=(0.05, 0.20),
            pepper_intensity=(0.02, 0.08),
            dilate_prob=0.7,
            dilate_iters=(1, 2),
            per_instance_variation=True,
        )

        # 4) add bright debris blobs
        img = add_blob_debris(
            img, mask,
            n_blobs=(8, 25),
            radius_range=(2.0, 12.0),
            intensity_range=(0.3, 0.9),
            blur_sigma_range=(0.3, 1.0),
            avoid_objects=False,
        )

        # 5) weak halo
        img = add_directional_halo(
            img, mask,
            xy_sigma=1.0,
            z_decay=3.0,
            xy_strength=0.03,
            z_strength=0.10,
            max_down_z=10,
        )

        # 6) weak edge blending
        img = apply_soft_edge_blending(
            img, mask,
            edge_sigma=(0.8, 0.8, 0.5),
            edge_width=1,
            blend_strength=0.4,
        )

        # 7) global PSF
        img = apply_anisotropic_psf(
            img,
            sigma_xyz=(1.3, 1.3, 4.0),
        )

        # 8) poisson
        img = add_poisson_noise(img, peak_photons=35.0)

        # 9) weak gaussian read noise
        img = add_gaussian_read_noise(img, sigma=0.006)

        # 10) normalize if needed, then export
        img = np.clip(img, 0.0, 1.0)
        img_u8 = final_clip_and_uint8(img)

        save=np.transpose(img_u8, (2, 1, 0)).astype(np.float32)
        tifffile.imwrite(
            f"{args.output_folder}/{args.output_stem}{i}.tif",
            save.astype(np.float32),
            imagej=True
        )