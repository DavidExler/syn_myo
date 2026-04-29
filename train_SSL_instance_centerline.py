import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tifffile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ZScoreNormalize:
    """Per-volume z-score normalization with clipping for stability."""

    def __init__(self, clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0)):
        self.clip_range = clip_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        mean = float(img.mean())
        std = float(img.std())
        std = max(std, 1e-6)
        img = (img - mean) / std
        if self.clip_range is not None:
            img = np.clip(img, self.clip_range[0], self.clip_range[1])
        return img.astype(np.float32)


def ensure_divisible_patch(patch_size: Sequence[int], levels: int) -> None:
    divisor = 2 ** levels
    for p in patch_size:
        if p % divisor != 0:
            raise ValueError(
                f"Patch dimension {p} must be divisible by {divisor} for {levels} downsampling levels."
            )



# ============================================================
# Datasets
# ============================================================

class Random3DCrop:
    def __init__(self, patch_size: Tuple[int, int, int]):
        self.patch_size = patch_size

    def __call__(self, img: np.ndarray, ann: Optional[np.ndarray] = None):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((pad_d, 0), (pad_h, 0), (pad_w, 0)), mode="edge")
            if ann is not None:
                ann = np.pad(ann, ((pad_d, 0), (pad_h, 0), (pad_w, 0)), mode="edge")
            d, h, w = img.shape

        d_start = np.random.randint(0, d - pd + 1)
        h_start = np.random.randint(0, h - ph + 1)
        w_start = np.random.randint(0, w - pw + 1)

        img_patch = img[d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        ann_patch = None
        if ann is not None:
            ann_patch = ann[d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        return img_patch, ann_patch


class SyntheticSegmentationDataset(Dataset):
    def __init__(
        self,
        syn_dir: str,
        cache_dir: str,
        patch_size: Tuple[int, int, int] = (64, 256, 256),
        image_indices: Optional[Sequence[int]] = None,
        patches_per_image: int = 32,
        normalize: bool = True,
        crop_attempts: int = 8,
        syn_prefix: str = "processed_transformed_syn_effects_",
        cache_prefix: str = "target_",
    ):
        if image_indices is None:
            raise ValueError("image_indices must be provided when using prefix-based loading.")

        self.image_indices = list(image_indices)
        self.syn_files = [Path(syn_dir) / f"{syn_prefix}{i}.tif" for i in self.image_indices]
        self.cache_files = [Path(cache_dir) / f"{cache_prefix}{i}.tif" for i in self.image_indices]

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.cropper = Random3DCrop(patch_size)
        self.normalizer = ZScoreNormalize() if normalize else None
        self.crop_attempts = crop_attempts

        self.images: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []

        for syn_file, cache_file in zip(self.syn_files, self.cache_files):
            if not syn_file.exists():
                raise FileNotFoundError(f"Synthetic image not found: {syn_file}")
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"Cached target not found: {cache_file}\n"
                    f"Run precomputation first."
                )

            img = tifffile.imread(syn_file).astype(np.float32)
            if self.normalizer is not None:
                img = self.normalizer(img)

            target = tifffile.imread(cache_file).astype(np.float32)

            # Keep only target channel 1 = centerline
            if target.ndim == 4:
                if target.shape[0] < 2:
                    raise ValueError(
                        f"Cached target {cache_file} must have at least 2 channels to select channel 1, got {target.shape}"
                    )
                target = target[1:2]   # (1, D, H, W)
            elif target.ndim == 3:
                target = target[None, ...]   # single-channel
            else:
                raise ValueError(
                    f"Cached target {cache_file} must have shape (3, D, H, W) or (D, H, W), got {target.shape}"
                )

            self.images.append(img)
            self.targets.append(target)

        if len(self.images) == 0:
            raise ValueError("No synthetic training samples were loaded.")

    def __len__(self) -> int:
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx: int):
        img_idx = idx % len(self.images)
        img = self.images[img_idx]
        target = self.targets[img_idx]
        target_volume = target[0]   # shape -> (D, H, W)

        pd, ph, pw = self.patch_size

        # pad before sampling
        d, h, w = img.shape
        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            img_pad = np.pad(img, ((pad_d, 0), (pad_h, 0), (pad_w, 0)), mode="edge")
            target_pad = np.pad(
                target,
                ((0, 0), (pad_d, 0), (pad_h, 0), (pad_w, 0)),
                mode="edge",
            )
        else:
            img_pad = img
            target_pad = target

        target_volume = target_pad[0]   # shape -> (D, H, W)
        d, h, w = img_pad.shape

        # old foreground-aware crop selection logic
        best_crop = None
        best_target_count = -1

        for _ in range(self.crop_attempts):
            d_start = np.random.randint(0, d - pd + 1)
            h_start = np.random.randint(0, h - ph + 1)
            w_start = np.random.randint(0, w - pw + 1)

            target_patch_try = target_volume[d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
            target_count = int(np.count_nonzero(target_patch_try > 0))

            if target_count > best_target_count:
                best_target_count = target_count
                best_crop = (d_start, h_start, w_start)

            if target_count > 0:
                break

        d_start, h_start, w_start = best_crop

        img_patch = img_pad[d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        target_patch = target_pad[:, d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]

        if np.random.rand() < 0.5:
            img_patch = np.flip(img_patch, axis=2).copy()
            target_patch = np.flip(target_patch, axis=3).copy()
        if np.random.rand() < 0.5:
            img_patch = np.flip(img_patch, axis=1).copy()
            target_patch = np.flip(target_patch, axis=2).copy()
        if np.random.rand() < 0.5:
            img_patch = np.flip(img_patch, axis=0).copy()
            target_patch = np.flip(target_patch, axis=1).copy()

        if np.random.rand() < 0.8:
            gain = np.random.uniform(0.9, 1.1)
            bias = np.random.uniform(-0.1, 0.1)
            noise_std = np.random.uniform(0.0, 0.05)
            img_patch = img_patch * gain + bias
            img_patch = img_patch + np.random.normal(0.0, noise_std, size=img_patch.shape).astype(np.float32)

        img_patch = torch.from_numpy(img_patch).unsqueeze(0).float()
        target_patch = torch.from_numpy(target_patch).float()

        return img_patch, target_patch


class RealSSLVolumeDataset(Dataset):
    def __init__(
        self,
        real_dir: str,
        patch_size: Tuple[int, int, int] = (64, 256, 256),
        patches_per_image: int = 32,
        normalize: bool = True,
    ):
        self.real_files = sorted(Path(real_dir).glob("*.tif"))
        if len(self.real_files) == 0:
            raise ValueError(f"No real TIFF files found in {real_dir}.")

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.cropper = Random3DCrop(patch_size)
        self.normalizer = ZScoreNormalize() if normalize else None

        self.images: List[np.ndarray] = []
        for file in self.real_files:
            img = tifffile.imread(file).astype(np.float32)
            if self.normalizer is not None:
                img = self.normalizer(img)
            self.images.append(img)

    def __len__(self) -> int:
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx: int):
        img_idx = idx % len(self.images)
        img = self.images[img_idx]
        img_patch, _ = self.cropper(img)
        img_patch = torch.from_numpy(img_patch).unsqueeze(0).float()
        return img_patch


class RealInferenceDataset(Dataset):
    def __init__(self, real_dir: str, normalize: bool = True):
        self.real_files = [f"{real_dir}/{i}.tif" for i in range(20)]
        if len(self.real_files) == 0:
            raise ValueError(f"No real TIFF files found in {real_dir}.")
        self.normalizer = ZScoreNormalize() if normalize else None

    def __len__(self) -> int:
        return len(self.real_files)

    def __getitem__(self, idx: int):
        path = self.real_files[idx]
        img = tifffile.imread(path).astype(np.float32)
        raw_shape = img.shape
        if self.normalizer is not None:
            img = self.normalizer(img)
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img, str(path), raw_shape


# ============================================================
# Model
# ============================================================

class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance"):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if norm == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == "group":
            groups = min(8, out_channels)
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, out_channels)
        else:
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance"):
        super().__init__()
        self.block1 = ConvNormAct(in_channels, out_channels, norm=norm)
        self.block2 = ConvNormAct(out_channels, out_channels, norm=norm)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.block1(x)
        x = self.block2(x)
        return x + residual


class Encoder3D(nn.Module):
    def __init__(self, in_channels: int = 1, features: Sequence[int] = (32, 64, 128, 256), norm: str = "instance"):
        super().__init__()
        self.features = list(features)
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        current_in = in_channels
        for feat in self.features:
            self.blocks.append(ResidualBlock3D(current_in, feat, norm=norm))
            current_in = feat

        self.bottleneck = ResidualBlock3D(self.features[-1], self.features[-1] * 2, norm=norm)
        self.out_channels = self.features[-1] * 2

    def forward(self, x: torch.Tensor):
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        return x, skips


class Decoder3D(nn.Module):
    def __init__(self, encoder_features: Sequence[int], out_channels: int, norm: str = "instance"):
        super().__init__()
        feats = list(encoder_features)
        bottleneck_channels = feats[-1] * 2

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        current = bottleneck_channels
        for feat in reversed(feats):
            self.upconvs.append(nn.ConvTranspose3d(current, feat, kernel_size=2, stride=2))
            self.dec_blocks.append(ResidualBlock3D(feat * 2, feat, norm=norm))
            current = feat

        self.head = nn.Conv3d(feats[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        skips = list(skips)[::-1]
        for up, block, skip in zip(self.upconvs, self.dec_blocks, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return self.head(x)


class SegmentationUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, features: Sequence[int] = (32, 64, 128, 256), norm: str = "instance"):
        super().__init__()
        self.encoder = Encoder3D(in_channels=in_channels, features=features, norm=norm)
        self.decoder = Decoder3D(encoder_features=features, out_channels=out_channels, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


class SSLReconstructionModel(nn.Module):
    """Encoder + lightweight decoder for masked reconstruction pretraining."""

    def __init__(self, in_channels: int = 1, features: Sequence[int] = (32, 64, 128, 256), norm: str = "instance"):
        super().__init__()
        self.encoder = Encoder3D(in_channels=in_channels, features=features, norm=norm)
        self.decoder = Decoder3D(encoder_features=features, out_channels=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


# ============================================================
# SSL masking
# ============================================================

def random_block_mask(x: torch.Tensor, mask_ratio: float = 0.5, block_size: Tuple[int, int, int] = (8, 16, 16)):
    """Mask coarse 3D blocks for reconstruction pretraining."""
    b, c, d, h, w = x.shape
    bd, bh, bw = block_size

    gd = math.ceil(d / bd)
    gh = math.ceil(h / bh)
    gw = math.ceil(w / bw)

    grid_mask = (torch.rand((b, 1, gd, gh, gw), device=x.device) < mask_ratio).float()
    mask = F.interpolate(grid_mask, size=(d, h, w), mode="nearest")
    x_masked = x.clone()
    x_masked = x_masked * (1.0 - mask)
    return x_masked, mask


# ============================================================
# Losses and metrics
# ============================================================

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    min_pos_weight: float = 1.0,
    max_pos_weight: float = 25.0,
) -> torch.Tensor:
    with torch.no_grad():
        pos = target.sum()
        neg = target.numel() - pos
        if pos > 0:
            pos_weight = torch.clamp(neg / (pos + 1e-6), min=min_pos_weight, max=max_pos_weight)
        else:
            pos_weight = torch.tensor(min_pos_weight, device=target.device)

    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def instance_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cl_logits = logits[:, 0]
    cl_gt = target[:, 0]

    cl_bce = weighted_bce_with_logits(
        cl_logits,
        cl_gt,
        min_pos_weight=1.0,
        max_pos_weight=40.0,
    )
    cl_dice = dice_loss_from_logits(cl_logits, cl_gt)

    return 1.5 * cl_bce + 1.0 * cl_dice


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean()


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return masked_diff.sum() / denom


# ============================================================
# Training / evaluation
# ============================================================

def evaluate_segmentation(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for img, ann in dataloader:
            img = img.to(device, non_blocking=True)
            ann = ann.to(device, non_blocking=True)

            logits = model(img)
            loss = instance_loss(logits, ann)

            dice = dice_score_from_logits(logits[:, 0], ann[:, 0])

            total_loss += float(loss.item())
            total_dice += float(dice.item())
            num_batches += 1

    return (
        total_loss / max(num_batches, 1),
        total_dice / max(num_batches, 1),
    )


def pretrain_ssl(
    model: SSLReconstructionModel,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: str,
    mask_ratio: float = 0.5,
    block_size: Tuple[int, int, int] = (8, 16, 16),
    resume: bool = False,
):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_loss = float("inf")

    if resume and Path(save_path).exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Resumed SSL training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for img in dataloader:
            img = img.to(device, non_blocking=True)
            masked_img, mask = random_block_mask(img, mask_ratio=mask_ratio, block_size=block_size)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                recon = model(masked_img)
                loss = reconstruction_loss(recon, img, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(len(dataloader), 1)
        print(f"[SSL] Epoch {epoch + 1}/{epochs} - Reconstruction Loss: {avg_loss:.6f}")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, save_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"encoder": model.encoder.state_dict()}, save_path.replace(".pth", "_best_encoder.pth"))
            print(f"  -> Saved best SSL encoder to {save_path.replace('.pth', '_best_encoder.pth')}")

        print(f"  -> Saved checkpoint to {save_path}")


def train_segmentation(
    model: SegmentationUNet3D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: str,
    freeze_encoder_epochs: int = 0,
    resume: bool = False,
):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_val_loss = float("inf")

    if resume and Path(save_path).exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed segmentation training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        freeze_encoder = epoch < freeze_encoder_epochs
        for param in model.encoder.parameters():
            param.requires_grad = not freeze_encoder

        model.train()
        running_loss = 0.0

        for img, ann in train_loader:
            img = img.to(device, non_blocking=True)
            ann = ann.to(device, non_blocking=True)

            if not torch.isfinite(img).all():
                print("Non-finite img detected")
                print("img min/max:", img.min().item(), img.max().item())
                raise RuntimeError("img contains NaN/Inf")

            if not torch.isfinite(ann).all():
                print("Non-finite target detected")
                print("ann min/max:", ann.min().item(), ann.max().item())
                raise RuntimeError("ann contains NaN/Inf")

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(img)
                loss = instance_loss(logits, ann)

            if not torch.isfinite(logits).all():
                print("Non-finite logits detected")
                print("logits min/max:", logits.min().item(), logits.max().item())
                raise RuntimeError("logits contains NaN/Inf")

            if not torch.isfinite(loss):
                print("Loss is NaN/Inf")
                print("img min/max:", img.min().item(), img.max().item())
                print("ann min/max:", ann.min().item(), ann.max().item())
                print("logits min/max:", logits.min().item(), logits.max().item())
                raise RuntimeError("loss is NaN/Inf")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_dice = evaluate_segmentation(model, val_loader, device)
        scheduler.step()

        print(
            f"[SEG] Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val Dice: {val_dice:.6f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, save_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path.replace(".pth", "_best.pth"))
            print(f"  -> Saved best segmentation model to {save_path.replace('.pth', '_best.pth')}")

        print(f"  -> Saved checkpoint to {save_path}")


# ============================================================
# Sliding-window inference
# ============================================================

def compute_starts(size: int, patch: int, stride: int) -> List[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    _, d, h, w = volume.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1.0 - overlap)))
    sh = max(1, int(ph * (1.0 - overlap)))
    sw = max(1, int(pw * (1.0 - overlap)))

    d_pad = max(d, pd)
    h_pad = max(h, ph)
    w_pad = max(w, pw)

    padded = torch.zeros((1, 1, d_pad, h_pad, w_pad), dtype=volume.dtype)
    padded[:, :, :d, :h, :w] = volume.unsqueeze(0)

    #pred_sum = torch.zeros((3, d_pad, h_pad, w_pad), dtype=torch.float32)
    pred_sum = torch.zeros((1, d_pad, h_pad, w_pad), dtype=torch.float32)
    weight_sum = torch.zeros((d_pad, h_pad, w_pad), dtype=torch.float32)

    weight_patch = torch.ones((pd, ph, pw), dtype=torch.float32)

    d_starts = compute_starts(d_pad, pd, sd)
    h_starts = compute_starts(h_pad, ph, sh)
    w_starts = compute_starts(w_pad, pw, sw)

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                patch = padded[:, :, ds:ds + pd, hs:hs + ph, ws:ws + pw].to(device)
                logits = model(patch)                            # (1, 1, pd, ph, pw)
                probs = torch.sigmoid(logits).squeeze(0).cpu()  # (1, pd, ph, pw)
                pred_sum[:, ds:ds + pd, hs:hs + ph, ws:ws + pw] += probs * weight_patch.unsqueeze(0)
                weight_sum[ds:ds + pd, hs:hs + ph, ws:ws + pw] += weight_patch

    pred = pred_sum / weight_sum.unsqueeze(0).clamp_min(1e-6)
    return pred[:, :d, :h, :w].numpy()


@torch.no_grad()
def run_inference(
    model: SegmentationUNet3D,
    dataloader: DataLoader,
    checkpoint_path: str,
    device: torch.device,
    save_dir: str,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    for img, path_str, _ in dataloader:
        img = img[0]
        pred = sliding_window_inference(model, img, patch_size=patch_size, overlap=overlap, device=device)
        stem = Path(path_str[0]).stem
        out_path = Path(save_dir) / f"{stem}_prediction.tif"
        tifffile.imwrite(out_path, pred.astype(np.float32))
        print(f"Saved {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="3D SSL pretraining + segmentation fine-tuning")
    parser.add_argument("--mode", choices=["ssl_pretrain", "train_seg", "infer"], required=True)

    parser.add_argument("--syn_dir", type=str, default="/srv/exler/ci_ws_24_emsig_technical_overview/data/syn")
    parser.add_argument("--ann_dir", type=str, default="/srv/exler/ci_ws_24_emsig_technical_overview/data/annotations")
    parser.add_argument("--real_dir", type=str, default="/srv/exler/ci_ws_24_emsig_technical_overview/data/real")
    parser.add_argument("--cached_centerlines", type=str, default="targets")

    parser.add_argument("--ssl_checkpoint", type=str, default="checkpoints/ssl_encoder.pth")
    parser.add_argument("--seg_checkpoint", type=str, default="checkpoints/seg_model.pth")
    parser.add_argument("--save_dir", type=str, default="predictions/segmentation")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch_d", type=int, default=64)
    parser.add_argument("--patch_h", type=int, default=256)
    parser.add_argument("--patch_w", type=int, default=256)
    parser.add_argument("--patches_per_image", type=int, default=32)

    parser.add_argument("--norm", choices=["instance", "batch", "group"], default="instance")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=10)
    parser.add_argument("--load_ssl_encoder", action="store_true")
    parser.add_argument("--inference_overlap", type=float, default=0.5)
    parser.add_argument("--resume_ssl", action="store_true", help="Resume SSL pretraining from checkpoint")
    parser.add_argument("--resume_seg", action="store_true", help="Resume segmentation training from checkpoint")

    parser.add_argument("--crop_attempts", type=int, default=8, help="How often to retry a crop to find foreground")

    args = parser.parse_args()
    cache_dir = Path(args.cached_centerlines)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = (args.patch_d, args.patch_h, args.patch_w)
    features = (32, 64, 128, 256)
    ensure_divisible_patch(patch_size, levels=len(features))

    print(f"Using device: {device}")
    print(f"Patch size: {patch_size}")

    if args.mode == "ssl_pretrain":
        dataset = RealSSLVolumeDataset(
            real_dir=args.real_dir,
            patch_size=patch_size,
            patches_per_image=args.patches_per_image,
            normalize=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )
        model = SSLReconstructionModel(in_channels=1, features=features, norm=args.norm)
        pretrain_ssl(
            model=model,
            dataloader=loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.ssl_checkpoint,
            resume=args.resume_ssl,
        )

    elif args.mode == "train_seg":
        num_images = len(list(Path(args.syn_dir).glob("*.tif")))
        if num_images < 6:
            raise ValueError("Need at least 6 synthetic images for the current split logic.")

        val_count = min(5, max(1, num_images // 5))
        val_indices = list(range(num_images - val_count, num_images))
        train_indices = list(range(0, num_images - val_count))

        print(f"Total synthetic images: {num_images}")
        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")

        train_dataset = SyntheticSegmentationDataset(
            syn_dir=args.syn_dir,
            cache_dir=cache_dir,
            syn_prefix= "processed_transformed_syn_effects_",
            cache_prefix= "centerline_",
            patch_size=patch_size,
            image_indices=train_indices,
            patches_per_image=args.patches_per_image,
            normalize=True,
            crop_attempts=args.crop_attempts,
        )
        val_dataset = SyntheticSegmentationDataset(
            syn_dir=args.syn_dir,
            cache_dir=cache_dir,
            syn_prefix= "processed_transformed_syn_effects_",
            cache_prefix= "centerline_",
            patch_size=patch_size,
            image_indices=val_indices,
            patches_per_image=max(8, args.patches_per_image // 2),
            normalize=True,
            crop_attempts=args.crop_attempts,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        model = SegmentationUNet3D(in_channels=1, out_channels=1, features=features, norm=args.norm)

        if args.load_ssl_encoder:
            ssl_checkpoint_path = args.ssl_checkpoint.replace(".pth", "_best_encoder.pth")
            if not Path(ssl_checkpoint_path).exists():
                ssl_checkpoint_path = args.ssl_checkpoint
            ssl_state = torch.load(ssl_checkpoint_path, map_location="cpu")
            model.encoder.load_state_dict(ssl_state["encoder"], strict=True)
            print(f"Loaded SSL encoder from {ssl_checkpoint_path}")

        train_segmentation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.seg_checkpoint,
            freeze_encoder_epochs=args.freeze_encoder_epochs if args.load_ssl_encoder else 0,
            resume=args.resume_seg,
        )

    elif args.mode == "infer":
        dataset = RealInferenceDataset(real_dir=args.real_dir, normalize=True)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = SegmentationUNet3D(in_channels=1, out_channels=1, features=features, norm=args.norm)
        run_inference(
            model=model,
            dataloader=loader,
            checkpoint_path=args.seg_checkpoint,
            device=device,
            save_dir=args.save_dir,
            patch_size=patch_size,
            overlap=args.inference_overlap,
        )


if __name__ == "__main__":
    main()