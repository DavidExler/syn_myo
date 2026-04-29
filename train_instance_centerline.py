import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tifffile
from pathlib import Path
import argparse
from scipy.ndimage import distance_transform_edt, maximum_filter, binary_fill_holes
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, maximum_filter

class SynDataset(Dataset):
    def __init__(
        self,
        syn_dir,
        cache_dir=None,
        file_stem="syn_",
        cache_file_stem="target_",
        patch_size=(64, 256, 256),
        inference_mode=False,
        image_indices=None,
    ):
        self.image_indices = []
        if image_indices is not None:
            self.image_indices = sorted(image_indices)  
            print(f"Using specified image indices: {self.image_indices}")
        else:
            files = list(Path(syn_dir).glob("*.tif"))
            self.image_indices = list(range(len(files)))
            print(f"Using all image indices: {self.image_indices}")
        if file_stem == "none":
            self.syn_files = [f"{syn_dir}/{i}.tif" for i in self.image_indices]
        else:
            self.syn_files = [f"{syn_dir}/{file_stem}{i}.tif" for i in self.image_indices]
        print(f"Syn files: {self.syn_files}")
        self.patch_size = patch_size
        self.inference_mode = inference_mode

        self.images = []
        self.targets = []

        if not self.inference_mode:
            if cache_dir is None:
                raise ValueError("cache_dir must be provided for training mode.")
            self.cache_dir = Path(cache_dir)
            self.cache_files = [f"{cache_dir}/{cache_file_stem}{i}.tif" for i in self.image_indices]
            print(f"Cache files: {self.cache_files}")
            assert len(self.syn_files) == len(self.cache_files), "Number of images and cached targets must match."

        for idx, syn_file in enumerate(self.syn_files):

            img = tifffile.imread(syn_file).astype(np.float32)
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            self.images.append(img)

            if not self.inference_mode:
                cache_file = self.cache_files[idx]

                if not Path(cache_file).exists():
                    raise FileNotFoundError(
                        f"Cached target not found: {cache_file}\n"
                        f"Run precompute_and_save_target_dir(...) first."
                    )

                target = tifffile.imread(cache_file).astype(np.uint8)  # (3, D, H, W)
                if target.ndim == 4:
                    target = target[1:2]   # (1, D, H, W)
                elif target.ndim == 3:
                    target = target[None, ...]  # single-channel cache
                else:
                    raise ValueError(f"Unexpected target shape {target.shape} in {cache_file}")

                self.targets.append(target)


                self.targets.append(target)

    def __len__(self):
        if self.inference_mode:
            return len(self.images)
        else:
            return len(self.images) * 10

    def __getitem__(self, idx):
        if self.inference_mode:
            img = self.images[idx]
            #dummy = np.zeros((3, *img.shape), dtype=np.float32)
            dummy = np.zeros((1, *img.shape), dtype=np.float32)
            img = torch.from_numpy(img).unsqueeze(0).float()
            dummy = torch.from_numpy(dummy).float()
            return img, dummy

        img_idx = idx // 10
        img = self.images[img_idx]
        target = self.targets[img_idx]   # (3, D, H, W)

        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        # Random crop; retry a few times to prefer patches with foreground
        best_crop = None
        best_fg = -1

        fg_volume = target[0]  # foreground channel

        for _ in range(8):
            d_start = np.random.randint(0, d - pd + 1)
            h_start = np.random.randint(0, h - ph + 1)
            w_start = np.random.randint(0, w - pw + 1)

            fg_patch_try = fg_volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            fg_count = np.count_nonzero(fg_patch_try > 0)

            if fg_count > best_fg:
                best_fg = fg_count
                best_crop = (d_start, h_start, w_start)

            if fg_count > 0:
                break

        d_start, h_start, w_start = best_crop

        img_patch = img[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        target_patch = target[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        img_patch = torch.from_numpy(img_patch).unsqueeze(0).float()
        target_patch = torch.from_numpy(target_patch).float()

        return img_patch, target_patch


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.features = features
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.conv_block(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = skip_connections[i // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

            x = torch.cat((skip, x), dim=1)
            x = self.decoder[i + 1](x)

        return self.final_conv(x)


def soft_dice_loss_from_logits(logits, target, eps=1e-8):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + eps) / (denom + eps))
    return dice.mean()


def weighted_bce_with_logits(logits, target, min_pos_weight=1.0, max_pos_weight=25.0):
    """
    Dynamic positive weighting for sparse targets like centerlines.
    """
    with torch.no_grad():
        pos = target.sum()
        neg = target.numel() - pos
        if pos > 0:
            pos_weight = torch.clamp(neg / (pos + 1e-8), min=min_pos_weight, max=max_pos_weight)
        else:
            pos_weight = torch.tensor(min_pos_weight, device=target.device)

    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight
    )


def loss_function(logits, target):
    """
    Channels:
      0 -> foreground
      1 -> centerline
      2 -> radius on centerline
    """
    cl_logits = logits[:, 0]
    cl_gt = target[:, 0]
    cl_bce = weighted_bce_with_logits(cl_logits, cl_gt, min_pos_weight=1.0, max_pos_weight=40.0)
    cl_dice = soft_dice_loss_from_logits(cl_logits, cl_gt)


    cl_bce = weighted_bce_with_logits(
        cl_logits,
        cl_gt,
        min_pos_weight=1.0,
        max_pos_weight=40.0
    )
    cl_dice = soft_dice_loss_from_logits(cl_logits, cl_gt)

    total_loss = 1.5 * cl_bce + 1.0 * cl_dice
    return total_loss


def create_optimizer(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr=lr)


def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for img, ann in dataloader:
            img, ann = img.to(device), ann.to(device)
            pred = model(img)
            loss = loss_fn(pred, ann)
            total_loss += loss.item()

            probs = torch.sigmoid(pred[:, 0])
            binary = (probs > 0.5).float()
            gt = ann[:, 0]
            dice = (2 * (binary * gt).sum() + 1e-8) / (binary.sum() + gt.sum() + 1e-8)
            total_dice += dice.item()

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    return avg_loss, avg_dice


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    epochs,
    device,
    resume=False,
    checkpoint_path="checkpoints/instance_model_centerline_noDA.pth",
    start_epoch=0
):
    model.to(device)
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    if resume:
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for img, ann in train_dataloader:
            img, ann = img.to(device), ann.to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, ann)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        val_loss, val_dice = evaluate_model(model, val_dataloader, loss_fn, device)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Dice: {val_dice:.4f}"
        )

        torch.save(model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path.replace(".pth", "_best.pth"))
            print(f"  -> Best model saved at epoch {best_epoch}")


def infer_model(
    model,
    dataloader,
    device,
    checkpoint_path,
    save_dir="predictions/instance_centerline_predictions",
    patch_size=(64, 256, 256)
):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    pd, ph, pw = patch_size

    with torch.no_grad():
        for i, (img, _) in enumerate(dataloader):
            img = img.squeeze(0).squeeze(0).cpu().numpy()
            D, H, W = img.shape

            if img.max() > 0:
                img = img / img.max()

            D_pad = ((D + pd - 1) // pd) * pd
            H_pad = ((H + ph - 1) // ph) * ph
            W_pad = ((W + pw - 1) // pw) * pw

            img_padded = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)
            img_padded[:D, :H, :W] = img

            #pred_full = np.zeros((3, D_pad, H_pad, W_pad), dtype=np.float32)
            pred_full = np.zeros((1, D_pad, H_pad, W_pad), dtype=np.float32)

            for d in range(0, D_pad, pd):
                for h in range(0, H_pad, ph):
                    for w in range(0, W_pad, pw):
                        patch = img_padded[d:d+pd, h:h+ph, w:w+pw]
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                        pred_patch = model(patch_tensor)
                        pred_patch = torch.sigmoid(pred_patch).squeeze(0).cpu().numpy()  
                        pred_full[:, d:d+pd, h:h+ph, w:w+pw] = pred_patch[0:1]
                        
            pred_cropped = pred_full[:, :D, :H, :W]

            tifffile.imwrite(
                os.path.join(save_dir, f"{i}_prediction.tif"),
                pred_cropped.astype(np.float32)
            )
            print(f"Saved prediction {i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Infer UNet3D model")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True, help="Mode: train or infer")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint for inference or resuming training")
    parser.add_argument("--syn_path", type=str, default="syn", help="Path to synthetic images")
    parser.add_argument("--ann_path", type=str, default="annotations", help="Path to annotations")
    parser.add_argument("--cache_path", type=str, default="cached_centerlines", help="Path to cached centerlines")
    parser.add_argument("--syn_stem", type=str, default="syn_effects_", help="Stem for synthetic image files")
    parser.add_argument("--cache_stem", type=str, default="target_", help="Stem for cached target files")
    parser.add_argument("--save_path", type=str, default="predictions/instance_centerline", help="output folder for inference")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to resume training from")
    parser.add_argument("--centerline_dilation", type=int, default=1, help="Dilate skeleton target slightly to ease learning")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D(in_channels=1, out_channels=1)

    syn_path = args.syn_path
    ann_path = args.ann_path
    save_path = args.save_path
    resume = args.resume 
    cache_dir = Path(args.cache_path)
    if args.mode == "train":
        num_images = len(list(Path(syn_path).glob("*.tif")))
        test_indices = list(range(max(0, num_images - 5), num_images))
        train_indices = list(range(max(0, num_images - 5)))
        if len(train_indices) > len(test_indices):
            tmp = train_indices
            train_indices = test_indices
            test_indices = tmp

        print(f"Total images: {num_images}")
        print(f"Train indices: {train_indices}")
        print(f"Test indices: {test_indices}")

        train_dataset = SynDataset(
            syn_dir=syn_path,
            cache_dir=ann_path,
            file_stem=args.syn_stem,
            cache_file_stem=args.cache_stem,
            patch_size=(64, 256, 256),
            inference_mode=False,
            image_indices=train_indices,
        )

        val_dataset = SynDataset(
            syn_dir=syn_path,
            cache_dir=ann_path,
            file_stem=args.syn_stem,
            cache_file_stem=args.cache_stem,
            patch_size=(64, 256, 256),
            inference_mode=False,
            image_indices=test_indices,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        optimizer = create_optimizer(model, lr=1e-5)
        loss_fn = loss_function

        train_model(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            loss_fn,
            epochs=150,
            device=device,
            resume=resume,
            checkpoint_path=args.checkpoint_path,
            start_epoch=args.start_epoch
        )

    elif args.mode == "infer":
        dataset = SynDataset(
            syn_dir='real',
            cache_dir=ann_path,
            patch_size=(64, 256, 256),
            inference_mode=True,
            file_stem=args.syn_stem,
            cache_file_stem=args.cache_stem,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        infer_model(
            model, 
            dataloader,
            device,
            args.checkpoint_path,
            patch_size=(64, 256, 256),
            save_dir=save_path
        )