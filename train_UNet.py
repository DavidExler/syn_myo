import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile
from pathlib import Path
import argparse


class SynDataset(Dataset):
    def __init__(self, syn_dir, ann_dir=None, patch_size=(64, 256, 256), inference_mode=False, image_indices=None):
        self.syn_files = sorted(list(Path(syn_dir).glob('*.tif')))
        #print(f"found files: {self.syn_files}")
        if ann_dir is not None:
            self.ann_files = sorted(list(Path(ann_dir).glob('*.tif')))
            assert len(self.syn_files) == len(self.ann_files)
        else:
            self.ann_files = None
        self.patch_size = patch_size
        self.inference_mode = inference_mode
        self.image_indices = image_indices  # For train/test split
        
        # Preload all images and annotations
        self.images = []
        self.annotations = []
        for idx, syn_file in enumerate(self.syn_files):
            # Skip images not in the indices list (for train/test split)
            if self.image_indices is not None and idx not in self.image_indices:
                continue
            img = tifffile.imread(syn_file).astype(np.float32) / 255.0
            self.images.append(img)
            if self.ann_files is not None:
                ann_file = self.ann_files[idx]
                ann = tifffile.imread(ann_file).astype(np.float32)
                ann = (ann > 0).astype(np.float32)  # Binary: 0 background, 1 foreground
                self.annotations.append(ann)
            else:
                self.annotations.append(None)  # or dummy

    def __len__(self):
        if self.inference_mode:
            return len(self.images)
        else:
            return len(self.images) * 10  # multiple patches per image

    def __getitem__(self, idx):
        if self.inference_mode:
            img = self.images[idx]
            ann = self.annotations[idx] if self.annotations[idx] is not None else np.zeros_like(img)
            img = torch.from_numpy(img).unsqueeze(0)
            ann = torch.from_numpy(ann)
            return img, ann
        else:
            img_idx = idx // 10
            img = self.images[img_idx]
            ann = self.annotations[img_idx]
            
            # Random crop
            d, h, w = img.shape
            pd, ph, pw = self.patch_size
            d_start = np.random.randint(0, d - pd + 1)
            h_start = np.random.randint(0, h - ph + 1)
            w_start = np.random.randint(0, w - pw + 1)
            
            img_patch = img[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            ann_patch = ann[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            
            img_patch = torch.from_numpy(img_patch).unsqueeze(0)
            ann_patch = torch.from_numpy(ann_patch)
            return img_patch, ann_patch

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.features = features
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)

        # Encoder
        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
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
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[i + 1](x)

        return self.final_conv(x)

def loss_function(pred, target):
    # Binary segmentation with BCEWithLogitsLoss
    bce = nn.BCEWithLogitsLoss()(pred.squeeze(1), target)
    # Dice Loss
    pred_sigmoid = torch.sigmoid(pred.squeeze(1))
    dice = 1 - (2 * (pred_sigmoid * target).sum() + 1e-8) / (pred_sigmoid.sum() + target.sum() + 1e-8)
    return bce + dice  # Combine BCE and Dice

def create_optimizer(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr=lr)

def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluate model on validation/test set and return loss and Dice score"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for img, ann in dataloader:
            img, ann = img.to(device), ann.to(device)
            pred = model(img)
            loss = loss_fn(pred, ann)
            total_loss += loss.item()
            
            # Calculate Dice score
            pred_sigmoid = torch.sigmoid(pred.squeeze(1))
            pred_binary = (pred_sigmoid > 0.5).float()
            dice = (2 * (pred_binary * ann).sum() + 1e-8) / (pred_binary.sum() + ann.sum() + 1e-8)
            total_dice += dice.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    return avg_loss, avg_dice

def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device, save_path='checkpoints/unet_model.pth'):
    model.to(device)
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, ann in train_dataloader:
            img, ann = img.to(device), ann.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, ann)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Evaluate on validation set
        val_loss, val_dice = evaluate_model(model, val_dataloader, loss_fn, device)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f'  -> Best model saved at epoch {best_epoch}')

def infer_model(model, dataloader, device, checkpoint_path, save_dir='predictions/unet_predictions', patch_size=(64, 256, 256)):
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    pd, ph, pw = patch_size
    
    with torch.no_grad():
        for i, (img, _) in enumerate(dataloader):
            img = img.squeeze(0).squeeze(0).cpu().numpy()  # shape: (D, H, W)
            D, H, W = img.shape
            img = (img / np.max(img)) * 255.0
            
            # Pad to multiples of patch_size
            D_pad = ((D + pd - 1) // pd) * pd
            H_pad = ((H + ph - 1) // ph) * ph
            W_pad = ((W + pw - 1) // pw) * pw
            
            img_padded = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)
            img_padded[:D, :H, :W] = img
            
            pred_full = np.zeros((D_pad, H_pad, W_pad), dtype=np.float32)
            
            for d in range(0, D_pad, pd):
                for h in range(0, H_pad, ph):
                    for w in range(0, W_pad, pw):
                        patch = img_padded[d:d+pd, h:h+ph, w:w+pw]
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, pd, ph, pw)
                        pred_patch = model(patch_tensor)
                        pred_patch = torch.sigmoid(pred_patch).squeeze(0).squeeze(0).cpu().numpy()  # (pd, ph, pw)
                        pred_full[d:d+pd, h:h+ph, w:w+pw] = pred_patch
            
            # Crop back to original size
            pred_cropped = pred_full[:D, :H, :W]
            
            tifffile.imwrite(os.path.join(save_dir, f'{i}_prediction.tif'), pred_cropped)
            print(f'Saved prediction {i}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Infer UNet3D model')
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], required=True, help='Mode: train or infer')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/unet_model.pth', help='Path to checkpoint for inference')
    parser.add_argument('--syn_path', type=str, default='syn', help='Path to synthetic images')
    parser.add_argument('--save_path', type=str, default='predictions/unet_predictions', help='Path to save predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model
    model = UNet3D(in_channels=1, out_channels=1)  # Binary segmentation
    syn_path = args.syn_path
    save_path = args.save_path
    if args.mode == 'train':
        # Create train/test split (5 images for test, rest for train)
        num_images = len(list(Path('syn').glob('*.tif')))
        test_indices = list(range(num_images - 5, num_images))  # Last 5 images for test
        train_indices = list(range(num_images - 5))  # First images for train
        
        print(f'Total images: {num_images}')
        print(f'Train indices: {train_indices}')
        print(f'Test indices: {test_indices}')
        
        # Create train and validation datasets
        train_dataset = SynDataset(syn_path, 'annotations', patch_size=(64, 256, 256), inference_mode=False, image_indices=train_indices)
        val_dataset = SynDataset(syn_path, 'annotations', patch_size=(64, 256, 256), inference_mode=False, image_indices=test_indices)
        
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Optimizer
        optimizer = create_optimizer(model, lr=1e-5)
        # Loss
        loss_fn = loss_function

        # Train
        train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs=150, device=device)
    elif args.mode == 'infer':
        dataset = SynDataset('real', ann_dir=None, patch_size=(64, 256, 256), inference_mode=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # Infer
        infer_model(model, dataloader, device, args.checkpoint_path, patch_size=(64, 256, 256), save_dir=save_path)