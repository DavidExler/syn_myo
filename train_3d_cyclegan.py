"""
3D CycleGAN Training Script for Medical/Microscopy Image Translation
Uses PyTorch and MONAI for handling 3D volumetric medical imaging data.
Loads images from 'real' and 'syn' folders for unpaired image-to-image translation.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tifffile
from pathlib import Path
import itertools
from datetime import datetime
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset paths
    DATA_ROOT = Path(__file__).parent
    REAL_DIR = DATA_ROOT / "real"
    SYN_DIR = DATA_ROOT / "syn"
    CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
    LOG_DIR = DATA_ROOT / "logs"
    
    # Create directories if they don't exist
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # Training hyperparameters
    BATCH_SIZE = 1
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999
    LAMBDA_CYCLE = 10.0  # Cycle consistency loss weight
    LAMBDA_IDENTITY = 5.0  # Identity loss weight
    
    # Data
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_DEPTH = 64  # z-dimension for 3D
    NUM_WORKERS = 0
    
    # Model
    N_RESIDUAL_BLOCKS = 9
    INPUT_CHANNELS = 1
    OUTPUT_CHANNELS = 1
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Setup logging configuration"""
    log_file = Config.LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================

class UnpairedVolumeDataset(Dataset):
    """
    Load unpaired 3D volumetric images from two directories.
    Supports TIFF files (single file per volume or multi-slice).
    """
    
    def __init__(self, real_dir, syn_dir, img_shape=(128, 128, 64)):
        self.img_shape = img_shape
        
        # Get sorted list of files
        self.real_files = sorted(glob.glob(os.path.join(real_dir, "*.tif"))) + \
                          sorted(glob.glob(os.path.join(real_dir, "*.tiff")))
        self.syn_files = sorted(glob.glob(os.path.join(syn_dir, "*.tif"))) + \
                         sorted(glob.glob(os.path.join(syn_dir, "*.tiff")))
        
        if not self.real_files or not self.syn_files:
            raise ValueError(f"No TIFF files found in {real_dir} or {syn_dir}")
    
    def __len__(self):
        return max(len(self.real_files), len(self.syn_files))
    
    def _load_and_preprocess(self, filepath):
        """Load TIFF file and preprocess to target shape"""
        try:
            # Load TIFF file (handles both single and multi-slice)
            img = tifffile.imread(filepath)
            
            # Ensure 3D shape
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            elif img.ndim == 4:
                img = img[0]  # Take first channel if 4D
            
            # Normalize to [0, 1]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)
            
            # Resize/crop to target shape
            img = self._resize_3d(img, self.img_shape)
            
            return torch.from_numpy(img).float().unsqueeze(0)  # Add channel dim
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros((1, *self.img_shape), dtype=torch.float32)
    
    def _resize_3d(self, img, target_shape):
        """Simple 3D resize by cropping or padding"""
        d, h, w = img.shape
        td, th, tw = target_shape
        
        # Center crop if larger
        if d > td:
            start = (d - td) // 2
            img = img[start:start+td, :, :]
        else:
            img = np.pad(img, ((0, td-d), (0, 0), (0, 0)), mode='constant')
        
        d, h, w = img.shape
        if h > th:
            start = (h - th) // 2
            img = img[:, start:start+th, :]
        else:
            img = np.pad(img, ((0, 0), (0, th-h), (0, 0)), mode='constant')
        
        d, h, w = img.shape
        if w > tw:
            start = (w - tw) // 2
            img = img[:, :, start:start+tw]
        else:
            img = np.pad(img, ((0, 0), (0, 0), (0, tw-w)), mode='constant')
        
        return img
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.real_files)
        syn_idx = idx % len(self.syn_files)
        
        real_img = self._load_and_preprocess(self.real_files[real_idx])
        syn_img = self._load_and_preprocess(self.syn_files[syn_idx])
        
        return {"real": real_img, "syn": syn_img}


# ============================================================================
# GENERATOR AND DISCRIMINATOR
# ============================================================================

class ResidualBlock3D(nn.Module):
    """3D Residual block for generator"""
    
    def __init__(self, in_channels):
        super(ResidualBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm3d(in_channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator3D(nn.Module):
    """3D CycleGAN Generator based on ResNet architecture"""
    
    def __init__(self, input_channels, output_channels, n_residual_blocks=9):
        super(Generator3D, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # Downsampling
        self.down1 = self._downsample_block(64, 128)
        self.down2 = self._downsample_block(128, 256)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock3D(256) for _ in range(n_residual_blocks)]
        )
        
        # Upsampling
        self.up1 = self._upsample_block(256, 128)
        self.up2 = self._upsample_block(128, 64)
        
        # Output convolution
        self.output = nn.Sequential(
            nn.Conv3d(64, output_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    @staticmethod
    def _downsample_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    @staticmethod
    def _upsample_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x


class Discriminator3D(nn.Module):
    """3D CycleGAN Discriminator (PatchGAN)"""
    
    def __init__(self, input_channels):
        super(Discriminator3D, self).__init__()
        
        self.model = nn.Sequential(
            self._conv_block(input_channels, 64, normalize=False),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
            nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)
        )
    
    @staticmethod
    def _conv_block(in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# TRAINING
# ============================================================================

class CycleGANTrainer:
    """Main training class for 3D CycleGAN"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.gen_real2syn = Generator3D(
            config.INPUT_CHANNELS, config.OUTPUT_CHANNELS, config.N_RESIDUAL_BLOCKS
        ).to(config.DEVICE)
        self.gen_syn2real = Generator3D(
            config.INPUT_CHANNELS, config.OUTPUT_CHANNELS, config.N_RESIDUAL_BLOCKS
        ).to(config.DEVICE)
        self.disc_real = Discriminator3D(config.INPUT_CHANNELS).to(config.DEVICE)
        self.disc_syn = Discriminator3D(config.INPUT_CHANNELS).to(config.DEVICE)
        
        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            itertools.chain(self.gen_real2syn.parameters(), self.gen_syn2real.parameters()),
            lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2)
        )
        self.optimizer_d = optim.Adam(
            itertools.chain(self.disc_real.parameters(), self.disc_syn.parameters()),
            lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2)
        )
        
        # Fake image buffer (for training stability)
        self.fake_real_buffer = []
        self.fake_syn_buffer = []
        self.buffer_size = 50
        
        self.logger.info(f"Models initialized on device: {config.DEVICE}")
        self.logger.info(f"Total Gen parameters: {sum(p.numel() for p in self.gen_real2syn.parameters())}")
    
    def update_fake_buffer(self, buffer, img, max_size=50):
        """Update fake image buffer for discriminator stability"""
        if len(buffer) < max_size:
            buffer.append(img.detach())
            return img
        else:
            if np.random.rand() > 0.5:
                idx = np.random.randint(0, max_size)
                temp = buffer[idx].clone()
                buffer[idx] = img.detach()
                return temp
            else:
                return img
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        gen_losses = []
        disc_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            real_img = batch["real"].to(self.config.DEVICE)
            syn_img = batch["syn"].to(self.config.DEVICE)
            
            # Generate fake images
            fake_syn = self.gen_real2syn(real_img)
            fake_real = self.gen_syn2real(syn_img)
            
            # Update discriminators
            disc_loss = self._update_discriminators(real_img, syn_img, fake_real, fake_syn)
            disc_losses.append(disc_loss)
            
            # Update generators
            gen_loss = self._update_generators(real_img, syn_img)
            gen_losses.append(gen_loss)
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.NUM_EPOCHS} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f}"
                )
        
        avg_gen_loss = np.mean(gen_losses)
        avg_disc_loss = np.mean(disc_losses)
        
        return avg_gen_loss, avg_disc_loss
    
    def _update_discriminators(self, real_img, syn_img, fake_real, fake_syn):
        """Update discriminators"""
        self.optimizer_d.zero_grad()
        
        # Discriminator for real images
        pred_real = self.disc_real(real_img)
        pred_fake_real = self.disc_real(fake_real.detach())
        loss_disc_real = self.criterion_gan(pred_real, torch.ones_like(pred_real)) + \
                         self.criterion_gan(pred_fake_real, torch.zeros_like(pred_fake_real))
        
        # Discriminator for synthetic images
        pred_syn = self.disc_syn(syn_img)
        pred_fake_syn = self.disc_syn(fake_syn.detach())
        loss_disc_syn = self.criterion_gan(pred_syn, torch.ones_like(pred_syn)) + \
                        self.criterion_gan(pred_fake_syn, torch.zeros_like(pred_fake_syn))
        
        loss_disc = loss_disc_real + loss_disc_syn
        loss_disc.backward()
        self.optimizer_d.step()
        
        return loss_disc.item()
    
    def _update_generators(self, real_img, syn_img):
        """Update generators with cycle consistency and identity losses"""
        self.optimizer_g.zero_grad()
        
        # Generate fake images
        fake_syn = self.gen_real2syn(real_img)
        fake_real = self.gen_syn2real(syn_img)
        
        # Cycle consistency
        cycled_real = self.gen_syn2real(fake_syn)
        cycled_syn = self.gen_real2syn(fake_real)
        
        loss_cycle_real = self.criterion_cycle(cycled_real, real_img)
        loss_cycle_syn = self.criterion_cycle(cycled_syn, syn_img)
        loss_cycle = loss_cycle_real + loss_cycle_syn
        
        # Identity loss (optional but helps preserve content)
        identity_real = self.gen_syn2real(real_img)
        identity_syn = self.gen_real2syn(syn_img)
        loss_identity_real = self.criterion_identity(identity_real, real_img)
        loss_identity_syn = self.criterion_identity(identity_syn, syn_img)
        loss_identity = loss_identity_real + loss_identity_syn
        
        # Adversarial loss
        pred_fake_syn = self.disc_syn(fake_syn)
        pred_fake_real = self.disc_real(fake_real)
        loss_adv_syn = self.criterion_gan(pred_fake_syn, torch.ones_like(pred_fake_syn))
        loss_adv_real = self.criterion_gan(pred_fake_real, torch.ones_like(pred_fake_real))
        loss_adv = loss_adv_syn + loss_adv_real
        
        # Total generator loss
        loss_gen = loss_adv + self.config.LAMBDA_CYCLE * loss_cycle + \
                   self.config.LAMBDA_IDENTITY * loss_identity
        
        loss_gen.backward()
        self.optimizer_g.step()
        
        return loss_gen.item()
    
    def save_checkpoint(self, epoch, loss_gen, loss_disc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'gen_real2syn': self.gen_real2syn.state_dict(),
            'gen_syn2real': self.gen_syn2real.state_dict(),
            'disc_real': self.disc_real.state_dict(),
            'disc_syn': self.disc_syn.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'loss_gen': loss_gen,
            'loss_disc': loss_disc,
        }
        
        checkpoint_path = self.config.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
        self.gen_real2syn.load_state_dict(checkpoint['gen_real2syn'])
        self.gen_syn2real.load_state_dict(checkpoint['gen_syn2real'])
        self.disc_real.load_state_dict(checkpoint['disc_real'])
        self.disc_syn.load_state_dict(checkpoint['disc_syn'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, train_loader):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.config.NUM_EPOCHS} epochs")
        
        for epoch in range(self.config.NUM_EPOCHS):
            gen_loss, disc_loss = self.train_epoch(train_loader, epoch + 1)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} - "
                f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}"
            )
            
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch + 1, gen_loss, disc_loss)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main function to run training"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*80)
    logger.info("Starting 3D CycleGAN Training")
    logger.info("="*80)
    
    # Create dataset and dataloader
    logger.info(f"Loading dataset from {Config.REAL_DIR} and {Config.SYN_DIR}")
    dataset = UnpairedVolumeDataset(
        real_dir=str(Config.REAL_DIR),
        syn_dir=str(Config.SYN_DIR),
        img_shape=(Config.IMG_DEPTH, Config.IMG_HEIGHT, Config.IMG_WIDTH),
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    train_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    
    # Initialize trainer
    trainer = CycleGANTrainer(Config)
    
    # Train
    trainer.train(train_loader)
    
    logger.info("="*80)
    logger.info("Training completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
