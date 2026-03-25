# 3D CycleGAN Training Script

A PyTorch-based implementation of 3D CycleGAN for unpaired medical/microscopy image-to-image translation.

## Overview

This project trains a 3D CycleGAN to translate between two domains (real ↔ synthetic) without requiring paired training data. It's designed to work with volumetric 3D images stored as TIFF files.

**Key Features:**
- ✅ Full 3D convolutional architecture (not 2D slices)
- ✅ Unpaired training (no need for corresponding image pairs)
- ✅ Cycle consistency loss for domain translation
- ✅ Identity loss to preserve image content
- ✅ Checkpoint saving and loading
- ✅ GPU acceleration with CUDA support
- ✅ Support for TIFF format input/output

## Repository Structure

```
.
├── train_3d_cyclegan.py          # Main training script
├── inference_3d_cyclegan.py      # Inference/translation script
├── requirements.txt              # Python dependencies
├── real/                         # Real domain images (your data)
├── syn/                          # Synthetic domain images (your data)
├── checkpoints/                  # Saved model checkpoints
└── logs/                         # Training logs
```

## Installation

### 1. Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

### 2. Install Dependencies

```bash
# CPU-only
pip install -r requirements.txt

# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. Prepare Data

Create TIFF files in the `real/` and `syn/` directories:

```bash
# Directory structure should look like:
# real/
#   ├── image_001.tif
#   ├── image_002.tif
#   └── ...
# syn/
#   ├── synthetic_001.tif
#   ├── synthetic_002.tif
#   └── ...
```

**Image Requirements:**
- Format: TIFF (8-bit, 16-bit, or 32-bit)
- Dimensions: Any 2D or 3D shape (will be automatically resized to 128×128×64)
- Content: Gray-scale microscopy or medical imaging data

## Usage

### Training

Basic training with default parameters:
```bash
python train_3d_cyclegan.py
```

The script will:
1. Load all TIFF images from `real/` and `syn/` folders
2. Train for 100 epochs (configurable)
3. Save checkpoints every 5 epochs in `checkpoints/` folder
4. Log training progress to `logs/` folder

**Configuration:**

Edit the `Config` class in `train_3d_cyclegan.py` to adjust:

```python
# Dataset paths
REAL_DIR = Path(__file__).parent / "real"
SYN_DIR = Path(__file__).parent / "syn"

# Training
BATCH_SIZE = 1                    # Decrease for low VRAM, increase for faster training
NUM_EPOCHS = 100                  # Number of training epochs
LEARNING_RATE = 0.0002           # Adam learning rate
LAMBDA_CYCLE = 10.0              # Cycle consistency loss weight
LAMBDA_IDENTITY = 5.0            # Identity loss weight

# Data
IMG_HEIGHT = 128                  # Input image height
IMG_WIDTH = 128                   # Input image width
IMG_DEPTH = 64                    # Input image depth (z-dimension)
NUM_WORKERS = 0                   # DataLoader workers (0 on Windows)

# Model
N_RESIDUAL_BLOCKS = 9            # Generator residual blocks
SAVE_INTERVAL = 5                # Save checkpoint every N epochs
```

### Inference

Translate images using a trained model:

```bash
# Real → Synthetic
python inference_3d_cyclegan.py checkpoints/epoch_100.pth input.tif output.tif --mode real2syn

# Synthetic → Real
python inference_3d_cyclegan.py checkpoints/epoch_100.pth input.tif output.tif --mode syn2real
```

**Arguments:**
- `checkpoint`: Path to trained model checkpoint
- `input_image`: Path to input TIFF image
- `output_image`: Path to save output TIFF image
- `--mode`: Translation direction (`real2syn` or `syn2real`, default: `real2syn`)
- `--device`: Computation device (`cuda` or `cpu`, default: auto-detect)

## Architecture Details

### Generator (3D ResNet)
- Input: 1-channel 3D volume (128×128×64)
- Architecture: Conv → Downsample → 9×ResBlock → Upsample → Conv
- Output: 1-channel translated 3D volume
- Activation: Tanh

### Discriminator (3D PatchGAN)
- Input: 1-channel 3D volume
- Architecture: 5×Conv + InstanceNorm + LeakyReLU layers
- Output: Patch-based discrimination scores
- Activation: Linear

### Loss Functions
1. **Adversarial Loss** (MSE): Genera real-looking images
2. **Cycle Consistency Loss** (L1): real → syn → real should match
3. **Identity Loss** (L1): Generator should preserve content

## Training Tips

### VRAM Requirements
- **6GB VRAM**: Batch size 1, 64×64×64 images
- **12GB VRAM**: Batch size 2, 128×128×64 images  
- **24GB+ VRAM**: Batch size 2-4, full resolution

### Hyperparameter Tuning
- **Training unstable?** Increase `LAMBDA_CYCLE` (encourages cycle consistency)
- **Poor image quality?** Increase `N_RESIDUAL_BLOCKS` (more model capacity)
- **Mode collapse?** Decrease `LEARNING_RATE` or increase `LAMBDA_IDENTITY`
- **Slow convergence?** Increase `BATCH_SIZE` or `LEARNING_RATE`

### Data Tips
1. **Dataset size**: Minimum 50 images per domain, 100+ recommended
2. **Class balance**: Keep real/syn image counts similar
3. **Diversity**: Ensure good variation in images (different angles, illuminations)
4. **Resolution**: Consistent image sizes work best

## Output Files

After training, you'll find:

```
checkpoints/
├── epoch_005.pth        # Checkpoint with generator and discriminator weights
├── epoch_010.pth
└── ...

logs/
└── training_20240101_120000.log  # Training logs with loss progression
```

### Checkpoint Contents
- Generator weights (real→syn)
- Generator weights (syn→real)
- Discriminator weights (real)
- Discriminator weights (syn)
- Optimizer states
- Training loss values

## Troubleshooting

### "No TIFF files found"
- Ensure images are in `real/` and `syn/` folders
- Check file extensions (.tif or .tiff)

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` (try 1)
- Reduce `IMG_DEPTH`, `IMG_HEIGHT`, `IMG_WIDTH`
- Use CPU instead: Set `DEVICE = torch.device("cpu")`

### Training loss diverging
- Decrease `LEARNING_RATE` (try 0.0001)
- Increase `LAMBDA_CYCLE` (try 20-50)
- Check data normalization

### Poor translation quality
- Train for more epochs
- Use more diverse training data
- Increase `N_RESIDUAL_BLOCKS` (try 15)
- Verify input images have good contrast

## References

This implementation is based on the following work:

**Primary References:**
1. CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
   - Zhu et al., ICCV 2017
   - https://github.com/junyanz/CycleGAN

2. 3D CycleGAN for Medical Imaging (davidiommi/3D-CycleGan-Pytorch-MedImaging)
   - https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging

3. Instance Normalization: The Missing Ingredient for Fast Stylization
   - Ulyanov et al., 2016

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```

## License

MIT License - Feel free to use for research and commercial purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `logs/` folder
3. Verify data format and paths
4. Check GPU/VRAM availability

---

**Last updated:** January 2024  
**Python Version:** 3.8+  
**PyTorch Version:** 2.0+
