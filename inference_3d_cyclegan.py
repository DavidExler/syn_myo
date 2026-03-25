"""
3D CycleGAN Inference Script
Translate images using trained generators
"""

import torch
import numpy as np
import tifffile
from pathlib import Path
import argparse
from train_3d_cyclegan import Generator3D, Config


class CycleGANInference:
    """Inference class for applying trained CycleGAN models"""
    
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to trained checkpoint
            device: torch device (cuda or cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create generators
        self.gen_real2syn = Generator3D(
            Config.INPUT_CHANNELS, Config.OUTPUT_CHANNELS, Config.N_RESIDUAL_BLOCKS
        ).to(self.device)
        self.gen_syn2real = Generator3D(
            Config.INPUT_CHANNELS, Config.OUTPUT_CHANNELS, Config.N_RESIDUAL_BLOCKS
        ).to(self.device)
        
        # Load weights
        self.gen_real2syn.load_state_dict(checkpoint['gen_real2syn'])
        self.gen_syn2real.load_state_dict(checkpoint['gen_syn2real'])
        
        # Set to eval mode
        self.gen_real2syn.eval()
        self.gen_syn2real.eval()
        
        print(f"Model loaded from {checkpoint_path}")
    
    def _preprocess_image(self, img_path, target_shape=(64, 128, 128)):
        """Load and preprocess image"""
        # Load image
        img = tifffile.imread(img_path)
        
        # Ensure 3D
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 4:
            img = img[0]
        
        # Normalize
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        # Resize
        d, h, w = img.shape
        td, th, tw = target_shape
        
        # Center crop or pad
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
        
        # Add batch and channel dims
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        return tensor
    
    def _postprocess_image(self, tensor):
        """Convert model output back to numpy"""
        # Remove batch and channel dims
        img = tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
        # Clip to valid range
        img = np.clip(img, -1, 1)
        # Scale to [0, 1]
        img = (img + 1) / 2
        # Scale to uint8 or uint16
        img = (img * 65535).astype(np.uint16)
        return img
    
    def translate_real_to_syn(self, input_path, output_path):
        """Translate from real to synthetic domain"""
        print(f"Translating {input_path} (real → syn)...")
        
        img_tensor = self._preprocess_image(input_path).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.gen_real2syn(img_tensor)
        
        output_img = self._postprocess_image(output_tensor)
        
        tifffile.imwrite(output_path, output_img)
        print(f"Saved to {output_path}")
    
    def translate_syn_to_real(self, input_path, output_path):
        """Translate from synthetic to real domain"""
        print(f"Translating {input_path} (syn → real)...")
        
        img_tensor = self._preprocess_image(input_path).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.gen_syn2real(img_tensor)
        
        output_img = self._postprocess_image(output_tensor)
        
        tifffile.imwrite(output_path, output_img)
        print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="3D CycleGAN Inference")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("input_image", type=str, help="Input image path")
    parser.add_argument("output_image", type=str, help="Output image path")
    parser.add_argument("--mode", type=str, choices=["real2syn", "syn2real"], 
                       default="real2syn", help="Translation mode")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = CycleGANInference(args.checkpoint, device=torch.device(args.device))
    
    # Translate
    if args.mode == "real2syn":
        inference.translate_real_to_syn(args.input_image, args.output_image)
    else:
        inference.translate_syn_to_real(args.input_image, args.output_image)


if __name__ == "__main__":
    main()
