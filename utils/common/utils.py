"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from skimage.metrics import structural_similarity
import h5py
import numpy as np
import torch
import random

def save_reconstructions(reconstructions, out_dir, targets=None, inputs=None):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])

def ssim_loss(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)
       ssim_loss is defined as (1 - ssim)
    """
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]
    return 1 - ssim

def seed_fix(n):
    """
    Fix random seed for reproducibility across all random number generators
    """
    import os
    
    # Set seeds for all random number generators
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    np.random.seed(n)
    random.seed(n)
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms with warn_only to avoid breaking existing operations
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")
        # Fallback to more conservative settings
        try:
            torch.use_deterministic_algorithms(False)
            # Just ensure the basic deterministic settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except:
            pass
    
    print(f"Random seed fixed to {n} for reproducibility")

def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader to ensure reproducibility
    when using multiple workers
    """
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


def center_crop(data, height, width):
    """
    Center crop the data to the specified shape and move it to the specified device.
    
    Parameters:
    - data: numpy array of shape (C, H, W)
    - height: desired output height
    - width: desired output width
    - device: 'cpu' or 'cuda'
    
    Returns:
    - cropped tensor on the specified device
    """
    # Convert to PyTorch tensor and move to device
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.to(device)

    _, h, w = data.shape

    # Height padding
    if h < height:
        pad_h1 = (height - h) // 2
        pad_h2 = (height - h) - pad_h1
        data = torch.nn.functional.pad(data, (0, 0, pad_h1, pad_h2), mode='constant', value=0)
        h = height

    # Width padding
    if w < width:
        pad_w1 = (width - w) // 2
        pad_w2 = (width - w) - pad_w1
        data = torch.nn.functional.pad(data, (pad_w1, pad_w2, 0, 0), mode='constant', value=0)
        w = width

    start_h = (h - height) // 2
    start_w = (w - width) // 2
    return data[:, start_h:start_h + height, start_w:start_w + width]
