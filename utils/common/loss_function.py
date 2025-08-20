"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import math


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class SSIM_L1_Loss(nn.Module):
    """
    SSIM + L1 loss module.
    """

    def __init__(self, alpha=0.8, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            alpha: SSIM loss 비중 (0~1), L1 loss 비중은 1-alpha
        """
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIMLoss(win_size=win_size, k1=k1, k2=k2)

    def forward(self, X, Y, data_range):
        ssim_loss = self.ssim(X, Y, data_range)
        l1_loss = torch.mean(torch.abs(X - Y))
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


def create_anatomical_mask(target, anatomy_type, threshold_brain=5e-5, threshold_knee=2e-5):
    """
    Create anatomical mask that matches FastMRI leaderboard evaluation strategy.
    
    Args:
        target: Target image tensor [B, H, W] or [H, W]
        anatomy_type: 'brain' or 'knee'
        threshold_brain: Threshold for brain tissue
        threshold_knee: Threshold for knee tissue
    
    Returns:
        mask: Binary mask tensor matching target shape
    """
    # Handle batch dimension
    if len(target.shape) == 3:
        batch_size = target.shape[0]
        masks = []
        for i in range(batch_size):
            single_mask = create_anatomical_mask(target[i], anatomy_type, threshold_brain, threshold_knee)
            masks.append(single_mask)
        return torch.stack(masks, dim=0)
    
    # Single image processing
    target_np = target.detach().cpu().numpy()
    
    # Apply threshold based on anatomy type
    threshold = threshold_brain if anatomy_type == 'brain' else threshold_knee
    mask = np.zeros_like(target_np)
    mask[target_np > threshold] = 1
    
    # Morphological operations to match evaluation
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    
    # Convert back to tensor
    return torch.from_numpy(mask.astype(np.float32)).to(target.device)


class AnatomicalSSIMLoss(nn.Module):
    """
    SSIM loss with anatomical masking - matches leaderboard evaluation
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, 
                 anatomy_type='brain', threshold_brain=5e-5, threshold_knee=2e-5):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            anatomy_type: 'brain' or 'knee'
            threshold_brain: Brain tissue threshold (matches evaluation)
            threshold_knee: Knee tissue threshold (matches evaluation)
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.anatomy_type = anatomy_type
        self.threshold_brain = threshold_brain
        self.threshold_knee = threshold_knee
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        # Create anatomical masks
        masks = create_anatomical_mask(Y, self.anatomy_type, self.threshold_brain, self.threshold_knee)
        
        # Apply masks to both prediction and target
        X_masked = X * masks
        Y_masked = Y * masks
        
        # Calculate SSIM on masked regions
        X_masked = X_masked.unsqueeze(1)
        Y_masked = Y_masked.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X_masked, self.w)
        uy = F.conv2d(Y_masked, self.w)
        uxx = F.conv2d(X_masked * X_masked, self.w)
        uyy = F.conv2d(Y_masked * Y_masked, self.w)
        uxy = F.conv2d(X_masked * Y_masked, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class AnatomicalSSIM_L1_Loss(nn.Module):
    """
    Anatomical SSIM + L1 loss module that matches evaluation strategy
    """

    def __init__(self, alpha=0.9, win_size: int = 7, k1: float = 0.01, k2: float = 0.03,
                 anatomy_type='brain', threshold_brain=5e-5, threshold_knee=2e-5):
        """
        Args:
            alpha: SSIM loss weight (0~1), L1 loss weight is 1-alpha
            anatomy_type: 'brain' or 'knee'
        """
        super().__init__()
        self.alpha = alpha
        self.anatomy_type = anatomy_type
        self.threshold_brain = threshold_brain
        self.threshold_knee = threshold_knee
        self.ssim = AnatomicalSSIMLoss(win_size=win_size, k1=k1, k2=k2, 
                                       anatomy_type=anatomy_type,
                                       threshold_brain=threshold_brain,
                                       threshold_knee=threshold_knee)

    def forward(self, X, Y, data_range):
        # Get anatomical masks
        masks = create_anatomical_mask(Y, self.anatomy_type, self.threshold_brain, self.threshold_knee)
        
        # Masked SSIM loss
        ssim_loss = self.ssim(X, Y, data_range)
        
        # Masked L1 loss
        X_masked = X * masks
        Y_masked = Y * masks
        l1_loss = torch.mean(torch.abs(X_masked - Y_masked))
        
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


class IndexBasedWeightedLoss(nn.Module):
    """
    Base class for index-based weighted loss functions.
    
    Applies weighting based on slice index following y = cos^2(x) where x = current_slice/max_slice * pi/2.
    This accounts for the fact that middle slices typically have larger masks and more complex structures,
    while edge slices have smaller masks and simpler shapes (easier to achieve high SSIM).
    """
    
    def __init__(self):
        super().__init__()
        self.weight_list = []
    
    def set_weight(self, num_slices):
        """
        Set weights for each slice based on cos^2 function.
        
        Args:
            num_slices (int): Total number of slices in the volume
        """
        self.weight_list = []
        for i in range(num_slices):
            # Weight proportional to mask size, which follows cos^2 pattern
            weight = math.cos(i * math.pi / (num_slices * 2)) ** 2
            self.weight_list.append(weight)
    
    def get_slice_weight(self, slice_idx, num_slices):
        """
        Get weight for a specific slice index.
        
        Args:
            slice_idx (int): Current slice index
            num_slices (int): Total number of slices
            
        Returns:
            float: Weight for the slice
        """
        if len(self.weight_list) != num_slices:
            self.set_weight(num_slices)
        
        if slice_idx < len(self.weight_list):
            return self.weight_list[slice_idx]
        else:
            # Fallback calculation if index is out of range
            return math.cos(slice_idx * math.pi / (num_slices * 2)) ** 2


class AreaBasedAnatomicalSSIMLoss(nn.Module):
    """
    Area-based weighted Anatomical SSIM loss.
    
    Instead of assuming cos^2 distribution, this calculates the actual anatomical
    area for each slice and weights the loss proportionally to the area size.
    
    This is much more precise and adaptive to individual anatomy variations.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, 
                 anatomy_type='brain', threshold_brain=5e-5, threshold_knee=2e-5):
        super().__init__()
        self.anatomical_ssim = AnatomicalSSIMLoss(
            win_size=win_size, k1=k1, k2=k2,
            anatomy_type=anatomy_type,
            threshold_brain=threshold_brain,
            threshold_knee=threshold_knee
        )
        self.anatomy_type = anatomy_type
        self.threshold_brain = threshold_brain
        self.threshold_knee = threshold_knee

    def calculate_anatomical_area(self, target):
        """
        Calculate the actual anatomical area for each slice in the batch.
        
        Args:
            target (torch.Tensor): Ground truth target [B, H, W]
            
        Returns:
            torch.Tensor: Anatomical area for each slice [B]
        """
        # Create anatomical masks
        masks = create_anatomical_mask(target, self.anatomy_type, 
                                     self.threshold_brain, self.threshold_knee)
        
        # Calculate area (number of pixels) for each slice
        batch_size = masks.shape[0]
        areas = []
        
        for i in range(batch_size):
            area = torch.sum(masks[i])  # Count pixels in anatomical region
            areas.append(area)
        
        return torch.stack(areas)

    def forward(self, prediction, target, data_range, normalize_weights=True):
        """
        Forward pass with area-based weighting.
        
        Args:
            prediction (torch.Tensor): Model prediction [B, H, W]
            target (torch.Tensor): Ground truth target [B, H, W]
            data_range (torch.Tensor): Data range for normalization [B]
            normalize_weights (bool): Whether to normalize weights to sum to 1
            
        Returns:
            torch.Tensor: Area-weighted anatomical SSIM loss
        """
        batch_size = prediction.shape[0]
        
        # Calculate actual anatomical areas for each slice
        areas = self.calculate_anatomical_area(target)
        
        # Convert to weights (area intensity) - pure area-based weighting
        weights = areas.float()
        
        # Normalize weights if requested (so they sum to batch_size)
        if normalize_weights and torch.sum(weights) > 0:
            weights = weights / torch.sum(weights) * batch_size
        
        # Handle edge case where all areas are zero (keep equal weights)
        if torch.sum(weights) == 0:
            weights = torch.ones_like(weights)
        
        # Calculate individual losses and apply pure area-based weighting
        weighted_losses = []
        
        for i in range(batch_size):
            # Compute loss for every slice (even if area=0, weight=0)
            single_pred = prediction[i:i+1]
            single_target = target[i:i+1]
            single_range = data_range[i:i+1]
            
            item_loss = self.anatomical_ssim(single_pred, single_target, single_range)
            weighted_loss = item_loss * weights[i]
            weighted_losses.append(weighted_loss)
        
        # Return mean of area-weighted losses (pure accuracy)
        return torch.mean(torch.stack(weighted_losses))


class AreaBasedAnatomicalSSIM_L1_Loss(nn.Module):
    """
    Area-based weighted Anatomical SSIM + L1 loss.
    
    Uses actual calculated anatomical areas instead of assumed cos^2 distribution.
    Much more adaptive and precise for individual anatomy variations.
    """

    def __init__(self, alpha=0.8, win_size: int = 7, k1: float = 0.01, k2: float = 0.03,
                 anatomy_type='brain', threshold_brain=5e-5, threshold_knee=2e-5):
        super().__init__()
        self.alpha = alpha
        self.anatomy_type = anatomy_type
        self.threshold_brain = threshold_brain
        self.threshold_knee = threshold_knee
        self.anatomical_ssim_l1 = AnatomicalSSIM_L1_Loss(
            alpha=alpha, win_size=win_size, k1=k1, k2=k2,
            anatomy_type=anatomy_type,
            threshold_brain=threshold_brain,
            threshold_knee=threshold_knee
        )

    def calculate_anatomical_area(self, target):
        """
        Calculate the actual anatomical area for each slice in the batch.
        
        Args:
            target (torch.Tensor): Ground truth target [B, H, W]
            
        Returns:
            torch.Tensor: Anatomical area for each slice [B]
        """
        # Create anatomical masks
        masks = create_anatomical_mask(target, self.anatomy_type, 
                                     self.threshold_brain, self.threshold_knee)
        
        # Calculate area (number of pixels) for each slice
        batch_size = masks.shape[0]
        areas = []
        
        for i in range(batch_size):
            area = torch.sum(masks[i])  # Count pixels in anatomical region
            areas.append(area)
        
        return torch.stack(areas)

    def forward(self, prediction, target, data_range, normalize_weights=True):
        """
        Forward pass with area-based weighting.
        
        Args:
            prediction (torch.Tensor): Model prediction [B, H, W]
            target (torch.Tensor): Ground truth target [B, H, W]
            data_range (torch.Tensor): Data range for normalization [B]
            normalize_weights (bool): Whether to normalize weights to sum to 1
            
        Returns:
            torch.Tensor: Area-weighted anatomical SSIM+L1 loss
        """
        batch_size = prediction.shape[0]
        
        # Calculate actual anatomical areas for each slice
        areas = self.calculate_anatomical_area(target)
        
        # Convert to weights (area intensity) - pure area-based weighting
        weights = areas.float()
        
        # Normalize weights if requested (so they sum to batch_size)
        if normalize_weights and torch.sum(weights) > 0:
            weights = weights / torch.sum(weights) * batch_size
        
        # Handle edge case where all areas are zero (keep equal weights)
        if torch.sum(weights) == 0:
            weights = torch.ones_like(weights)
        
        # Calculate individual losses and apply pure area-based weighting
        weighted_losses = []
        
        for i in range(batch_size):
            # Compute loss for every slice (even if area=0, weight=0)
            single_pred = prediction[i:i+1]
            single_target = target[i:i+1]
            single_range = data_range[i:i+1]
            
            item_loss = self.anatomical_ssim_l1(single_pred, single_target, single_range)
            weighted_loss = item_loss * weights[i]
            weighted_losses.append(weighted_loss)
        
        # Return mean of area-weighted losses (pure accuracy)
        return torch.mean(torch.stack(weighted_losses))

class IndexBasedAnatomicalSSIMLoss(IndexBasedWeightedLoss):
    """
    Index-based weighted Anatomical SSIM loss.
    
    Combines anatomical masking with slice index-based weighting:
    Loss = SSIM_loss * cos^2(current_slice_index / max_slice_index * pi/2)
    
    NOTE: This is the old approach - consider using AreaBasedAnatomicalSSIMLoss instead
    for more precise, data-driven weighting.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, 
                 anatomy_type='brain', threshold_brain=5e-5, threshold_knee=2e-5):
        super().__init__()
        self.anatomical_ssim = AnatomicalSSIMLoss(
            win_size=win_size, k1=k1, k2=k2,
            anatomy_type=anatomy_type,
            threshold_brain=threshold_brain,
            threshold_knee=threshold_knee
        )

    def forward(self, prediction, target, data_range, slice_indices=None, num_slices=None):
        """
        Forward pass with index-based weighting.
        
        Args:
            prediction (torch.Tensor): Model prediction [B, H, W]
            target (torch.Tensor): Ground truth target [B, H, W]
            data_range (torch.Tensor): Data range for normalization [B]
            slice_indices (torch.Tensor, optional): Slice indices for each item in batch [B]
            num_slices (int, optional): Total number of slices in volume
            
        Returns:
            torch.Tensor: Index-weighted anatomical SSIM loss
        """
        # Compute base anatomical SSIM loss
        base_loss = self.anatomical_ssim(prediction, target, data_range)
        
        # If no slice information provided, return unweighted loss
        if slice_indices is None or num_slices is None:
            return base_loss
        
        # Apply index-based weighting
        batch_size = prediction.shape[0]
        weighted_losses = []
        
        for i in range(batch_size):
            slice_idx = slice_indices[i].item() if torch.is_tensor(slice_indices[i]) else slice_indices[i]
            weight = self.get_slice_weight(slice_idx, num_slices)
            
            # Compute loss for individual item
            single_pred = prediction[i:i+1]
            single_target = target[i:i+1] 
            single_range = data_range[i:i+1]
            
            item_loss = self.anatomical_ssim(single_pred, single_target, single_range)
            weighted_loss = item_loss * weight
            weighted_losses.append(weighted_loss)
        
        # Return mean of weighted losses
        return torch.mean(torch.stack(weighted_losses))


class IndexBasedAnatomicalSSIM_L1_Loss(IndexBasedWeightedLoss):
    """
    Index-based weighted Anatomical SSIM + L1 loss.
    
    Combines anatomical masking, SSIM+L1 loss, and slice index-based weighting.
    """

    def __init__(self, alpha=0.8, win_size: int = 7, k1: float = 0.01, k2: float = 0.03,
                 anatomy_type='brain', threshold_brain=5e-5, threshold_knee=2e-5):
        super().__init__()
        self.alpha = alpha
        self.anatomical_ssim_l1 = AnatomicalSSIM_L1_Loss(
            alpha=alpha, win_size=win_size, k1=k1, k2=k2,
            anatomy_type=anatomy_type,
            threshold_brain=threshold_brain,
            threshold_knee=threshold_knee
        )

    def forward(self, prediction, target, data_range, slice_indices=None, num_slices=None):
        """
        Forward pass with index-based weighting.
        
        Args:
            prediction (torch.Tensor): Model prediction [B, H, W]
            target (torch.Tensor): Ground truth target [B, H, W]
            data_range (torch.Tensor): Data range for normalization [B]
            slice_indices (torch.Tensor, optional): Slice indices for each item in batch [B]
            num_slices (int, optional): Total number of slices in volume
            
        Returns:
            torch.Tensor: Index-weighted anatomical SSIM+L1 loss
        """
        # Compute base anatomical SSIM+L1 loss
        base_loss = self.anatomical_ssim_l1(prediction, target, data_range)
        
        # If no slice information provided, return unweighted loss
        if slice_indices is None or num_slices is None:
            return base_loss
        
        # Apply index-based weighting
        batch_size = prediction.shape[0]
        weighted_losses = []
        
        for i in range(batch_size):
            slice_idx = slice_indices[i].item() if torch.is_tensor(slice_indices[i]) else slice_indices[i]
            weight = self.get_slice_weight(slice_idx, num_slices)
            
            # Compute loss for individual item
            single_pred = prediction[i:i+1]
            single_target = target[i:i+1]
            single_range = data_range[i:i+1]
            
            item_loss = self.anatomical_ssim_l1(single_pred, single_target, single_range)
            weighted_loss = item_loss * weight
            weighted_losses.append(weighted_loss)
        
        # Return mean of weighted losses
        return torch.mean(torch.stack(weighted_losses))
    




# ---------------------- Sobel edges (for optional edge loss) ----------------------
def sobel_edges(x):
    # x: (N,1,H,W)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ex = F.conv2d(x, kx, padding=1)
    ey = F.conv2d(x, ky, padding=1)
    return torch.sqrt(ex * ex + ey * ey + 1e-12)


# ---------------------- Composite loss ----------------------
class SobelLoss(nn.Module):
    """
    L = λ1 * L1 + λ2 * (1 - SSIM) + λ3 * Fourier_HF + λ4 * Edge(Sobel)
    Accepts (N,1,H,W) magnitudes OR (N,2,H,W) complex-as-channels (will convert to magnitude).
    """
    def __init__(self,
                 weight_l1=0.1,
                 weight_ssim=0.7,
                 weight_fourier=0.0,
                 weight_edge=0.2,
                 ssim_win=7,
                 hp_ratio=0.15,
                 gamma=2.0,
                 anatomy_type='brain'):
        super().__init__()
        self.ssim = AnatomicalSSIMLoss(win_size=ssim_win, anatomy_type=anatomy_type)
        self.weight_l1 = weight_l1
        self.weight_ssim = weight_ssim
        self.weight_fourier = weight_fourier
        self.weight_edge = weight_edge
        self.hp_ratio = hp_ratio
        self.gamma = gamma

    @staticmethod
    def _to_mag(t):
        # t: (N,1,H,W) or (N,2,H,W)
        if t.dim() != 4:
            raise ValueError("Expected NCHW")
        if t.size(1) == 1:
            return t
        elif t.size(1) == 2:
            mag = torch.sqrt(t[:,0:1]**2 + t[:,1:2]**2 + 1e-12)
            return mag
        else:
            raise ValueError("Channel must be 1 (mag) or 2 (complex).")

    def forward(self, pred, target, data_range=None):
        """
        pred, target: (N,1,H,W) magnitude OR (N,2,H,W) complex-as-channels
        data_range: scalar or (N,1,1,1); if None, computed from target
        """

        x = pred
        y = target

        # Main terms
        l1 = F.l1_loss(x, y)
        ssim_loss = self.ssim(x, y, data_range=data_range)

        # Optional edge term
        edge = F.l1_loss(sobel_edges(x), sobel_edges(y))

        return self.weight_l1 * l1 + self.weight_ssim * ssim_loss + self.weight_edge * edge
