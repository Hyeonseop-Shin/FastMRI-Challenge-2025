import torch
import numpy as np
from utils.data.augmentation import spatial_augmentation

def to_tensor(data):
    """Convert numpy array to torch tensor"""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        return torch.tensor(data)

class DataTransform:
    def __init__(self, isforward, max_key, augmentation=False, anatomy_type='knee'):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentation = augmentation
        self.anatomy_type = anatomy_type
        
        self.aug_params = {
            'flip_prop': 0.0, 
            'rotate_prop': 0.0, 
            'rotate_range': (3, 8),
            'scale_prop': 0.0, 
            'scale_range': (0.95, 1.05),
            'shift_prop': 0.0, 
            'shift_range': (3, 8),
            # New brightness/contrast parameters
            'brightness_prop': 1.0,
            'brightness_range': (0.8, 1.2),
            'contrast_prop': 0.5,
            'contrast_range': (0.8, 1.2),
            'anatomy_type': anatomy_type,
            # New k-space augmentation parameter
            'enable_kspace_intensity_aug': True
        }
    
    def set_brightness_contrast_augmentation(self, brightness_prop=1.0, contrast_prop=0.5, 
                                           brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        """
        Configure brightness and contrast augmentation parameters.
        
        Args:
            brightness_prop: Probability of applying brightness augmentation (0.0 to 1.0)
            contrast_prop: Probability of applying contrast augmentation (0.0 to 1.0)
            brightness_range: Range for brightness factor (e.g., (0.8, 1.2))
            contrast_range: Range for contrast factor (e.g., (0.8, 1.2))
        """
        self.aug_params.update({
            'brightness_prop': brightness_prop,
            'brightness_range': brightness_range,
            'contrast_prop': contrast_prop,
            'contrast_range': contrast_range
        })
    
    def enable_kspace_intensity_augmentation(self, enable=True):
        """
        Enable or disable k-space intensity augmentation.
        
        When enabled, brightness/contrast augmentation is applied via:
        1. K-space (multi-coil) → Image domain (per coil)
        2. Apply augmentation to each coil's image consistently  
        3. Image domain → K-space (per coil)
        4. RSS preserves the augmentation effects
        
        This ensures both k-space and target are consistently augmented.
        
        Args:
            enable: Whether to enable k-space intensity augmentation
        """
        self.aug_params['enable_kspace_intensity_aug'] = enable
    
    def __call__(self, mask, input, target, attrs, fname, slice):
        
        if self.augmentation:
            input, target = spatial_augmentation(kspace=input, image=target, **self.aug_params)

        if self.isforward:
            target = maximum = -1
        else:
            # maximum = np.max(target)
            maximum = attrs[self.max_key]
            target = to_tensor(target)
    
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace.float(), target, maximum, fname, slice
