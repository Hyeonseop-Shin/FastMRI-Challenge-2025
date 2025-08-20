
import numpy as np
from typing import Union, Tuple
from skimage.transform import rotate, AffineTransform, warp, rescale


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered IFFT2 (complex output)"""
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

def fft2c(image: np.ndarray) -> np.ndarray:
    """Centered FFT2 (complex input)"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def rotate_kspace(kspace: np.ndarray, angle: float) -> np.ndarray:
    image = ifft2c(kspace)

    real_part = np.stack([rotate(img.real, angle, resize=False, mode='edge', preserve_range=True) for img in image])
    imag_part = np.stack([rotate(img.imag, angle, resize=False, mode='edge', preserve_range=True) for img in image])

    rotated_image = real_part + 1j * imag_part
    return fft2c(rotated_image)


def shift_kspace(kspace: np.ndarray, shift: Union[int, Tuple[int,int]]) -> np.ndarray:
    if isinstance(shift, int):
        shift = (shift, shift)

    dx, dy = shift
    C, H, W = kspace.shape

    yy, xx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')  # shape (H, W)
    phase = np.exp(-2j * np.pi * (-dy * yy + dx * xx))  # shape (H, W)

    return kspace * phase[None, :, :]


def scale_kspace(kspace: np.ndarray, factor: Union[float, Tuple[float,float]]) -> np.ndarray:
    if isinstance(factor, (int, float)):
        factor = (factor, factor)
    image = ifft2c(kspace)
    image_scaled = []

    for img in image:
        real_scaled = rescale(img.real, scale=factor, mode='edge', anti_aliasing=True, preserve_range=True).T
        imag_scaled = rescale(img.imag, scale=factor, mode='edge', anti_aliasing=True, preserve_range=True).T

        real_cropped = center_crop(real_scaled, img.shape)
        imag_cropped = center_crop(imag_scaled, img.shape)

        image_scaled.append(real_cropped + 1j * imag_cropped)

    return np.transpose(fft2c(np.stack(image_scaled)), (0, 2, 1))


def flip_kspace(kspace: np.ndarray) -> np.ndarray:
    return np.conj(np.flip(kspace, axis=-2).copy())


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    return rotate(image, angle, resize=False, mode='edge', preserve_range=True)


def shift_image(image: np.ndarray, shift: Union[int, Tuple[int,int]]) -> np.ndarray:
    if isinstance(shift, int):
        shift = (-shift, shift)
    tform = AffineTransform(translation=shift)

    return warp(image, tform, mode='edge', preserve_range=True)


def scale_image(image: np.ndarray, factor: Union[float, Tuple[float,float]]) -> np.ndarray:
    if isinstance(factor, float):
        factor = (factor, factor)

    scaled = rescale(image, scale=factor, mode='edge', anti_aliasing=True, preserve_range=True)
    return center_crop(scaled, crop_size=image.shape)

def flip_image(image: np.ndarray) -> np.ndarray:
    return np.fliplr(image).copy()


def center_crop(image: np.ndarray, crop_size: Union[int, Tuple[int,int]]):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    H, W = image.shape
    target_w, target_h = crop_size

    start_y = max((H - target_h) // 2, 0)
    end_y = start_y + min(H, target_h)
    start_x = max((W - target_w) // 2, 0)
    end_x = start_x + min(W, target_w)

    cropped = image[start_y:end_y, start_x:end_x]

    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]

    if pad_h > 0 or pad_w > 0:
        pad_top = max(pad_h // 2, 0)
        pad_bottom = max(pad_h - pad_top, 0)
        pad_left = max(pad_w // 2, 0)
        pad_right = max(pad_w - pad_left, 0)

        cropped = np.pad(cropped, pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=0)

    return cropped


def rss(kspace: np.ndarray) -> np.ndarray:
    """
    Convert multi-coil k-space to image using Root-Sum-of-Squares (RSS).
    kspace shape: [coil, height, width]
    """
    image_per_coil = ifft2c(kspace)  # shape: [coil, H, W]
    rss_image = np.sqrt(np.sum(np.abs(image_per_coil) ** 2, axis=0))  # shape: [H, W]
    return rss_image


def kspace_intensity_augmentation(
        kspace: np.ndarray,
        brightness_factor: float = 1.0,
        contrast_factor: float = 1.0,
        anatomy_type: str = 'brain'
    ) -> np.ndarray:
    """
    Apply brightness/contrast augmentation to k-space via multi-coil image domain.
    
    Idea:
    1. K-space (multi-coil) → Image domain (per coil)
    2. Apply brightness/contrast to each coil's image consistently
    3. Image domain → K-space (per coil)
    4. RSS preserves the augmentation effects
    
    Args:
        kspace: Multi-coil k-space data, shape [coil, height, width]
        brightness_factor: Brightness multiplication factor
        contrast_factor: Contrast multiplication factor
        anatomy_type: For anatomical mask creation
    
    Returns:
        Augmented k-space with same shape as input
    """
    if brightness_factor == 1.0 and contrast_factor == 1.0:
        return kspace  # No augmentation needed
    
    # Step 1: Convert k-space to image domain for each coil
    coil_images = ifft2c(kspace)  # shape: [coil, H, W], complex values
    
    # Step 2: Convert to magnitude images for augmentation
    coil_magnitudes = np.abs(coil_images)  # shape: [coil, H, W], real values
    coil_phases = np.angle(coil_images)    # shape: [coil, H, W], preserve phase
    
    # Step 3: Create RSS image to get anatomical mask
    rss_image = np.sqrt(np.sum(coil_magnitudes ** 2, axis=0))  # shape: [H, W]
    
    # Step 4: Create anatomical mask (same logic as brightness_contrast_augmentation)
    mask = np.zeros(rss_image.shape)
    if anatomy_type == 'knee':
        mask[rss_image > 2e-5] = 1
    elif anatomy_type == 'brain':
        mask[rss_image > 5e-5] = 1
    else:
        mask[rss_image > 5e-5] = 1
    
    # Apply morphological operations
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    
    # Step 5: Apply augmentation to each coil's magnitude consistently
    augmented_magnitudes = []
    
    for coil_idx in range(coil_magnitudes.shape[0]):
        coil_mag = coil_magnitudes[coil_idx]  # shape: [H, W]
        
        # Apply brightness augmentation to anatomical area
        if brightness_factor != 1.0:
            masked_area = coil_mag * mask
            brightened_area = masked_area * brightness_factor
            coil_mag = coil_mag * (1 - mask) + brightened_area
        
        # Apply contrast augmentation to anatomical area
        if contrast_factor != 1.0:
            masked_area = coil_mag * mask
            if mask.sum() > 0:  # Avoid division by zero
                mean_val = np.mean(masked_area[mask > 0])
                contrasted_area = (masked_area - mean_val * mask) * contrast_factor + mean_val * mask
                coil_mag = coil_mag * (1 - mask) + contrasted_area
        
        # Ensure non-negative values
        coil_mag = np.maximum(coil_mag, 0)
        augmented_magnitudes.append(coil_mag)
    
    augmented_magnitudes = np.stack(augmented_magnitudes, axis=0)  # shape: [coil, H, W]
    
    # Step 6: Reconstruct complex images with original phases
    augmented_coil_images = augmented_magnitudes * np.exp(1j * coil_phases)
    
    # Step 7: Convert back to k-space
    augmented_kspace = fft2c(augmented_coil_images)
    
    return augmented_kspace

def brightness_contrast_augmentation(
        image: np.ndarray,
        anatomy_type: str = 'brain',
        brightness_prop: float = 0.1,
        contrast_prop: float = 0.1,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
    """
    Apply brightness and contrast augmentation ON the anatomical mask area.
    The mask area gets augmented, but the mask boundaries stay the same.
    
    Args:
        image: Input image array
        anatomy_type: 'brain' or 'knee' for different mask thresholds
        brightness_prop: Probability for brightness augmentation (0.0 to 1.0)
        contrast_prop: Probability for contrast augmentation (0.0 to 1.0)
        brightness_range: Range for brightness factor
        contrast_range: Range for contrast factor
    
    Returns:
        Augmented image with brightness/contrast applied to anatomical areas
    """
    if brightness_prop == 0.0 and contrast_prop == 0.0:
        return image
    
    # Create anatomical mask (same logic as EvalMRI.py)
    mask = np.zeros(image.shape)
    if anatomy_type == 'knee':
        mask[image > 2e-5] = 1
    elif anatomy_type == 'brain':
        mask[image > 5e-5] = 1
    else:
        # Default to brain threshold
        mask[image > 5e-5] = 1
    
    # Apply morphological operations (matching EvalMRI.py)
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    
    # Start with original image
    augmented_image = image.copy()
    
    # Apply brightness augmentation to the masked area
    if brightness_prop > 0.0 and np.random.random() < brightness_prop:
        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        # Apply brightness only to the anatomical area
        masked_area = image * mask
        brightened_area = masked_area * brightness_factor
        # Keep non-masked areas unchanged, update masked areas
        augmented_image = image * (1 - mask) + brightened_area
    
    # Apply contrast augmentation to the masked area
    if contrast_prop > 0.0 and np.random.random() < contrast_prop:
        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        # Apply contrast only to the anatomical area
        masked_area = augmented_image * mask
        mean_val = np.mean(masked_area[mask > 0])  # Mean of only the masked region
        contrasted_area = (masked_area - mean_val * mask) * contrast_factor + mean_val * mask
        # Keep non-masked areas unchanged, update masked areas
        augmented_image = augmented_image * (1 - mask) + contrasted_area
    
    # Ensure non-negative values
    augmented_image = np.maximum(augmented_image, 0)
    
    return augmented_image


def spatial_augmentation(
        kspace: np.ndarray, 
        image: np.ndarray,
        flip_prop=0.5,
        rotate_prop=0.4, rotate_range=(3, 8),
        scale_prop=0.4, scale_range=(0.95, 1.05),
        shift_prop=0.4, shift_range=(3, 8),
        # New brightness/contrast parameters
        brightness_prop=0.0, brightness_range=(0.8, 1.2),
        contrast_prop=0.0, contrast_range=(0.8, 1.2),
        anatomy_type='brain',
        # New parameter for k-space augmentation
        enable_kspace_intensity_aug=False
    ) -> Tuple[np.ndarray, np.ndarray]:

    # if np.random.random() < rotate_prop:
    #     angle = np.random.uniform(rotate_range[0], rotate_range[1])
    #     if np.random.random() < 0.5:
    #         angle = -angle
    #     kspace = rotate_kspace(kspace, angle=angle)
    #     image = rotate_image(image, angle)
    
    # if np.random.random() < scale_prop:
    #     scale_factor = tuple(np.random.uniform(scale_range[0], scale_range[1], 2))
    #     kspace = scale_kspace(kspace, scale_factor)
    #     image = scale_image(image, scale_factor)

    # if np.random.random() < shift_prop:
    #     shift_step = tuple(np.random.randint(shift_range[0], shift_range[1], 2))
    #     kspace = shift_kspace(kspace, shift_step)
    #     image = shift_image(image, shift_step)

    # Apply brightness and contrast augmentation
    if brightness_prop > 0.0 or contrast_prop > 0.0:
        if enable_kspace_intensity_aug:
            # NEW: Idea - augment k-space via multi-coil image domain
            brightness_factor = 1.0
            contrast_factor = 1.0
            
            if brightness_prop > 0.0 and np.random.random() < brightness_prop:
                brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
            
            if contrast_prop > 0.0 and np.random.random() < contrast_prop:
                contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
            
            # Apply to k-space (which will also affect the target image consistently)
            if brightness_factor != 1.0 or contrast_factor != 1.0:
                kspace = kspace_intensity_augmentation(
                    kspace,
                    brightness_factor=brightness_factor,
                    contrast_factor=contrast_factor,
                    anatomy_type=anatomy_type
                )
                
                # Also update the target image to match the k-space augmentation
                image = brightness_contrast_augmentation(
                    image,
                    anatomy_type=anatomy_type,
                    brightness_prop=1.0,  # Force application with same factors
                    contrast_prop=1.0,
                    brightness_range=(brightness_factor, brightness_factor),  # Use same factor
                    contrast_range=(contrast_factor, contrast_factor)
                )
        else:
            # ORIGINAL: Only augment target image (preserves anatomical mask)
            image = brightness_contrast_augmentation(
                image, 
                anatomy_type=anatomy_type,
                brightness_prop=brightness_prop,
                contrast_prop=contrast_prop,
                brightness_range=brightness_range,
                contrast_range=contrast_range
            )

    if np.random.random() < flip_prop:
        kspace = flip_kspace(kspace)
        image = flip_image(image)
    
    return kspace, image


if __name__ == "__main__":
    kspace_path = "/root/fastMRI/datasets/train/kspace/brain_acc4_1.h5"
    img_path = "/root/fastMRI/datasets/train/image/brain_acc4_1.h5"

    import h5py
    import numpy

    with h5py.File(kspace_path, 'r') as f:
        kspace = np.array(f['kspace'][:])
    with h5py.File(img_path, 'r') as f:
        img = np.array(f['image_label'][:])
        attrs = dict(f.attrs)

    print(kspace.shape)
    print(img.shape)

    print(np.min(kspace.real), np.max(kspace.real))
    print(np.min(kspace.imag), np.max(kspace.imag))
    print(np.max(img))

    print(np.mean(kspace))
    print(np.mean(img))

    kspace = flip_kspace(kspace)
    img = flip_image(img)

    print(kspace.shape)
    print(img.shape)

    print(np.min(kspace.real), np.max(kspace.real))
    print(np.min(kspace.imag), np.max(kspace.imag))
    print(np.max(img))

    print(np.mean(kspace))
    print(np.mean(img))

    print(attrs)