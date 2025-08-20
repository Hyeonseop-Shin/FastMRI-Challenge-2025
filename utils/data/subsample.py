
import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        # Use a seeded random state for reproducible mask generation
        self.rng = np.random.RandomState(2025)  # Fixed seed for reproducible masks

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = np.random.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-1]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            num_ones = (num_cols - num_low_freqs) // acceleration

            mask_side = np.zeros(num_cols - num_low_freqs)
            indices = np.random.choice(num_cols - num_low_freqs, num_ones, replace=False)
            mask_side[indices] = True
            
            mask = np.zeros(num_cols)
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[:pad] = mask_side[:pad]
            mask[pad : pad + num_low_freqs] = True
            mask[pad + num_low_freqs:] = mask_side[pad:]

        return mask


class EquispacedMaskFunc(MaskFunc):

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-1]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols)
            pad = (num_cols - num_low_freqs) // 2
            mask[pad : pad + num_low_freqs] = True

            mask_side = np.zeros(num_cols)
            offset  = np.random.randint(acceleration)
            mask_side[offset::acceleration] = True

            mask[:pad] = mask_side[:pad]
            # mask[pad + num_low_freqs:] = mask_side[::-1][-(num_cols-pad-num_low_freqs):]
            mask[pad + num_low_freqs:] = mask_side[pad:num_cols - num_low_freqs]

        return mask


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    else:
        raise Exception(f"{mask_type_str} not supported")
