import h5py
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

import torch

def to_tensor(data):
    """Convert numpy array to torch tensor"""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        return torch.tensor(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    
    def __call__(self, mask, input, target, attrs, fname, slice, num_slices):
        
        if self.isforward:
            target = maximum = -1
        else:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
    
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace.float(), target, maximum, fname, slice, num_slices


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, anatomy='all'):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        if anatomy != 'all':
            kspace_files = [f for f in kspace_files if anatomy in f.name]

        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind, num_slices) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice, num_slices = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice, num_slices)


def create_eval_loaders(data_path, args, shuffle=False, isforward=False, anatomy='all'):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        anatomy=anatomy
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
