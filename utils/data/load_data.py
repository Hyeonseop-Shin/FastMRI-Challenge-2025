import h5py
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Union, Iterator
import logging

from utils.data.subsample import EquispacedMaskFunc, RandomMaskFunc
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Constants
FORWARD_MODE_PLACEHOLDER = -1
DEFAULT_KSPACE_KEY = 'kspace'
DEFAULT_TARGET_KEY = 'image_label'
DEFAULT_MASK_KEY = 'mask'


class DatasetConfig:
    """Configuration class for dataset parameters"""
    
    def __init__(self, args, 
                 isforward: bool = False,
                 slice_moe_num: int = 0):
        self.isforward = isforward
        self.max_key = args.max_key if not isforward else FORWARD_MODE_PLACEHOLDER
        self.target_key = args.target_key if not isforward else FORWARD_MODE_PLACEHOLDER
        self.input_key = args.input_key
        self.batch_size = args.batch_size
        self.use_moe = args.use_moe
        self.num_folds = args.num_folds
        self.k_fold = args.k_fold
        self.acc = args.acc
        self.anatomy = args.anatomy
        self.slice_moe_total = args.slice_moe if args.use_moe else None
        self.slice_moe_num = slice_moe_num
        self.random_mask_prop = args.random_mask_prop
        self.use_random_mask = args.use_random_mask


class SliceData(Dataset):
    """Unified dataset class - handles both standard and indexed loading"""
    
    def __init__(self, 
                 kspace_root: Union[str, Path], 
                 image_root: Optional[Union[str, Path]] = None, 
                 file_list: Optional[List[str]] = None, 
                 transform=None, 
                 input_key: str = DEFAULT_KSPACE_KEY, 
                 target_key: str = DEFAULT_TARGET_KEY, 
                 forward: bool = False,
                 acc: int = 4,
                 slice_moe_num: int = 0,
                 slice_moe_total: int = 1,
                 use_random_mask: bool = False,
                 random_mask_prop: float = 0.2):
        self.kspace_root = Path(kspace_root)
        self.image_root = Path(image_root) if image_root and not forward else None
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward

        self.slice_moe_num = slice_moe_num
        self.slice_moe_total = slice_moe_total

        self.use_random_mask = use_random_mask
        self.random_mask_prop = random_mask_prop
        center_fractions = [0.08]
        self.eq_mask_generator = EquispacedMaskFunc(center_fractions=center_fractions, accelerations=[acc])
        if use_random_mask:
            self.random_mask_generator = RandomMaskFunc(center_fractions=center_fractions, accelerations=[acc])

        self.examples: List[Tuple[str, int]] = []
        self._build_examples(file_list)
    
    def _build_examples(self, file_list: Optional[List[str]]):
        """Build examples from index file or all files"""
        if file_list:
            self._build_from_file_list(file_list=file_list)
        else:
            self._build_from_directory()
    
    def _build_from_file_list(self, file_list: List[str]):
        """Build examples from provided file list"""
        for input_path in file_list:
            if not os.path.exists(input_path):
                logger.warning(f"File not found: {input_path}")
                continue
            
            try:
                self._add_slices_from_file(input_path=input_path)
            except Exception as e:
                logger.error(f"Error processing file {input_path}: {e}")
                continue
    
    def _build_from_directory(self):
        """Build examples from all files in kspace root directory"""
        try:
            for input_name in os.listdir(self.kspace_root):
                input_path = self.kspace_root / input_name
                if input_path.suffix.lower() == '.h5':
                    self._add_slices_from_file(input_path=str(input_path))
        except Exception as e:
            logger.error(f"Error reading directory {self.kspace_root}: {e}")
    
    def _add_slices_from_file(self, input_path: str):
        """Add all slices from a single file to examples"""
        try:
            with h5py.File(input_path, "r") as hf:
                if self.input_key not in hf:
                    logger.warning(f"Key '{self.input_key}' not found in {input_path}")
                    return
                
                num_slices = hf[self.input_key].shape[0]
            
            start_slice = int(self.slice_moe_num / self.slice_moe_total * num_slices)
            end_slice = int((self.slice_moe_num + 1) / self.slice_moe_total * num_slices)
            end_slice = num_slices if end_slice > num_slices else end_slice

            slice_moe_range = range(start_slice, end_slice)
            for slice_ind in slice_moe_range:
                self.examples.append((input_path, slice_ind))
                
        except Exception as e:
            logger.error(f"Error reading HDF5 file {input_path}: {e}")

    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, index: int):
        input_path, slice_ind = self.examples[index]

        # Load kspace data
        try:
            input_data, mask = self._load_kspace_data(input_path=input_path, slice_ind=slice_ind)
            if self.use_random_mask and np.random.random() < self.random_mask_prop:
                mask = self.random_mask_generator(shape=input_data.shape)
            else:
                mask = self.eq_mask_generator(shape=input_data.shape)
        except Exception as e:
            logger.error(f"Error loading kspace data from {input_path}, slice {slice_ind}: {e}")
            raise
        
        # Load target data
        target, attrs = self._load_target_data(input_path=input_path, slice_ind=slice_ind)

        # Extract filename
        fname = os.path.basename(input_path)
        
        return self.transform(mask, input_data, target, attrs, fname, slice_ind)
    
    def _load_kspace_data(self, input_path: str, slice_ind: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load kspace data and mask from file"""
        with h5py.File(input_path, "r") as hf:
            input_data = hf[self.input_key][slice_ind]
            mask = np.array(hf[DEFAULT_MASK_KEY])
        return input_data, mask
    
    def _load_target_data(self, input_path: str, slice_ind: int) -> Tuple[Union[np.ndarray, int], Union[dict, int]]:
        """Load target data and attributes"""
        if self.forward:
            return FORWARD_MODE_PLACEHOLDER, FORWARD_MODE_PLACEHOLDER
        
        try:
            image_path = input_path.replace("kspace", "image")
            with h5py.File(image_path, "r") as hf:
                target = hf[self.target_key][slice_ind]
                attrs = dict(hf.attrs)
            return target, attrs
        except Exception as e:
            logger.error(f"Error loading target data from {image_path}, slice {slice_ind}: {e}")
            raise


def split_k_folds(file_list: List[str], num_folds: int = 5) -> List[List[str]]:
    """Split file list into k folds for cross-validation"""
    file_list_copy = file_list.copy()  # Don't modify the original list

    np.random.shuffle(file_list_copy)

    fold_size = len(file_list_copy) // num_folds
    folds = [file_list_copy[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    
    # Handle remaining files
    if len(file_list_copy) % num_folds != 0:
        folds[-1].extend(file_list_copy[num_folds * fold_size:])
    
    return folds


def _get_file_list(config: DatasetConfig, kspace_root: Path, index_file: Optional[str] = None) -> List[str]:
    """Get list of files to process based on configuration"""
    if config.use_moe and index_file:
        with open(index_file, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
    else:
        file_list = [str(p) for p in kspace_root.glob('*.h5')]
        if not file_list:
            raise ValueError(f"No files found in {kspace_root}")
    return file_list


def _create_dataset(kspace_root: Path, 
                   image_root: Optional[Path], 
                   file_list: Optional[List[str]], 
                   config: DatasetConfig, 
                   augmentation: bool = False) -> SliceData:
    """Create a SliceData dataset with given parameters"""
    return SliceData(
        kspace_root=kspace_root,
        image_root=image_root,
        file_list=file_list,
        transform=DataTransform(config.isforward, config.max_key, augmentation=augmentation),
        input_key=config.input_key,
        target_key=config.target_key,
        forward=config.isforward,
        acc=config.acc,
        slice_moe_num=config.slice_moe_num,
        slice_moe_total=config.slice_moe_total,
        random_mask_prop=config.random_mask_prop,
        use_random_mask=config.use_random_mask
    )


def create_data_loaders(train_data_path: Union[str, Path], 
                        val_data_path: Union[str, Path], 
                        args, 
                        index_file: Optional[str] = None, 
                        shuffle: bool = False, 
                        isforward: bool = False, 
                        augmentation: bool = False,
                        slice_moe_num: int = 0) -> Iterator[Tuple[DataLoader, DataLoader]]:
    """Create standard data loader with k-fold cross-validation"""
    config = DatasetConfig(args, isforward, slice_moe_num)
    
    # Set up paths
    train_kspace_root = Path(train_data_path) / "kspace"
    train_image_root = Path(train_data_path) / "image" if not isforward else None
    val_kspace_root = Path(val_data_path) / "kspace"
    val_image_root = Path(val_data_path) / "image" if not isforward else None

    # Get file list and create folds
    file_list = _get_file_list(config, train_kspace_root, index_file)
    folds = split_k_folds(file_list, config.num_folds)
    fold_num = config.num_folds if config.k_fold else 1

    for val_fold in range(fold_num):
        # Create training file list (all folds except validation fold)
        train_folds = [f for i, f in enumerate(folds) if i != val_fold]
        train_file_list = [item for sublist in train_folds for item in sublist]  # Flatten

        # Create datasets
        train_dataset = _create_dataset(
            train_kspace_root, train_image_root, 
            train_file_list if config.use_moe else None, 
            config, augmentation
        )
        
        val_dataset = _create_dataset(
            val_kspace_root, val_image_root,
            folds[val_fold] if config.use_moe else None,
            config, augmentation
        )

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=shuffle)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=shuffle)

        yield train_loader, val_loader

def create_eval_loaders(data_path: Union[str, Path],
                        args,
                        isforward: bool = False,
                        shuffle: bool = False) -> DataLoader:

    config = DatasetConfig(args, isforward)

    kspace_root = Path(data_path) / "kspace"
    image_root = Path(data_path) / "image" if not isforward else None

    # No file list used for eval
    eval_dataset = _create_dataset(
        kspace_root=kspace_root,
        image_root=image_root,
        file_list=None,
        config=config,
        augmentation=False
    )

    loader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
    )

    return loader
