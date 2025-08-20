import os
import h5py
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch

class MRIClassifier:
    """MRI data classifier for anatomy and acceleration"""
    
    def __init__(self, brain_width_range=(384, 400), mask_threshold=95):
        self.brain_width_range = brain_width_range
        self.mask_threshold = mask_threshold
    
    def classify_by_path(self, h5_path):
        """Classify MRI data into anatomy and acceleration types"""
        with h5py.File(h5_path, 'r') as f:
            kspace = f['kspace'][:]
            mask = f['mask'][:]
        
        return self.classify(kspace, mask)
    
    def classify(self, kspace, mask):
        """Classify MRI data from kspace and mask arrays"""
        # Handle both PyTorch tensors and NumPy arrays
        
        if hasattr(mask, 'detach'):
            # PyTorch tensor
            mask = mask.detach().cpu().numpy()
        elif isinstance(mask, torch.Tensor):
            # PyTorch tensor without detach (shouldn't happen but be safe)
            mask = mask.cpu().numpy()
        else:
            # Already NumPy array or similar
            mask = np.array(mask)
        
        # Anatomy classification (brain/knee)
        # Handle complex kspace: if last dim is 2 (real/imag), use shape[-2] for width
        if kspace.shape[-1] == 2:
            width = kspace.shape[-2]  # Complex format: [..., height, width, 2]
        else:
            width = kspace.shape[-1]  # Standard format: [..., height, width]
        
        anatomy = 'brain' if self.brain_width_range[0] <= width <= self.brain_width_range[1] else 'knee'
        
        # Acceleration classification
        ones_count = int(np.sum(mask))
        acceleration = 'acc4' if ones_count >= self.mask_threshold else 'acc8'
        
        return {
            'anatomy': anatomy,
            'acceleration': acceleration, 
            'class_label': f"{acceleration}-{anatomy}"
        }

def classify_and_index(train_path, val_path, output_base, brain_width_range=(390, 400), mask_threshold=95, print_freq=10):
    """
    Classify MRI files and create index files for each class
    
    Returns:
        dict: {class_label: [file_list]}
        list: paths to generated index files
    """
    classifier = MRIClassifier(brain_width_range, mask_threshold)
    
    # Step 1: Classify all files
    records = []
    train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.h5')]
    val_files = [os.path.join(val_path, f) for f in os.listdir(val_path) if f.endswith('.h5')]
    total_files = train_files + val_files

    print(f"Classifying {len(total_files)} files...")
    for file_path in tqdm(sorted(total_files)):

        classification = classifier.classify_by_path(file_path)
        records.append({
            'file': os.path.basename(file_path),
            'anatomy': classification['anatomy'],
            'acceleration': classification['acceleration'],
            'class': classification['class_label'],
            'path': file_path
        })
    
    # Step 2: Group by class and save index files
    df = pd.DataFrame(records)
    class_groups = df.groupby('class')['path'].apply(list).to_dict()

    # Step 3: Merge acceleration
    class_groups['acc4-brain'] = class_groups['acc4-brain'] + class_groups['acc8-brain']
    class_groups['acc8-brain'] = class_groups['acc4-brain']
    class_groups['acc4-knee'] = class_groups['acc4-knee'] + class_groups['acc8-knee']
    class_groups['acc8-knee'] = class_groups['acc4-knee']
    
    os.makedirs(output_base, exist_ok=True)
    index_files = []
    
    for class_label, path_list in class_groups.items():
        index_file_path = os.path.join(output_base, f"{class_label}.txt")
        index_files.append(index_file_path)
        
        with open(index_file_path, 'w') as f:
            for path in path_list:
                f.write(path + '\n')

        print(f"Created index: {index_file_path} ({len(path_list)} files)")
    
    # Step 3: Print statistics
    total = sum(len(files) for files in class_groups.values())
    print("\n=== MoE Classification Statistics ===")
    for class_label, path_list in class_groups.items():
        count = len(path_list)
        percentage = (count/total)*100
        print(f"{class_label}: {count} files ({percentage:.1f}%)")
    print("=====================================")
    
    # Save summary CSV
    csv_path = os.path.join(output_base, "classification_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved classification summary: {csv_path}")
    
    return class_groups, index_files

# Backward compatibility
def classify_and_split(input_folder, output_base, brain_width_range=(390, 400)):
    """Backward compatibility wrapper"""
    return classify_and_index(input_folder, output_base, brain_width_range)

# Example usage
# if __name__ == '__main__':
#     input_folder = "/root/Data/train/kspace"
#     output_base = "/root/Data/train/class_indices"
#     class_groups, index_files = classify_and_index(input_folder, output_base)
