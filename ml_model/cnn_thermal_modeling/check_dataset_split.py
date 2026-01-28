"""
Check component split in HDF5 dataset.
"""

import h5py
import numpy as np
from pathlib import Path

dataset_file = Path(__file__).resolve().parent / "datasets" / "HBridge_cnn_dataset.h5"

print("="*80)
print("CHECKING DATASET COMPONENT SPLIT")
print("="*80)

with h5py.File(dataset_file, 'r') as f:
    n_components = f['sand_temps'].shape[1]
    print(f"\nTotal components: {n_components}")
    
    if 'metadata/train_component_indices' in f:
        train_indices = f['metadata/train_component_indices'][:]
        val_indices = np.array([i for i in range(n_components) if i not in train_indices])
        
        print(f"Training components: {len(train_indices)}")
        print(f"  Indices: {train_indices.tolist()}")
        
        print(f"\nValidation components: {len(val_indices)}")
        print(f"  Indices: {val_indices.tolist()}")
        
        # Get component names if available
        if 'metadata/component_names' in f:
            comp_names = [name.decode('utf-8') for name in f['metadata/component_names'][:]]
            print(f"\nValidation component names:")
            for idx in val_indices:
                if idx < len(comp_names):
                    print(f"  [{idx}] {comp_names[idx]}")
    else:
        print("\n⚠️  No component split metadata found!")
        print("Dataset may not have been created with component-level validation.")
