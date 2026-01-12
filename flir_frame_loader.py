"""
===============================================================================
FLIR FRAME LOADER - Thermal Image Sequence Processor
===============================================================================
Load and process FLIR thermal frame sequences from ResearchIR CSV exports.

Purpose:
    - Load frame-by-frame thermal data from ResearchIR exports
    - Build 3D NumPy arrays: [n_frames, height, width]
    - Extract temporal metadata (timestamps, frame indices)
    - Provide efficient frame access for CNN training

ResearchIR Export Format:
    - Folder contains multiple CSV files (one per frame)
    - Each CSV: Full pixel temperature matrix at specific time point
    - Filename pattern: Usually includes frame number or timestamp
    - Typical interval: 15 seconds between frames

Usage:
    from flir_frame_loader import FLIRFrameLoader
    
    loader = FLIRFrameLoader()
    frames, metadata = loader.load_sequence("ResearchIR_Outputs_HBridge_15s")
    
    # Access specific frame
    frame_t30 = loader.get_frame_at_time(30.0)  # Frame at 30 seconds
    
    # Export for fast loading
    loader.export_frame_stack("HBridge_frames.npy")

Created: January 12, 2026
===============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import re


class FLIRFrameLoader:
    """
    Loader for FLIR thermal frame sequences from ResearchIR CSV exports.
    
    Handles frame-by-frame thermal data, builds temporal sequences,
    and provides efficient access for CNN training.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize FLIR frame loader.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.frames = None
        self.timestamps = None
        self.metadata = {}
        self.frame_files = []
    
    def load_sequence(self, folder_path: str, 
                     frame_pattern: str = "*.csv",
                     max_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load complete thermal frame sequence from folder.
        
        Args:
            folder_path: Path to ResearchIR export folder with frame CSVs
            frame_pattern: Glob pattern for frame files (default: "*.csv")
            max_frames: Maximum frames to load (None = load all)
        
        Returns:
            Tuple of (frames, timestamps)
                frames: NumPy array [n_frames, height, width]
                timestamps: NumPy array [n_frames] in seconds
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all frame CSV files
        self.frame_files = sorted(folder.glob(frame_pattern))
        
        if not self.frame_files:
            raise ValueError(f"No frame files found matching {frame_pattern} in {folder_path}")
        
        if max_frames:
            self.frame_files = self.frame_files[:max_frames]
        
        if self.verbose:
            print(f"\n[FLIR FRAME LOADER]")
            print(f"  Folder: {folder_path}")
            print(f"  Found {len(self.frame_files)} frame files")
        
        # Load first frame to get dimensions
        first_frame = self._load_single_frame(self.frame_files[0])
        height, width = first_frame.shape
        n_frames = len(self.frame_files)
        
        if self.verbose:
            print(f"  Frame dimensions: {height} x {width}")
            print(f"  Total frames: {n_frames}")
        
        # Preallocate array
        self.frames = np.zeros((n_frames, height, width), dtype=np.float32)
        self.timestamps = np.zeros(n_frames, dtype=np.float32)
        
        # Load all frames
        for idx, frame_file in enumerate(self.frame_files):
            self.frames[idx] = self._load_single_frame(frame_file)
            self.timestamps[idx] = self._extract_timestamp(frame_file, idx)
            
            if self.verbose and (idx + 1) % 10 == 0:
                print(f"  Loaded {idx + 1}/{n_frames} frames...")
        
        # Store metadata
        self.metadata = {
            'n_frames': n_frames,
            'height': height,
            'width': width,
            'folder_path': str(folder),
            'frame_files': [str(f) for f in self.frame_files],
            'time_start': self.timestamps[0],
            'time_end': self.timestamps[-1],
            'time_interval_avg': np.mean(np.diff(self.timestamps)) if n_frames > 1 else 0
        }
        
        if self.verbose:
            print(f"  ✓ Loaded complete sequence")
            print(f"  Time range: {self.timestamps[0]:.1f}s - {self.timestamps[-1]:.1f}s")
            print(f"  Avg interval: {self.metadata['time_interval_avg']:.1f}s\n")
        
        return self.frames, self.timestamps
    
    def _load_single_frame(self, file_path: Path) -> np.ndarray:
        """
        Load single thermal frame from CSV file.
        
        Args:
            file_path: Path to frame CSV file
        
        Returns:
            NumPy array [height, width] with thermal data
        """
        try:
            # ResearchIR CSV format has 5 header lines, then blank line, then data
            # Skip first 6 lines (header + blank)
            frame_data = np.loadtxt(file_path, delimiter=',', skiprows=6)
            return frame_data.astype(np.float32)
            
        except Exception as e:
            # If that fails, try pandas (handles headers better)
            df = pd.read_csv(file_path, header=None, skiprows=6)
            return df.values.astype(np.float32)
    
    def _extract_timestamp(self, file_path: Path, frame_index: int) -> float:
        """
        Extract timestamp from filename or use frame index.
        
        Common patterns:
            - "frame_0030.csv" -> 30 seconds (assumes 1s interval)
            - "thermal_450s.csv" -> 450 seconds
            - "Rec-000009-328_13_10_18_329_5.csv" -> Use index * 15s
        
        Args:
            file_path: Path to frame file
            frame_index: Sequential frame index (0-based)
        
        Returns:
            Timestamp in seconds
        """
        filename = file_path.stem  # Filename without extension
        
        # Try to extract number from filename
        # Look for patterns like: _###, ###s, -###
        patterns = [
            r'_(\d+)s',      # "_450s" format
            r'_(\d+)$',      # "_0030" format (end of filename)
            r'-(\d+)',       # "-450" format
            r'(\d+)s',       # "450s" format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return float(match.group(1))
        
        # If no timestamp found, assume 15s intervals (ResearchIR default)
        # Frame 0 = 0s, Frame 1 = 15s, Frame 2 = 30s, etc.
        return frame_index * 15.0
    
    def get_frame_at_time(self, time_s: float, interpolate: bool = False) -> np.ndarray:
        """
        Get frame at specific time point.
        
        Args:
            time_s: Time in seconds
            interpolate: If True, interpolate between frames (not implemented yet)
        
        Returns:
            Frame at requested time [height, width]
        """
        if self.frames is None:
            raise ValueError("No frames loaded. Call load_sequence() first.")
        
        # Find closest frame
        idx = np.argmin(np.abs(self.timestamps - time_s))
        
        return self.frames[idx]
    
    def get_frame_by_index(self, index: int) -> np.ndarray:
        """
        Get frame by index.
        
        Args:
            index: Frame index (0-based)
        
        Returns:
            Frame at index [height, width]
        """
        if self.frames is None:
            raise ValueError("No frames loaded. Call load_sequence() first.")
        
        return self.frames[index]
    
    def export_frame_stack(self, output_file: str):
        """
        Export frame stack to NumPy .npy file for fast loading.
        
        Args:
            output_file: Path to output .npy file
        """
        if self.frames is None:
            raise ValueError("No frames loaded. Call load_sequence() first.")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save frames and metadata
        np.savez_compressed(
            output_path.with_suffix('.npz'),
            frames=self.frames,
            timestamps=self.timestamps,
            **self.metadata
        )
        
        if self.verbose:
            print(f"  ✓ Exported frame stack: {output_path.with_suffix('.npz')}")
            print(f"  File size: {output_path.with_suffix('.npz').stat().st_size / 1e6:.1f} MB")
    
    def load_from_npz(self, npz_file: str):
        """
        Load previously exported frame stack.
        
        Args:
            npz_file: Path to .npz file from export_frame_stack()
        """
        data = np.load(npz_file, allow_pickle=True)
        
        self.frames = data['frames']
        self.timestamps = data['timestamps']
        
        # Restore metadata
        self.metadata = {
            key: data[key].item() if data[key].ndim == 0 else data[key]
            for key in data.files
            if key not in ['frames', 'timestamps']
        }
        
        if self.verbose:
            print(f"\n[FLIR FRAME LOADER]")
            print(f"  Loaded from: {npz_file}")
            print(f"  Frames: {self.frames.shape[0]}")
            print(f"  Dimensions: {self.frames.shape[1]} x {self.frames.shape[2]}")
            print(f"  Time range: {self.timestamps[0]:.1f}s - {self.timestamps[-1]:.1f}s\n")
    
    def get_temperature_range(self) -> Tuple[float, float]:
        """
        Get min/max temperature across all frames.
        
        Returns:
            Tuple of (min_temp, max_temp) in °C
        """
        if self.frames is None:
            raise ValueError("No frames loaded.")
        
        return float(np.min(self.frames)), float(np.max(self.frames))
    
    def get_frame_statistics(self, frame_index: int) -> Dict:
        """
        Get statistics for specific frame.
        
        Args:
            frame_index: Frame index
        
        Returns:
            Dictionary with statistics (min, max, mean, std)
        """
        if self.frames is None:
            raise ValueError("No frames loaded.")
        
        frame = self.frames[frame_index]
        
        return {
            'index': frame_index,
            'timestamp': self.timestamps[frame_index],
            'min_temp': float(np.min(frame)),
            'max_temp': float(np.max(frame)),
            'mean_temp': float(np.mean(frame)),
            'std_temp': float(np.std(frame))
        }


def main():
    """
    Example usage: Load FLIR frame sequences for HBridge and LoadShedding.
    """
    loader = FLIRFrameLoader(verbose=True)
    
    # Test with HBridge data
    print("\n" + "="*80)
    print("  LOADING HBRIDGE THERMAL SEQUENCE")
    print("="*80)
    
    frames_hb, times_hb = loader.load_sequence(
        "inputs/ResearchIR_Outputs_HBridge_15s",
        max_frames=None  # Load all frames
    )
    
    print(f"\nHBridge Summary:")
    print(f"  Frames loaded: {frames_hb.shape}")
    print(f"  Temperature range: {loader.get_temperature_range()}")
    
    # Export for fast loading
    loader.export_frame_stack("outputs/flir_frames/HBridge_thermal_frames.npz")
    
    # Test with LoadShedding data
    print("\n" + "="*80)
    print("  LOADING LOADSHEDDING THERMAL SEQUENCE")
    print("="*80)
    
    frames_ls, times_ls = loader.load_sequence(
        "inputs/ResearchIR_Outputs_Load_Shedding",
        max_frames=None
    )
    
    print(f"\nLoadShedding Summary:")
    print(f"  Frames loaded: {frames_ls.shape}")
    print(f"  Temperature range: {loader.get_temperature_range()}")
    
    # Export
    loader.export_frame_stack("outputs/flir_frames/LoadShedding_thermal_frames.npz")
    
    print("\n" + "="*80)
    print("  FRAME LOADING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
