"""
===============================================================================
CNN DATA PREPROCESSOR - Training Dataset Builder for Spatial Thermal CNN
===============================================================================
Preprocess and align FLIR thermal frames with thermistor ground truth data.

Purpose:
    - Load FLIR frame sequences (from flir_frame_loader)
    - Load thermistor time series (from Phase 6 calibration database)
    - Load ROI pixel maps (from roi_pixel_mapper)
    - Detect temporal offset via cross-correlation
    - Interpolate thermistor data to FLIR timestamps
    - Build ROI masks for supervised learning
    - Export HDF5 dataset for CNN training

HDF5 Dataset Structure:
    /flir_frames: [n_frames, H, W] - Input thermal images
    /sand_temps: [n_frames, n_components] - Ground truth thermistor temps
    /roi_masks: [n_components, H, W] - Binary masks for each ROI
    /metadata: Timestamps, component names, alignment info

Workflow:
    1. Load FLIR frames (640x480, 15s intervals)
    2. Load thermistor CSV (Phase 6 calibration data)
    3. Cross-correlate to find time offset
    4. Interpolate thermistor to FLIR timestamps
    5. Generate ROI masks from pixel maps
    6. Export aligned HDF5 dataset

Created: January 12, 2026
===============================================================================
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from scipy import signal
from scipy.interpolate import interp1d
from flir_frame_loader import FLIRFrameLoader


class CNNDataPreprocessor:
    """
    Preprocessor for building CNN training datasets from FLIR and thermistor data.
    
    Handles temporal alignment, ROI mask generation, and HDF5 export.
    """
    
    def __init__(self, image_width: int = 640, image_height: int = 480, verbose: bool = True):
        """
        Initialize CNN data preprocessor.
        
        Args:
            image_width: FLIR image width (pixels)
            image_height: FLIR image height (pixels)
            verbose: Enable verbose output
        """
        self.image_width = image_width
        self.image_height = image_height
        self.verbose = verbose
        
        # Data containers
        self.flir_frames = None
        self.flir_timestamps = None
        self.thermistor_data = None
        self.roi_pixel_map = None
        self.roi_masks = None
        
        # Alignment results
        self.time_offset = 0.0
        self.aligned_sand_temps = None
    
    def load_flir_frames(self, flir_folder: str, max_frames: Optional[int] = None):
        """
        Load FLIR thermal frame sequence.
        
        Args:
            flir_folder: Path to ResearchIR exports or .npz file
            max_frames: Maximum frames to load (None = all)
        """
        if self.verbose:
            print("\n" + "="*80)
            print("  STEP 1: LOADING FLIR FRAMES")
            print("="*80)
        
        loader = FLIRFrameLoader(verbose=self.verbose)
        
        # Check if .npz file (pre-exported) or folder
        flir_path = Path(flir_folder)
        if flir_path.suffix == '.npz':
            loader.load_from_npz(str(flir_path))
        else:
            loader.load_sequence(str(flir_path), max_frames=max_frames)
        
        self.flir_frames = loader.frames
        self.flir_timestamps = loader.timestamps
        
        if self.verbose:
            print(f"  ✓ FLIR frames loaded: {self.flir_frames.shape}")
            print(f"  ✓ Time range: {self.flir_timestamps[0]:.1f}s - {self.flir_timestamps[-1]:.1f}s\n")
    
    def load_thermistor_data(self, thermistor_csv: str, time_column: str = 'Time (s)', 
                            component_prefix: str = 'Temp_'):
        """
        Load thermistor ground truth data from Phase 6 calibration.
        
        Args:
            thermistor_csv: Path to thermistor time series CSV
            time_column: Name of time column
            component_prefix: Prefix for component temperature columns
        """
        if self.verbose:
            print("="*80)
            print("  STEP 2: LOADING THERMISTOR DATA")
            print("="*80)
        
        # Store CSV path for debug reporting
        self.thermistor_csv_path = Path(thermistor_csv)
        
        df = pd.read_csv(thermistor_csv)
        
        # Extract time and temperature columns
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in {thermistor_csv}")
        
        # Find all temperature columns
        temp_cols = [col for col in df.columns if col.startswith(component_prefix)]
        
        if not temp_cols:
            raise ValueError(f"No temperature columns found with prefix '{component_prefix}'")
        
        self.thermistor_data = {
            'time': df[time_column].values,
            'components': {col.replace(component_prefix, ''): df[col].values for col in temp_cols},
            'component_names': [col.replace(component_prefix, '') for col in temp_cols]
        }
        
        if self.verbose:
            print(f"  ✓ Loaded thermistor data")
            print(f"  Components: {len(self.thermistor_data['component_names'])}")
            print(f"  Component list: {', '.join(self.thermistor_data['component_names'])}")
            print(f"  Time range: {self.thermistor_data['time'][0]:.1f}s - {self.thermistor_data['time'][-1]:.1f}s")
            print(f"  Sample rate: {len(self.thermistor_data['time'])} samples\n")
    
    def load_roi_pixel_map(self, pixel_map_csv: str):
        """
        Load ROI pixel map from roi_pixel_mapper output.
        
        Args:
            pixel_map_csv: Path to pixel map CSV
        """
        if self.verbose:
            print("="*80)
            print("  STEP 3: LOADING ROI PIXEL MAP")
            print("="*80)
        
        df = pd.read_csv(pixel_map_csv)
        
        # Parse pixel_list column (stored as string representation of list)
        df['pixel_list_parsed'] = df['pixel_list'].apply(eval)
        
        self.roi_pixel_map = df
        
        if self.verbose:
            print(f"  ✓ Loaded pixel map")
            print(f"  Components: {len(df)}")
            print(f"  Total pixels across all ROIs: {df['pixel_count'].sum()}\n")
    
    def detect_time_offset(self, reference_component: Optional[str] = None) -> float:
        """
        Detect time offset between FLIR and thermistor via cross-correlation.
        
        Uses mean FLIR temperature vs thermistor temperature to find optimal offset.
        Resamples both to common time grid before correlation to handle different sampling rates.
        
        Args:
            reference_component: Specific component to use (None = use mean of all)
        
        Returns:
            Time offset in seconds (positive = FLIR leads thermistor)
        """
        if self.verbose:
            print("="*80)
            print("  STEP 4: DETECTING TIME OFFSET (CROSS-CORRELATION)")
            print("="*80)
        
        # Calculate mean FLIR temperature per frame
        flir_mean_temps = np.mean(self.flir_frames, axis=(1, 2))
        
        # Get thermistor reference signal
        if reference_component:
            if reference_component not in self.thermistor_data['components']:
                raise ValueError(f"Component '{reference_component}' not found in thermistor data")
            thermistor_signal_full = self.thermistor_data['components'][reference_component]
        else:
            # Use mean of all thermistor components
            all_temps = np.array([temps for temps in self.thermistor_data['components'].values()])
            thermistor_signal_full = np.mean(all_temps, axis=0)
        
        thermistor_time_full = self.thermistor_data['time']
        
        # Resample thermistor to match FLIR sampling rate for fair comparison
        # Use a common time grid that covers the overlap region
        flir_start = self.flir_timestamps[0]
        flir_end = self.flir_timestamps[-1]
        therm_start = thermistor_time_full[0]
        therm_end = thermistor_time_full[-1]
        
        # Create search window - allow offset up to ±1 hour
        search_window = 3600  # seconds
        search_start = max(flir_start - search_window, therm_start)
        search_end = min(flir_end + search_window, therm_end)
        
        # Resample both to common 1 Hz grid for correlation
        common_dt = 1.0  # 1 Hz sampling for correlation
        common_times = np.arange(search_start, search_end, common_dt)
        
        # Interpolate FLIR to common grid
        flir_interp = interp1d(self.flir_timestamps, flir_mean_temps, 
                              kind='linear', bounds_error=False, fill_value=np.nan)
        flir_resampled = flir_interp(common_times)
        
        # Interpolate thermistor to common grid
        therm_interp = interp1d(thermistor_time_full, thermistor_signal_full,
                               kind='linear', bounds_error=False, fill_value=np.nan)
        therm_resampled = therm_interp(common_times)
        
        # Remove NaN values (edges where extrapolation not valid)
        valid_mask = ~(np.isnan(flir_resampled) | np.isnan(therm_resampled))
        if np.sum(valid_mask) < 10:
            print("  Warning: Insufficient overlap between FLIR and thermistor data")
            print("  Using zero offset (no alignment)")
            self.time_offset = 0.0
            return self.time_offset
        
        flir_valid = flir_resampled[valid_mask]
        therm_valid = therm_resampled[valid_mask]
        
        # Cross-correlate on resampled data (normalize by removing mean)
        correlation = signal.correlate(therm_valid - np.mean(therm_valid), 
                                      flir_valid - np.mean(flir_valid), 
                                      mode='full')
        lags = signal.correlation_lags(len(therm_valid), len(flir_valid), mode='full')
        
        # Find peak correlation
        peak_idx = np.argmax(correlation)
        lag_samples = lags[peak_idx]
        
        # Convert lag to time offset
        # Positive lag = FLIR leads thermistor (FLIR happened earlier)
        self.time_offset = lag_samples * common_dt
        
        if self.verbose:
            print(f"  ✓ Cross-correlation complete")
            print(f"  Peak correlation: {correlation[peak_idx]:.0f}")
            print(f"  Time offset: {self.time_offset:.1f}s")
            print(f"  Interpretation: FLIR {'leads' if self.time_offset > 0 else 'lags'} thermistor by {abs(self.time_offset):.1f}s")
            print(f"  FLIR time range: {flir_start:.1f}s - {flir_end:.1f}s")
            print(f"  Thermistor time range: {therm_start:.1f}s - {therm_end:.1f}s\n")
        
        return self.time_offset
    
    def align_and_interpolate(self):
        """
        Align thermistor data to FLIR timestamps using detected offset and interpolation.
        
        Creates aligned_sand_temps: [n_frames, n_components]
        Also tracks outliers for debug reporting.
        """
        if self.verbose:
            print("="*80)
            print("  STEP 5: ALIGNING AND INTERPOLATING DATA")
            print("="*80)
        
        # Adjust FLIR timestamps by offset
        adjusted_flir_times = self.flir_timestamps + self.time_offset
        
        n_frames = len(adjusted_flir_times)
        n_components = len(self.thermistor_data['component_names'])
        
        self.aligned_sand_temps = np.zeros((n_frames, n_components), dtype=np.float32)
        
        # Initialize outlier tracking list
        self.outlier_records = []
        
        # Interpolate each component
        for idx, comp_name in enumerate(self.thermistor_data['component_names']):
            thermistor_temps = self.thermistor_data['components'][comp_name]
            thermistor_times = self.thermistor_data['time']
            
            # Create interpolator (linear interpolation, extrapolate for out-of-bounds)
            interpolator = interp1d(thermistor_times, thermistor_temps, 
                                   kind='linear', fill_value='extrapolate')
            
            # Interpolate to FLIR timestamps
            self.aligned_sand_temps[:, idx] = interpolator(adjusted_flir_times)
            
            # Track outliers (temperatures outside typical range -100 to 300°C)
            outlier_mask = (thermistor_temps < -100) | (thermistor_temps > 300)
            if np.any(outlier_mask):
                outlier_count = np.sum(outlier_mask)
                min_temp = np.min(thermistor_temps)
                max_temp = np.max(thermistor_temps)
                
                self.outlier_records.append({
                    'thermistor_file': str(self.thermistor_csv_path),
                    'component_name': comp_name,
                    'min_temp': min_temp,
                    'max_temp': max_temp,
                    'outlier_count': outlier_count,
                    'total_samples': len(thermistor_temps),
                    'outlier_percentage': 100 * outlier_count / len(thermistor_temps)
                })
        
        if self.verbose:
            print(f"  ✓ Interpolation complete")
            print(f"  Aligned temperatures: {self.aligned_sand_temps.shape}")
            print(f"  Temperature range: {np.min(self.aligned_sand_temps):.1f}°C - {np.max(self.aligned_sand_temps):.1f}°C")
            if self.outlier_records:
                print(f"  WARNING: Found {len(self.outlier_records)} components with outliers")
            print()
    
    def generate_roi_masks(self):
        """
        Generate binary ROI masks from pixel map.
        
        Creates roi_masks: [n_components, H, W]
        Each mask is 1 at ROI pixels, 0 elsewhere.
        """
        if self.verbose:
            print("="*80)
            print("  STEP 6: GENERATING ROI MASKS")
            print("="*80)
        
        n_components = len(self.roi_pixel_map)
        self.roi_masks = np.zeros((n_components, self.image_height, self.image_width), dtype=np.uint8)
        
        for idx, row in self.roi_pixel_map.iterrows():
            pixel_list = row['pixel_list_parsed']
            
            # Set mask pixels to 1
            for x, y in pixel_list:
                if 0 <= x < self.image_width and 0 <= y < self.image_height:
                    self.roi_masks[idx, y, x] = 1
        
        if self.verbose:
            print(f"  ✓ ROI masks generated")
            print(f"  Mask shape: {self.roi_masks.shape}")
            print(f"  Total ROI pixels: {np.sum(self.roi_masks)}")
            print(f"  Coverage: {100 * np.sum(self.roi_masks) / (self.image_height * self.image_width):.2f}%\n")
    
    def export_hdf5(self, output_file: str):
        """
        Export aligned dataset to HDF5 for CNN training.
        
        Args:
            output_file: Path to output .h5 file
        """
        if self.verbose:
            print("="*80)
            print("  STEP 7: EXPORTING HDF5 DATASET")
            print("="*80)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # Main datasets
            f.create_dataset('flir_frames', data=self.flir_frames, compression='gzip')
            f.create_dataset('sand_temps', data=self.aligned_sand_temps, compression='gzip')
            f.create_dataset('roi_masks', data=self.roi_masks, compression='gzip')
            f.create_dataset('timestamps', data=self.flir_timestamps)
            
            # Metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['n_frames'] = len(self.flir_timestamps)
            meta_group.attrs['n_components'] = len(self.thermistor_data['component_names'])
            meta_group.attrs['image_width'] = self.image_width
            meta_group.attrs['image_height'] = self.image_height
            meta_group.attrs['time_offset'] = self.time_offset
            meta_group.attrs['time_start'] = self.flir_timestamps[0]
            meta_group.attrs['time_end'] = self.flir_timestamps[-1]
            
            # Component names (store as string array)
            component_names_str = [name.encode('utf-8') for name in self.thermistor_data['component_names']]
            meta_group.create_dataset('component_names', data=component_names_str)
            
            # ROI pixel map (store as DataFrame CSV in string format)
            roi_csv_str = self.roi_pixel_map.to_csv(index=False)
            meta_group.create_dataset('roi_pixel_map_csv', data=roi_csv_str.encode('utf-8'))
        
        if self.verbose:
            file_size_mb = Path(output_file).stat().st_size / 1e6
            print(f"  ✓ HDF5 dataset exported")
            print(f"  File: {output_file}")
            print(f"  Size: {file_size_mb:.1f} MB")
            print(f"  Datasets: flir_frames, sand_temps, roi_masks, timestamps, metadata\n")
        
        # Export debug report if outliers were detected
        if hasattr(self, 'outlier_records') and self.outlier_records:
            self.export_debug_report(output_file)
    
    def export_debug_report(self, dataset_file: str):
        """
        Export debug report CSV with outlier information.
        
        Args:
            dataset_file: Path to the HDF5 dataset (used to determine output path)
        """
        # Save to outputs/debug_report.csv
        outputs_dir = Path(__file__).parent / 'outputs'
        outputs_dir.mkdir(exist_ok=True)
        debug_csv = outputs_dir / 'debug_report.csv'
        
        if self.outlier_records:
            df = pd.DataFrame(self.outlier_records)
            df.to_csv(debug_csv, index=False)
            
            if self.verbose:
                print("="*80)
                print("  DEBUG REPORT: OUTLIER DETECTION")
                print("="*80)
                print(f"  Found {len(self.outlier_records)} components with extreme values")
                print(f"  Report saved to: {debug_csv}")
                print(f"\n  Summary:")
                print(f"  - Total components with outliers: {len(self.outlier_records)}")
                print(f"  - Temperature range: {df['min_temp'].min():.1f}°C to {df['max_temp'].max():.1f}°C")
                print(f"  - Total outlier samples: {df['outlier_count'].sum()}")
                print()

    
    def build_dataset(self, flir_folder: str, thermistor_csv: str, 
                     pixel_map_csv: str, output_h5: str,
                     reference_component: Optional[str] = None,
                     max_frames: Optional[int] = None,
                     time_column: str = 'Time (s)',
                     component_prefix: str = 'Temp_'):
        """
        Complete end-to-end dataset building pipeline.
        
        Args:
            flir_folder: Path to FLIR frames folder or .npz
            thermistor_csv: Path to thermistor CSV
            pixel_map_csv: Path to ROI pixel map CSV
            output_h5: Path to output HDF5 file
            reference_component: Component for time alignment (None = use mean)
            max_frames: Max frames to load (None = all)
            time_column: Name of time column in thermistor CSV
            component_prefix: Prefix for component temperature columns
        """
        print("\n" + "="*80)
        print("  CNN TRAINING DATASET BUILDER")
        print("="*80 + "\n")
        
        # Execute pipeline
        self.load_flir_frames(flir_folder, max_frames=max_frames)
        self.load_thermistor_data(thermistor_csv, time_column=time_column, 
                                 component_prefix=component_prefix)
        self.load_roi_pixel_map(pixel_map_csv)
        self.detect_time_offset(reference_component=reference_component)
        self.align_and_interpolate()
        self.generate_roi_masks()
        self.export_hdf5(output_h5)
        
        print("="*80)
        print("  DATASET BUILDING COMPLETE ✓")
        print("="*80 + "\n")
        
        return output_h5


def load_dataset(h5_file: str) -> Dict:
    """
    Load HDF5 dataset for training.
    
    Args:
        h5_file: Path to HDF5 file
    
    Returns:
        Dictionary with all dataset components
    """
    with h5py.File(h5_file, 'r') as f:
        dataset = {
            'flir_frames': f['flir_frames'][:],
            'sand_temps': f['sand_temps'][:],
            'roi_masks': f['roi_masks'][:],
            'timestamps': f['timestamps'][:],
            'metadata': {
                'n_frames': f['metadata'].attrs['n_frames'],
                'n_components': f['metadata'].attrs['n_components'],
                'image_width': f['metadata'].attrs['image_width'],
                'image_height': f['metadata'].attrs['image_height'],
                'time_offset': f['metadata'].attrs['time_offset'],
                'time_start': f['metadata'].attrs['time_start'],
                'time_end': f['metadata'].attrs['time_end'],
                'component_names': [name.decode('utf-8') for name in f['metadata/component_names'][:]],
            }
        }
    
    return dataset


def main():
    """
    Example usage: Build CNN training datasets for HBridge and LoadShedding.
    """
    
    # HBridge Dataset
    print("\n" + "="*80)
    print("  BUILDING HBRIDGE CNN DATASET")
    print("="*80 + "\n")
    
    preprocessor_hb = CNNDataPreprocessor(verbose=True)
    
    preprocessor_hb.build_dataset(
        flir_folder="outputs/flir_frames/HBridge_thermal_frames.npz",
        thermistor_csv="outputs/phase6_calibration/HBridge_thermistor_data.csv",
        pixel_map_csv="outputs/roi_pixel_maps/HBridge_pixel_map.csv",
        output_h5="ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5",
        reference_component=None,  # Use mean of all components
        max_frames=None  # Use all frames
    )
    
    # LoadShedding Dataset
    print("\n" + "="*80)
    print("  BUILDING LOADSHEDDING CNN DATASET")
    print("="*80 + "\n")
    
    preprocessor_ls = CNNDataPreprocessor(verbose=True)
    
    preprocessor_ls.build_dataset(
        flir_folder="outputs/flir_frames/LoadShedding_thermal_frames.npz",
        thermistor_csv="outputs/phase6_calibration/LoadShedding_thermistor_data.csv",
        pixel_map_csv="outputs/roi_pixel_maps/LoadShedding_pixel_map.csv",
        output_h5="ml_model/cnn_thermal_modeling/datasets/LoadShedding_cnn_dataset.h5",
        reference_component=None,
        max_frames=None
    )
    
    print("\n" + "="*80)
    print("  ALL DATASETS BUILT SUCCESSFULLY ✓")
    print("="*80 + "\n")
    
    # Test loading
    print("Testing dataset loading...")
    dataset = load_dataset("ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5")
    print(f"  ✓ Loaded HBridge dataset: {dataset['flir_frames'].shape}")


if __name__ == "__main__":
    main()
