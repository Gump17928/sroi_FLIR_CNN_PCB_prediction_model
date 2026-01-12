"""
===============================================================================
THERMISTOR DATA LOADER
===============================================================================
Data loading utilities for thermistor and FLIR thermal measurements.

Classes:
- ThermalDataLoader: Load FLIR camera and USB thermistor CSV files
- SteadyStateAnalyzer: Calculate steady-state temperature values

Used by: phase6_thermal_calibration.py, viz_phase6_validation.py
Data sources: FLIR ResearchIR CSV exports, USB thermistor DAQami CSV files

Author: Thermal Analysis Pipeline
Date: November 25, 2025
Updated: December 2, 2025 - Split from thermistor_validation.py
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from typing import Tuple, List, Dict, Optional
import re

# Configure matplotlib for LaTeX-style rendering
plt.rcParams.update({
    'text.usetex': False,  # Set to True if LaTeX is installed
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


class ThermalDataLoader:
    """Loads and parses FLIR camera and USB thermistor CSV files.
    
    Supports loading single files or merging multiple USB device files
    (e.g., USB-TEMP Device 0 + USB-TEMP-AI Device 1) into a combined dataset.
    """
    
    @staticmethod
    def load_temp_csv(csv_file: Path, legend_override: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Load temperature CSV file (FLIR or USB thermistor format).
        
        Args:
            csv_file: Path to CSV file
            legend_override: Optional list of legend names for FLIR data
            
        Returns:
            x: Time vector (seconds)
            Y: Data matrix (N x M) where N=samples, M=channels
            labels: Channel labels (1 x M)
            meta: Metadata dict with folder, base, type, scanRate
        """
        csv_file = Path(csv_file)
        meta = {
            'folder': str(csv_file.parent),
            'base': csv_file.stem,
            'type': '',
            'scanRate': np.nan
        }
        
        # Read all lines to detect file type
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Detect USB-TEMP format (contains "Sample,Date/Time" header)
        usb_header_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Sample,Date/Time"):
                usb_header_line = i
                break
        
        is_usb = usb_header_line is not None
        meta['type'] = 'usb' if is_usb else 'flir'
        
        if is_usb:
            return ThermalDataLoader._load_usb_csv(csv_file, lines, usb_header_line, meta)
        else:
            return ThermalDataLoader._load_flir_csv(csv_file, legend_override, meta)
    
    @staticmethod
    def load_multi_device_csv(csv_files: List[Path], device_prefixes: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Load and merge multiple USB thermistor CSV files from different devices.
        
        Handles channel naming conflicts by prefixing channels with device identifier.
        For example, if both Device 0 and Device 1 have AI4, they become:
        - Device0_AI4 (°C)
        - Device1_AI4 (°C)
        
        Args:
            csv_files: List of paths to USB thermistor CSV files
            device_prefixes: Optional list of device prefixes (e.g., ['Device0', 'Device1'])
                           If None, auto-detected from filenames or numbered sequentially
            
        Returns:
            x: Merged time vector (seconds) - uses first file's time base
            Y: Merged data matrix (N x M) where M = sum of all channels across devices
            labels: Merged channel labels with device prefixes to avoid conflicts
            meta: Metadata dict with combined information
        """
        if not csv_files:
            raise ValueError("No CSV files provided")
        
        # Convert to Path objects
        csv_files = [Path(f) for f in csv_files]
        
        # Auto-detect device prefixes from filenames if not provided
        if device_prefixes is None:
            device_prefixes = []
            for csv_file in csv_files:
                # Try to extract device number from filename
                # Pattern: "USB-TEMP (Device 0)" or "USB-TEMP-AI (Device 1)"
                match = re.search(r'Device\s+(\d+)', csv_file.name, re.IGNORECASE)
                if match:
                    device_prefixes.append(f"Device{match.group(1)}")
                else:
                    # Fallback: use sequential numbering
                    device_prefixes.append(f"Device{len(device_prefixes)}")
        
        # Load all files
        all_data = []
        reference_time = None
        
        for i, (csv_file, dev_prefix) in enumerate(zip(csv_files, device_prefixes)):
            print(f"  Loading {dev_prefix}: {csv_file.name}")
            x, Y, labels, meta = ThermalDataLoader.load_temp_csv(csv_file)
            
            # Use first file's time vector as reference
            if reference_time is None:
                reference_time = x
            
            # Check time vector compatibility (allow small differences)
            if len(x) != len(reference_time):
                print(f"    Warning: {dev_prefix} has {len(x)} samples vs {len(reference_time)} in reference")
                # Resample or truncate to match reference
                min_len = min(len(x), len(reference_time))
                Y = Y[:min_len, :]
                x = x[:min_len]
                reference_time = reference_time[:min_len]
                
                # Retroactively trim all previously loaded devices
                for prev_data in all_data:
                    prev_data['Y'] = prev_data['Y'][:min_len, :]
            
            # Prefix labels with device identifier to avoid conflicts
            prefixed_labels = [f"{dev_prefix}_{label}" for label in labels]
            
            all_data.append({
                'Y': Y,
                'labels': prefixed_labels,
                'meta': meta,
                'device': dev_prefix
            })
            
            print(f"    Loaded {Y.shape[1]} channels: {', '.join(prefixed_labels)}")
        
        # Merge all data horizontally
        merged_Y = np.hstack([d['Y'] for d in all_data])
        merged_labels = []
        for d in all_data:
            merged_labels.extend(d['labels'])
        
        # Create combined metadata
        merged_meta = {
            'folder': str(csv_files[0].parent),
            'base': 'multi_device',
            'type': 'usb_multi',
            'scanRate': all_data[0]['meta'].get('scanRate', np.nan),
            'devices': [d['device'] for d in all_data],
            'source_files': [str(f) for f in csv_files]
        }
        
        print(f"\n  Merged: {merged_Y.shape[1]} total channels from {len(csv_files)} devices")
        
        return reference_time, merged_Y, merged_labels, merged_meta
    
    @staticmethod
    def _load_usb_csv(csv_file: Path, lines: List[str], header_line: int, meta: Dict) -> Tuple:
        """Load USB thermistor CSV format."""
        # Extract scan rate if present
        for line in lines:
            match = re.search(r'Scan\s*Rate\s*:\s*([0-9]*\.?[0-9]+)', line, re.IGNORECASE)
            if match:
                meta['scanRate'] = float(match.group(1))
                break
        
        # Read data starting from header line
        df = pd.read_csv(csv_file, skiprows=header_line)
        
        # Get time vector (prefer Date/Time column)
        time_cols = [col for col in df.columns if 'date' in col.lower() and 'time' in col.lower()]
        if time_cols:
            # Parse datetime and convert to relative seconds
            # Try common formats to avoid warning (including fractional seconds)
            dt = None
            for fmt in ['%m/%d/%Y %I:%M:%S.%f %p', '%m/%d/%Y %I:%M:%S %p', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                try:
                    dt = pd.to_datetime(df[time_cols[0]], format=fmt, errors='coerce')
                    if dt.notna().any():
                        break
                except:
                    continue
            # Fall back to automatic parsing if formats don't work
            if dt is None or dt.isna().all():
                dt = pd.to_datetime(df[time_cols[0]], errors='coerce')
            x = (dt - dt.iloc[0]).dt.total_seconds().values
        elif 'Sample' in df.columns and not np.isnan(meta['scanRate']):
            # Use sample number and scan rate
            samples = df['Sample'].values
            x = (samples - samples[0]) / meta['scanRate']
        else:
            raise ValueError("USB-TEMP: need Date/Time or Sample+Scan Rate columns")
        
        # Extract AI* columns (analog input channels)
        ai_cols = [col for col in df.columns if col.startswith('AI')]
        if not ai_cols:
            raise ValueError("No AI* columns found in USB thermistor file")
        
        Y = df[ai_cols].values
        
        # Create labels from header (first row contains channel descriptions)
        headers = lines[header_line].strip().split(',')
        labels = []
        for ai_col in ai_cols:
            # Extract AI number (e.g., 'AI0' from 'AI0_C')
            match = re.search(r'AI(\d+)', ai_col)
            if match:
                ai_num = match.group(0)  # e.g., 'AI0'
                # Find matching header
                matching = [h for h in headers if h.startswith(ai_num)]
                if matching:
                    labels.append(matching[0])
                    continue
            labels.append(ai_col)
        
        return x, Y, labels, meta
    
    @staticmethod
    def _load_flir_csv(csv_file: Path, legend_override: Optional[List[str]], meta: Dict) -> Tuple:
        """Load FLIR camera CSV format."""
        df = pd.read_csv(csv_file)
        
        # Get time vector (reltime column)
        if 'reltime' not in df.columns:
            raise ValueError("FLIR CSV: missing 'reltime' column")
        
        x = df['reltime'].values
        x = x - x[0]  # Zero relative time
        
        # Exclude frame, abstime, reltime columns from data
        exclude_cols = ['frame', 'abstime', 'reltime']
        data_cols = [col for col in df.columns if col.lower() not in [e.lower() for e in exclude_cols]]
        
        Y = df[data_cols].values
        
        # Labels: use override if provided, else column names
        if legend_override and len(legend_override) >= len(df.columns):
            # Map override to data columns (excluding frame/time columns)
            col_indices = [i for i, col in enumerate(df.columns) if col in data_cols]
            labels = [legend_override[i] for i in col_indices]
        else:
            # Use first line of CSV as labels if possible
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()
            if first_line:
                cols = first_line.split(',')
                if len(cols) == len(df.columns):
                    labels = [cols[i] for i, col in enumerate(df.columns) if col in data_cols]
                else:
                    labels = data_cols
            else:
                labels = data_cols
        
        return x, Y, labels, meta


class SteadyStateAnalyzer:
    """Analyzes temperature data to find steady-state values."""
    
    @staticmethod
    def steady_value(t: np.ndarray, y: np.ndarray, 
                     smooth_win: int = 7, 
                     min_dur: float = 120.0,
                     slope_frac: float = 0.15,
                     flat_frac: float = 0.02,
                     center: str = 'median') -> Dict:
        """
        Find steady-state value from time series.
        
        Args:
            t: Time vector (seconds)
            y: Temperature data
            smooth_win: Smoothing window size
            min_dur: Minimum duration for steady state (seconds)
            slope_frac: Slope threshold as fraction of (range(y)/range(t))
            flat_frac: Flatness threshold as fraction of range(y)
            center: 'median' or 'mean' for steady-state value
            
        Returns:
            Dictionary with 'value', 'idx' (start, end), 'info'
        """
        t = t.flatten()
        y = y.flatten()
        
        # Calculate time step
        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            dt = (t[-1] - t[0]) / max(1, len(t) - 1)
        
        # Smooth for slope stability
        if smooth_win > 1:
            y_smooth = np.convolve(y, np.ones(smooth_win)/smooth_win, mode='same')
        else:
            y_smooth = y
        
        # Calculate slope threshold
        y_range = np.ptp(y)  # range
        t_range = t[-1] - t[0]
        eps_slope = slope_frac * (y_range / max(1e-9, t_range))
        
        # Find regions with low slope
        dy = np.gradient(y_smooth, t)
        mask = np.abs(dy) <= eps_slope
        
        # Find last run of at least min_dur
        min_n = max(1, int(min_dur / max(dt, 1e-9)))
        idx_run = SteadyStateAnalyzer._last_run(mask, min_n)
        
        if idx_run is None or len(idx_run) == 0:
            idx_run = np.arange(max(0, len(t) - min_n), len(t))
        
        # Refine for flatness
        flat_thr = flat_frac * y_range
        while len(idx_run) > min_n and np.ptp(y[idx_run]) > flat_thr:
            idx_run = idx_run[1:]
        
        # Calculate steady-state value
        if center == 'mean':
            value = np.nanmean(y[idx_run])
        else:
            value = np.nanmedian(y[idx_run])
        
        return {
            'value': value,
            'idx': (idx_run[0], idx_run[-1]),
            'info': {
                'eps_slope': eps_slope,
                'min_n': min_n,
                'range_y': y_range,
                'flat_thr': flat_thr
            }
        }
    
    @staticmethod
    def _last_run(mask: np.ndarray, min_n: int) -> Optional[np.ndarray]:
        """Find indices of last run of True values with minimum length."""
        # Pad mask for edge detection
        m = np.concatenate([[False], mask, [False]])
        
        # Find start and end of runs
        diff = np.diff(m.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        if len(starts) == 0:
            return None
        
        # Find runs >= min_n
        lengths = ends - starts + 1
        valid = np.where(lengths >= min_n)[0]
        
        if len(valid) == 0:
            return None
        
        # Return last valid run
        last_idx = valid[-1]
        return np.arange(starts[last_idx], ends[last_idx] + 1)
    """Main class for comparing FLIR and thermistor measurements."""
    
    def __init__(self, 
                 fig_size: Tuple[float, float] = (7.0, 3.5),
                 line_width: float = 2.0,
                 dpi: int = 600,
                 smooth_win: int = 20,
                 y_tick_step: float = 1.0,
                 y_tick_step_small: float = 0.01,
                 data_shift: float = 10.0):
        """
        Initialize validator with plot parameters.
        
        Args:
            fig_size: Figure size in inches (width, height)
            line_width: Line width for plots
            dpi: Resolution for exported images
            smooth_win: Smoothing window size (samples)
            y_tick_step: Y-axis tick step for temperature plots
            y_tick_step_small: Y-axis tick step for ratio plots
            data_shift: Seconds to skip at start (avoid transients)
        """
        self.fig_size = fig_size
        self.line_width = line_width
        self.dpi = dpi
        self.smooth_win = smooth_win
        self.y_tick_step = y_tick_step
        self.y_tick_step_small = y_tick_step_small
        self.data_shift = data_shift
        
        # Default color scheme
        self.default_colors = plt.cm.tab10.colors
    
    def plot_pair_compare(self,
                         x_a: np.ndarray, y_a: np.ndarray, labels_a: List[str], meta_a: Dict,
                         x_b: np.ndarray, y_b: np.ndarray, labels_b: List[str], meta_b: Dict,
                         pairs: List[Tuple],
                         tiles: int = 2,
                         legend_ab: Optional[List[str]] = None,
                         ss_mode: str = 'separate',
                         export_dir: Optional[Path] = None,
                         export_base: Optional[str] = None) -> None:
        """
        Create pair comparison plots for matched channels.
        
        Args:
            x_a, y_a, labels_a, meta_a: Dataset A (time, data, labels, metadata)
            x_b, y_b, labels_b, meta_b: Dataset B
            pairs: List of tuples (patternA, patternB, baseTag, labelA, labelB)
            tiles: Number of subplot tiles (2 = overlay + delta)
            legend_ab: Optional custom legend labels [A_name, B_name]
            ss_mode: 'separate' or 'overlap' for steady-state calculation
            export_dir: Output directory for plots
            export_base: Base name for exported files
        """
        if export_dir is None:
            export_dir = Path(meta_a['folder'])
        if export_base is None:
            export_base = meta_a['base']
        
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate legend labels
        lbl = self._make_labels(meta_a, meta_b, legend_ab)
        
        for pair in pairs:
            pat_a, pat_b, base_tag = pair[0], pair[1], pair[2]
            
            # Find matching channels
            idx_a = self._find_label_idx(labels_a, pat_a)
            idx_b = self._find_label_idx(labels_b, pat_b)
            
            if idx_a is None or idx_b is None:
                print(f"Warning: Could not find pair {pat_a} <-> {pat_b}, skipping...")
                continue
            
            y_a_single = y_a[:, idx_a]
            y_b_single = y_b[:, idx_b]
            
            # Clean and deduplicate
            x_a_clean, y_a_clean = self._dedupe_sort_mean(x_a, y_a_single)
            x_b_clean, y_b_clean = self._dedupe_sort_mean(x_b, y_b_single)
            
            # Create union timebase
            t_all = np.unique(np.concatenate([x_a_clean, x_b_clean]))
            y_a_union = np.interp(t_all, x_a_clean, y_a_clean)
            y_b_union = np.interp(t_all, x_b_clean, y_b_clean)
            
            # Apply smoothing
            if self.smooth_win > 1:
                kernel = np.ones(self.smooth_win) / self.smooth_win
                y_a_union = np.convolve(y_a_union, kernel, mode='same')
                y_b_union = np.convolve(y_b_union, kernel, mode='same')
            
            # Calculate steady-state values
            ss_a = SteadyStateAnalyzer.steady_value(x_a_clean, y_a_clean, 
                                                    smooth_win=max(5, self.smooth_win//2))
            ss_b = SteadyStateAnalyzer.steady_value(x_b_clean, y_b_clean,
                                                    smooth_win=max(5, self.smooth_win//2))
            
            # Overlap data for measured delta/ratio
            valid_mask = np.isfinite(y_a_union) & np.isfinite(y_b_union)
            t_overlap = t_all[valid_mask]
            y_a_overlap = y_a_union[valid_mask]
            y_b_overlap = y_b_union[valid_mask]
            
            # Calculate steady-state delta and ratio
            if ss_mode == 'overlap' and len(t_overlap) >= 3:
                delta_meas = y_b_overlap - y_a_overlap
                ss_delta = SteadyStateAnalyzer.steady_value(t_overlap, delta_meas,
                                                            smooth_win=max(3, self.smooth_win//3))
                ss_delta_val = ss_delta['value']
            else:
                ss_delta_val = ss_b['value'] - ss_a['value']
            
            # Calculate full delta (with hybrid tail substitution)
            t_end_a = x_a_clean[-1]
            t_end_b = x_b_clean[-1]
            y_a_fill = y_a_union.copy()
            y_b_fill = y_b_union.copy()
            y_a_fill[t_all > t_end_a] = ss_a['value']
            y_b_fill[t_all > t_end_b] = ss_b['value']
            delta_full = y_b_fill - y_a_fill
            
            # Create figure
            fig, axes = plt.subplots(1, tiles, figsize=self.fig_size)
            if tiles == 1:
                axes = [axes]
            
            # Left plot: Overlay
            self._render_overlay(axes[0], t_all, y_a_union, y_b_union,
                               base_tag, lbl, ss_a, ss_b)
            
            # Right plot: Delta T
            if tiles >= 2:
                self._render_delta(axes[1], t_all, delta_full,
                                 base_tag, lbl, ss_delta_val)
            
            plt.tight_layout()
            
            # Export PNG and PDF
            safe_tag = re.sub(r'[^A-Za-z0-9]+', '_', base_tag)
            out_png = export_dir / f"{export_base}_{safe_tag}_pair_compare.png"
            out_pdf = export_dir / f"{export_base}_{safe_tag}_pair_compare.pdf"
            fig.savefig(out_png, dpi=self.dpi, bbox_inches='tight')
            fig.savefig(out_pdf, bbox_inches='tight')
            print(f"Saved: {out_png}")
            print(f"Saved: {out_pdf}")
            plt.close(fig)
    
    def _render_overlay(self, ax, t, y_a, y_b, base_tag, lbl, ss_a, ss_b):
        """Render overlay plot of two temperature series."""
        # Find data start index (skip initial transient)
        start_idx = np.searchsorted(t, self.data_shift)
        
        # Plot data
        ax.plot(t, y_a, '-', linewidth=self.line_width, 
               color=self.default_colors[0], label=lbl['A'])
        ax.plot(t, y_b, '-', linewidth=self.line_width,
               color=self.default_colors[1], label=lbl['B'])
        
        # Steady-state lines
        color_a_light = tuple(0.7 * c + 0.3 for c in self.default_colors[0])
        color_b_light = tuple(0.7 * c + 0.3 for c in self.default_colors[1])
        ax.axhline(ss_a['value'], linestyle='--', linewidth=1.0,
                  color=color_a_light, label=f"{lbl['A']} SS")
        ax.axhline(ss_b['value'], linestyle='--', linewidth=1.0,
                  color=color_b_light, label=f"{lbl['B']} SS")
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, alpha=0.3)
        
        # Set limits with tighter zoom on actual data range
        ax.set_xlim(t[start_idx], t[-1])
        
        # Get data range after shift, including steady-state lines
        y_data = np.concatenate([y_a[start_idx:], y_b[start_idx:]])
        y_data_with_ss = np.concatenate([y_data, [ss_a['value'], ss_b['value']]])
        
        # Use 1st and 99th percentile to avoid outliers affecting zoom
        y_min = np.nanpercentile(y_data_with_ss, 1)
        y_max = np.nanpercentile(y_data_with_ss, 99)
        
        # Ensure steady-state values are included
        y_min = min(y_min, ss_a['value'], ss_b['value'])
        y_max = max(y_max, ss_a['value'], ss_b['value'])
        
        # Very tight margins: just 2°C total padding
        y_min_plot = y_min - 1.0
        y_max_plot = y_max + 1.0
        
        # Round to tick values but keep tight
        y_lo = np.floor(y_min_plot)
        y_hi = np.ceil(y_max_plot)
        if y_hi == y_lo:
            y_hi = y_lo + 2
        
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(np.arange(y_lo, y_hi + self.y_tick_step/2, self.y_tick_step))
        
        # Position legend to avoid data (try upper right, then lower right, then best)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        ax.set_title(f"{base_tag}: {lbl['title']}", fontsize=8)
    
    def _render_delta(self, ax, t, delta, base_tag, lbl, ss_delta_val):
        """Render delta T plot."""
        # Find data start index
        start_idx = np.searchsorted(t, self.data_shift)
        
        # Plot data
        has_data = np.any(np.isfinite(delta))
        if has_data:
            ax.plot(t, delta, '-', linewidth=self.line_width,
                   color=self.default_colors[2], label=lbl['delta'])
            d_min = np.nanmin(delta[start_idx:])
            d_max = np.nanmax(delta[start_idx:])
        else:
            d_min = ss_delta_val
            d_max = ss_delta_val
        
        # Steady-state line
        color_light = tuple(0.7 * c + 0.3 for c in self.default_colors[2])
        ax.axhline(ss_delta_val, linestyle='--', linewidth=1.0,
                  color=color_light, label=f"SS {lbl['delta']}")
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ΔT (°C)')
        ax.grid(True, alpha=0.3)
        
        # Set limits with tighter zoom
        ax.set_xlim(t[start_idx], t[-1])
        
        # Use percentiles to avoid outliers
        if has_data:
            d_min_p = np.nanpercentile(delta[start_idx:], 1)
            d_max_p = np.nanpercentile(delta[start_idx:], 99)
        else:
            d_min_p = ss_delta_val
            d_max_p = ss_delta_val
        
        # Include steady-state value
        d_min_plot = min(d_min_p, ss_delta_val) - 0.5  # Just 0.5°C below
        d_max_plot = max(d_max_p, ss_delta_val) + 0.5  # Just 0.5°C above
        
        # Round but keep very tight
        d_lo = np.floor(d_min_plot)
        d_hi = np.ceil(d_max_plot)
        if d_hi == d_lo:
            d_hi = d_lo + 1
        
        ax.set_ylim(d_lo, d_hi)
        ax.set_yticks(np.arange(d_lo, d_hi + self.y_tick_step/2, self.y_tick_step))
        
        # Position legend to avoid data
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        ax.set_title(f"{base_tag}: ΔT", fontsize=8)
    
    def _make_labels(self, meta_a: Dict, meta_b: Dict, override_ab: Optional[List[str]]) -> Dict:
        """Generate legend labels for datasets."""
        if override_ab and len(override_ab) >= 2:
            a_name = override_ab[0]
            b_name = override_ab[1]
        else:
            a_name = 'FLIR' if meta_a['type'] == 'flir' else 'Thermistor'
            b_name = 'FLIR' if meta_b['type'] == 'flir' else 'Thermistor'
        
        return {
            'A': a_name,
            'B': b_name,
            'delta': f"ΔT ({b_name}-{a_name})",
            'ratio': f"{b_name}/{a_name}",
            'title': f"{a_name} vs {b_name}"
        }
    
    @staticmethod
    def _find_label_idx(labels: List[str], pattern: str) -> Optional[int]:
        """Find index of label matching pattern (case-insensitive)."""
        # Try exact match first
        for i, lbl in enumerate(labels):
            if pattern.lower() in lbl.lower():
                return i
        
        # Try regex match
        for i, lbl in enumerate(labels):
            if re.search(pattern, lbl, re.IGNORECASE):
                return i
        
        return None
    
    @staticmethod
    def _dedupe_sort_mean(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove NaN, sort by time, and average duplicate timestamps."""
        # Remove invalid data
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        
        # Sort
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        # Average duplicates
        x_unique, inverse = np.unique(x, return_inverse=True)
        y_unique = np.array([np.mean(y[inverse == i]) for i in range(len(x_unique))])
        
        return x_unique, y_unique


def main():
    """
    Main execution function for thermistor validation.
    
    This function demonstrates the typical workflow:
    1. Load FLIR camera data
    2. Load USB thermistor data (Air and Sand conditions)
    3. Compare matched measurement pairs
    4. Export comparison plots
    """
    # Define input files (configurable)
    input_dir = Path(__file__).parent / "inputs"
    output_dir = Path(__file__).parent / "outputs"
    
    # Default input files
    flir_file = input_dir / "Test_3_FLIR_Camera_Results_5_of_7.csv"
    air_file = input_dir / "Test_3_AIR_usb_temp_DAQami.csv"
    sand_file = input_dir / "Test_3_SAND_usb_temp_DAQami.csv"
    
    # Check if files exist
    if not flir_file.exists():
        print(f"Error: FLIR file not found: {flir_file}")
        return
    if not air_file.exists():
        print(f"Error: Air thermistor file not found: {air_file}")
        return
    
    has_sand = sand_file.exists()
    
    print("=" * 80)
    print("THERMISTOR VALIDATION ANALYSIS")
    print("=" * 80)
    print(f"\nLoading data files...")
    print(f"  FLIR:  {flir_file.name}")
    print(f"  Air:   {air_file.name}")
    if has_sand:
        print(f"  Sand:  {sand_file.name}")
    
    # Load data
    loader = ThermalDataLoader()
    x1, y1, labels1, meta1 = loader.load_temp_csv(flir_file)
    x2, y2, labels2, meta2 = loader.load_temp_csv(air_file)
    
    print(f"\nFLIR data: {len(x1)} samples, {y1.shape[1]} channels")
    print(f"  Channels: {', '.join(labels1[:5])}..." if len(labels1) > 5 else f"  Channels: {', '.join(labels1)}")
    print(f"\nAir thermistor data: {len(x2)} samples, {y2.shape[1]} channels")
    print(f"  Channels: {', '.join(labels2[:5])}..." if len(labels2) > 5 else f"  Channels: {', '.join(labels2)}")
    
    if has_sand:
        x3, y3, labels3, meta3 = loader.load_temp_csv(sand_file)
        print(f"\nSand thermistor data: {len(x3)} samples, {y3.shape[1]} channels")
        print(f"  Channels: {', '.join(labels3[:5])}..." if len(labels3) > 5 else f"  Channels: {', '.join(labels3)}")
    
    # Initialize validator
    validator = ThermistorValidator(
        fig_size=(7.0, 3.5),
        line_width=2.0,
        dpi=600,
        smooth_win=20,
        y_tick_step=1.0,
        data_shift=10.0
    )
    
    # Define comparison pairs
    # Format: (FLIR_pattern, USB_pattern, display_name, air_label, sand_label)
    pair_flir_comp = [
        ('Box 3', 'AI0', 'LDO2', 'therm-air', 'therm-sand'),
        ('Box 1', 'AI1', 'PS2', 'therm-air', 'therm-sand'),
        ('SMD RES1', 'AI6', 'SMD_RES1', 'therm-air', 'therm-sand'),
        ('LED1', 'AI7', 'LED1', 'therm-air', 'therm-sand'),
    ]
    
    pair_therm_comp = [
        ('AI0', 'AI0', 'LDO2', 'therm-air', 'therm-sand'),
        ('AI1', 'AI1', 'PS2', 'therm-air', 'therm-sand'),
        ('AI4', 'AI4', 'Ambient', 'therm-air', 'therm-sand'),
        ('AI6', 'AI6', 'SMD_RES1', 'therm-air', 'therm-sand'),
        ('AI7', 'AI7', 'LED1', 'therm-air', 'therm-sand'),
    ]
    
    # Generate comparison plots
    if has_sand:
        print("\n" + "=" * 80)
        print("Generating FLIR vs Sand thermistor comparison plots...")
        print("=" * 80)
        validator.plot_pair_compare(
            x1, y1, labels1, meta1,
            x3, y3, labels3, meta3,
            pair_flir_comp,
            tiles=2,
            legend_ab=['FLIR Air', 'Sand'],
            export_dir=output_dir,
            export_base=f"{meta2['base']}_FLIR_Air_vs_Sand"
        )
        
        print("\n" + "=" * 80)
        print("Generating Air vs Sand thermistor comparison plots...")
        print("=" * 80)
        validator.plot_pair_compare(
            x2, y2, labels2, meta2,
            x3, y3, labels3, meta3,
            pair_therm_comp,
            tiles=2,
            legend_ab=['Air', 'Sand'],
            export_dir=output_dir,
            export_base=f"{meta3['base']}_vsSand"
        )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files saved to: {output_dir}")
    print("\nGenerated plots:")
    for png_file in sorted(output_dir.glob("*.png")):
        print(f"  - {png_file.name}")


if __name__ == "__main__":
    main()
