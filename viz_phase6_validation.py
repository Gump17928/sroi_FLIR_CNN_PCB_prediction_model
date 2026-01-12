"""
===============================================================================
PHASE 6 VISUALIZATION - Thermistor Validation Plots & Configuration Validation
===============================================================================
Visualization functions for Phase 6 (Thermal Calibration):
- FLIR vs thermistor comparison plots
- Air vs sand cooling comparisons
- Delta T analysis plots
- Steady-state validation visualizations
- Configuration validation functions

Used by: phase6_thermal_calibration.py, researchir_post_processor.py
Data sources: FLIR thermal measurements + USB thermistor ground truth

Author: Thermal Analysis Pipeline
Date: November 25, 2025
Updated: December 5, 2025 - Added configuration validation functions
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
import json

# Import data loaders
from loader_thermistor import ThermalDataLoader, SteadyStateAnalyzer

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


class ThermistorValidator:
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


def validate_channel_names(pairs: List, therm_files: any, verbose: bool = False) -> Dict[str, any]:
    """
    Validate that thermistor channel names follow correct conventions.
    
    For multi-device setups, channels MUST have Device{N}_ prefix.
    For single-device setups, channels should NOT have Device prefix.
    
    Supports two pair formats:
    - Array format: [flir_roi, therm_air_chan, component, type] 
    - Dict format: {"flir_roi": "...", "therm_air_chan": "...", ...}
    
    Args:
        pairs: List of component pairs (either list or dict format)
        therm_files: Either single file path (str) or list of file paths
        verbose: If True, print detailed channel information
        
    Returns:
        Dict with validation results:
            - 'valid': bool indicating if all channels are valid
            - 'is_multi_device': bool indicating if multi-device setup detected
            - 'errors': List of error messages
            - 'warnings': List of warning messages
    """
    result = {
        'valid': True,
        'is_multi_device': False,
        'errors': [],
        'warnings': []
    }
    
    # Determine if multi-device setup
    if isinstance(therm_files, list) and len(therm_files) > 1:
        result['is_multi_device'] = True
    
    # Check channel naming conventions
    for i, pair in enumerate(pairs):
        # Handle both array and dict formats
        if isinstance(pair, list):
            # Array format: [flir_roi, therm_air_chan, component, type]
            if len(pair) < 2:
                result['errors'].append(f"Pair {i}: Invalid array format (need at least 2 elements)")
                result['valid'] = False
                continue
            flir_roi = pair[0]
            air_chan = pair[1]
            sand_chan = pair[1] if len(pair) > 1 else ''  # Use same channel for sand
        elif isinstance(pair, dict):
            # Dict format
            flir_roi = pair.get('flir_roi', 'unknown')
            air_chan = pair.get('therm_air_chan', '')
            sand_chan = pair.get('therm_sand_chan', '')
        else:
            result['errors'].append(f"Pair {i}: Unknown pair format (not list or dict)")
            result['valid'] = False
            continue
        
        has_device_prefix_air = air_chan.startswith('Device')
        has_device_prefix_sand = sand_chan.startswith('Device') if sand_chan else False
        
        if result['is_multi_device']:
            # Multi-device: MUST have Device prefix
            if not has_device_prefix_air:
                result['valid'] = False
                result['errors'].append(
                    f"Pair {i} ({flir_roi}): "
                    f"Multi-device setup requires 'Device{{N}}_' prefix for therm_air_chan: '{air_chan}'"
                )
            if sand_chan and not has_device_prefix_sand:
                result['valid'] = False
                result['errors'].append(
                    f"Pair {i} ({flir_roi}): "
                    f"Multi-device setup requires 'Device{{N}}_' prefix for therm_sand_chan: '{sand_chan}'"
                )
        else:
            # Single-device: Should NOT have Device prefix (but warn, don't fail)
            if has_device_prefix_air:
                result['warnings'].append(
                    f"Pair {i} ({flir_roi}): "
                    f"Single-device setup typically doesn't use 'Device{{N}}_' prefix: '{air_chan}'"
                )
            if sand_chan and has_device_prefix_sand:
                result['warnings'].append(
                    f"Pair {i} ({flir_roi}): "
                    f"Single-device setup typically doesn't use 'Device{{N}}_' prefix: '{sand_chan}'"
                )
    
    if verbose:
        print(f"  Multi-device setup: {'Yes' if result['is_multi_device'] else 'No'}")
        print(f"  Channel validation: {'✓ Pass' if result['valid'] else '✗ Fail'}")
        if result['errors']:
            for err in result['errors']:
                print(f"    ERROR: {err}")
        if result['warnings']:
            for warn in result['warnings']:
                print(f"    WARNING: {warn}")
    
    return result


def validate_session_config(session: Dict, check_files_exist: bool = True, verbose: bool = False) -> Dict[str, any]:
    """
    Validate a single session configuration structure.
    
    Checks:
    - Required fields present (name, pcb, flir_file, therm_air_file, pairs)
    - Files exist (if check_files_exist=True)
    - At least one component pair defined
    - Channel naming conventions
    
    Args:
        session: Session configuration dictionary
        check_files_exist: If True, verify all file paths exist
        verbose: If True, print detailed validation output
        
    Returns:
        Dict with validation results:
            - 'valid': bool indicating if session config is valid
            - 'errors': List of error messages
            - 'warnings': List of warning messages
            - 'stats': Dict with statistics (num_pairs, num_files, etc.)
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required fields
    required_fields = ['name', 'pcb', 'flir_file', 'therm_air_file', 'pairs']
    for field in required_fields:
        if field not in session:
            result['valid'] = False
            result['errors'].append(f"Missing required field: '{field}'")
    
    if not result['valid']:
        return result  # Can't proceed without required fields
    
    # Check pairs list
    pairs = session['pairs']
    if not isinstance(pairs, list) or len(pairs) == 0:
        result['valid'] = False
        result['errors'].append("No component pairs defined in 'pairs' list")
        return result
    
    result['stats']['num_pairs'] = len(pairs)
    
    # Validate each pair has required fields (handle both array and dict formats)
    for i, pair in enumerate(pairs):
        if isinstance(pair, list):
            # Array format: [flir_roi, therm_air_chan, component, type]
            if len(pair) < 2:
                result['valid'] = False
                result['errors'].append(f"Pair {i}: Array format requires at least 2 elements [flir_roi, therm_air_chan]")
        elif isinstance(pair, dict):
            # Dict format: validate required fields
            required_pair_fields = ['flir_roi', 'therm_air_chan']
            for field in required_pair_fields:
                if field not in pair:
                    result['valid'] = False
                    result['errors'].append(f"Pair {i}: Missing required field '{field}'")
        else:
            result['valid'] = False
            result['errors'].append(f"Pair {i}: Invalid format (must be list or dict)")
    
    # Check file paths exist
    if check_files_exist:
        # FLIR file
        flir_path = Path(session['flir_file'])
        if not flir_path.exists():
            result['valid'] = False
            result['errors'].append(f"FLIR file not found: {flir_path}")
        
        # Thermistor air files (single or list)
        air_files = session['therm_air_file']
        if isinstance(air_files, str):
            air_files = [air_files]
        
        result['stats']['num_air_files'] = len(air_files)
        for air_file in air_files:
            air_path = Path(air_file)
            if not air_path.exists():
                result['valid'] = False
                result['errors'].append(f"Air thermistor file not found: {air_path}")
        
        # Thermistor sand files (optional, single or list)
        if 'therm_sand_file' in session and session['therm_sand_file']:
            sand_files = session['therm_sand_file']
            if isinstance(sand_files, str):
                sand_files = [sand_files]
            
            result['stats']['num_sand_files'] = len(sand_files)
            for sand_file in sand_files:
                sand_path = Path(sand_file)
                if not sand_path.exists():
                    result['warnings'].append(f"Sand thermistor file not found: {sand_path}")
    
    # Validate channel naming conventions
    therm_files = session['therm_air_file']
    channel_validation = validate_channel_names(pairs, therm_files, verbose=False)
    
    if not channel_validation['valid']:
        result['valid'] = False
        result['errors'].extend(channel_validation['errors'])
    
    result['warnings'].extend(channel_validation['warnings'])
    result['stats']['is_multi_device'] = channel_validation['is_multi_device']
    
    if verbose:
        print(f"\nSession: {session['name']}")
        print(f"  PCB: {session['pcb']}")
        print(f"  Component pairs: {result['stats']['num_pairs']}")
        print(f"  Multi-device: {'Yes' if result['stats'].get('is_multi_device') else 'No'}")
        print(f"  Validation: {'✓ Pass' if result['valid'] else '✗ Fail'}")
        
        if result['errors']:
            print(f"  Errors:")
            for err in result['errors']:
                print(f"    - {err}")
        
        if result['warnings']:
            print(f"  Warnings:")
            for warn in result['warnings']:
                print(f"    - {warn}")
    
    return result


def validate_multi_session_config(config_path: str, check_files_exist: bool = True, verbose: bool = False) -> Dict[str, any]:
    """
    Validate multi-session configuration file structure.
    
    Checks:
    - JSON is valid and parseable
    - Required top-level structure (sessions list)
    - Each session has valid configuration
    - File paths exist (if check_files_exist=True)
    
    Args:
        config_path: Path to JSON configuration file
        check_files_exist: If True, verify all file paths exist
        verbose: If True, print detailed validation output
        
    Returns:
        Dict with validation results:
            - 'valid': bool indicating if entire config is valid
            - 'config': Parsed config dict (or None if parse failed)
            - 'errors': List of error messages
            - 'warnings': List of warning messages
            - 'session_results': List of per-session validation results
            - 'stats': Dict with aggregate statistics
    """
    result = {
        'valid': True,
        'config': None,
        'errors': [],
        'warnings': [],
        'session_results': [],
        'stats': {'total_sessions': 0, 'total_pairs': 0}
    }
    
    config_path = Path(config_path)
    
    # Check config file exists
    if not config_path.exists():
        result['valid'] = False
        result['errors'].append(f"Configuration file not found: {config_path}")
        return result
    
    # Parse JSON
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        result['config'] = config
    except json.JSONDecodeError as e:
        result['valid'] = False
        result['errors'].append(f"Invalid JSON format: {e}")
        return result
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error reading config file: {e}")
        return result
    
    # Check top-level structure
    if 'sessions' not in config:
        result['valid'] = False
        result['errors'].append("Missing required top-level field: 'sessions'")
        return result
    
    sessions = config['sessions']
    if not isinstance(sessions, list) or len(sessions) == 0:
        result['valid'] = False
        result['errors'].append("'sessions' must be a non-empty list")
        return result
    
    result['stats']['total_sessions'] = len(sessions)
    
    # Validate each session
    for i, session in enumerate(sessions):
        session_result = validate_session_config(session, check_files_exist, verbose=verbose)
        result['session_results'].append(session_result)
        
        if not session_result['valid']:
            result['valid'] = False
            result['errors'].append(f"Session {i+1} validation failed")
        
        # Accumulate stats
        result['stats']['total_pairs'] += session_result['stats'].get('num_pairs', 0)
        
        # Accumulate warnings
        result['warnings'].extend([f"Session {i+1}: {w}" for w in session_result['warnings']])
    
    if verbose and not any([s['valid'] for s in result['session_results']]):
        print(f"\nConfiguration file: {config_path}")
        print(f"  Total sessions: {result['stats']['total_sessions']}")
        print(f"  Total component pairs: {result['stats']['total_pairs']}")
        print(f"  Overall validation: {'✓ Pass' if result['valid'] else '✗ Fail'}")
    
    return result


def validate_session_outputs(session_dir: Path, expected_components: List[str], verbose: bool = False) -> Dict[str, any]:
    """
    Validate that a calibration session produced expected outputs.
    
    Checks:
    - raw_inputs_overview.png exists
    - detailed_plots/ folder exists
    - Expected number of PNG/PDF files (2 per component: flir_vs_air, flir_vs_sand)
    - No missing component plots
    
    Args:
        session_dir: Path to session output directory (calibration_database/{SessionName})
        expected_components: List of component names that should have plots
        verbose: If True, print detailed validation output
        
    Returns:
        Dict with validation results:
            - 'valid': bool indicating if all expected outputs exist
            - 'errors': List of error messages
            - 'warnings': List of warning messages
            - 'stats': Dict with file counts
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'has_overview': False,
            'has_detailed_folder': False,
            'png_count': 0,
            'pdf_count': 0,
            'missing_components': []
        }
    }
    
    session_dir = Path(session_dir)
    
    # Check session directory exists
    if not session_dir.exists():
        result['valid'] = False
        result['errors'].append(f"Session directory not found: {session_dir}")
        return result
    
    # Check raw inputs overview
    overview_file = session_dir / 'raw_inputs_overview.png'
    if overview_file.exists():
        result['stats']['has_overview'] = True
    else:
        result['valid'] = False
        result['errors'].append("Missing raw_inputs_overview.png")
    
    # Check detailed plots folder
    detailed_dir = session_dir / 'detailed_plots'
    if detailed_dir.exists():
        result['stats']['has_detailed_folder'] = True
        
        # Count files
        png_files = list(detailed_dir.glob('*.png'))
        pdf_files = list(detailed_dir.glob('*.pdf'))
        
        result['stats']['png_count'] = len(png_files)
        result['stats']['pdf_count'] = len(pdf_files)
        
        # Check for missing components
        for component in expected_components:
            # Look for both air and sand plots
            air_png = detailed_dir / f"{component}_flir_vs_air.png"
            sand_png = detailed_dir / f"{component}_flir_vs_sand.png"
            
            if not air_png.exists():
                result['warnings'].append(f"Missing air plot for component: {component}")
                if component not in result['stats']['missing_components']:
                    result['stats']['missing_components'].append(component)
            
            if not sand_png.exists():
                result['warnings'].append(f"Missing sand plot for component: {component}")
                if component not in result['stats']['missing_components']:
                    result['stats']['missing_components'].append(component)
        
        # Check expected file counts (should be 2 PNG + 2 PDF per component)
        expected_png = len(expected_components) * 2  # air + sand
        expected_pdf = len(expected_components) * 2
        
        if result['stats']['png_count'] < expected_png:
            result['warnings'].append(
                f"Expected {expected_png} PNG files, found {result['stats']['png_count']}"
            )
        
        if result['stats']['pdf_count'] < expected_pdf:
            result['warnings'].append(
                f"Expected {expected_pdf} PDF files, found {result['stats']['pdf_count']}"
            )
    else:
        result['valid'] = False
        result['errors'].append("Missing detailed_plots/ folder")
    
    if verbose:
        print(f"\nSession output validation: {session_dir.name}")
        print(f"  Raw overview: {'✓' if result['stats']['has_overview'] else '✗'}")
        print(f"  Detailed plots folder: {'✓' if result['stats']['has_detailed_folder'] else '✗'}")
        if result['stats']['has_detailed_folder']:
            print(f"    PNG files: {result['stats']['png_count']}")
            print(f"    PDF files: {result['stats']['pdf_count']}")
        print(f"  Validation: {'✓ Pass' if result['valid'] else '✗ Fail'}")
        
        if result['errors']:
            print(f"  Errors:")
            for err in result['errors']:
                print(f"    - {err}")
        
        if result['warnings']:
            print(f"  Warnings:")
            for warn in result['warnings']:
                print(f"    - {warn}")
    
    return result


def validate_test_config(test_config: Dict, test_id: str, board_name: str, verbose: bool = False) -> Dict:
    """
    Validate a single test configuration within a board.
    
    Parameters:
        test_config: Dict containing test configuration
        test_id: Test identifier for error messages
        board_name: Board name for context
        verbose: If True, print detailed validation info
    
    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'component_count': 0,
            'is_multi_device': False,
            'has_air': False,
            'has_sand': False,
            'missing_files': []
        }
    }
    
    # Check for empty/placeholder tests
    if 'comment' in test_config and not test_config.get('flir_file'):
        result['warnings'].append(f"Test {test_id}: Placeholder test (no data)")
        return result
    
    # Check required fields (flir_file is optional when using ResearchIR_Outputs parsing)
    required_fields = ['test_id', 'therm_air_file', 'pairs']
    for field in required_fields:
        if field not in test_config:
            result['valid'] = False
            result['errors'].append(f"Test {test_id}: Missing required field '{field}'")
    
    if not result['valid']:
        return result
    
    # Validate FLIR file path if provided (optional)
    if 'flir_file' in test_config:
        flir_file = test_config['flir_file']
        if isinstance(flir_file, str):
            if not Path(flir_file).exists():
                result['stats']['missing_files'].append(flir_file)
                result['warnings'].append(f"Test {test_id}: FLIR file not found: {flir_file} (will use ResearchIR_Outputs if available)")
        else:
            result['errors'].append(f"Test {test_id}: flir_file must be a string, got {type(flir_file)}")
            result['valid'] = False
    
    # Validate thermistor air files (can be string or list for multi-device)
    therm_air = test_config['therm_air_file']
    if isinstance(therm_air, list):
        result['stats']['is_multi_device'] = True
        for i, file in enumerate(therm_air):
            if not Path(file).exists():
                result['stats']['missing_files'].append(file)
                result['errors'].append(f"Test {test_id}: Air thermistor file {i} not found: {file}")
                result['valid'] = False
            else:
                result['stats']['has_air'] = True
    elif isinstance(therm_air, str):
        if not Path(therm_air).exists():
            result['stats']['missing_files'].append(therm_air)
            result['errors'].append(f"Test {test_id}: Air thermistor file not found: {therm_air}")
            result['valid'] = False
        else:
            result['stats']['has_air'] = True
    else:
        result['errors'].append(f"Test {test_id}: therm_air_file must be string or list, got {type(therm_air)}")
        result['valid'] = False
    
    # Validate thermistor sand files (optional)
    if 'therm_sand_file' in test_config:
        therm_sand = test_config['therm_sand_file']
        if isinstance(therm_sand, list):
            for i, file in enumerate(therm_sand):
                if not Path(file).exists():
                    result['stats']['missing_files'].append(file)
                    result['warnings'].append(f"Test {test_id}: Sand thermistor file {i} not found: {file}")
                else:
                    result['stats']['has_sand'] = True
        elif isinstance(therm_sand, str):
            if not Path(therm_sand).exists():
                result['stats']['missing_files'].append(therm_sand)
                result['warnings'].append(f"Test {test_id}: Sand thermistor file not found: {therm_sand}")
            else:
                result['stats']['has_sand'] = True
    
    # Validate pairs
    if not isinstance(test_config['pairs'], list):
        result['valid'] = False
        result['errors'].append(f"Test {test_id}: 'pairs' must be a list")
        return result
    
    if len(test_config['pairs']) == 0:
        result['valid'] = False
        result['errors'].append(f"Test {test_id}: No measurement pairs defined")
        return result
    
    result['stats']['component_count'] = len(test_config['pairs'])
    
    # Validate each pair and check for multi-device channel naming
    components_seen = set()
    channels_seen = set()
    
    for i, pair in enumerate(test_config['pairs']):
        # Support both dict and array formats
        if isinstance(pair, dict):
            flir_roi = pair.get('flir_roi')
            therm_channel = pair.get('therm_channel')
            component = pair.get('component')
            comp_type = pair.get('type')
        elif isinstance(pair, list) and len(pair) >= 3:
            flir_roi, therm_channel, component = pair[0], pair[1], pair[2]
            comp_type = pair[3] if len(pair) > 3 else None
        else:
            result['errors'].append(f"Test {test_id}: Pair {i} has invalid format")
            result['valid'] = False
            continue
        
        # Check for duplicates
        if component in components_seen:
            result['warnings'].append(f"Test {test_id}: Duplicate component '{component}' in pairs")
        components_seen.add(component)
        
        if therm_channel in channels_seen:
            result['warnings'].append(f"Test {test_id}: Duplicate channel '{therm_channel}' in pairs")
        channels_seen.add(therm_channel)
        
        # Check multi-device channel naming
        if result['stats']['is_multi_device']:
            if not re.match(r'Device\d+_AI\d+', therm_channel):
                result['errors'].append(
                    f"Test {test_id}: Multi-device test requires 'Device{{N}}_AI{{M}}' format "
                    f"for channel '{therm_channel}' (component '{component}')"
                )
                result['valid'] = False
    
    # Validate channel naming matches device count
    if result['stats']['is_multi_device']:
        # Extract device numbers from channels
        device_numbers = set()
        for pair in test_config['pairs']:
            if isinstance(pair, dict):
                channel = pair.get('therm_channel', '')
            elif isinstance(pair, list):
                channel = pair[1]
            else:
                continue
            
            match = re.match(r'Device(\d+)_AI\d+', channel)
            if match:
                device_numbers.add(int(match.group(1)))
        
        # Check device count matches file count
        therm_air = test_config['therm_air_file']
        if isinstance(therm_air, list):
            expected_devices = len(therm_air)
            if device_numbers and max(device_numbers) + 1 > expected_devices:
                result['warnings'].append(
                    f"Test {test_id}: Channel names reference Device{max(device_numbers)}, "
                    f"but only {expected_devices} device files provided"
                )
    
    if verbose:
        status = '✓' if result['valid'] else '✗'
        print(f"\n{status} Test {test_id} ({board_name}):")
        print(f"    Components: {result['stats']['component_count']}")
        print(f"    Multi-device: {result['stats']['is_multi_device']}")
        print(f"    Air data: {'✓' if result['stats']['has_air'] else '✗'}")
        print(f"    Sand data: {'✓' if result['stats']['has_sand'] else '(none)'}")
        
        if result['errors']:
            print(f"    Errors:")
            for err in result['errors']:
                print(f"      - {err}")
        if result['warnings']:
            print(f"    Warnings:")
            for warn in result['warnings']:
                print(f"      - {warn}")
    
    return result


def validate_board_config(board_config: Dict, verbose: bool = False) -> Dict:
    """
    Validate a board configuration with multiple tests.
    
    Parameters:
        board_config: Dict containing board configuration with tests
        verbose: If True, print detailed validation info
    
    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'board_name': board_config.get('name', 'Unknown'),
            'test_count': 0,
            'total_components': 0,
            'tests_passed': 0,
            'tests_failed': 0
        }
    }
    
    # Check required board fields
    if 'name' not in board_config:
        result['valid'] = False
        result['errors'].append("Board missing 'name' field")
    
    if 'tests' not in board_config:
        result['valid'] = False
        result['errors'].append(f"Board '{result['stats']['board_name']}' missing 'tests' field")
        return result
    
    if not isinstance(board_config['tests'], list):
        result['valid'] = False
        result['errors'].append(f"Board '{result['stats']['board_name']}' 'tests' must be a list")
        return result
    
    result['stats']['test_count'] = len(board_config['tests'])
    
    # Validate each test
    for test in board_config['tests']:
        test_id = test.get('test_id', 'Unknown')
        test_result = validate_test_config(test, test_id, result['stats']['board_name'], verbose=verbose)
        
        # Aggregate stats
        if test_result['valid']:
            result['stats']['tests_passed'] += 1
        else:
            result['stats']['tests_failed'] += 1
            result['valid'] = False
        
        result['stats']['total_components'] += test_result['stats']['component_count']
        result['errors'].extend(test_result['errors'])
        result['warnings'].extend(test_result['warnings'])
    
    if verbose and not any(t.get('comment') for t in board_config['tests']):
        status = '✓' if result['valid'] else '✗'
        print(f"\n{status} Board: {result['stats']['board_name']}")
        print(f"    Tests: {result['stats']['test_count']} "
              f"({result['stats']['tests_passed']} passed, {result['stats']['tests_failed']} failed)")
        print(f"    Total components: {result['stats']['total_components']}")
    
    return result


def validate_multi_board_config(config_file: str, verbose: bool = False) -> Dict:
    """
    Validate complete multi-board configuration file.
    
    Parameters:
        config_file: Path to JSON config file
        verbose: If True, print detailed validation info
    
    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'board_count': 0,
            'test_count': 0,
            'component_count': 0,
            'boards_passed': 0,
            'boards_failed': 0
        }
    }
    
    # Load config file
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        result['valid'] = False
        result['errors'].append(f"Config file not found: {config_file}")
        return result
    except json.JSONDecodeError as e:
        result['valid'] = False
        result['errors'].append(f"Invalid JSON in config file: {e}")
        return result
    
    if verbose:
        print(f"=" * 80)
        print(f"Validating multi-board configuration: {Path(config_file).name}")
        print(f"=" * 80)
    
    # Check top-level structure
    if 'boards' not in config:
        result['valid'] = False
        result['errors'].append("Config file missing 'boards' field")
        return result
    
    if not isinstance(config['boards'], list):
        result['valid'] = False
        result['errors'].append("'boards' must be a list")
        return result
    
    result['stats']['board_count'] = len(config['boards'])
    
    # Validate each board
    for board in config['boards']:
        board_result = validate_board_config(board, verbose=verbose)
        
        # Aggregate stats
        if board_result['valid']:
            result['stats']['boards_passed'] += 1
        else:
            result['stats']['boards_failed'] += 1
            result['valid'] = False
        
        result['stats']['test_count'] += board_result['stats']['test_count']
        result['stats']['component_count'] += board_result['stats']['total_components']
        result['errors'].extend(board_result['errors'])
        result['warnings'].extend(board_result['warnings'])
    
    # Print summary
    if verbose:
        print(f"\n" + "=" * 80)
        print(f"VALIDATION SUMMARY")
        print(f"=" * 80)
        print(f"Boards: {result['stats']['board_count']} "
              f"({result['stats']['boards_passed']} ✓, {result['stats']['boards_failed']} ✗)")
        print(f"Tests: {result['stats']['test_count']}")
        print(f"Components: {result['stats']['component_count']}")
        print(f"\nOverall: {'✓ VALIDATION PASSED' if result['valid'] else '✗ VALIDATION FAILED'}")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for i, err in enumerate(result['errors'][:10], 1):  # Show first 10
                print(f"  {i}. {err}")
            if len(result['errors']) > 10:
                print(f"  ... and {len(result['errors']) - 10} more errors")
        
        if result['warnings']:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for i, warn in enumerate(result['warnings'][:10], 1):  # Show first 10
                print(f"  {i}. {warn}")
            if len(result['warnings']) > 10:
                print(f"  ... and {len(result['warnings']) - 10} more warnings")
        
        print(f"=" * 80)
    
    return result


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
