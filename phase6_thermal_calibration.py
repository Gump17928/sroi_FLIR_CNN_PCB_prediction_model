"""
Phase 6: Thermal Calibration Module
====================================
Learns thermal relationships from PCBs with both FLIR camera and thermistor measurements.

This module:
1. Compares FLIR camera measurements to ground-truth thermistor data
2. Calculates component-type-specific calibration offsets (FLIR vs Thermistor)
3. Models air vs sand cooling effectiveness by component type
4. Exports calibration database for applying to FLIR-only measurements
5. Supports incremental calibration data collection across multiple test sessions
6. Enables multi-session aggregation for batch processing
7. Tracks test sessions and timestamps for convergence analysis

Key Features:
- Incremental Mode: Append new measurements to existing calibration database
- Multi-Session Mode: Process multiple test sessions in a single run
- Session Tracking: Track which test contributed which measurements
- Convergence Analysis: Automatic quality assessment and visualization
- Flexible Data: Support air-only sessions (sand cooling optional)

Key Outputs:
- thermal_calibration_points.csv: Individual calibration measurements with session tracking
- thermal_calibration_by_type.csv: Aggregate statistics per component type
- thermal_calibration_metadata.json: Complete calibration metadata including sessions
- convergence_analysis/: Optional convergence plots and quality reports

Author: Thermal Analysis Pipeline
Date: November 25, 2025
Updated: December 2, 2025 - Added incremental calibration support
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt

# Import data loaders and visualization
import sys
sys.path.insert(0, str(Path(__file__).parent))
from loader_thermistor import ThermalDataLoader, SteadyStateAnalyzer
from viz_phase6_validation import ThermistorValidator


class ThermalCalibrator:
    """Calibrates FLIR measurements against thermistor ground truth.
    
    Supports incremental calibration across multiple test sessions:
    - Load existing calibration database
    - Append new measurement pairs
    - Process multiple sessions in batch
    - Track session attribution and timestamps
    - Automatic convergence analysis integration
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize thermal calibrator.
        
        Args:
            output_dir: Directory for calibration output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calibration data storage
        self.calibration_pairs = []
        self.component_type_calibrations = {}
        self.cooling_model = {}
        
    def load_existing_calibration(self, csv_file: Path) -> int:
        """
        Load existing calibration points from previous test sessions.
        
        Args:
            csv_file: Path to existing thermal_calibration_points.csv
            
        Returns:
            Number of calibration points loaded
        """
        if not csv_file.exists():
            print(f"\nNo existing calibration file found at {csv_file}")
            print("Starting fresh calibration database...")
            return 0
            
        try:
            existing_df = pd.read_csv(csv_file)
            
            # Ensure required columns exist
            required_cols = ['pcb', 'component_name', 'component_type', 'flir_pattern', 
                           'therm_pattern', 'flir_ss', 'air_ss', 'flir_offset']
            missing_cols = [col for col in required_cols if col not in existing_df.columns]
            
            if missing_cols:
                print(f"\nWarning: Existing calibration file missing columns: {missing_cols}")
                print("Starting fresh calibration database...")
                return 0
            
            # Convert DataFrame to list of dicts and add to calibration_pairs
            existing_points = existing_df.to_dict('records')
            self.calibration_pairs.extend(existing_points)
            
            print(f"\n{'='*80}")
            print(f"Loaded Existing Calibration Data")
            print(f"{'='*80}")
            print(f"File: {csv_file}")
            print(f"Calibration points loaded: {len(existing_points)}")
            
            # Show summary by component type
            type_counts = existing_df['component_type'].value_counts()
            print(f"\nCalibration points by component type:")
            for comp_type, count in type_counts.items():
                print(f"  {comp_type}: {count} samples")
            
            # Show test sessions if column exists
            if 'test_session' in existing_df.columns:
                session_counts = existing_df['test_session'].value_counts()
                print(f"\nTest sessions in database:")
                for session, count in session_counts.items():
                    print(f"  {session}: {count} measurements")
            
            return len(existing_points)
            
        except Exception as e:
            print(f"\nError loading existing calibration: {e}")
            print("Starting fresh calibration database...")
            return 0
    
    def generate_raw_input_overview(self,
                                   x_flir, y_flir, labels_flir,
                                   x_air, y_air, labels_air,
                                   x_sand, y_sand, labels_sand,
                                   has_sand: bool,
                                   session_name: str,
                                   output_dir: Path,
                                   tested_components: List[str] = None) -> None:
        """
        Generate overview plot showing all input channels for visual verification.
        
        This creates a 3-subplot figure showing:
        - All FLIR camera channels (ROI boxes) - filtered to tested components if provided
        - All Air thermistor channels (including multi-device merged channels)
        - All Sand thermistor channels (if available)
        
        Especially useful for multi-device sessions to verify 8+ channels merged correctly.
        
        Args:
            x_flir, y_flir, labels_flir: FLIR data
            x_air, y_air, labels_air: Air thermistor data
            x_sand, y_sand, labels_sand: Sand thermistor data
            has_sand: Whether sand data is available
            session_name: Name of test session for plot title
            output_dir: Directory to save plot
            tested_components: List of FLIR ROI names to plot (filters to only tested components)
        """
        print(f"\nGenerating raw input overview plot...")
        
        # Create figure with subplots
        n_subplots = 3 if has_sand else 2
        fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 4*n_subplots))
        if n_subplots == 2:
            axes = [axes[0], axes[1], None]
        
        # Plot FLIR channels - filter to only tested components if specified
        ax = axes[0]
        plotted_count = 0
        
        # Determine time scale for FLIR data
        flir_max_time = x_flir[-1] - x_flir[0]
        if flir_max_time > 7200:  # > 2 hours
            x_flir_display = x_flir / 3600
            flir_time_label = 'Time (hours)'
        else:
            x_flir_display = x_flir / 60
            flir_time_label = 'Time (minutes)'
        
        for i in range(y_flir.shape[1]):
            flir_label = labels_flir[i]
            # If tested_components is provided, only plot those channels
            if tested_components is not None:
                if flir_label not in tested_components:
                    continue
            ax.plot(x_flir_display, y_flir[:, i], label=flir_label, linewidth=1.5, alpha=0.8)
            plotted_count += 1
        
        channel_desc = f"{plotted_count} tested components" if tested_components else f"{y_flir.shape[1]} channels"
        ax.set_xlabel(flir_time_label, fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('FLIR SROI Measurements', fontsize=11)
        ax.set_xlim(x_flir_display[0], x_flir_display[-1])  # Fit data tightly
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot Air thermistor channels
        ax = axes[1]
        
        # Determine time scale for Air data
        air_max_time = x_air[-1] - x_air[0]
        if air_max_time > 7200:  # > 2 hours
            x_air_display = x_air / 3600
            air_time_label = 'Time (hrs)'
        else:
            x_air_display = x_air / 60
            air_time_label = 'Time (min)'
        
        for i in range(y_air.shape[1]):
            ax.plot(x_air_display, y_air[:, i], label=labels_air[i], linewidth=1.5, alpha=0.8)
        ax.set_xlabel(air_time_label, fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Thermistor Measurements (Air)', fontsize=11)
        ax.set_xlim(x_air_display[0], x_air_display[-1])  # Fit data tightly
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot Sand thermistor channels (if available)
        if has_sand:
            ax = axes[2]
            
            # Determine time scale for Sand data
            sand_max_time = x_sand[-1] - x_sand[0]
            if sand_max_time > 7200:  # > 2 hours
                x_sand_display = x_sand / 3600
                sand_time_label = 'Time (hrs)'
            else:
                x_sand_display = x_sand / 60
                sand_time_label = 'Time (min)'
            
            for i in range(y_sand.shape[1]):
                ax.plot(x_sand_display, y_sand[:, i], label=labels_sand[i], linewidth=1.5, alpha=0.8)
            ax.set_xlabel(sand_time_label, fontsize=12)
            ax.set_ylabel('Temperature (°C)', fontsize=12)
            ax.set_title('Thermistor Measurements (Embedded)', fontsize=11)
            ax.set_xlim(x_sand_display[0], x_sand_display[-1])  # Fit data tightly
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Use subplots_adjust instead of tight_layout to avoid warnings with legends outside axes
        plt.subplots_adjust(right=0.75, hspace=0.35)
        
        # Save plot
        plot_file = output_dir / "raw_inputs_overview.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {plot_file}")
        print(f"  FLIR channels: {y_flir.shape[1]}, Air channels: {y_air.shape[1]}" + 
              (f", Sand channels: {y_sand.shape[1]}" if has_sand else ""))
        
    def add_measurement_pair(self,
                            flir_file: Path,
                            therm_air_file,  # Can be Path or List[Path]
                            therm_sand_file,  # Can be Path, List[Path], or None
                            pairs: List[Tuple[str, str, str, str]],
                            pcb_name: str = "Unknown",
                            test_session: Optional[str] = None) -> None:
        """
        Add calibration measurements from a PCB with FLIR and thermistor data.
        
        Supports both single thermistor files and multiple USB device files per condition.
        For multi-device setups, channels are automatically prefixed with device identifier
        (e.g., Device0_AI4, Device1_AI4) to avoid conflicts.
        
        Args:
            flir_file: Path to FLIR camera CSV export
            therm_air_file: Path or List[Path] to USB thermistor CSV (air cooling)
                           List for multi-device (e.g., [Device0.csv, Device1.csv])
            therm_sand_file: Path, List[Path], or None for USB thermistor CSV (sand cooling)
            pairs: List of (flir_pattern, therm_pattern, component_name, component_type)
                  For multi-device, therm_pattern should include device prefix (e.g., 'Device0_AI4')
            pcb_name: Name of the PCB for tracking
            test_session: Name/ID of test session for incremental tracking
        """
        print(f"\n{'='*80}")
        print(f"Processing Calibration Data: {pcb_name}")
        print(f"{'='*80}")
        
        # Create session-specific output directory
        # Note: self.output_dir is already the calibration_database directory
        if test_session:
            session_output_dir = self.output_dir / test_session
            session_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Session output directory: {session_output_dir}")
        else:
            session_output_dir = self.output_dir
            session_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data - support both single file and multi-file modes
        loader = ThermalDataLoader()
        x_flir, y_flir, labels_flir, meta_flir = loader.load_temp_csv(flir_file)
        
        # Load air thermistor data (single or multi-device)
        if isinstance(therm_air_file, list):
            print(f"\nLoading Air thermistor data from {len(therm_air_file)} devices:")
            x_air, y_air, labels_air, meta_air = loader.load_multi_device_csv(therm_air_file)
        else:
            x_air, y_air, labels_air, meta_air = loader.load_temp_csv(therm_air_file)
        
        # Load sand thermistor data (single or multi-device, optional)
        has_sand = False
        if therm_sand_file is not None:
            if isinstance(therm_sand_file, list):
                if all(Path(f).exists() for f in therm_sand_file):
                    print(f"\nLoading Sand thermistor data from {len(therm_sand_file)} devices:")
                    x_sand, y_sand, labels_sand, meta_sand = loader.load_multi_device_csv(therm_sand_file)
                    has_sand = True
            else:
                if Path(therm_sand_file).exists():
                    x_sand, y_sand, labels_sand, meta_sand = loader.load_temp_csv(therm_sand_file)
                    has_sand = True
        
        print(f"\nLoaded FLIR data: {len(x_flir)} samples, {y_flir.shape[1]} channels")
        print(f"Loaded Air thermistor data: {len(x_air)} samples, {y_air.shape[1]} channels")
        if has_sand:
            print(f"Loaded Sand thermistor data: {len(x_sand)} samples, {y_sand.shape[1]} channels")
        
        # Extract tested component names from pairs for FLIR filtering
        tested_components = []
        for pair in pairs:
            if isinstance(pair, dict):
                flir_roi = pair.get('flir_roi')
            elif isinstance(pair, (list, tuple)) and len(pair) >= 1:
                flir_roi = pair[0]
            else:
                continue
            if flir_roi:
                tested_components.append(flir_roi)
        
        print(f"Filtering FLIR plot to {len(tested_components)} tested components: {tested_components}")
        
        # Generate raw input overview plot for visual verification
        self.generate_raw_input_overview(
            x_flir, y_flir, labels_flir,
            x_air, y_air, labels_air,
            x_sand if has_sand else None,
            y_sand if has_sand else None,
            labels_sand if has_sand else None,
            has_sand,
            test_session if test_session else pcb_name,
            session_output_dir,
            tested_components=tested_components
        )
        
        # Calculate initial temperatures (first 30 seconds) for all channels
        # This establishes ambient baseline before component heating
        print(f"\nCalculating initial temperatures (first 30 seconds)...")
        initial_temps_air = {}
        for i, label in enumerate(labels_air):
            initial_temps_air[label] = self._calculate_initial_temp(x_air, y_air[:, i], time_window=30.0)
        
        initial_temps_sand = {}
        if has_sand:
            for i, label in enumerate(labels_sand):
                initial_temps_sand[label] = self._calculate_initial_temp(x_sand, y_sand[:, i], time_window=30.0)
        
        print(f"  Air channels: {len(initial_temps_air)} initial temps calculated")
        if has_sand:
            print(f"  Sand channels: {len(initial_temps_sand)} initial temps calculated")
        
        # Process each measurement pair
        for pair in pairs:
            # Support both tuple and dict formats
            if isinstance(pair, dict):
                flir_pat = pair.get('flir_roi')
                therm_pat = pair.get('therm_channel')
                comp_name = pair.get('component')
                comp_type = pair.get('type')
            elif isinstance(pair, (list, tuple)) and len(pair) >= 4:
                flir_pat, therm_pat, comp_name, comp_type = pair[0], pair[1], pair[2], pair[3]
            else:
                print(f"\n  Warning: Invalid pair format: {pair}, skipping...")
                continue
            
            print(f"\n  Processing: {comp_name} ({comp_type})")
            
            # Find matching channels
            idx_flir = self._find_label_idx(labels_flir, flir_pat)
            idx_air = self._find_label_idx(labels_air, therm_pat)
            
            if idx_flir is None:
                print(f"    Warning: FLIR pattern '{flir_pat}' not found, skipping...")
                continue
            if idx_air is None:
                print(f"    Warning: Thermistor pattern '{therm_pat}' not found, skipping...")
                continue
            
            # Calculate steady-state values
            ss_flir = SteadyStateAnalyzer.steady_value(x_flir, y_flir[:, idx_flir])
            ss_air = SteadyStateAnalyzer.steady_value(x_air, y_air[:, idx_air])
            
            # Get initial temperatures
            air_initial = initial_temps_air.get(labels_air[idx_air], np.nan)
            sand_initial = np.nan
            
            # FLIR offset (how much FLIR differs from thermistor air)
            offset_flir = ss_flir['value'] - ss_air['value']
            
            print(f"    FLIR SS: {ss_flir['value']:.2f}°C")
            print(f"    Air Thermistor SS: {ss_air['value']:.2f}°C")
            print(f"    Air Initial: {air_initial:.2f}°C")
            print(f"    FLIR Offset: {offset_flir:+.2f}°C")
            
            # Sand cooling benefit (if available)
            cooling_benefit = None
            ss_sand_val = None
            if has_sand:
                idx_sand = self._find_label_idx(labels_sand, therm_pat)
                if idx_sand is not None:
                    ss_sand = SteadyStateAnalyzer.steady_value(x_sand, y_sand[:, idx_sand])
                    ss_sand_val = ss_sand['value']
                    sand_initial = initial_temps_sand.get(labels_sand[idx_sand], np.nan)
                    cooling_benefit = ss_air['value'] - ss_sand['value']
                    cooling_ratio = ss_sand['value'] / ss_air['value'] if ss_air['value'] > 0 else 1.0
                    print(f"    Sand Thermistor SS: {ss_sand_val:.2f}°C")
                    print(f"    Sand Initial: {sand_initial:.2f}°C")
                    print(f"    Cooling Benefit: {cooling_benefit:+.2f}°C")
                    print(f"    Cooling Ratio: {cooling_ratio:.4f}")
            
            # Extract board_name and test_id from test_session
            # Format: "BoardName_TestID" or legacy "SessionName"
            board_name = None
            test_id = None
            if test_session and '_' in test_session:
                parts = test_session.split('_', 1)
                board_name = parts[0]
                test_id = parts[1] if len(parts) > 1 else None
            
            # Store calibration data
            cal_point = {
                'pcb': pcb_name,
                'board_name': board_name,
                'test_id': test_id,
                'component_name': comp_name,
                'component_type': comp_type,
                'flir_pattern': flir_pat,
                'therm_pattern': therm_pat,
                'flir_ss': ss_flir['value'],
                'air_ss': ss_air['value'],
                'air_initial': air_initial,
                'sand_ss': ss_sand_val,
                'sand_initial': sand_initial,
                'flir_offset': offset_flir,
                'cooling_benefit': cooling_benefit,
                'cooling_ratio': cooling_ratio if has_sand and idx_sand is not None else None,
                'test_session': test_session if test_session else f"{pcb_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat()
            }
            self.calibration_pairs.append(cal_point)
        
        # Generate detailed comparison plots (FLIR vs Sand, Air vs Sand)
        # TEMPORARILY DISABLED - causing datetime parsing issues with large files
        # if has_sand:
        #     print(f"\nGenerating detailed comparison plots for session...")
        #     detailed_plots_dir = session_output_dir / "detailed_plots"
        #     detailed_plots_dir.mkdir(parents=True, exist_ok=True)
        #     self.generate_detailed_comparison_plots(
        #         flir_file=flir_file,
        #         therm_air_file=therm_air_file,
        #         therm_sand_file=therm_sand_file,
        #         pairs=pairs,
        #         session_name=test_session if test_session else pcb_name,
        #         plot_dir=detailed_plots_dir
        #     )
        # else:
        #     print(f"\nSkipping detailed comparison plots (no sand data available)")
        print(f"\nSkipping detailed comparison plots (temporarily disabled)")
    
    def add_measurement_sessions(self, sessions: List[Dict]) -> None:
        """
        Process multiple test sessions in a single calibration run.
        
        Supports both single-device and multi-device USB thermistor configurations.
        
        Args:
            sessions: List of session dictionaries, each containing:
                - name: Test session name/ID
                - pcb: PCB name
                - flir_file: Path to FLIR CSV
                - therm_air_file: Path or List[Path] to air thermistor CSV(s)
                - therm_sand_file: Optional Path or List[Path] to sand thermistor CSV(s)
                - pairs: List of (flir_pattern, therm_pattern, comp_name, comp_type)
                  For multi-device, use device-prefixed patterns (e.g., 'Device0_AI4')
                
        Example (Single Device):
            sessions = [
                {
                    'name': 'LoadShedding_Week1',
                    'pcb': 'LoadShedding_RevA',
                    'flir_file': 'test1_flir.csv',
                    'therm_air_file': 'test1_air.csv',
                    'therm_sand_file': None,
                    'pairs': [('R1', 'AI1', 'R1', 'Resistor'), ...]
                }
            ]
        
        Example (Multi-Device):
            sessions = [
                {
                    'name': 'LoadShedding_Week2_MultiDevice',
                    'pcb': 'LoadShedding_RevA',
                    'flir_file': 'test2_flir.csv',
                    'therm_air_file': ['Device0_air.csv', 'Device1_air.csv'],
                    'therm_sand_file': ['Device0_sand.csv', 'Device1_sand.csv'],
                    'pairs': [
                        ('R8', 'Device0_AI1', 'R8', 'Resistor'),
                        ('C12', 'Device1_AI4', 'C12', 'Capacitor')
                    ]
                }
            ]
        """
        print(f"\n{'='*80}")
        print(f"Processing Multiple Test Sessions")
        print(f"{'='*80}")
        print(f"Total sessions to process: {len(sessions)}\n")
        
        initial_count = len(self.calibration_pairs)
        
        for i, session in enumerate(sessions, 1):
            print(f"\n[Session {i}/{len(sessions)}] {session['name']}")
            print(f"{'-'*80}")
            
            # Validate required fields (flir_file is optional when using ResearchIR_Outputs)
            required_fields = ['name', 'pcb', 'therm_air_file', 'pairs']
            missing = [f for f in required_fields if f not in session]
            if missing:
                print(f"  ERROR: Missing required fields: {missing}")
                print(f"  Skipping session...")
                continue
            
            # Process this session
            try:
                # Handle both single files and lists of files
                air_file = session['therm_air_file']
                if isinstance(air_file, list):
                    air_file = [Path(f) for f in air_file]
                else:
                    air_file = Path(air_file)
                
                sand_file = session.get('therm_sand_file')
                if sand_file is not None:
                    if isinstance(sand_file, list):
                        sand_file = [Path(f) for f in sand_file]
                    else:
                        sand_file = Path(sand_file)
                
                self.add_measurement_pair(
                    flir_file=Path(session['flir_file']),
                    therm_air_file=air_file,
                    therm_sand_file=sand_file,
                    pairs=session['pairs'],
                    pcb_name=session['pcb'],
                    test_session=session['name']
                )
            except Exception as e:
                print(f"  ERROR processing session: {e}")
                print(f"  Skipping session...")
                import traceback
                traceback.print_exc()
                continue
        
        new_points = len(self.calibration_pairs) - initial_count
        print(f"\n{'='*80}")
        print(f"Multi-Session Processing Complete")
        print(f"{'='*80}")
        print(f"New calibration points added: {new_points}")
        print(f"Total calibration points: {len(self.calibration_pairs)}")
    
    def add_measurement_boards(self, boards: List[Dict]) -> None:
        """
        Process multiple boards with hierarchical test organization.
        
        Supports the new board->test structure where each board has multiple tests,
        and each test can have different component mappings for the same channels.
        
        Args:
            boards: List of board dictionaries, each containing:
                - name: Board name
                - pcb: PCB file name
                - tests: List of test dictionaries with:
                    - test_id: Test identifier
                    - flir_file: Path to FLIR CSV
                    - therm_air_file: Path or List[Path] to air thermistor CSV(s)
                    - therm_sand_file: Optional Path or List[Path] to sand thermistor CSV(s)
                    - pairs: List of measurement pairs (flir_roi, therm_channel, component, type)
        
        Example:
            boards = [
                {
                    'name': 'LoadShedding',
                    'pcb': 'LoadShedding_RevE.brd',
                    'tests': [
                        {
                            'test_id': 'Test1_Air',
                            'flir_file': 'test1_flir.csv',
                            'therm_air_file': 'test1_air.csv',
                            'pairs': [{'flir_roi': 'U3', 'therm_channel': 'AI0', 'component': 'U3', 'type': 'fet'}]
                        }
                    ]
                }
            ]
        """
        print(f"\n{'='*80}")
        print(f"Processing Board-Based Multi-Test Configuration")
        print(f"{'='*80}")
        print(f"Total boards: {len(boards)}\n")
        
        initial_count = len(self.calibration_pairs)
        total_tests = sum(len(board.get('tests', [])) for board in boards)
        test_counter = 0
        
        for board in boards:
            board_name = board.get('name', 'Unknown')
            pcb_file = board.get('pcb', f"{board_name}.brd")
            tests = board.get('tests', [])
            
            print(f"\n{'='*80}")
            print(f"BOARD: {board_name}")
            print(f"{'='*80}")
            print(f"PCB: {pcb_file}")
            print(f"Tests in board: {len(tests)}\n")
            
            for test in tests:
                test_counter += 1
                test_id = test.get('test_id', f'Test{test_counter}')
                
                # Skip placeholder tests
                if 'comment' in test and not test.get('flir_file'):
                    print(f"[{test_counter}/{total_tests}] {test_id}: Skipping (placeholder test)")
                    continue
                
                print(f"\n[{test_counter}/{total_tests}] {board_name} / {test_id}")
                print(f"{'-'*80}")
                
                # Validate required fields (flir_file is optional when using ResearchIR_Outputs)
                # Accept either 'pairs' or 'air_pairs' for backward compatibility
                required_fields = ['test_id', 'therm_air_file']
                if 'pairs' not in test and 'air_pairs' not in test:
                    print(f"  ERROR: Missing 'pairs' or 'air_pairs' field")
                    print(f"  Skipping test...")
                    continue
                
                missing = [f for f in required_fields if f not in test]
                if missing:
                    print(f"  ERROR: Missing required fields: {missing}")
                    print(f"  Skipping test...")
                    continue
                
                # Get pairs - use air_pairs if available, otherwise fallback to pairs
                pairs = test.get('air_pairs', test.get('pairs', []))
                
                # Process this test
                try:
                    # Handle both single files and lists of files
                    air_file = test['therm_air_file']
                    if isinstance(air_file, list):
                        air_file = [Path(f) for f in air_file]
                    else:
                        air_file = Path(air_file)
                    
                    sand_file = test.get('therm_sand_file')
                    if sand_file is not None:
                        if isinstance(sand_file, list):
                            sand_file = [Path(f) for f in sand_file]
                        else:
                            sand_file = Path(sand_file)
                    
                    # Create session name from board and test
                    session_name = f"{board_name}_{test_id}"
                    
                    # Get FLIR file: use test['flir_file'] if provided, otherwise use parsed ResearchIR_Outputs
                    flir_file = test.get('flir_file')
                    if flir_file is None:
                        # Use the parsed FLIR data from ResearchIR_Outputs
                        # Use board_name (from board config) instead of pcb file name
                        # Construct path: outputs/{timestamp}_P6-7/{board_name}/{board_name}_FLIR_AllComponents.csv
                        
                        # Try to find the parsed FLIR file in outputs using board_name
                        import glob
                        possible_paths = glob.glob(f"outputs/*_P6-7/{board_name}/{board_name}_FLIR_AllComponents.csv")
                        if possible_paths:
                            flir_file = sorted(possible_paths)[-1]  # Use most recent
                            print(f"  Using parsed FLIR data: {flir_file}")
                        else:
                            print(f"  ERROR: No FLIR file specified and no parsed data found for {board_name}")
                            print(f"  Skipping test...")
                            continue
                    
                    self.add_measurement_pair(
                        flir_file=Path(flir_file),
                        therm_air_file=air_file,
                        therm_sand_file=sand_file,
                        pairs=pairs,  # Use the pairs variable we extracted earlier
                        pcb_name=pcb_file,
                        test_session=session_name
                    )
                except Exception as e:
                    print(f"  ERROR processing test: {e}")
                    print(f"  Skipping test...")
                    import traceback
                    traceback.print_exc()
                    continue
        
        new_points = len(self.calibration_pairs) - initial_count
        print(f"\n{'='*80}")
        print(f"Multi-Board Processing Complete")
        print(f"{'='*80}")
        print(f"Boards processed: {len(boards)}")
        print(f"Tests processed: {test_counter}")
        print(f"New calibration points added: {new_points}")
        print(f"Total calibration points: {len(self.calibration_pairs)}")
    
    def compute_component_type_calibrations(self) -> Dict:
        """
        Compute calibration parameters grouped by component type.
        
        Returns:
            Dictionary with component type calibrations
        """
        print(f"\n{'='*80}")
        print("Computing Component Type Calibrations")
        print(f"{'='*80}")
        
        if not self.calibration_pairs:
            print("Warning: No calibration data available!")
            return {}
        
        # Group by component type
        df = pd.DataFrame(self.calibration_pairs)
        
        for comp_type in df['component_type'].unique():
            type_data = df[df['component_type'] == comp_type]
            
            # FLIR offset statistics
            flir_offset_mean = type_data['flir_offset'].mean()
            flir_offset_std = type_data['flir_offset'].std()
            flir_offset_median = type_data['flir_offset'].median()
            
            # Cooling benefit statistics (if available)
            cooling_data = type_data.dropna(subset=['cooling_benefit'])
            if len(cooling_data) > 0:
                cooling_benefit_mean = cooling_data['cooling_benefit'].mean()
                cooling_benefit_std = cooling_data['cooling_benefit'].std()
                cooling_ratio_mean = cooling_data['cooling_ratio'].mean()
                cooling_ratio_std = cooling_data['cooling_ratio'].std()
            else:
                cooling_benefit_mean = None
                cooling_benefit_std = None
                cooling_ratio_mean = None
                cooling_ratio_std = None
            
            self.component_type_calibrations[comp_type] = {
                'count': len(type_data),
                'flir_offset_mean': flir_offset_mean,
                'flir_offset_std': flir_offset_std,
                'flir_offset_median': flir_offset_median,
                'cooling_benefit_mean': cooling_benefit_mean,
                'cooling_benefit_std': cooling_benefit_std,
                'cooling_ratio_mean': cooling_ratio_mean,
                'cooling_ratio_std': cooling_ratio_std,
                'air_ss_mean': type_data['air_ss'].mean(),
                'air_ss_std': type_data['air_ss'].std(),
                # Enhanced metadata for validation
                'board_sources': ','.join(type_data['pcb'].unique()) if 'pcb' in type_data else 'Unknown',
                'sample_count_per_board': type_data.groupby('pcb').size().to_dict() if 'pcb' in type_data else {},
                'confidence': 'High' if len(type_data) >= 10 else ('Medium' if len(type_data) >= 5 else 'Low')
            }
            
            print(f"\n{comp_type}:")
            print(f"  Samples: {len(type_data)}")
            print(f"  FLIR Offset: {flir_offset_mean:+.2f} ± {flir_offset_std:.2f}°C (median: {flir_offset_median:+.2f}°C)")
            print(f"  Air SS Temp: {type_data['air_ss'].mean():.2f} ± {type_data['air_ss'].std():.2f}°C")
            if cooling_benefit_mean is not None:
                print(f"  Cooling Benefit: {cooling_benefit_mean:+.2f} ± {cooling_benefit_std:.2f}°C")
                print(f"  Cooling Ratio: {cooling_ratio_mean:.4f} ± {cooling_ratio_std:.4f}")
        
        return self.component_type_calibrations
    
    def perform_cross_validation(self) -> Dict:
        """
        Perform cross-validation between boards to measure calibration accuracy.
        
        Trains calibration on one board, validates on another.
        Returns validation metrics (RMSE, MAE, R²) per component type.
        """
        import numpy as np
        
        print(f"\n{'='*80}")
        print("Cross-Validation Analysis")
        print(f"{'='*80}")
        
        if not self.calibration_pairs:
            print("Warning: No calibration data available!")
            return {}
        
        df = pd.DataFrame(self.calibration_pairs)
        
        # Check if we have multiple boards
        if 'pcb' not in df.columns:
            print("Warning: No PCB information in calibration data!")
            return {}
        
        unique_boards = df['pcb'].unique()
        if len(unique_boards) < 2:
            print(f"Warning: Need at least 2 boards for cross-validation. Found: {unique_boards}")
            return {}
        
        print(f"\nBoards available: {list(unique_boards)}")
        
        validation_results = {}
        
        # For each pair of boards, train on one and validate on the other
        for train_board in unique_boards:
            for val_board in unique_boards:
                if train_board == val_board:
                    continue
                
                print(f"\n--- Train on {train_board} -> Validate on {val_board} ---")
                
                # Split data
                train_data = df[df['pcb'] == train_board]
                val_data = df[df['pcb'] == val_board]
                
                # Build calibration from training board
                train_cal = {}
                for comp_type in train_data['component_type'].unique():
                    type_data = train_data[train_data['component_type'] == comp_type]
                    train_cal[comp_type] = {
                        'flir_offset_mean': type_data['flir_offset'].mean(),
                        'cooling_ratio_mean': type_data['cooling_ratio'].mean() if 'cooling_ratio' in type_data and not type_data['cooling_ratio'].isna().all() else None
                    }
                
                # Validate on validation board
                predictions_air = []
                actuals_air = []
                predictions_sand = []
                actuals_sand = []
                component_types = []
                
                for _, row in val_data.iterrows():
                    comp_type = row['component_type']
                    if comp_type not in train_cal:
                        continue  # Skip types not in training set
                    
                    # Predict air temperature
                    flir_raw = row['flir_ss']
                    offset = train_cal[comp_type]['flir_offset_mean']
                    predicted_air = flir_raw + offset
                    actual_air = row['air_ss']
                    
                    predictions_air.append(predicted_air)
                    actuals_air.append(actual_air)
                    component_types.append(comp_type)
                    
                    # Predict sand temperature if available
                    if 'sand_ss' in row and not pd.isna(row['sand_ss']):
                        cooling_ratio = train_cal[comp_type]['cooling_ratio_mean']
                        if cooling_ratio is not None:
                            predicted_sand = predicted_air * cooling_ratio
                            actual_sand = row['sand_ss']
                            predictions_sand.append(predicted_sand)
                            actuals_sand.append(actual_sand)
                
                # Calculate metrics for air predictions
                if len(predictions_air) > 0:
                    predictions_air = np.array(predictions_air)
                    actuals_air = np.array(actuals_air)
                    
                    # RMSE: Root Mean Squared Error
                    rmse_air = np.sqrt(np.mean((predictions_air - actuals_air)**2))
                    
                    # MAE: Mean Absolute Error
                    mae_air = np.mean(np.abs(predictions_air - actuals_air))
                    
                    # R²: Coefficient of determination
                    ss_res = np.sum((actuals_air - predictions_air)**2)  # Residual sum of squares
                    ss_tot = np.sum((actuals_air - np.mean(actuals_air))**2)  # Total sum of squares
                    r2_air = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    
                    print(f"  Air Temperature Validation ({len(predictions_air)} components):")
                    print(f"    RMSE: {rmse_air:.3f}°C")
                    print(f"    MAE:  {mae_air:.3f}°C")
                    print(f"    R²:   {r2_air:.4f}")
                    
                    validation_results[f"{train_board}_to_{val_board}_air"] = {
                        'train_board': train_board,
                        'val_board': val_board,
                        'condition': 'air',
                        'samples': len(predictions_air),
                        'rmse': rmse_air,
                        'mae': mae_air,
                        'r2': r2_air
                    }
                
                # Calculate metrics for sand predictions
                if len(predictions_sand) > 0:
                    predictions_sand = np.array(predictions_sand)
                    actuals_sand = np.array(actuals_sand)
                    
                    # RMSE: Root Mean Squared Error
                    rmse_sand = np.sqrt(np.mean((predictions_sand - actuals_sand)**2))
                    
                    # MAE: Mean Absolute Error
                    mae_sand = np.mean(np.abs(predictions_sand - actuals_sand))
                    
                    # R²: Coefficient of determination
                    ss_res = np.sum((actuals_sand - predictions_sand)**2)
                    ss_tot = np.sum((actuals_sand - np.mean(actuals_sand))**2)
                    r2_sand = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    
                    print(f"  Sand Temperature Validation ({len(predictions_sand)} components):")
                    print(f"    RMSE: {rmse_sand:.3f}°C")
                    print(f"    MAE:  {mae_sand:.3f}°C")
                    print(f"    R²:   {r2_sand:.4f}")
                    
                    validation_results[f"{train_board}_to_{val_board}_sand"] = {
                        'train_board': train_board,
                        'val_board': val_board,
                        'condition': 'sand',
                        'samples': len(predictions_sand),
                        'rmse': rmse_sand,
                        'mae': mae_sand,
                        'r2': r2_sand
                    }
        
        return validation_results
    
    def export_calibration_database(self, prefix: str = "thermal_calibration", 
                                   run_convergence_analysis: bool = False) -> None:
        """
        Export calibration results to CSV and JSON files.
        
        Args:
            prefix: Filename prefix for output files
            run_convergence_analysis: If True, run convergence analysis after export
        """
        print(f"\n{'='*80}")
        print("Exporting Calibration Database")
        print(f"{'='*80}")
        
        # Export individual calibration points
        df_points = pd.DataFrame(self.calibration_pairs)
        
        # Sort by timestamp if available for chronological ordering
        if 'timestamp' in df_points.columns:
            df_points = df_points.sort_values('timestamp')
        
        points_file = self.output_dir / f"{prefix}_points.csv"
        df_points.to_csv(points_file, index=False)
        print(f"\nSaved: {points_file}")
        print(f"  {len(df_points)} calibration points")
        
        # Show test session summary if column exists
        if 'test_session' in df_points.columns:
            session_counts = df_points['test_session'].value_counts()
            print(f"\n  Test sessions in database:")
            for session, count in session_counts.items():
                print(f"    {session}: {count} measurements")
        
        # Show board/test summary if columns exist
        if 'board_name' in df_points.columns and 'test_id' in df_points.columns:
            board_test_summary = df_points.groupby(['board_name', 'test_id']).size()
            print(f"\n  Board/Test breakdown:")
            for (board, test), count in board_test_summary.items():
                if board and test:
                    print(f"    {board} / {test}: {count} measurements")
        
        # Export component type calibrations
        df_types = pd.DataFrame.from_dict(self.component_type_calibrations, orient='index')
        df_types.index.name = 'component_type'
        types_file = self.output_dir / f"{prefix}_by_type.csv"
        df_types.to_csv(types_file)
        print(f"\nSaved: {types_file}")
        print(f"  {len(df_types)} component types")
        
        # Export complete metadata as JSON
        metadata = {
            'calibration_date': datetime.now().isoformat(),
            'num_calibration_points': len(self.calibration_pairs),
            'component_types': list(self.component_type_calibrations.keys()),
            'pcbs_used': list(set([p['pcb'] for p in self.calibration_pairs])),
            'test_sessions': list(df_points['test_session'].unique()) if 'test_session' in df_points.columns else [],
            'component_type_calibrations': self.component_type_calibrations,
            'usage_notes': {
                'flir_offset_mean': 'Add this to FLIR measurement to get corrected temperature',
                'cooling_benefit_mean': 'Subtract this from air temp to predict sand cooling temp',
                'cooling_ratio_mean': 'Multiply air temp by this to predict sand cooling temp'
            }
        }
        
        json_file = self.output_dir / f"{prefix}_metadata.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved: {json_file}")
        
        # Create visualization
        self._create_calibration_plots(prefix)
        
        # Run convergence analysis if requested
        if run_convergence_analysis:
            self._run_convergence_analysis(points_file)
    
    def _run_convergence_analysis(self, calibration_file: Path) -> None:
        """
        Run convergence analysis on calibration data.
        
        Args:
            calibration_file: Path to thermal_calibration_points.csv
        """
        try:
            from calibration_convergence_analysis import CalibrationConvergenceAnalyzer
            
            print(f"\n{'='*80}")
            print("Running Calibration Convergence Analysis")
            print(f"{'='*80}")
            
            # Create convergence analysis subdirectory
            conv_dir = self.output_dir / "convergence_analysis"
            conv_dir.mkdir(exist_ok=True)
            
            # Run analysis
            analyzer = CalibrationConvergenceAnalyzer(
                calibration_file=str(calibration_file),
                output_dir=str(conv_dir)
            )
            
            report = analyzer.run_complete_analysis()
            
            print(f"\nConvergence analysis complete!")
            print(f"  Output directory: {conv_dir}")
            print(f"  Report: {conv_dir / 'calibration_convergence_report.json'}")
            
        except ImportError:
            print(f"\n[INFO] Calibration convergence analysis module not found.")
            print(f"  To analyze convergence, run:")
            print(f"    python calibration_convergence_analysis.py --calibration_file {calibration_file}")
        except Exception as e:
            print(f"\n[WARNING] Error running convergence analysis: {e}")
            print(f"  You can manually run:")
            print(f"    python calibration_convergence_analysis.py --calibration_file {calibration_file}")
    
    def run_validation_workflow(self) -> None:
        """
        Run complete validation workflow for calibration quality assessment.
        
        Performs:
        1. Cross-validation (train on one board, validate on other)
        2. Creates validation visualization plots
        3. Exports calibration quality metrics CSV
        
        This should be called after compute_component_type_calibrations()
        and before export_calibration_database() for validation mode,
        or after all calibration data is collected.
        """
        print(f"\n{'='*80}")
        print("Running Calibration Validation Workflow")
        print(f"{'='*80}")
        
        # Check if we have calibration data
        if not self.calibration_pairs:
            print("[WARNING] No calibration data available for validation")
            return
        
        # Convert to DataFrame
        df_calibration = pd.DataFrame(self.calibration_pairs)
        
        # Check if we have multiple boards
        unique_boards = df_calibration['pcb'].unique()
        if len(unique_boards) < 2:
            print(f"[INFO] Only {len(unique_boards)} board(s) available. Validation requires >=2 boards.")
            print(f"  Boards: {list(unique_boards)}")
            print(f"  Skipping cross-validation (need 2+ boards to train/validate)")
            return
        
        print(f"Found {len(unique_boards)} boards for cross-validation:")
        for board in unique_boards:
            count = len(df_calibration[df_calibration['pcb'] == board])
            print(f"  - {board}: {count} calibration points")
        
        # Perform cross-validation
        print(f"\n{'='*80}")
        print("Step 1: Cross-Validation")
        print(f"{'='*80}")
        validation_results = self.perform_cross_validation()
        
        if not validation_results:
            print("[WARNING] Cross-validation produced no results")
            return
        
        # Create validation visualizations
        print(f"\n{'='*80}")
        print("Step 2: Creating Validation Visualizations")
        print(f"{'='*80}")
        
        try:
            from viz_calibration_validation import create_validation_plots, export_quality_metrics_csv
            
            create_validation_plots(
                validation_results=validation_results,
                calibration_data=df_calibration,
                output_dir=self.output_dir
            )
            
            export_quality_metrics_csv(
                validation_results=validation_results,
                calibration_data=df_calibration,
                output_dir=self.output_dir
            )
            
            print("\nValidation workflow complete!")
            print(f"  Validation plot: {self.output_dir / 'calibration_validation_summary.png'}")
            print(f"  Quality metrics: {self.output_dir / 'calibration_quality_metrics.csv'}")
            
        except ImportError as e:
            print(f"[ERROR] Could not import validation visualization module: {e}")
            print(f"  Make sure viz_calibration_validation.py is in the same directory")
        except Exception as e:
            print(f"[ERROR] Error during validation visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_calibration_plots(self, prefix: str) -> None:
        """Create visualization of calibration results."""
        if not self.calibration_pairs:
            return
        
        df = pd.DataFrame(self.calibration_pairs)
        
        # Plot 1: FLIR offset by component type
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # FLIR offset box plot
        ax = axes[0]
        comp_types = df['component_type'].unique()
        offset_data = [df[df['component_type'] == ct]['flir_offset'].values for ct in comp_types]
        ax.boxplot(offset_data, labels=comp_types)
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('FLIR Offset (°C)')
        ax.set_title('FLIR vs Thermistor Offset by Component Type')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Cooling benefit box plot
        ax = axes[1]
        cooling_df = df.dropna(subset=['cooling_benefit'])
        if len(cooling_df) > 0:
            comp_types_cool = cooling_df['component_type'].unique()
            cooling_data = [cooling_df[cooling_df['component_type'] == ct]['cooling_benefit'].values 
                          for ct in comp_types_cool]
            ax.boxplot(cooling_data, labels=comp_types_cool)
            ax.set_ylabel('Cooling Benefit (°C)')
            ax.set_title('Air vs Sand Cooling Benefit by Component Type')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Sand Cooling Data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Air vs Sand Cooling Benefit')
        
        plt.tight_layout()
        plot_file = self.output_dir / f"{prefix}_summary.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved: {plot_file}")
    
    def generate_detailed_comparison_plots(self,
                                          flir_file: Path,
                                          therm_air_file,  # Can be Path or List[Path]
                                          therm_sand_file,  # Can be Path, List[Path], or None
                                          pairs: List[Tuple[str, str, str, str]],
                                          session_name: Optional[str] = None,
                                          plot_dir: Optional[Path] = None,
                                          output_subdir: str = "detailed_plots") -> List[str]:
        """
        Generate detailed time-series comparison plots showing FLIR, Air, and Sand data.
        
        This creates the Test_X style comparison plots for visual validation:
        - FLIR vs Sand comparison plots (for each component with thermistor)
        - Air vs Sand comparison plots (for each component with thermistor)
        
        Args:
            flir_file: Path to FLIR camera CSV
            therm_air_file: Path to Air thermistor CSV
            therm_sand_file: Path to Sand thermistor CSV (optional)
            pairs: List of (flir_pattern, therm_pattern, component_name, component_type)
            output_subdir: Subdirectory name for detailed plots
            
        Returns:
            List of generated plot file paths
        """
        print(f"\n{'='*80}")
        print("Generating Detailed Comparison Plots")
        if session_name:
            print(f"Session: {session_name}")
        print(f"{'='*80}")
        
        # Create output subdirectory
        if plot_dir is None:
            plot_dir = self.output_dir / output_subdir
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if sand file exists
        has_sand = False
        if therm_sand_file is not None:
            if isinstance(therm_sand_file, list):
                has_sand = all(Path(f).exists() for f in therm_sand_file)
            else:
                has_sand = Path(therm_sand_file).exists()
        
        if not has_sand:
            print("Warning: No sand thermistor file found, skipping detailed plots")
            return []
        
        # Load data using ThermalDataLoader
        loader = ThermalDataLoader()
        x_flir, y_flir, labels_flir, meta_flir = loader.load_temp_csv(flir_file)
        
        # Load air thermistor data (single or multi-device)
        if isinstance(therm_air_file, list):
            x_air, y_air, labels_air, meta_air = loader.load_multi_device_csv(therm_air_file)
        else:
            x_air, y_air, labels_air, meta_air = loader.load_temp_csv(therm_air_file)
        
        # Load sand thermistor data (single or multi-device)
        if isinstance(therm_sand_file, list):
            x_sand, y_sand, labels_sand, meta_sand = loader.load_multi_device_csv(therm_sand_file)
        else:
            x_sand, y_sand, labels_sand, meta_sand = loader.load_temp_csv(therm_sand_file)
        
        print(f"\nLoaded data for comparison plots:")
        print(f"  FLIR: {len(x_flir)} samples, {y_flir.shape[1]} channels")
        print(f"  Air:  {len(x_air)} samples, {y_air.shape[1]} channels")
        print(f"  Sand: {len(x_sand)} samples, {y_sand.shape[1]} channels")
        
        # Initialize validator for plotting
        validator = ThermistorValidator(
            fig_size=(7.0, 3.5),
            line_width=2.0,
            dpi=600,
            smooth_win=20,
            y_tick_step=1.0,
            data_shift=10.0
        )
        
        # Build comparison pair lists from calibration pairs
        # Format for plot_pair_compare: (pattern1, pattern2, display_name, label1, label2)
        pair_flir_comp = []  # FLIR vs Sand
        pair_therm_comp = []  # Air vs Sand
        
        for pair in pairs:
            # Support both tuple and dict formats
            if isinstance(pair, dict):
                flir_pat = pair.get('flir_roi')
                therm_pat = pair.get('therm_channel')
                comp_name = pair.get('component')
                comp_type = pair.get('type')
            elif isinstance(pair, (list, tuple)) and len(pair) >= 4:
                flir_pat, therm_pat, comp_name, comp_type = pair[0], pair[1], pair[2], pair[3]
            else:
                continue
            
            # FLIR vs Sand (only if component has thermistor measurement)
            pair_flir_comp.append((flir_pat, therm_pat, f"{comp_name}", 'FLIR-air', 'therm-sand'))
            
            # Air vs Sand (thermistor to thermistor)
            pair_therm_comp.append((therm_pat, therm_pat, f"{comp_name}", 'therm-air', 'therm-sand'))
        
        # Add ambient measurement for Air vs Sand comparison
        pair_therm_comp.append(('AI4', 'AI4', 'Ambient', 'therm-air', 'therm-sand'))
        
        generated_plots = []
        
        # Generate FLIR vs Sand comparison plots
        print(f"\nGenerating FLIR vs Sand comparison plots...")
        print(f"  Components: {len(pair_flir_comp)}")
        validator.plot_pair_compare(
            x_flir, y_flir, labels_flir, meta_flir,
            x_sand, y_sand, labels_sand, meta_sand,
            pair_flir_comp,
            tiles=2,
            legend_ab=['FLIR Air', 'Sand'],
            export_dir=plot_dir,
            export_base=f"{meta_air['base']}_FLIR_Air_vs_Sand"
        )
        
        # Track generated FLIR vs Sand plots (PNG and PDF)
        for _, _, comp_name, _, _ in pair_flir_comp:
            plot_base = f"{meta_air['base']}_FLIR_Air_vs_Sand_{comp_name}_pair_compare"
            generated_plots.append(str(plot_dir / f"{plot_base}.png"))
            generated_plots.append(str(plot_dir / f"{plot_base}.pdf"))
        
        # Generate Air vs Sand comparison plots
        print(f"\nGenerating Air vs Sand comparison plots...")
        print(f"  Components: {len(pair_therm_comp)}")
        validator.plot_pair_compare(
            x_air, y_air, labels_air, meta_air,
            x_sand, y_sand, labels_sand, meta_sand,
            pair_therm_comp,
            tiles=2,
            legend_ab=['Air', 'Sand'],
            export_dir=plot_dir,
            export_base=f"{meta_sand['base']}_vsSand"
        )
        
        # Track generated Air vs Sand plots (PNG and PDF)
        for _, _, comp_name, _, _ in pair_therm_comp:
            plot_base = f"{meta_sand['base']}_vsSand_{comp_name}_pair_compare"
            generated_plots.append(str(plot_dir / f"{plot_base}.png"))
            generated_plots.append(str(plot_dir / f"{plot_base}.pdf"))
        
        print(f"\nGenerated {len(generated_plots)} detailed comparison plots (PNG + PDF)")
        print(f"  Output directory: {plot_dir}")
        
        # Export plot manifest CSV
        manifest_file = plot_dir / "plot_manifest.csv"
        df_manifest = pd.DataFrame({
            'plot_file': [Path(p).name for p in generated_plots],
            'plot_path': generated_plots,
            'plot_type': ['FLIR_vs_Sand' if 'FLIR_Air_vs_Sand' in p else 'Air_vs_Sand' for p in generated_plots],
            'format': [Path(p).suffix[1:] for p in generated_plots]
        })
        df_manifest.to_csv(manifest_file, index=False)
        print(f"\nSaved plot manifest: {manifest_file}")
        print(f"  {len(df_manifest)} plot files listed (PNG + PDF)")
        
        return generated_plots
    
    @staticmethod
    def _calculate_initial_temp(x: np.ndarray, y: np.ndarray, time_window: float = 30.0) -> float:
        """
        Calculate initial temperature from first N seconds of data.
        
        Takes mean temperature from first time_window seconds to establish
        baseline ambient temperature before significant component heating.
        
        Args:
            x: Time array (seconds)
            y: Temperature array (°C) for single channel
            time_window: Time window in seconds (default 30.0)
        
        Returns:
            Mean temperature over initial time window (°C)
        """
        if len(x) == 0 or len(y) == 0:
            return np.nan
        
        # Find samples within initial time window
        initial_mask = x <= (x[0] + time_window)
        
        if np.sum(initial_mask) == 0:
            # If no samples in window, use first sample
            return y[0]
        
        # Return mean of initial samples
        return np.mean(y[initial_mask])
    
    @staticmethod
    def _find_label_idx(labels: List[str], pattern: str) -> Optional[int]:
        """Find index of label matching pattern (case-insensitive)."""
        import re
        for i, lbl in enumerate(labels):
            if pattern.lower() in lbl.lower():
                return i
        for i, lbl in enumerate(labels):
            if re.search(pattern, lbl, re.IGNORECASE):
                return i
        return None


def main():
    """
    Main execution for thermal calibration.
    
    Processes PCBs with both FLIR and thermistor measurements to build
    calibration database for correcting FLIR-only measurements.
    """
    print("="*80)
    print("THERMAL CALIBRATION - PHASE 6")
    print("="*80)
    print("\nBuilding calibration database from thermistor-validated measurements...")
    
    # Initialize calibrator
    calibrator = ThermalCalibrator()
    
    # Define input files
    input_dir = Path(__file__).parent / "inputs"
    output_dir = Path(__file__).parent / "outputs"
    
    # Load component mapping from JSON
    mapping_file = Path(__file__).parent / "loadshedding_thermistor_mapping.json"
    
    print(f"\nLoading component mapping from: {mapping_file}")
    with open(mapping_file, 'r') as f:
        mapping_config = json.load(f)
    
    # LoadShedding AC Switch PCB (calibration reference)
    # Use comprehensive ResearchIR FLIR data (all 61 components)
    flir_file = output_dir / "LoadShedding_FLIR_AllComponents.csv"
    air_file = input_dir / "Test_3_AIR_usb_temp_DAQami.csv"
    sand_file = input_dir / "Test_3_SAND_usb_temp_DAQami.csv"
    
    # Build measurement pairs from JSON mapping
    # Format: (FLIR_pattern, Thermistor_pattern, Component_Name, Component_Type)
    pairs = []
    thermistor_mapping = mapping_config["thermistor_to_flir_mapping"]
    
    for thermistor_channel, mapping_data in thermistor_mapping.items():
        flir_component = mapping_data["flir_component"]
        component_type = mapping_data["component_type"]
        component_name = f"{flir_component}_{component_type}"
        
        # Skip J1 since it's ambient measurement, not a real component
        if flir_component == "J1":
            print(f"  Skipping {thermistor_channel} -> {flir_component} (Ambient measurement)")
            continue
        
        pairs.append((flir_component, thermistor_channel, component_name, component_type))
        print(f"  Mapping {thermistor_channel} -> {flir_component} ({component_type})")
    
    print(f"\nTotal measurement pairs: {len(pairs)}")
    
    # Add LoadShedding calibration data
    calibrator.add_measurement_pair(
        flir_file=flir_file,
        therm_air_file=air_file,
        therm_sand_file=sand_file,
        pairs=pairs,
        pcb_name="LoadShedding_AC_Switch"
    )
    
    # Generate detailed comparison plots (Test_X style time-series plots)
    # DISABLED - Not needed for multi-test workflow
    # calibrator.generate_detailed_comparison_plots(
    #     flir_file=flir_file,
    #     therm_air_file=air_file,
    #     therm_sand_file=sand_file,
    #     pairs=pairs,
    #     output_subdir="calibration_database/detailed_plots"
    # )
    
    # Compute component-type calibrations
    calibrator.compute_component_type_calibrations()
    
    # Export calibration database
    calibrator.export_calibration_database(prefix="thermal_calibration")
    
    print(f"\n{'='*80}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*80}")
    print("\nCalibration database ready for Phase 7 (Thermal Prediction)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
