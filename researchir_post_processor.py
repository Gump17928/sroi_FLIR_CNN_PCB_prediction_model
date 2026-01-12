"""
===============================================================================
RESEARCHIR THERMAL DATA POST-PROCESSOR
===============================================================================
Main orchestrator for thermal post-processing workflow.

Coordinates all 7 phases:
1. Data Loading - Load and parse ResearchIR exports
2. Filtering - Remove camera artifacts, analyze transients
3. Component Analysis - Statistical analysis and export
4. Spatial Coupling - Thermal interaction analysis
5. Potting Risk - Failure prediction for embedded systems
6. Thermal Calibration - Build calibration database from thermistor measurements
   - Incremental Mode: Append to existing database (--append_calibration)
   - Multi-Session: Batch process multiple tests (--multi_session_config)
   - Convergence Analysis: Track quality improvement (--convergence_analysis)
7. Thermal Prediction - Apply calibration to predict temperatures

Incremental Calibration Features:
  - Accumulate calibration data across multiple test sessions
  - Track which session contributed which measurements
  - Visualize convergence and quality improvement
  - Identify component types needing more data

Usage:
    # Standard processing
    python researchir_post_processor.py
    
    # Incremental calibration (Week 1)
    python researchir_post_processor.py --thermal_modeling \
        --test_session "Week1" --convergence_analysis
    
    # Incremental calibration (Week 2 - append)
    python researchir_post_processor.py --thermal_modeling \
        --append_calibration --test_session "Week2" --convergence_analysis
    
    # Multi-session batch processing
    python researchir_post_processor.py --thermal_modeling \
        --multi_session_config config.json --convergence_analysis

Updated: December 2, 2025 - Added incremental calibration support
===============================================================================
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Import phase modules
import phase1_data_loading as phase1
import phase2_filtering as phase2
import phase3_component_analysis as phase3
import phase4_spatial_coupling as phase4
import phase5_potting_risk as phase5
import viz_phase2_filtering
import viz_phase3_statistics
import viz_phase4_coupling
import config_ui

# Import thermal modeling phases (6 & 7)
try:
    import phase6_thermal_calibration as phase6
    import phase7_thermal_prediction as phase7
    from loader_researchir import ResearchIRStatsParser
    import json
    from datetime import datetime
    THERMAL_MODELING_AVAILABLE = True
except ImportError:
    THERMAL_MODELING_AVAILABLE = False
    print("Warning: Thermal modeling phases (6 & 7) not available")

# Import machine learning phase (8)
try:
    import phase8_ml_training as phase8
    import viz_phase8_ml_results
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: Machine learning phase (8) not available")


def _export_thermistor_timeseries_csv(thermistor_data: Dict, output_file: Path, medium: str = "sand"):
    """
    Export compiled thermistor time series to CSV for ML model training.
    
    Creates a CSV with Time (s) column and one column per component, suitable for
    CNN thermal modeling where thermistor measurements serve as ground truth.
    
    Args:
        thermistor_data: Dict mapping component_testID -> DataFrame with Time, Temperature, Component
        output_file: Path to output CSV file
        medium: Medium type ("sand" or "air") for logging
    """
    if not thermistor_data:
        print(f"    No {medium} thermistor data to export")
        return
    
    try:
        # Find all unique components and tests
        component_test_pairs = {}
        for key, df in thermistor_data.items():
            component = df['Component'].iloc[0] if 'Component' in df.columns else key.split('_')[0]
            test = df['Test'].iloc[0] if 'Test' in df.columns else key.split('_')[-1]
            component_test_pairs[key] = (component, test)
        
        # Find common time base (use the one with most samples)
        max_samples = 0
        reference_time = None
        for key, df in thermistor_data.items():
            if len(df) > max_samples:
                max_samples = len(df)
                reference_time = df['Time'].values
        
        # Start with time column
        merged_df = pd.DataFrame({'Time (s)': reference_time})
        
        # Add each component's temperature as a column
        for key, df in thermistor_data.items():
            component, test = component_test_pairs[key]
            
            # Interpolate to common time base if needed
            if len(df['Time']) != len(reference_time) or not np.allclose(df['Time'].values, reference_time):
                # Use interpolation to align to common time base
                interp_func = interp1d(df['Time'].values, df['Temperature'].values, 
                                      kind='linear', fill_value='extrapolate')
                temps = interp_func(reference_time)
            else:
                temps = df['Temperature'].values
            
            # Column name: just component name
            # If same component appears multiple times, keep first occurrence
            col_name = component
            counter = 1
            while col_name in merged_df.columns:
                col_name = f"{component}_{counter}"
                counter += 1
            
            merged_df[col_name] = temps
        
        # Export to CSV
        merged_df.to_csv(output_file, index=False)
        print(f"    ✓ Exported {len(merged_df.columns)-1} components to {output_file.name}")
        print(f"      Time range: {merged_df['Time (s)'].min():.1f}s - {merged_df['Time (s)'].max():.1f}s")
        print(f"      Total samples: {len(merged_df)}")
        
    except Exception as e:
        print(f"    ✗ Error exporting {medium} thermistor data: {e}")
        import traceback
        traceback.print_exc()


def _create_three_condition_plots_from_tests(cal_flir_file, test_configs, calibration_pcb, outputs_path, plot_settings=None):
    """
    Helper function to create 3-condition comparison plots from multiple tests.
    Combines data from ALL tests to show all measured components.
    
    Args:
        cal_flir_file: Path to calibration FLIR CSV
        test_configs: List of test configurations with therm files and pairs
        calibration_pcb: Name of calibration PCB
        outputs_path: Base output directory path
        plot_settings: Optional dict with plot customization (y_axis_min, y_axis_max, etc.)
    """
    from loader_thermistor import ThermalDataLoader
    import viz_phase2_filtering
    import pandas as pd
    
    # Load FLIR data once (shared across all tests)
    flir_df = pd.read_csv(cal_flir_file)
    time_col = 'reltime' if 'reltime' in flir_df.columns else 'time_s'
    
    # Convert FLIR dataframe to component dictionaries
    flir_data = {}
    for comp in flir_df.columns:
        if comp in ['frame', 'reltime', 'time_s', 'Image']:
            continue
        flir_data[comp] = pd.DataFrame({
            'Time': flir_df[time_col].values,
            'Temperature': flir_df[comp].values
        })
    
    # Aggregate data from ALL tests
    # Note: Same component may be tested in Air Test X but Sand Test Y
    # So we build separate mappings for air and sand across all tests
    air_data = {}
    sand_data = {}
    loader = ThermalDataLoader()
    
    # Process each test for AIR measurements
    print(f"  Collecting AIR measurements from all tests...")
    for test_config in test_configs:
        therm_air_file = test_config['therm_air_file']
        # Use air_pairs if available, otherwise fallback to pairs
        pairs = test_config.get('air_pairs', test_config.get('pairs', []))
        test_id = test_config.get('test_id', 'unknown')
        
        # Load air thermistor data for this test (skip if loading fails)
        try:
            x_air, y_air, labels_air, meta_air = (
                loader.load_temp_csv(therm_air_file) if not isinstance(therm_air_file, list) 
                else loader.load_multi_device_csv(therm_air_file)
            )
        except Exception as e:
            print(f"    Skipping {test_id} air data: {e}")
            continue
        
        # Map air thermistor channels to components using pairs
        for pair in pairs:
            if isinstance(pair, dict):
                flir_comp = pair['flir_roi']
                therm_channel = pair['therm_channel']
            else:
                flir_comp, therm_channel, _, _ = pair
            
            # Find thermistor channel index in air data
            air_idx = None
            for i, label in enumerate(labels_air):
                if therm_channel in label:
                    air_idx = i
                    break
            
            # Create unique key: component_testID to allow same component in multiple tests
            unique_key = f"{flir_comp}_{test_id}"
            
            # Add component air data
            if air_idx is not None:
                air_data[unique_key] = pd.DataFrame({
                    'Time': x_air,
                    'Temperature': y_air[:, air_idx],
                    'Component': flir_comp,
                    'Test': test_id
                })
                print(f"    {flir_comp} ({test_id}): Air data")
    
    # Process each test for SAND measurements
    print(f"  Collecting SAND measurements from all tests...")
    for test_config in test_configs:
        therm_sand_file = test_config.get('therm_sand_file')
        # Use sand_pairs if available, otherwise fallback to pairs
        pairs = test_config.get('sand_pairs', test_config.get('pairs', []))
        test_id = test_config.get('test_id', 'unknown')
        
        if not therm_sand_file:
            continue
        
        # Load sand thermistor data for this test (skip if loading fails)
        try:
            x_sand, y_sand, labels_sand, meta_sand = (
                loader.load_temp_csv(therm_sand_file) if not isinstance(therm_sand_file, list) 
                else loader.load_multi_device_csv(therm_sand_file)
            )
        except Exception as e:
            print(f"    Skipping {test_id} sand data: {e}")
            continue
        
        # Map sand thermistor channels to components using pairs
        for pair in pairs:
            if isinstance(pair, dict):
                flir_comp = pair['flir_roi']
                therm_channel = pair['therm_channel']
            else:
                flir_comp, therm_channel, _, _ = pair
            
            # Find thermistor channel index in sand data
            sand_idx = None
            for i, label in enumerate(labels_sand):
                if therm_channel in label:
                    sand_idx = i
                    break
            
            # Create unique key: component_testID to allow same component in multiple tests
            unique_key = f"{flir_comp}_{test_id}"
            
            # Add component sand data
            if sand_idx is not None:
                sand_data[unique_key] = pd.DataFrame({
                    'Time': x_sand,
                    'Temperature': y_sand[:, sand_idx],
                    'Component': flir_comp,
                    'Test': test_id
                })
                print(f"    {flir_comp} ({test_id}): Sand data")
    
    # =========================================================================
    # EXPORT COMPILED THERMISTOR TIME SERIES FOR ML MODEL (CNN TRAINING)
    # =========================================================================
    print(f"\n  Exporting compiled thermistor time series for ML model...")
    
    # Export SAND thermistor data (for CNN training ground truth)
    if sand_data:
        _export_thermistor_timeseries_csv(
            sand_data, 
            outputs_path / f"{calibration_pcb}_thermistor_timeseries.csv",
            medium="sand"
        )
    
    # Export AIR thermistor data (optional, for reference)
    if air_data:
        _export_thermistor_timeseries_csv(
            air_data, 
            outputs_path / f"{calibration_pcb}_air_thermistor_timeseries.csv",
            medium="air"
        )
    
    # Build list of component names from FLIR data
    flir_component_names = set(flir_data.keys())
    
    # Extract component names from air and sand unique keys
    air_components_list = []
    for key in air_data.keys():
        if 'Component' in air_data[key].columns:
            comp_name = air_data[key]['Component'].iloc[0]
        else:
            comp_name = key.split('_')[0]
        air_components_list.append(comp_name)
    air_component_names = set(air_components_list)
    
    sand_components_list = []
    for key in sand_data.keys():
        if 'Component' in sand_data[key].columns:
            comp_name = sand_data[key]['Component'].iloc[0]
        else:
            comp_name = key.split('_')[0]
        sand_components_list.append(comp_name)
    sand_component_names = set(sand_components_list)
    
    # Find components that exist in all three datasets
    overlapping_components = list(flir_component_names & air_component_names & sand_component_names)
    
    if not overlapping_components:
        print("  No overlapping components found between FLIR, Air, and Sand data")
        return
    
    print(f"  Found {len(overlapping_components)} overlapping components: {sorted(overlapping_components)}")
    print(f"  Total air measurements: {len(air_data)}")
    print(f"  Total sand measurements: {len(sand_data)}")
    
    # Create output directory
    output_dir = outputs_path / "three_condition_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate main combined plot
    output_files, valid_entries = viz_phase2_filtering.create_three_condition_comparison(
        flir_data=flir_data,
        air_data=air_data,
        sand_data=sand_data,
        component_names=overlapping_components,
        output_dir=str(output_dir),
        pcb_name=calibration_pcb,
        plot_settings=plot_settings
    )
    
    print(f"  Created {len(output_files)} main plot files in {output_dir}")
    
    # Generate individual component detail plots
    if valid_entries:
        detail_files = viz_phase2_filtering.create_component_detail_plots(
            flir_data=flir_data,
            air_data=air_data,
            sand_data=sand_data,
            valid_entries=valid_entries,
            output_dir=str(output_dir),
            pcb_name=calibration_pcb,
            plot_settings=plot_settings
        )
        print(f"  Created {len(detail_files)} component detail plots")
    
    # Generate thermal metrics CSV and max temperature plot
    # Run filtering and analysis on FLIR data to get component statistics
    import phase1_data_loading as phase1
    import phase2_filtering as phase2
    import viz_phase3_statistics
    
    # Classify components by type
    grouped_components = phase1.classify_components(flir_data, custom_groups=None, debug=False)
    
    # Apply filtering and analyze temperature transients
    filtered_flir_data = phase2.apply_filtering_to_component_data(flir_data, filter_type='median', debug=False)
    analysis_results = phase2.analyze_temperature_transients(flir_data, filter_type='median', debug=False)
    
    # Generate thermal metrics visualization (max temp plot + CSV export)
    print(f"  Generating thermal metrics for {calibration_pcb}...")
    metrics_files = []
    viz_phase3_statistics.create_ieee_plots(
        component_data=filtered_flir_data,
        grouped_components=grouped_components,
        analysis_results=analysis_results,
        output_dir=str(output_dir),
        pcb_name=calibration_pcb,
        debug=False
    )
    print(f"  Exported thermal metrics: {output_dir}/{calibration_pcb}_component_thermal_metrics.csv")


def _create_three_condition_plots(cal_flir_file, therm_air_file, therm_sand_file, 
                                  pairs, calibration_pcb, outputs_path):
    """
    Helper function for single-test mode (backward compatibility).
    Wraps the multi-test function for a single test configuration.
    """
    test_config = {
        'therm_air_file': therm_air_file,
        'therm_sand_file': therm_sand_file,
        'pairs': pairs
    }
    _create_three_condition_plots_from_tests(
        cal_flir_file=cal_flir_file,
        test_configs=[test_config],
        calibration_pcb=calibration_pcb,
        outputs_path=outputs_path
    )


def generate_output_folder(workflow: str, debug: bool = False, base_dir: str = "outputs") -> str:
    """
    Generate timestamped output folder name.
    
    Format: MMDD_HHMM_<workflow>[_debug]
    Example: 1126_1430_P1-7
    
    Args:
        workflow: Workflow type ('phases_1_5', 'phases_6_7', 'full_pipeline')
        debug: Whether debug mode is enabled
        base_dir: Base outputs directory
        
    Returns:
        Full path to auto-generated output directory
    """
    now = datetime.now()
    timestamp = now.strftime("%m%d_%H%M")
    
    # Workflow abbreviations
    workflow_abbrev = {
        'phases_1_5': 'P1-5',
        'phases_6_7': 'P6-7',
        'full_pipeline': 'P1-7'
    }
    
    abbrev = workflow_abbrev.get(workflow, 'P1-7')
    folder_name = f"{timestamp}_{abbrev}"
    
    if debug:
        folder_name += "_debug"
    
    return str(Path(base_dir) / folder_name)

# Component type definitions for consistent use across phases
COMPONENT_TYPES = phase1.DEFAULT_COMPONENT_TYPES


def process_researchir_data(input_folder: str, output_dir: str,
                           coordinates_file: Optional[str] = None,
                           proximity_threshold: float = 10.0,
                           filter_type: str = 'median',
                           spatial_enabled: bool = True,
                           debug: bool = False) -> Dict:
    """
    Complete ResearchIR thermal data post-processing workflow (Phases 1-5)
    
    Executes all 5 phases of thermal analysis:
    1. Load data from ResearchIR exports
    2. Apply filtering and analyze transients
    3. Generate statistical summaries
    4. Analyze spatial thermal coupling
    5. Assess potting/embedding risk
    
    Args:
        input_folder: Path to ResearchIR export folder
        output_dir: Path for output files
        coordinates_file: Optional CSV with component X,Y coordinates
        proximity_threshold: Distance threshold for thermal neighbors (mm)
        filter_type: Filter to apply ('median', 'savgol', 'none', etc.)
        spatial_enabled: Enable spatial coupling analysis
        debug: Enable verbose debug output
    
    Returns:
        Dictionary with processing results and output file paths
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(" RESEARCHIR THERMAL DATA POST-PROCESSING (Phases 1-5)")
    print("="*60)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_dir}")
    print(f"Filter type:   {filter_type}")
    print(f"Spatial analysis: {'Enabled' if spatial_enabled and coordinates_file else 'Disabled'}")
    print("="*60)
    
    # Extract PCB name from input folder path for plot labeling
    # Example: "inputs/ResearchIR_Outputs_LoadShedding" -> "LoadShedding"
    pcb_name = None
    if input_folder:
        folder_name = os.path.basename(input_folder.rstrip('/\\'))
        if 'ResearchIR_Outputs_' in folder_name:
            pcb_name = folder_name.split('ResearchIR_Outputs_')[-1]
        else:
            pcb_name = folder_name
    
    # =========================================================================
    # PHASE 1: DATA LOADING
    # =========================================================================
    print("\n[PHASE 1] Loading ResearchIR export files...")
    component_data = phase1.load_researchir_files(input_folder, debug=debug)
    
    print("\n[PHASE 1] Classifying components by type...")
    grouped_components = phase1.classify_components(component_data, debug=debug)
    
    # =========================================================================
    # PHASE 2: FILTERING
    # =========================================================================
    print(f"\n[PHASE 2] Applying {filter_type} filtering...")
    filtered_component_data = phase2.apply_filtering_to_component_data(
        component_data, filter_type=filter_type, debug=debug)
    
    print("\n[PHASE 2] Analyzing temperature transients...")
    analysis_results = phase2.analyze_temperature_transients(
        component_data, filter_type=filter_type, debug=debug)
    
    # =========================================================================
    # PHASE 3: COMPONENT ANALYSIS
    # =========================================================================
    print("\n[PHASE 3] Exporting MATLAB data...")
    mat_file = phase3.export_matlab_data(filtered_component_data,
                                        grouped_components,
                                        analysis_results,
                                        output_dir)
    
    print("\n[PHASE 3] Creating summary CSV...")
    csv_file = phase3.export_summary_csv(grouped_components,
                                         analysis_results,
                                         output_dir,
                                         COMPONENT_TYPES)
    
    # =========================================================================
    # VISUALIZATION: IEEE-FORMAT PLOTS
    # =========================================================================
    print("\n[VISUALIZATION] Creating IEEE-format summary plots...")
    plot_files = viz_phase3_statistics.create_ieee_plots(filtered_component_data, grouped_components,
                                        analysis_results, output_dir, pcb_name=pcb_name, debug=debug)    # =========================================================================
    # VISUALIZATION: FILTERING COMPARISON
    # =========================================================================
    comparison_files = []
    if filter_type != 'none':
        print(f"\n[VISUALIZATION] Creating {filter_type} filtering comparison plots...")
        comparison_files = viz_phase2_filtering.create_filtering_comparison_plots(
            component_data, filtered_component_data, grouped_components,
            output_dir, filter_type=filter_type, debug=debug)
        
        print(f"\n[VISUALIZATION] Creating multi-filter comparison grid...")
        multifilter_grid, comparison_df = viz_phase2_filtering.create_multifilter_component_grid(
            component_data, grouped_components, output_dir, 
            filter_params={}, debug=debug)
    
    # =========================================================================
    # MATLAB: USAGE SCRIPT
    # =========================================================================
    print("\n[MATLAB] Creating MATLAB usage script...")
    phase3.create_matlab_usage_script(output_dir, grouped_components, COMPONENT_TYPES)
    
    # =========================================================================
    # PHASE 4: SPATIAL COUPLING (if coordinates provided)
    # =========================================================================
    coupling_metrics = {}
    component_coordinates = {}
    proximity_matrix = {}
    coupling_plots = []
    
    if spatial_enabled and coordinates_file:
        print("\n[PHASE 4] Loading component coordinates...")
        component_coordinates = phase4.load_component_coordinates(coordinates_file, debug=debug)
        
        if component_coordinates:
            print(f"\n[PHASE 4] Building proximity matrix (threshold: {proximity_threshold}mm)...")
            proximity_matrix = phase4.build_proximity_matrix(
                component_data, component_coordinates, proximity_threshold)
            
            if proximity_matrix:
                print("\n[PHASE 4] Calculating thermal coupling metrics...")
                coupling_metrics = phase4.calculate_thermal_coupling_metrics(
                    filtered_component_data, proximity_matrix, analysis_results)
                
                print(f"  Analyzed thermal coupling for {len(coupling_metrics)} components")
                
                print("\n[PHASE 4] Creating thermal coupling visualizations...")
                coupling_plots = viz_phase4_coupling.create_thermal_coupling_visualizations(
                    filtered_component_data, coupling_metrics, analysis_results,
                    output_dir, component_coordinates, debug=debug)
    
    # =========================================================================
    # PHASE 5: POTTING RISK (requires coupling metrics from Phase 4)
    # =========================================================================
    risk_csv = None
    risk_df = None
    
    if coupling_metrics:
        print("\n[PHASE 5] Analyzing potted condition failure risk...")
        risk_csv, risk_df = phase5.analyze_potted_condition_risk(
            coupling_metrics, analysis_results, output_dir)
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    results = {
        'input_folder': input_folder,
        'output_dir': output_dir,
        'total_components': len(component_data),
        'component_groups': grouped_components,
        'analysis_results': analysis_results,
        'coupling_metrics': coupling_metrics,
        'risk_analysis_df': risk_df,
        'output_files': {
            'plots': plot_files,
            'comparison_plots': comparison_files,
            'coupling_plots': coupling_plots,
            'matlab': mat_file,
            'summary_csv': csv_file,
            'risk_csv': risk_csv
        }
    }
    
    print("\n" + "="*60)
    print(" PHASES 1-5 COMPLETE")
    print("="*60)
    print(f"Total components processed: {len(component_data)}")
    print(f"Component groups: {len(grouped_components)}")
    if coupling_metrics:
        print(f"Spatial coupling analysis: {len(coupling_metrics)} components")
    print(f"\nOutput files saved to: {output_dir}")
    print("="*60)
    
    return results


def validate_thermal_config(multi_session_config: str = None, 
                           verbose: bool = False) -> bool:
    """
    Validate thermal modeling configuration without processing.
    
    Supports both legacy session-based and new board-based configs.
    
    Performs comprehensive validation checks:
    - Configuration file structure and JSON syntax
    - Required fields present in all sessions/tests
    - File paths exist (FLIR, thermistor files)
    - Channel naming conventions (Device prefix for multi-device)
    - Component pair definitions
    
    Args:
        multi_session_config: Path to JSON config file (sessions or boards)
        verbose: Show detailed validation output
        
    Returns:
        True if all validation checks pass, False otherwise
    """
    from viz_phase6_validation import (validate_multi_session_config, 
                                        validate_multi_board_config)
    import json
    
    print("="*80)
    print(" THERMAL CALIBRATION CONFIGURATION VALIDATION")
    print("="*80)
    
    if not multi_session_config:
        print("\n✗ Error: No configuration file specified")
        print("  Use --multi_session_config <config.json> to specify config file")
        return False
    
    print(f"\nValidating configuration: {multi_session_config}")
    print("-"*80)
    
    # Detect config format (session-based vs board-based)
    try:
        with open(multi_session_config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"\n✗ Error: Failed to load config file: {e}")
        return False
    
    is_board_based = 'boards' in config
    is_session_based = 'sessions' in config
    
    if is_board_based:
        print(f"  Config type: Board-based (hierarchical)")
        result = validate_multi_board_config(
            config_file=multi_session_config,
            verbose=verbose
        )
    elif is_session_based:
        print(f"  Config type: Session-based (legacy)")
        result = validate_multi_session_config(
            config_path=multi_session_config,
            check_files_exist=True,
            verbose=verbose
        )
    else:
        print("\n✗ Error: Config must have either 'boards' or 'sessions' field")
        return False
    
    # Print summary (unified format for both types)
    if not verbose:
        print("\n" + "="*80)
        print(" VALIDATION SUMMARY")
        print("="*80)
        
        if result['valid']:
            print("\n✓ ALL VALIDATION CHECKS PASSED")
            
            if is_board_based:
                print(f"\n  Total boards: {result['stats']['board_count']}")
                print(f"  Total tests: {result['stats']['test_count']}")
                print(f"  Total components: {result['stats']['component_count']}")
            else:
                print(f"\n  Total sessions: {result['stats']['total_sessions']}")
                print(f"  Total component pairs: {result['stats']['total_pairs']}")
                
                # Show session breakdown
                print(f"\n  Session Breakdown:")
                for i, session_result in enumerate(result['session_results'], 1):
                    session_name = result['config']['sessions'][i-1]['name']
                    num_pairs = session_result['stats']['num_pairs']
                    is_multi = session_result['stats'].get('is_multi_device', False)
                    device_str = "(multi-device)" if is_multi else "(single-device)"
                    print(f"    {i}. {session_name}: {num_pairs} components {device_str}")
            
            print(f"\n  Ready to process!")
            print(f"  Run without --validate to execute full workflow.")
            
        else:
            print("\n✗ VALIDATION FAILED")
            print(f"\n  Found {len(result['errors'])} error(s)")
            
            # Show errors
            if result['errors']:
                print("\n  Errors:")
                for err in result['errors'][:20]:  # Show first 20
                    print(f"    • {err}")
                if len(result['errors']) > 20:
                    print(f"    ... and {len(result['errors']) - 20} more errors")
        
        # Show warnings (even if validation passed)
        if result['warnings']:
            print(f"\n  ⚠ Warnings ({len(result['warnings'])}):")
            for warn in result['warnings'][:20]:  # Show first 20
                print(f"    • {warn}")
            if len(result['warnings']) > 20:
                print(f"    ... and {len(result['warnings']) - 20} more warnings")
        
        print("\n" + "="*80)
    
    return result['valid']


def process_thermal_modeling(calibration_pcb: str, prediction_pcb: str,
                           calibration_mapping: str,
                           inputs_dir: str = "inputs",
                           outputs_dir: str = "outputs",
                           debug: bool = False,
                           append_calibration: bool = False,
                           test_session: str = None,
                           multi_session_config: str = None,
                           convergence_analysis: bool = False) -> Dict:
    """
    Execute thermal modeling workflow (Phases 6 & 7).
    
    Phase 6: Calibration using PCB with both FLIR and thermistor data
    Phase 7: Prediction using calibration to estimate temperatures on FLIR-only PCB
    
    CALIBRATION MODES
    ==================
    1. Standard: Single test session, replace existing calibration
    2. Incremental: Single test session, append to existing calibration
    3. Multi-Session: Batch process multiple test sessions at once
    
    CONFIGURATION EXAMPLES
    ======================
    
    Single-Device Configuration (e.g., USB-TEMP with 6 channels):
    -------------------------------------------------------------
    {
      "sessions": [{
        "name": "Batch1_Test3_LoadShedding",
        "pcb": "Load_Shedding",
        "flir_file": "outputs/1205_1106_P1-7/Load_Shedding/Test_3_FLIR_Camera_Results_5_of_7.csv",
        "therm_air_file": "inputs/AIR_usb_temp_20241205_1200.csv",
        "therm_sand_file": "inputs/SAND_usb_temp_20241205_1200.csv",
        "pairs": [
          {"flir_roi": "U3", "therm_air_chan": "AI0", "therm_sand_chan": "AI0"},
          {"flir_roi": "U1", "therm_air_chan": "AI1", "therm_sand_chan": "AI1"},
          {"flir_roi": "J1", "therm_air_chan": "AI2", "therm_sand_chan": "AI2"}
        ]
      }]
    }
    
    Multi-Device Configuration (e.g., USB-TEMP + USB-TEMP-AI = 8 channels):
    -----------------------------------------------------------------------
    IMPORTANT: When using multiple thermistor devices, channel names MUST include 
    device prefix (Device0_AI0, Device1_AI4, etc.) to avoid conflicts!
    
    {
      "sessions": [{
        "name": "Batch2_MultiDevice_LoadShedding",
        "pcb": "Load_Shedding",
        "flir_file": "outputs/1205_1106_P1-7/Load_Shedding/Load_Shedding_FLIR_AllComponents.csv",
        "therm_air_file": [
          "inputs/AIR_usb_temp_device0_20241205.csv",    # Device 0: AI0-AI5 (6 ch)
          "inputs/AIR_usb_temp_device1_20241205.csv"     # Device 1: AI4-AI5 (2 ch)
        ],
        "therm_sand_file": [
          "inputs/SAND_usb_temp_device0_20241205.csv",
          "inputs/SAND_usb_temp_device1_20241205.csv"
        ],
        "pairs": [
          {"flir_roi": "U2", "therm_air_chan": "Device0_AI0", "therm_sand_chan": "Device0_AI0"},
          {"flir_roi": "R5", "therm_air_chan": "Device0_AI1", "therm_sand_chan": "Device0_AI1"},
          {"flir_roi": "C3", "therm_air_chan": "Device1_AI4", "therm_sand_chan": "Device1_AI4"},
          {"flir_roi": "L1", "therm_air_chan": "Device1_AI5", "therm_sand_chan": "Device1_AI5"}
        ]
      }]
    }
    
    EXPECTED OUTPUT STRUCTURE
    =========================
    outputs/{timestamp}_P6-7/
    ├── calibration_database/
    │   ├── {SessionName}/                      # One folder per session
    │   │   ├── raw_inputs_overview.png         # All FLIR + thermistor channels
    │   │   └── detailed_plots/                 # Per-component comparisons
    │   │       ├── {Component}_flir_vs_air.png
    │   │       ├── {Component}_flir_vs_air.pdf
    │   │       ├── {Component}_flir_vs_sand.png
    │   │       └── {Component}_flir_vs_sand.pdf
    │   ├── thermal_calibration_points.csv      # Aggregate calibration database
    │   ├── thermal_calibration_by_type.csv     # Per-component-type calibrations
    │   └── thermal_calibration_summary.png     # Calibration quality visualization
    └── {prediction_pcb}/
        └── thermal_predictions.csv             # Phase 7 predictions
    
    CHANNEL NAMING CONVENTIONS
    ==========================
    Single Device:  Use channel names as-is (AI0, AI1, AI2, ...)
    Multi-Device:   Use Device{N}_{Channel} format (Device0_AI0, Device1_AI4, ...)
                    The loader automatically prefixes channels when merging multiple files.
    
    TROUBLESHOOTING
    ===============
    - "Channel not found": Check device prefix if using multiple devices
    - "Empty calibration database": Verify FLIR ROI names match thermistor mappings
    - "Merge conflict": Ensure multi-device files use unique device IDs
    - "Missing plots": Check that session name is unique and valid for filesystem
    
    Args:
        calibration_pcb: Name of PCB to use for calibration (e.g., "Load_Shedding")
        prediction_pcb: Name of PCB to predict temperatures for (e.g., "HBridge_15s")
        calibration_mapping: Path to thermistor mapping JSON file
        inputs_dir: Input directory containing ResearchIR and thermistor data
        outputs_dir: Output directory for results
        debug: Enable debug output
        append_calibration: If True, load existing calibration database and append new 
                          measurements instead of replacing. Enables incremental data 
                          collection across multiple weeks/test sessions.
        test_session: Optional test session name/ID for tracking (e.g., "LoadShedding_Week1").
                     Used to identify which test session contributed which measurements.
                     Auto-generated if not provided.
        multi_session_config: Path to JSON config file containing multiple test sessions.
                            When provided, processes all sessions in batch and ignores
                            single-session parameters. Format: {"sessions": [{...}, ...]}
        convergence_analysis: If True, automatically run convergence analysis after 
                            calibration to visualize quality improvement and identify
                            component types needing more calibration data.
    
    Returns:
        Dictionary with processing results and output file paths
        
    CLI Usage Examples:
        # Validate configuration before running
        python researchir_post_processor.py --thermal_modeling \\
            --multi_session_config my_config.json --validate
        
        # Standard multi-session processing
        python researchir_post_processor.py --thermal_modeling \\
            --multi_session_config my_config.json --convergence_analysis
        
        # Incremental mode (append to existing calibration)
        python researchir_post_processor.py --thermal_modeling \\
            --append_calibration --test_session "Week2"
    """
    inputs_path = Path(inputs_dir)
    
    # Auto-generate output directory if not specified
    if outputs_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%m%d_%H%M')
        outputs_path = Path(f"outputs/{timestamp}_P6-7")
        print(f"\nAuto-generated output directory: {outputs_path}")
    else:
        outputs_path = Path(outputs_dir)
    
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(" THERMAL MODELING WORKFLOW (Phases 6 & 7)")
    print("="*80)
    print(f"Calibration PCB: {calibration_pcb}")
    print(f"Prediction PCB:  {prediction_pcb}")
    print(f"Mapping file:    {calibration_mapping}")
    print("="*80)
    
    # =========================================================================
    # PARSE RESEARCHIR DATA (both PCBs)
    # =========================================================================
    print("\n[DATA PARSING] Parsing ResearchIR Stats files...")
    
    # Parse calibration PCB FLIR data
    cal_researchir_dir = inputs_path / f"ResearchIR_Outputs_{calibration_pcb}"
    if not cal_researchir_dir.exists():
        raise FileNotFoundError(f"Calibration ResearchIR directory not found: {cal_researchir_dir}")
    
    print(f"\n  Parsing {calibration_pcb} FLIR data...")
    cal_parser = ResearchIRStatsParser(cal_researchir_dir)
    cal_flir_df = cal_parser.parse_all_frames()
    
    cal_output_dir = outputs_path / calibration_pcb
    cal_output_dir.mkdir(parents=True, exist_ok=True)
    cal_flir_file = cal_output_dir / f"{calibration_pcb}_FLIR_AllComponents.csv"
    cal_flir_df.to_csv(cal_flir_file, index=False)
    print(f"  Saved: {cal_flir_file}")
    
    # Parse prediction PCB FLIR data
    pred_researchir_dir = inputs_path / f"ResearchIR_Outputs_{prediction_pcb}"
    if not pred_researchir_dir.exists():
        raise FileNotFoundError(f"Prediction ResearchIR directory not found: {pred_researchir_dir}")
    
    print(f"\n  Parsing {prediction_pcb} FLIR data...")
    pred_parser = ResearchIRStatsParser(pred_researchir_dir)
    pred_flir_df = pred_parser.parse_all_frames()
    
    pred_output_dir = outputs_path / prediction_pcb
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    pred_flir_file = pred_output_dir / f"{prediction_pcb}_FLIR_AllComponents.csv"
    pred_flir_df.to_csv(pred_flir_file, index=False)
    print(f"  Saved: {pred_flir_file}")
    
    # =========================================================================
    # PHASE 6: THERMAL CALIBRATION
    # =========================================================================
    # Supports three calibration modes:
    #   1. Standard: Single test session, replace existing calibration
    #   2. Incremental: Single test session, append to existing calibration
    #   3. Multi-Session: Batch process multiple test sessions
    # 
    # Incremental mode enables week-by-week data collection where each test
    # session adds new measurements to the calibration database. The system
    # tracks which session contributed which measurements for convergence
    # analysis and quality assessment.
    # =========================================================================
    print("\n" + "="*80)
    print("[PHASE 6] THERMAL CALIBRATION")
    print("="*80)
    
    # Find thermistor files
    therm_air_file = list(inputs_path.glob("*AIR*usb_temp*.csv"))
    therm_sand_file = list(inputs_path.glob("*SAND*usb_temp*.csv"))
    
    if not therm_air_file:
        raise FileNotFoundError("No AIR thermistor file found")
    
    therm_air_file = therm_air_file[0]
    therm_sand_file = therm_sand_file[0] if therm_sand_file else None
    
    print(f"\nUsing thermistor files:")
    print(f"  Air:  {therm_air_file.name}")
    if therm_sand_file:
        print(f"  Sand: {therm_sand_file.name}")
    
    # Load mapping configuration
    mapping_path = Path(calibration_mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        mapping_config = json.load(f)
    
    # Build measurement pairs from mapping
    pairs = []
    for therm_channel, mapping_data in mapping_config["thermistor_to_flir_mapping"].items():
        flir_comp = mapping_data["flir_component"]
        comp_type = mapping_data["component_type"]
        
        # Skip ambient measurements
        if "ambient" in mapping_data.get("notes", "").lower():
            print(f"  Skipping {therm_channel} -> {flir_comp} (ambient)")
            continue
        
        pairs.append((flir_comp, therm_channel, f"{flir_comp}_{comp_type}", comp_type))
        print(f"  Pair: {therm_channel} -> {flir_comp} ({comp_type})")
    
    # Run Phase 6 calibration
    cal_db_dir = outputs_path / "calibration_database"
    calibrator = phase6.ThermalCalibrator(output_dir=cal_db_dir)
    
    # Incremental Mode: Load existing calibration database if appending
    # This enables week-by-week data collection where new measurements are
    # added to the existing database rather than replacing it
    if append_calibration:
        existing_file = cal_db_dir / "thermal_calibration_points.csv"
        calibrator.load_existing_calibration(existing_file)
    
    # Multi-Session Mode: Process multiple test sessions from config file
    # Each session can have different components, thermistor channels, and
    # sand cooling availability (air-only sessions supported)
    # 
    # Supports two config formats:
    # 1. Legacy session-based: {"sessions": [...]}
    # 2. New board-based: {"boards": [{name, pcb, tests: [...]}]}
    if multi_session_config:
        print(f"\n[MULTI-SESSION MODE] Loading config: {multi_session_config}")
        with open(multi_session_config, 'r') as f:
            multi_config = json.load(f)
        
        # Detect config format and process accordingly
        if 'boards' in multi_config:
            print(f"  Config format: Board-based (hierarchical)")
            calibrator.add_measurement_boards(multi_config['boards'])
            
            # Generate three-condition comparison plots for each board (all tests combined)
            print("\n[VISUALIZATION] Creating three-condition comparison plots...")
            for board_config in multi_config['boards']:
                pcb_name = board_config['name']
                pcb_file = board_config.get('pcb', f"{pcb_name}.brd")
                tests = board_config.get('tests', [])
                
                if not tests:
                    print(f"  Skipping {pcb_name}: No tests found")
                    continue
                
                print(f"\n  Processing board: {pcb_name}")
                
                # Parse this board's FLIR data
                board_researchir_dir = inputs_path / f"ResearchIR_Outputs_{pcb_name}"
                if not board_researchir_dir.exists():
                    print(f"  WARNING: FLIR directory not found: {board_researchir_dir}")
                    print(f"  Skipping {pcb_name} visualization...")
                    continue
                
                print(f"  Parsing {pcb_name} FLIR data...")
                board_parser = ResearchIRStatsParser(board_researchir_dir)
                board_flir_df = board_parser.parse_all_frames()
                
                board_output_dir = outputs_path / pcb_name
                board_output_dir.mkdir(parents=True, exist_ok=True)
                board_flir_file = board_output_dir / f"{pcb_name}_FLIR_AllComponents.csv"
                board_flir_df.to_csv(board_flir_file, index=False)
                print(f"  Saved: {board_flir_file}")
                
                # Extract plot settings for this board
                plot_settings = board_config.get('plot_settings', {})
                
                _create_three_condition_plots_from_tests(
                    cal_flir_file=board_flir_file,
                    test_configs=tests,
                    calibration_pcb=pcb_name,
                    outputs_path=outputs_path,
                    plot_settings=plot_settings
                )
                
        elif 'sessions' in multi_config:
            print(f"  Config format: Session-based (legacy)")
            calibrator.add_measurement_sessions(multi_config['sessions'])
        else:
            raise ValueError("Config must contain either 'boards' or 'sessions' field")
    else:
        # Single session mode
        calibrator.add_measurement_pair(
            flir_file=cal_flir_file,
            therm_air_file=therm_air_file,
            therm_sand_file=therm_sand_file,
            pairs=pairs,
            pcb_name=calibration_pcb,
            test_session=test_session
        )
        
        # Generate detailed comparison plots (Test_X style time-series plots)
        # DISABLED - Not needed for multi-test workflow
        # calibrator.generate_detailed_comparison_plots(
        #     flir_file=cal_flir_file,
        #     therm_air_file=therm_air_file,
        #     therm_sand_file=therm_sand_file,
        #     pairs=pairs,
        #     output_subdir="detailed_plots"
        # )
        
        # Generate three-condition comparison plots (FLIR + Air + Sand)
        print("\n[VISUALIZATION] Creating three-condition comparison plots...")
        _create_three_condition_plots(
            cal_flir_file=cal_flir_file,
            therm_air_file=therm_air_file,
            therm_sand_file=therm_sand_file,
            pairs=pairs,
            calibration_pcb=calibration_pcb,
            outputs_path=outputs_path
        )
    
    calibrator.compute_component_type_calibrations()
    
    # Check if validation mode is enabled in config
    if multi_session_config:
        with open(multi_session_config, 'r') as f:
            multi_config = json.load(f)
        
        calibration_mode = multi_config.get('calibration_mode', 'normal')
        
        if calibration_mode == 'validation':
            print(f"\n[VALIDATION MODE] Running calibration quality assessment...")
            calibrator.run_validation_workflow()
    
    calibrator.export_calibration_database(
        prefix="thermal_calibration",
        run_convergence_analysis=convergence_analysis
    )
    
    print(f"\nCalibration database created:")
    print(f"  {cal_db_dir / 'thermal_calibration_points.csv'}")
    print(f"  {cal_db_dir / 'thermal_calibration_by_type.csv'}")
    print(f"  {cal_db_dir / 'thermal_calibration_metadata.json'}")
    
    # Store calibration database path for Phase 8
    calibration_points_file = cal_db_dir / 'thermal_calibration_points.csv'
    
    # =========================================================================
    # PHASE 7: THERMAL PREDICTION
    # =========================================================================
    print("\n" + "="*80)
    print("[PHASE 7] THERMAL PREDICTION")
    print("="*80)
    
    # Load calibration database
    calibration_file = cal_db_dir / "thermal_calibration_by_type.csv"
    
    # Load component type patterns
    patterns_file = Path(__file__).parent / "component_type_patterns.json"
    if patterns_file.exists():
        with open(patterns_file, 'r') as f:
            patterns_config = json.load(f)
        component_patterns = patterns_config["patterns"]
        print(f"\nUsing component patterns from: {patterns_file.name}")
    else:
        # Default patterns
        component_patterns = {
            "PowerSupply": [r"^PS\d+$", r"^CONV$"],
            "IC": [r"^U\d+$", r"^IC\d+$"],
            "Resistor": [r"^R\d+$"],
            "LED": [r"^DL\d+$", r"^CR\d+$"]
        }
        print("\nUsing default component patterns")
    
    # Run Phase 7 prediction
    print(f"\nPredicting temperatures for {prediction_pcb}...")
    predictor = phase7.ThermalPredictor(calibration_file=calibration_file, output_dir=pred_output_dir, verbose=debug)
    
    # Set component type mapping from patterns
    # Build mapping from component patterns by checking each component name against patterns
    # This will be done automatically in predict_from_flir using _determine_component_type()
    
    # Run prediction
    df_predictions = predictor.predict_from_flir(
        flir_file=pred_flir_file,
        pcb_name=prediction_pcb,
        use_median_offset=True
    )
    
    # Export results
    predictor.export_predictions(df_predictions, prefix=f"thermal_prediction_{prediction_pcb}")
    
    print(f"\nPredictions exported to: {pred_output_dir}")
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    results = {
        'calibration_pcb': calibration_pcb,
        'prediction_pcb': prediction_pcb,
        'calibration_database': str(cal_db_dir),
        'calibration_points_file': str(calibration_points_file),
        'prediction_output_dir': str(pred_output_dir),
        'output_files': {
            'calibration_points': str(cal_db_dir / 'thermal_calibration_points.csv'),
            'calibration_by_type': str(cal_db_dir / 'thermal_calibration_by_type.csv'),
        }
    }
    
    print("\n" + "="*80)
    print(" PHASES 6-7 COMPLETE")
    print("="*80)
    
    return results


def process_ml_training(calibration_points_file: str,
                       pcb_filter: str = "Load_Shedding",
                       outputs_dir: str = "outputs",
                       debug: bool = False) -> Dict:
    """
    Execute machine learning workflow (Phase 8).
    
    Train OLS linear regression model to predict sand embedded temperatures
    from FLIR air measurements using component type as additional feature.
    
    Model: ΔT_sand = β₀ + β₁·ΔT_flir_air + β₂·is_IC + β₃·is_Resistor + ... + ε
    
    Args:
        calibration_points_file: Path to thermal_calibration_points.csv (from Phase 6)
        pcb_filter: PCB name to filter training data (e.g., "Load_Shedding")
        outputs_dir: Output directory for model and visualizations
        debug: Enable debug output
    
    Returns:
        Dictionary with processing results and output file paths
    """
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(" MACHINE LEARNING WORKFLOW (Phase 8)")
    print("="*80)
    print(f"Training PCB: {pcb_filter}")
    print(f"Calibration data: {calibration_points_file}")
    print("="*80)
    
    # =========================================================================
    # PHASE 8: ML TRAINING
    # =========================================================================
    print("\n" + "="*80)
    print("[PHASE 8] MACHINE LEARNING TRAINING")
    print("="*80)
    
    # Initialize ML predictor (outputs_dir already includes ml_model if needed)
    predictor = phase8.ThermalMLPredictor(output_dir=str(outputs_path), verbose=True)
    
    # Load training data
    predictor.load_training_data(
        calibration_csv=calibration_points_file,
        pcb_filter=pcb_filter
    )
    
    # Calculate delta T for each component
    predictor.calculate_delta_t()
    
    # Prepare feature matrix with one-hot encoded component types
    predictor.prepare_feature_matrix()
    
    # Train OLS model
    predictor.train_model()
    
    # Calculate performance metrics
    metrics = predictor.calculate_metrics()
    
    # Export model and metrics
    predictor.export_model()
    predictor.export_metrics()
    predictor.export_training_data()
    
    # =========================================================================
    # VISUALIZATION: ML RESULTS
    # =========================================================================
    print("\n[VISUALIZATION] Creating ML model plots...")
    
    # Get predictions for visualization
    y_true, y_pred, component_types = predictor.get_predictions_for_visualization()
    component_names = predictor.component_names
    
    # Create predicted vs actual plot
    plot_path = viz_phase8_ml_results.create_predicted_vs_actual_plot(
        y_true=y_true,
        y_pred=y_pred,
        component_types=component_types,
        output_dir=str(outputs_path),
        pcb_name=pcb_filter,
        component_names=component_names,
        metrics=metrics
    )
    
    # Create regression feature plot (shows actual model relationship)
    flir_delta_t = predictor.df_features['delta_t_flir_air'].values
    regression_feature_path = viz_phase8_ml_results.create_regression_feature_plot(
        flir_delta_t=flir_delta_t,
        y_true=y_true,
        y_pred=y_pred,
        component_types=component_types,
        output_dir=str(outputs_path),
        pcb_name=pcb_filter
    )
    
    # Create residual plot (optional diagnostic)
    residual_path = viz_phase8_ml_results.create_residual_plot(
        y_true=y_true,
        y_pred=y_pred,
        component_types=component_types,
        output_dir=str(outputs_path),
        pcb_name=pcb_filter
    )
    
    # Create component type comparison (optional diagnostic)
    type_comparison_path = viz_phase8_ml_results.create_component_type_comparison(
        y_true=y_true,
        y_pred=y_pred,
        component_types=component_types,
        output_dir=str(outputs_path),
        pcb_name=pcb_filter
    )
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    results = {
        'pcb_name': pcb_filter,
        'n_components': len(y_true),
        'metrics': metrics,
        'output_dir': str(outputs_path),
        'output_files': {
            'model': str(outputs_path / f"{pcb_filter}_thermal_ml_model.pkl"),
            'metrics': str(outputs_path / f"{pcb_filter}_ml_metrics.csv"),
            'training_data': str(outputs_path / f"{pcb_filter}_ml_training_data.csv"),
            'plot_predicted_vs_actual': plot_path,
            'plot_residuals': residual_path,
            'plot_type_comparison': type_comparison_path
        }
    }
    
    print("\n" + "="*80)
    print(" PHASE 8 COMPLETE")
    print("="*80)
    print(f"Training components: {len(y_true)}")
    print(f"Model R² Score: {metrics['R²']:.4f}")
    print(f"Model RMSE: {metrics['RMSE']:.2f} °C")
    print(f"Model MAE: {metrics['MAE']:.2f} °C")
    print(f"\nOutput files saved to: {outputs_path}")
    print("="*80)
    
    return results


def process_ml_validation(trained_model_path: str,
                         calibration_db_path: str,
                         test_pcb: str,
                         train_pcb: str,
                         outputs_dir: str,
                         debug: bool = False) -> Dict:
    """
    Phase 8b: Cross-board ML model validation.
    
    Uses model trained on one board to predict temperatures on another board
    and compares to actual measurements for validation.
    
    Args:
        trained_model_path: Path to trained .pkl model file
        calibration_db_path: Path to thermal_calibration_points.csv
        test_pcb: PCB name to validate on (e.g., "HBridge")
        train_pcb: PCB name used for training (for plot labels)
        outputs_dir: Base output directory
        debug: Enable verbose debug output
    
    Returns:
        Dictionary with validation results and output paths
    """
    from phase8_ml_training import validate_cross_board
    
    print("\n" + "="*80)
    print(" PHASE 8b: ML MODEL VALIDATION (Cross-Board)")
    print("="*80)
    
    # Validate inputs
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"Trained model not found: {trained_model_path}")
    
    if not os.path.exists(calibration_db_path):
        raise FileNotFoundError(f"Calibration database not found: {calibration_db_path}")
    
    # outputs_dir already points to ml_model directory, use it directly
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Run cross-board validation
    validation_results = validate_cross_board(
        trained_model_path=trained_model_path,
        calibration_db_path=calibration_db_path,
        test_pcb=test_pcb,
        output_dir=outputs_dir,
        verbose=True
    )
    
    # Generate validation plots
    if ML_AVAILABLE:
        from viz_phase8_ml_results import (
            create_cross_board_validation_plot,
            create_residual_plot,
            create_validation_percent_error_plots
        )
        
        print(f"\n[VISUALIZATION] Creating validation plots...")
        
        # Predicted vs Actual plot
        create_cross_board_validation_plot(
            y_true=validation_results['y_true'],
            y_pred=validation_results['y_pred'],
            component_types=validation_results['component_types'],
            train_pcb=train_pcb,
            test_pcb=test_pcb,
            metrics=validation_results['metrics'],
            output_dir=outputs_dir
        )
        
        # Residual plot
        create_residual_plot(
            y_true=validation_results['y_true'],
            y_pred=validation_results['y_pred'],
            component_types=validation_results['component_types'],
            output_dir=outputs_dir,
            pcb_name=f"{test_pcb}_validation"
        )
        
        # Percent error analysis plots (4 plots)
        error_plot_paths = create_validation_percent_error_plots(
            y_true=validation_results['y_true'],
            y_pred=validation_results['y_pred'],
            component_names=validation_results['component_names'],
            component_types=validation_results['component_types'],
            test_pcb=test_pcb,
            output_dir=outputs_dir
        )
    
    # Print summary with percent error statistics
    error_summary = validation_results.get('error_summary', {})
    
    print(f"\n{'='*80}")
    print(f" PHASE 8b VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Test R² Score: {validation_results['metrics']['R²']:.4f}")
    print(f"Test RMSE: {validation_results['metrics']['RMSE']:.2f} °C")
    print(f"Test MAE: {validation_results['metrics']['MAE']:.2f} °C")
    
    if error_summary:
        print(f"\nPERCENT ERROR ANALYSIS:")
        print(f"  Mean Error:        {error_summary['Mean_Percent_Error']:+.1f}%")
        print(f"  Median Error:      {error_summary['Median_Percent_Error']:+.1f}%")
        print(f"  MAPE:              {error_summary['MAPE']:.1f}%")
        print(f"  Within ±10%:       {error_summary['Components_Within_10pct']} ({error_summary['Percent_Within_10pct']})")
        print(f"  Within ±20%:       {error_summary['Components_Within_20pct']} ({error_summary['Percent_Within_20pct']})")
        print(f"  Worst prediction:  {error_summary['Max_Error_Component']} ({error_summary['Max_Error_Value']})")
    
    print(f"\nOutput files saved to: {outputs_dir}")
    print("="*80)
    
    return validation_results


def main():
    """Main entry point with argument parsing and configuration UI"""
    
    parser = argparse.ArgumentParser(
        description='Post-process ResearchIR thermal data exports'
    )
    
    parser.add_argument('--input', type=str, default='inputs/ResearchIR_Outputs_HBridge_15s',
                       help='Input folder with ResearchIR CSV/TXT exports')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: auto-generate with timestamp)')
    parser.add_argument('--coordinates', type=str, default='inputs/hbridge_pcb_components_enhanced.csv',
                       help='CSV file with component X,Y coordinates (mm)')
    parser.add_argument('--proximity_threshold', type=float, default=10.0,
                       help='Distance threshold in mm for thermal coupling (default: 10.0)')
    parser.add_argument('--filter', type=str, default='median',
                       choices=['none', 'median', 'savgol', 'lowpass', 'outlier', 'hybrid'],
                       help='Filtering method (default: median)')
    parser.add_argument('--no_spatial', action='store_true',
                       help='Disable spatial thermal coupling analysis')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--no_ui', action='store_true',
                       help='Skip interactive configuration UI')
    
    # Workflow selection arguments
    parser.add_argument('--thermal_modeling', action='store_true',
                       help='Run thermal modeling workflow only (Phases 6 & 7)')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='Run complete pipeline (Phases 1-7)')
    
    # Thermal modeling configuration
    parser.add_argument('--calibration_pcb', type=str, default='Load_Shedding',
                       help='PCB name for calibration (has thermistors)')
    parser.add_argument('--prediction_pcb', type=str, default='HBridge_15s',
                       help='PCB name for prediction (FLIR only)')
    parser.add_argument('--mapping_file', type=str, default='loadshedding_thermistor_mapping.json',
                       help='Thermistor mapping JSON file')
    
    # Incremental calibration arguments
    parser.add_argument('--append_calibration', action='store_true',
                       help='Append to existing calibration database instead of replacing')
    parser.add_argument('--test_session', type=str, default=None,
                       help='Test session name/ID for tracking (e.g., LoadShedding_Week1)')
    parser.add_argument('--multi_session_config', type=str, default='All_Boards_multi_test_config.json',
                       help='JSON config file with multiple test sessions to process')
    parser.add_argument('--convergence_analysis', action='store_true',
                       help='Run convergence analysis after calibration')
    
    # Validation arguments
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration without processing (dry-run mode)')
    parser.add_argument('--verbose_validation', action='store_true',
                       help='Show detailed validation output with file counts and channel lists')
    
    # Machine learning arguments (Phase 8)
    parser.add_argument('--ml_training', action='store_true',
                       help='Run machine learning training workflow only (Phase 8)')
    parser.add_argument('--ml_pcb', type=str, default='Load_Shedding',
                       help='PCB name to use for ML training (default: Load_Shedding)')
    parser.add_argument('--ml_validate_pcb', type=str,
                       help='PCB name to validate trained model against (e.g., HBridge)')
    
    args = parser.parse_args()
    
    # Handle --validate flag (validation mode)
    if args.validate:
        if not THERMAL_MODELING_AVAILABLE:
            print("Error: Thermal modeling validation requires Phase 6 & 7 modules!")
            sys.exit(1)
        
        # Validate thermal config
        validation_passed = validate_thermal_config(
            multi_session_config=args.multi_session_config,
            verbose=args.verbose_validation
        )
        
        # Exit with appropriate status code
        sys.exit(0 if validation_passed else 1)
    
    # Handle legacy --thermal_modeling flag (Phases 6-7 only, no UI)
    if args.thermal_modeling:
        if not THERMAL_MODELING_AVAILABLE:
            print("Error: Thermal modeling phases (6 & 7) not available!")
            sys.exit(1)
        
        try:
            results = process_thermal_modeling(
                calibration_pcb=args.calibration_pcb,
                prediction_pcb=args.prediction_pcb,
                calibration_mapping=args.mapping_file,
                inputs_dir="inputs",
                outputs_dir=args.output,
                debug=args.debug,
                append_calibration=args.append_calibration,
                test_session=args.test_session,
                multi_session_config=args.multi_session_config,
                convergence_analysis=args.convergence_analysis
            )
            print("\nPhases 6-7 completed successfully!")
            return
        except Exception as e:
            print(f"\nError during thermal modeling: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # Handle --ml_training flag (Phase 8 only, no UI)
    if args.ml_training:
        if not ML_AVAILABLE:
            print("Error: Machine learning phase (8) not available!")
            sys.exit(1)
        
        # Determine output directory
        if args.output is None:
            output_dir = generate_output_folder('phase_8', debug=args.debug)
        else:
            output_dir = args.output
        
        # Find calibration points file from most recent Phase 6 output
        # Check if multi_session_config is used
        if args.multi_session_config:
            # Look for calibration_database in outputs
            cal_points_file = Path("outputs") / "calibration_database" / "thermal_calibration_points.csv"
        else:
            # Search for most recent calibration points file
            pattern = str(Path("outputs") / "*" / "calibration_database" / "thermal_calibration_points.csv")
            matches = glob.glob(pattern)
            
            if matches:
                # Sort by modification time, get most recent
                matches.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                cal_points_file = Path(matches[0])
                print(f"\nUsing most recent calibration database: {cal_points_file.parent.parent.name}")
            else:
                cal_points_file = Path("outputs") / "calibration_database" / "thermal_calibration_points.csv"
        
        if not cal_points_file.exists():
            print(f"Error: Calibration points file not found: {cal_points_file}")
            print("\nPlease run Phase 6 (thermal calibration) first:")
            print("  python researchir_post_processor.py --thermal_modeling")
            sys.exit(1)
        
        try:
            results = process_ml_training(
                calibration_points_file=str(cal_points_file),
                pcb_filter=args.ml_pcb,
                outputs_dir=output_dir,
                debug=args.debug
            )
            
            # Handle ML validation if specified
            if args.ml_validate_pcb:
                print("\n>>> EXECUTING PHASE 8b (ML Validation)")
                
                # Find trained model
                model_files = glob.glob(os.path.join(output_dir, '*_thermal_ml_model.pkl'))
                if not model_files:
                    print("ERROR: No trained model found for validation")
                else:
                    trained_model = model_files[0]
                    
                    validation_results = process_ml_validation(
                        trained_model_path=trained_model,
                        calibration_db_path=str(cal_points_file),
                        test_pcb=args.ml_validate_pcb,
                        train_pcb=args.ml_pcb,
                        outputs_dir=output_dir,
                        debug=args.debug
                    )
            
            print("\nPhase 8 completed successfully!")
            return
        except Exception as e:
            print(f"\nError during ML training: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # Handle --full_pipeline flag (Phases 1-7, no UI)
    if args.full_pipeline:
        if not THERMAL_MODELING_AVAILABLE:
            print("Error: Thermal modeling phases (6 & 7) not available!")
            sys.exit(1)
        
        # If using multi_session_config, skip the simple path and use main workflow
        if not args.multi_session_config:
            try:
                print("\n" + "="*80)
                print(" RUNNING COMPLETE PIPELINE (Phases 1-7)")
                print("="*80)
                
                # Phases 1-5
                print("\n>>> EXECUTING PHASES 1-5 (Standard Analysis)")
                process_researchir_data(
                    input_folder=args.input,
                    output_dir=args.output,
                    coordinates_file=args.coordinates,
                    proximity_threshold=args.proximity_threshold,
                    filter_type=args.filter,
                    spatial_enabled=not args.no_spatial,
                    debug=args.debug
                )
                
                # Phases 6-7
                print("\n>>> EXECUTING PHASES 6-7 (Thermal Modeling)")
                process_thermal_modeling(
                    calibration_pcb=args.calibration_pcb,
                    prediction_pcb=args.prediction_pcb,
                    calibration_mapping=args.mapping_file,
                    inputs_dir="inputs",
                    outputs_dir=args.output,
                    debug=args.debug
                )
                
                print("\n" + "="*80)
                print(" COMPLETE PIPELINE FINISHED (Phases 1-7)")
                print("="*80)
                print("\nAll phases completed successfully!")
                return
            except Exception as e:
                print(f"\nError during pipeline execution: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
    
    # Determine workflow for auto-generated output folder
    if args.thermal_modeling:
        workflow = 'phases_6_7'
    elif args.full_pipeline:
        workflow = 'full_pipeline'
    else:
        workflow = 'full_pipeline'
    
    # Auto-generate output directory if not specified
    if args.output is None:
        output_dir = generate_output_folder(workflow, debug=args.debug)
    else:
        output_dir = args.output
    
    # Configuration dictionary
    config = {
        'workflow': workflow,
        'input_folder': args.input,
        'output_dir': output_dir,
        'coordinates_file': args.coordinates,
        'proximity_threshold': args.proximity_threshold,
        'filter_type': args.filter,
        'spatial_enabled': not args.no_spatial,
        'calibration_pcb': args.calibration_pcb,
        'prediction_pcb': args.prediction_pcb,
        'mapping_file': args.mapping_file,
        'multi_session_config': args.multi_session_config,
        'append_calibration': args.append_calibration,
        'test_session': args.test_session,
        'convergence_analysis': args.convergence_analysis,
        'debug': args.debug,
        'auto_output': args.output is None  # Track if output was auto-generated
    }
    
    # Show interactive UI unless disabled
    if not args.no_ui:
        config = config_ui.display_configuration_ui(config)
        if config is None:
            print("Processing cancelled by user.")
            return
    
    # Execute selected workflow
    try:
        workflow = config.get('workflow', 'phases_1_5')
        
        if workflow == 'phases_1_5':
            # Run Phases 1-5 only
            results = process_researchir_data(
                input_folder=config['input_folder'],
                output_dir=config['output_dir'],
                coordinates_file=config['coordinates_file'],
                proximity_threshold=config['proximity_threshold'],
                filter_type=config['filter_type'],
                spatial_enabled=config['spatial_enabled'],
                debug=config['debug']
            )
            print("\nPhases 1-5 completed successfully!")
        
        elif workflow == 'phases_6_7':
            # Run Phases 6-7 only
            if not THERMAL_MODELING_AVAILABLE:
                print("Error: Thermal modeling phases (6 & 7) not available!")
                sys.exit(1)
            
            results = process_thermal_modeling(
                calibration_pcb=config['calibration_pcb'],
                prediction_pcb=config['prediction_pcb'],
                calibration_mapping=config['mapping_file'],
                inputs_dir="inputs",
                outputs_dir=config['output_dir'],
                debug=config['debug']
            )
            print("\nPhases 6-7 completed successfully!")
        
        elif workflow == 'full_pipeline':
            # Run complete pipeline: Phases 1-7
            if not THERMAL_MODELING_AVAILABLE:
                print("Error: Thermal modeling phases (6 & 7) not available!")
                sys.exit(1)
            
            print("\n" + "="*80)
            print(" RUNNING COMPLETE PIPELINE (Phases 1-7)")
            print("="*80)
            
            # Check if using multi-session config with multiple boards
            multi_session_config_path = config.get('multi_session_config')
            boards_to_process = []
            
            if multi_session_config_path:
                import json
                with open(multi_session_config_path, 'r') as f:
                    multi_config = json.load(f)
                
                if 'boards' in multi_config:
                    # Multi-board configuration - process each board through Phase 1-5
                    for board_config in multi_config['boards']:
                        board_name = board_config['name']
                        
                        # Build coordinates file path from board name
                        # Convention: inputs/{board_name}_components_enhanced.csv
                        coordinates_file = f"inputs/{board_name}_components_enhanced.csv"
                        
                        boards_to_process.append({
                            'name': board_name,
                            'input_folder': f"inputs/ResearchIR_Outputs_{board_name}",
                            'coordinates': coordinates_file
                        })
                else:
                    # Single board configuration
                    boards_to_process.append({
                        'name': config.get('prediction_pcb', 'Board'),
                        'input_folder': config['input_folder'],
                        'coordinates': config.get('coordinates_file')
                    })
            else:
                # No multi-session config, use single input folder
                boards_to_process.append({
                    'name': config.get('prediction_pcb', 'Board'),
                    'input_folder': config['input_folder'],
                    'coordinates': config['coordinates_file']
                })
            
            # Run Phases 1-5 for each board
            print(f"\n>>> EXECUTING PHASES 1-5 for {len(boards_to_process)} board(s)")
            for board_info in boards_to_process:
                print(f"\n{'='*80}")
                print(f" Processing: {board_info['name']}")
                print(f"{'='*80}")
                
                results_1_5 = process_researchir_data(
                    input_folder=board_info['input_folder'],
                    output_dir=config['output_dir'],
                    coordinates_file=board_info['coordinates'],
                    proximity_threshold=config['proximity_threshold'],
                    filter_type=config['filter_type'],
                    spatial_enabled=config['spatial_enabled'],
                    debug=config['debug']
                )
            
            # Then run Phases 6-7 (uses all boards' data)
            print("\n>>> EXECUTING PHASES 6-7 (Thermal Modeling)")
            results_6_7 = process_thermal_modeling(
                calibration_pcb=config['calibration_pcb'],
                prediction_pcb=config['prediction_pcb'],
                calibration_mapping=config['mapping_file'],
                inputs_dir="inputs",
                outputs_dir=config['output_dir'],
                multi_session_config=config.get('multi_session_config'),
                append_calibration=config.get('append_calibration', False),
                test_session=config.get('test_session'),
                convergence_analysis=config.get('convergence_analysis', False),
                debug=config['debug']
            )
            
            # Phase 8: ML Training (if ML modules available)
            if ML_AVAILABLE:
                print("\n>>> EXECUTING PHASE 8 (Machine Learning)")
                
                # Get calibration points file from Phase 6 results
                calibration_points_file = results_6_7.get('calibration_points_file')
                if not calibration_points_file:
                    calibration_points_file = str(Path(config['output_dir']) / "calibration_database" / "thermal_calibration_points.csv")
                
                # Check if file exists
                if Path(calibration_points_file).exists():
                    ml_output_dir = str(Path(config['output_dir']) / 'ml_model')
                    
                    results_8 = process_ml_training(
                        calibration_points_file=calibration_points_file,
                        pcb_filter=config.get('calibration_pcb', 'Load_Shedding'),
                        outputs_dir=ml_output_dir,
                        debug=config['debug']
                    )
                    
                    # Phase 8b: Cross-board validation (auto-validate on HBridge)
                    print("\n>>> EXECUTING PHASE 8b (ML Cross-Board Validation)")
                    
                    # Find trained model
                    model_files = glob.glob(os.path.join(ml_output_dir, '*_thermal_ml_model.pkl'))
                    if model_files:
                        trained_model = model_files[0]
                        
                        try:
                            validation_results = process_ml_validation(
                                trained_model_path=trained_model,
                                calibration_db_path=calibration_points_file,
                                test_pcb='HBridge',
                                train_pcb='Load_Shedding',
                                outputs_dir=ml_output_dir,
                                debug=config['debug']
                            )
                            
                            print("\n" + "="*80)
                            print(" PHASE 8b COMPLETE (Cross-Board Validation)")
                            print("="*80)
                            print(f"Test board: HBridge")
                            print(f"Validation R²: {validation_results['metrics']['R²']:.4f}")
                            print(f"Validation RMSE: {validation_results['metrics']['RMSE']:.2f} °C")
                            print(f"Validation MAE: {validation_results['metrics']['MAE']:.2f} °C")
                            print("="*80)
                        except Exception as e:
                            print(f"Warning: Cross-board validation failed: {e}")
                            if config['debug']:
                                import traceback
                                traceback.print_exc()
                    else:
                        print("Warning: No trained model found for validation")
                else:
                    print(f"  Warning: Calibration points file not found, skipping Phase 8")
                    print(f"  Expected: {calibration_points_file}")
            else:
                print("\n>>> SKIPPING PHASE 8 (Machine Learning modules not available)")
            
            print("\n" + "="*80)
            print(" COMPLETE PIPELINE FINISHED (Phases 1-8)")
            print("="*80)
            print("\nAll phases completed successfully!")
        
        else:
            print(f"Error: Unknown workflow '{workflow}'")
            sys.exit(1)
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        if config['debug']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
