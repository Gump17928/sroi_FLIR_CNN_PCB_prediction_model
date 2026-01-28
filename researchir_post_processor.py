"""
ResearchIR Thermal Data Post-Processor
Main orchestrator for thermal analysis pipeline (Phases 1-8)

Phases:
  1-2: Data loading & filtering
  3-5: Component analysis, spatial coupling, potting risk
  6-7: Thermal calibration & prediction (requires thermistors)
  8:   Machine learning (U-Net CNN spatial prediction)

Key Features:
  - Incremental calibration across test sessions
  - Multi-session batch processing
  - FLIR frame filtering for ML training
  - Cross-PCB validation support

Usage: python researchir_post_processor.py [--thermal_modeling | --ml_training | --full_pipeline]
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
    import phase8_ml_training  # U-Net CNN wrapper
    import viz_phase8_ml_results
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"Warning: Machine learning phase (8) not available: {e}")


def filter_all_flir_frames():
    """
    Filter all ResearchIR FLIR frames using temporal median (kernel=5).
    Creates *_filtered directories for ML training. ~4 min/board, ~350MB RAM.
    """
    inputs_path = Path("inputs")
    
    print("\n" + "="*80)
    print("FLIR FRAME FILTERING FOR ML TRAINING")
    print("="*80)
    
    # Find folders to filter (exclude already filtered)
    all_folders = list(inputs_path.glob("ResearchIR_Outputs_*"))
    folders_to_filter = [f for f in all_folders if not f.name.endswith("_filtered")]
    
    if not folders_to_filter:
        print("No ResearchIR folders found. Looking for: inputs/ResearchIR_Outputs_*/")
        return 0
    
    print(f"Found {len(folders_to_filter)} board(s):", ", ".join(f.name for f in folders_to_filter))
    
    # Check for existing filtered folders
    existing_filtered = [(f, f.parent / f"{f.name}_filtered") 
                        for f in folders_to_filter 
                        if (f.parent / f"{f.name}_filtered").exists()]
    
    if existing_filtered:
        print(f"\n⚠ Found {len(existing_filtered)} existing filtered folder(s)")
        response = input("Re-filter? [y/N]: ").strip().lower()
        if response != 'y':
            print("Using existing filtered folders")
            return 0
        
        import shutil
        for _, filt in existing_filtered:
            shutil.rmtree(filt)
    
    # Filter each board
    print("\nFiltering frames (temporal median, kernel=5)...")
    boards_filtered = 0
    
    for i, input_folder in enumerate(folders_to_filter, 1):
        output_folder = input_folder.parent / f"{input_folder.name}_filtered"
        print(f"[{i}/{len(folders_to_filter)}] {input_folder.name} → {output_folder.name}")
        
        try:
            phase2.filter_flir_frames_for_ml(
                input_folder=str(input_folder),
                output_folder=str(output_folder),
                kernel_size=5
            )
            boards_filtered += 1
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*80}\nFiltered {boards_filtered}/{len(folders_to_filter)} boards successfully")
    print("Filtered frames ready for ML training\n" + "="*80)
    return boards_filtered


def _export_thermistor_timeseries_csv(thermistor_data: Dict, output_file: Path, medium: str = "sand"):
    """Export thermistor time series to CSV with Time(s) + component columns for ML training."""
    if not thermistor_data:
        print(f"    No {medium} thermistor data to export")
        return
    
    try:
        # Extract component names and find longest time series as reference
        component_test_pairs = {
            key: (df['Component'].iloc[0] if 'Component' in df.columns else key.split('_')[0],
                  df['Test'].iloc[0] if 'Test' in df.columns else key.split('_')[-1])
            for key, df in thermistor_data.items()
        }
        
        reference_time = max((df['Time'].values for df in thermistor_data.values()), key=len)
        merged_df = pd.DataFrame({'Time (s)': reference_time})
        
        # Interpolate each component to common time base
        for key, df in thermistor_data.items():
            component, _ = component_test_pairs[key]
            
            if len(df['Time']) != len(reference_time) or not np.allclose(df['Time'].values, reference_time):
                temp_series = pd.Series(index=df['Time'].values, data=df['Temperature'].values)
                aligned_series = temp_series.reindex(reference_time)
                aligned_series = aligned_series.interpolate(method='linear', limit_area='inside')
                aligned_series = aligned_series.fillna(method='ffill').fillna(method='bfill')
                temps = aligned_series.values
            else:
                temps = df['Temperature'].values
            
            # Handle duplicate component names
            col_name = component
            counter = 1
            while col_name in merged_df.columns:
                col_name = f"{component}_{counter}"
                counter += 1
            merged_df[col_name] = temps
        
        merged_df.to_csv(output_file, index=False)
        print(f"    ✓ Exported {len(merged_df.columns)-1} components, "
              f"{len(merged_df)} samples ({merged_df['Time (s)'].min():.1f}-{merged_df['Time (s)'].max():.1f}s)")
        
    except Exception as e:
        print(f"    ✗ Error exporting {medium} thermistor data: {e}")


def _create_three_condition_plots_from_tests(cal_flir_file, test_configs, calibration_pcb, outputs_path, plot_settings=None):
    """Create 3-condition plots combining data from multiple tests."""
    from loader_thermistor import ThermalDataLoader
    import viz_phase2_filtering, pandas as pd, phase1_data_loading as phase1, phase2_filtering as phase2, viz_phase3_statistics
    
    # Load FLIR data
    flir_df = pd.read_csv(cal_flir_file)
    time_col = 'reltime' if 'reltime' in flir_df.columns else 'time_s'
    flir_data = {comp: pd.DataFrame({'Time': flir_df[time_col].values, 'Temperature': flir_df[comp].values})
                 for comp in flir_df.columns if comp not in ['frame', 'reltime', 'time_s', 'Image']}
    
    # Helper to load thermistor data
    def load_therm_data(test_configs, medium_key, pairs_key):
        data, loader = {}, ThermalDataLoader()
        for cfg in test_configs:
            if not (therm_file := cfg.get(medium_key)): continue
            pairs = cfg.get(pairs_key, cfg.get('pairs', []))
            test_id = cfg.get('test_id', 'unknown')
            
            try:
                x, y, labels, _ = (loader.load_temp_csv(therm_file) if not isinstance(therm_file, list) 
                                   else loader.load_multi_device_csv(therm_file))
            except Exception as e:
                continue
            
            for pair in pairs:
                flir_comp = pair['flir_roi'] if isinstance(pair, dict) else pair[0]
                therm_chan = pair['therm_channel'] if isinstance(pair, dict) else pair[1]
                idx = next((i for i, lbl in enumerate(labels) if therm_chan in lbl), None)
                if idx is not None:
                    data[f"{flir_comp}_{test_id}"] = pd.DataFrame({
                        'Time': x, 'Temperature': y[:, idx], 'Component': flir_comp, 'Test': test_id})
        return data
    
    air_data = load_therm_data(test_configs, 'therm_air_file', 'air_pairs')
    sand_data = load_therm_data(test_configs, 'therm_sand_file', 'sand_pairs')
    
    # Export thermistor CSVs for ML
    if sand_data:
        _export_thermistor_timeseries_csv(sand_data, outputs_path / f"{calibration_pcb}_thermistor_timeseries.csv", "sand")
    if air_data:
        _export_thermistor_timeseries_csv(air_data, outputs_path / f"{calibration_pcb}_air_thermistor_timeseries.csv", "air")
    
    # Find overlapping components
    get_comps = lambda d: {(v['Component'].iloc[0] if 'Component' in v.columns else k.split('_')[0]) for k,v in d.items()}
    overlapping = list(set(flir_data.keys()) & get_comps(air_data) & get_comps(sand_data))
    
    if not overlapping:
        print("  No overlapping components found")
        return
    
    print(f"  Found {len(overlapping)} overlapping components, {len(air_data)} air, {len(sand_data)} sand measurements")
    
    # Generate plots
    output_dir = outputs_path / "three_condition_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files, valid_entries = viz_phase2_filtering.create_three_condition_comparison(
        flir_data, air_data, sand_data, overlapping, str(output_dir), calibration_pcb, plot_settings)
    
    if valid_entries:
        detail_files = viz_phase2_filtering.create_component_detail_plots(
            flir_data, air_data, sand_data, valid_entries, str(output_dir), calibration_pcb, plot_settings)
    
    # Generate thermal metrics
    grouped_components = phase1.classify_components(flir_data, None, False)
    filtered_flir_data = phase2.apply_filtering_to_component_data(flir_data, 'median', False)
    analysis_results = phase2.analyze_temperature_transients(flir_data, 'median', False)
    viz_phase3_statistics.create_ieee_plots(filtered_flir_data, grouped_components, analysis_results, str(output_dir), calibration_pcb, False)
    print(f"  Created plots and metrics in {output_dir}")


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
    """Generate timestamped output folder (format: MMDD_HHMM_<workflow>[_debug])."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    workflow_abbrev = {'phases_1_5': 'P1-5', 'phases_6_7': 'P6-7', 'full_pipeline': 'P1-7'}
    abbrev = workflow_abbrev.get(workflow, 'P1-7')
    folder_name = f"{timestamp}_{abbrev}{'_debug' if debug else ''}"
    return str(Path(base_dir) / folder_name)

# Component type definitions for consistent use across phases
COMPONENT_TYPES = phase1.DEFAULT_COMPONENT_TYPES


def process_researchir_data(input_folder: str, output_dir: str,
                           coordinates_file: Optional[str] = None,
                           proximity_threshold: float = 10.0,
                           filter_type: str = 'median',
                           spatial_enabled: bool = True,
                           debug: bool = False) -> Dict:
    """Complete ResearchIR thermal processing (Phases 1-5: load, filter, analyze, spatial coupling, potting risk)."""
    os.makedirs(output_dir, exist_ok=True)
    spatial_status = 'Enabled' if spatial_enabled and coordinates_file else 'Disabled'
    print(f"{'='*60}\n RESEARCHIR THERMAL DATA POST-PROCESSING (Phases 1-5)\n{'='*60}")
    print(f"Input: {input_folder}  |  Output: {output_dir}  |  Filter: {filter_type}  |  Spatial: {spatial_status}\n{'='*60}")
    
    # Extract PCB name from input folder
    pcb_name = None
    if input_folder:
        folder_name = os.path.basename(input_folder.rstrip('/\\'))
        pcb_name = folder_name.split('ResearchIR_Outputs_')[-1] if 'ResearchIR_Outputs_' in folder_name else folder_name
    
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
    
    print("\n[PHASE 2] Exporting filtered temperatures and filter comparison...")
    phase2.export_filtered_temperatures(filtered_component_data, output_dir=output_dir, board_name=pcb_name)
    phase2.export_filter_comparison(component_data, filter_type=filter_type, output_dir=output_dir, board_name=pcb_name, max_components=5)
    
    # =========================================================================
    # PHASE 3: COMPONENT ANALYSIS
    # =========================================================================
    print("\n[PHASE 3] Exporting MATLAB data, summary CSV, and component statistics...")
    mat_file = phase3.export_matlab_data(filtered_component_data, grouped_components, analysis_results, output_dir)
    csv_file = phase3.export_summary_csv(grouped_components, analysis_results, output_dir, COMPONENT_TYPES)
    phase3.export_component_statistics(filtered_component_data, output_dir=output_dir, board_name=pcb_name)
    
    # =========================================================================
    # VISUALIZATION & MATLAB EXPORTS
    # =========================================================================
    print("\n[VISUALIZATION] Creating IEEE plots, filter comparisons, and MATLAB script...")
    plot_files = viz_phase3_statistics.create_ieee_plots(filtered_component_data, grouped_components,
                                        analysis_results, output_dir, pcb_name=pcb_name, debug=debug)
    comparison_files = []
    if filter_type != 'none':
        comparison_files = viz_phase2_filtering.create_filtering_comparison_plots(
            component_data, filtered_component_data, grouped_components, output_dir, filter_type=filter_type, debug=debug)
        multifilter_grid, comparison_df = viz_phase2_filtering.create_multifilter_component_grid(
            component_data, grouped_components, output_dir, filter_params={}, debug=debug)
    phase3.create_matlab_usage_script(output_dir, grouped_components, COMPONENT_TYPES)
    
    # =========================================================================
    # PHASE 4: SPATIAL COUPLING (if coordinates provided)
    # =========================================================================
    coupling_metrics = {}
    component_coordinates = {}
    proximity_matrix = {}
    coupling_plots = []
    
    if spatial_enabled and coordinates_file:
        print(f"\n[PHASE 4] Spatial coupling analysis (threshold: {proximity_threshold}mm)...")
        component_coordinates = phase4.load_component_coordinates(coordinates_file, debug=debug)
        
        if component_coordinates:
            proximity_matrix = phase4.build_proximity_matrix(component_data, component_coordinates, proximity_threshold)
            
            if proximity_matrix:
                coupling_metrics = phase4.calculate_thermal_coupling_metrics(filtered_component_data, proximity_matrix, analysis_results)
                print(f"  Analyzed {len(coupling_metrics)} components, exporting matrix and visualizations...")
                phase4.export_coupling_matrix(proximity_matrix, filtered_component_data, output_dir=output_dir, board_name=pcb_name)
                coupling_plots = viz_phase4_coupling.create_thermal_coupling_visualizations(
                    filtered_component_data, coupling_metrics, analysis_results, output_dir, component_coordinates, debug=debug)
    
    # =========================================================================
    # PHASE 5: POTTING RISK (requires coupling metrics from Phase 4)
    # =========================================================================
    risk_csv = None
    risk_df = None
    
    if coupling_metrics:
        print("\n[PHASE 5] Analyzing potted condition failure risk...")
        risk_csv, risk_df = phase5.analyze_potted_condition_risk(
            coupling_metrics, analysis_results, output_dir)
    
    # Results summary
    coupling_info = f", {len(coupling_metrics)} spatial couplings" if coupling_metrics else ""
    print(f"\n{'='*60}\n PHASES 1-5 COMPLETE: {len(component_data)} components, {len(grouped_components)} groups{coupling_info}\n Output: {output_dir}\n{'='*60}")
    
    return {
        'input_folder': input_folder, 'output_dir': output_dir, 'total_components': len(component_data),
        'component_groups': grouped_components, 'analysis_results': analysis_results,
        'coupling_metrics': coupling_metrics, 'risk_analysis_df': risk_df,
        'output_files': {'plots': plot_files, 'comparison_plots': comparison_files, 'coupling_plots': coupling_plots,
                        'matlab': mat_file, 'summary_csv': csv_file, 'risk_csv': risk_csv}
    }


def validate_thermal_config(multi_session_config: str = None, verbose: bool = False) -> bool:
    """Validate thermal modeling config (session-based or board-based) without processing."""
    from viz_phase6_validation import validate_multi_session_config, validate_multi_board_config
    import json
    
    print(f"{'='*80}\n THERMAL CALIBRATION CONFIGURATION VALIDATION\n{'='*80}")
    
    if not multi_session_config:
        print("\n✗ Error: No configuration file specified\n  Use --multi_session_config <config.json>")
        return False
    
    print(f"\nValidating: {multi_session_config}\n{'-'*80}")
    
    try:
        with open(multi_session_config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"\n✗ Error: Failed to load config file: {e}")
        return False
    
    is_board_based = 'boards' in config
    is_session_based = 'sessions' in config
    
    if is_board_based:
        print("  Config type: Board-based (hierarchical)")
        result = validate_multi_board_config(config_file=multi_session_config, verbose=verbose)
    elif is_session_based:
        print("  Config type: Session-based (legacy)")
        result = validate_multi_session_config(config_path=multi_session_config, check_files_exist=True, verbose=verbose)
    else:
        print("\n✗ Error: Config must have either 'boards' or 'sessions' field")
        return False
    
    # Print summary
    if not verbose:
        print(f"\n{'='*80}\n VALIDATION SUMMARY\n{'='*80}")
        
        if result['valid']:
            print("\n✓ ALL VALIDATION CHECKS PASSED")
            
            if is_board_based:
                print(f"\n  Boards: {result['stats']['board_count']}  |  Tests: {result['stats']['test_count']}  |  Components: {result['stats']['component_count']}")
            else:
                print(f"\n  Sessions: {result['stats']['total_sessions']}  |  Component pairs: {result['stats']['total_pairs']}\n\n  Session Breakdown:")
                for i, session_result in enumerate(result['session_results'], 1):
                    session_name = result['config']['sessions'][i-1]['name']
                    num_pairs = session_result['stats']['num_pairs']
                    device_str = "(multi-device)" if session_result['stats'].get('is_multi_device', False) else "(single-device)"
                    print(f"    {i}. {session_name}: {num_pairs} components {device_str}")
            
            print("\n  Ready to process! Run without --validate to execute workflow.")
            
        else:
            print(f"\n✗ VALIDATION FAILED - {len(result['errors'])} error(s)")
            if result['errors']:
                print("\n  Errors:")
                for err in result['errors'][:20]:
                    print(f"    • {err}")
                if len(result['errors']) > 20:
                    print(f"    ... and {len(result['errors']) - 20} more errors")
        
        if result['warnings']:
            print(f"\n  ⚠ Warnings ({len(result['warnings'])}):")
            for warn in result['warnings'][:20]:
                print(f"    • {warn}")
            if len(result['warnings']) > 20:
                print(f"    ... and {len(result['warnings']) - 20} more warnings")
        
        print(f"\n{'='*80}")
    
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
    Phase 6: Calibrate using PCB with both FLIR and thermistor data
    Phase 7: Predict temperatures using calibration on FLIR-only PCB
    
    Modes: Standard (single session), Incremental (append), Multi-Session (batch process from JSON config)
    Multi-device thermistor: Use Device{N}_{Channel} naming (Device0_AI0, Device1_AI4, etc.)
    See multi_session_config_example.json for config format.
    """
    inputs_path = Path(inputs_dir)
    
    if outputs_dir is None:
        timestamp = datetime.now().strftime('%m%d_%H%M')
        outputs_path = Path(f"outputs/{timestamp}_P6-7")
        print(f"\\nAuto-generated output directory: {outputs_path}")
    else:
        outputs_path = Path(outputs_dir)
    
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}\\n THERMAL MODELING WORKFLOW (Phases 6 & 7)\\n{'='*80}")
    print(f"Calibration: {calibration_pcb}  |  Prediction: {prediction_pcb}  |  Mapping: {calibration_mapping}\\n{'='*80}")
    
    # =========================================================================
    # PARSE RESEARCHIR DATA (both PCBs)
    # =========================================================================
    print("\n[DATA PARSING] Parsing ResearchIR Stats files...")
    
    # Parse calibration PCB
    cal_researchir_dir = inputs_path / f"ResearchIR_Outputs_{calibration_pcb}"
    if not cal_researchir_dir.exists():
        raise FileNotFoundError(f"Calibration ResearchIR directory not found: {cal_researchir_dir}")
    
    cal_parser = ResearchIRStatsParser(cal_researchir_dir)
    cal_flir_df = cal_parser.parse_all_frames()
    cal_output_dir = outputs_path / calibration_pcb
    cal_output_dir.mkdir(parents=True, exist_ok=True)
    cal_flir_file = cal_output_dir / f"{calibration_pcb}_FLIR_AllComponents.csv"
    cal_flir_df.to_csv(cal_flir_file, index=False)
    print(f"  {calibration_pcb}: {cal_flir_file}")
    
    # Parse prediction PCB
    pred_researchir_dir = inputs_path / f"ResearchIR_Outputs_{prediction_pcb}"
    if not pred_researchir_dir.exists():
        raise FileNotFoundError(f"Prediction ResearchIR directory not found: {pred_researchir_dir}")
    
    pred_parser = ResearchIRStatsParser(pred_researchir_dir)
    pred_flir_df = pred_parser.parse_all_frames()
    pred_output_dir = outputs_path / prediction_pcb
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    pred_flir_file = pred_output_dir / f"{prediction_pcb}_FLIR_AllComponents.csv"
    pred_flir_df.to_csv(pred_flir_file, index=False)
    print(f"  {prediction_pcb}: {pred_flir_file}")
    
    # =========================================================================
    # PHASE 6: THERMAL CALIBRATION
    # =========================================================================
    print(f"\n{'='*80}\n[PHASE 6] THERMAL CALIBRATION\n{'='*80}")
    
    # Find thermistor files
    therm_air_file = list(inputs_path.glob("*AIR*usb_temp*.csv"))
    therm_sand_file = list(inputs_path.glob("*SAND*usb_temp*.csv"))
    
    if not therm_air_file:
        raise FileNotFoundError("No AIR thermistor file found")
    
    therm_air_file = therm_air_file[0]
    therm_sand_file = therm_sand_file[0] if therm_sand_file else None
    
    therm_files_msg = f"Air: {therm_air_file.name}" + (f" | Sand: {therm_sand_file.name}" if therm_sand_file else "")
    print(f"\nThermistor files: {therm_files_msg}")
    
    # Load mapping and build pairs
    mapping_path = Path(calibration_mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        mapping_config = json.load(f)
    
    pairs = []
    for therm_channel, mapping_data in mapping_config["thermistor_to_flir_mapping"].items():
        if "ambient" in mapping_data.get("notes", "").lower():
            continue
        flir_comp = mapping_data["flir_component"]
        comp_type = mapping_data["component_type"]
        pairs.append((flir_comp, therm_channel, f"{flir_comp}_{comp_type}", comp_type))
    print(f"  Built {len(pairs)} measurement pairs")
    
    # Run Phase 6 calibration
    cal_db_dir = outputs_path / "calibration_database"
    calibrator = phase6.ThermalCalibrator(output_dir=cal_db_dir)
    
    if append_calibration:
        existing_file = cal_db_dir / "thermal_calibration_points.csv"
        calibrator.load_existing_calibration(existing_file)
    
    # Multi-Session Mode: Process sessions from config file
    if multi_session_config:
        print(f"\n[MULTI-SESSION MODE] Loading: {multi_session_config}")
        with open(multi_session_config, 'r') as f:
            multi_config = json.load(f)
        
        if 'boards' in multi_config:
            print("  Config format: Board-based (hierarchical)")
            calibrator.add_measurement_boards(multi_config['boards'])
            
            # Generate three-condition comparison plots for each board
            print("\n[VISUALIZATION] Creating three-condition comparison plots...")
            for board_config in multi_config['boards']:
                pcb_name = board_config['name']
                tests = board_config.get('tests', [])
                
                if not tests:
                    continue
                
                # Parse board FLIR data
                board_researchir_dir = inputs_path / f"ResearchIR_Outputs_{pcb_name}"
                if not board_researchir_dir.exists():
                    print(f"  WARNING: Skipping {pcb_name} - FLIR directory not found")
                    continue
                
                board_parser = ResearchIRStatsParser(board_researchir_dir)
                board_flir_df = board_parser.parse_all_frames()
                board_output_dir = outputs_path / pcb_name
                board_output_dir.mkdir(parents=True, exist_ok=True)
                board_flir_file = board_output_dir / f"{pcb_name}_FLIR_AllComponents.csv"
                board_flir_df.to_csv(board_flir_file, index=False)
                print(f"  {pcb_name}: Saved {board_flir_file}")
                
                plot_settings = board_config.get('plot_settings', {})
                _create_three_condition_plots_from_tests(board_flir_file, tests, pcb_name, outputs_path, plot_settings)
                
        elif 'sessions' in multi_config:
            print("  Config format: Session-based (legacy)")
            calibrator.add_measurement_sessions(multi_config['sessions'])
        else:
            raise ValueError("Config must contain either 'boards' or 'sessions' field")
    else:
        # Single session mode
        calibrator.add_measurement_pair(cal_flir_file, therm_air_file, therm_sand_file, pairs, calibration_pcb, test_session)
        print("\n[VISUALIZATION] Creating three-condition comparison plots...")
        _create_three_condition_plots(cal_flir_file, therm_air_file, therm_sand_file, pairs, calibration_pcb, outputs_path)
    
    calibrator.compute_component_type_calibrations()
    
    # Check for validation mode in config
    if multi_session_config:
        with open(multi_session_config, 'r') as f:
            if json.load(f).get('calibration_mode') == 'validation':
                print("\n[VALIDATION MODE] Running calibration quality assessment...")
                calibrator.run_validation_workflow()
    
    calibrator.export_calibration_database(prefix="thermal_calibration", run_convergence_analysis=convergence_analysis)
    calibration_points_file = cal_db_dir / 'thermal_calibration_points.csv'
    print(f"\nCalibration database: {cal_db_dir}")
    
    # =========================================================================
    # PHASE 7: THERMAL PREDICTION
    # =========================================================================
    print(f"\n{'='*80}\n[PHASE 7] THERMAL PREDICTION\n{'='*80}")
    
    calibration_file = cal_db_dir / "thermal_calibration_by_type.csv"
    
    # Load component patterns
    patterns_file = Path(__file__).parent / "component_type_patterns.json"
    if patterns_file.exists():
        with open(patterns_file, 'r') as f:
            component_patterns = json.load(f)["patterns"]
        print(f"Using component patterns: {patterns_file.name}")
    else:
        component_patterns = {"PowerSupply": [r"^PS\d+$", r"^CONV$"], "IC": [r"^U\d+$", r"^IC\d+$"],
                              "Resistor": [r"^R\d+$"], "LED": [r"^DL\d+$", r"^CR\d+$"]}
        print("Using default component patterns")
    
    print(f"\nPredicting for {prediction_pcb}...")
    predictor = phase7.ThermalPredictor(calibration_file=calibration_file, output_dir=pred_output_dir, verbose=debug)
    df_predictions = predictor.predict_from_flir(pred_flir_file, prediction_pcb, use_median_offset=True)
    predictor.export_predictions(df_predictions, prefix=f"thermal_prediction_{prediction_pcb}")
    print(f"Predictions exported to: {pred_output_dir}")
    
    # Results summary
    print(f"\n{'='*80}\n PHASES 6-7 COMPLETE\n{'='*80}")
    
    return {
        'calibration_pcb': calibration_pcb, 'prediction_pcb': prediction_pcb,
        'calibration_database': str(cal_db_dir), 'calibration_points_file': str(calibration_points_file),
        'prediction_output_dir': str(pred_output_dir),
        'output_files': {'calibration_points': str(cal_db_dir / 'thermal_calibration_points.csv'),
                        'calibration_by_type': str(cal_db_dir / 'thermal_calibration_by_type.csv')}
    }


def process_ml_training(calibration_points_file: str,
                       pcb_filter: str = "Load_Shedding",
                       outputs_dir: str = "outputs",
                       debug: bool = False) -> Dict:
    """
    Execute machine learning workflow (Phase 8).
    
    Train OLS linear regression model to predict sand embedded temperatures
    from FLIR air measurements using component type as additional feature.
    
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
    print("\nNOTE: Linear regression model temporarily disabled")
    print("Will be replaced with CNN-based thermal prediction wrapper")
    print("See ml_model/cnn_thermal_modeling/ for new U-Net approach")
    print("="*80)
    
    # Placeholder return to maintain API compatibility
    return {
        'pcb_name': pcb_filter,
        'output_dir': str(outputs_path),
        'status': 'disabled - use CNN model instead'
    }
    
    # =========================================================================
    # LEGACY CODE - DISABLED
    # =========================================================================
    # The code below is commented out pending replacement with CNN approach
    
    # print("\n" + "="*80)
    # print("[PHASE 8] MACHINE LEARNING TRAINING")
    # print("="*80)
    # 
    # # Initialize ML predictor (outputs_dir already includes ml_model if needed)
    # predictor = phase8.ThermalMLPredictor(output_dir=str(outputs_path), verbose=True)
    # 
    # # Load training data
    # predictor.load_training_data(
    #     calibration_csv=calibration_points_file,
    #     pcb_filter=pcb_filter
    # )
    # 
    # # Calculate delta T for each component
    # predictor.calculate_delta_t()
    # 
    # # Prepare feature matrix with one-hot encoded component types
    # predictor.prepare_feature_matrix()
    # 
    # # Train OLS model
    # predictor.train_model()
    # 
    # # Calculate performance metrics
    # metrics = predictor.calculate_metrics()
    


def process_ml_validation(trained_model_path: str,
                         calibration_db_path: str,
                         test_pcb: str,
                         train_pcb: str,
                         outputs_dir: str,
                         debug: bool = False) -> Dict:
    """Cross-board ML model validation (Phase 8b).
    
    Uses model trained on one board to predict temperatures on another board.
    
    Args:
        trained_model_path: Path to trained .pkl model file
        calibration_db_path: Path to thermal_calibration_points.csv
        test_pcb: PCB name to validate on
        train_pcb: PCB name used for training
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
    
    # =========================================================================
    # PRE-PROCESSING MENU: FLIR FRAME FILTERING
    # =========================================================================
    # Check if user wants to filter FLIR frames before main pipeline
    # This is a one-time setup step for ML training that should be run
    # before building the CNN dataset
    # =========================================================================
    
    # Only show pre-processing menu if no command-line arguments provided
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print(" RESEARCHIR POST-PROCESSOR")
        print("="*80)
        print("\nPRE-PROCESSING OPTIONS:")
        print("  [F] Filter FLIR frames for ML training (one-time setup)")
        print("  [C] Continue to main pipeline")
        print("\nFiltering removes camera refocusing artifacts using temporal median filter.")
        print("Required before training CNN model. Takes ~8 minutes for 2 boards.")
        
        choice = input("\nSelect option [F/C]: ").strip().upper()
        
        if choice == 'F':
            # Run filtering and exit
            filter_all_flir_frames()
            print("\nFiltering complete. Exiting.")
            print("Re-run this script to continue with main pipeline.")
            return
        elif choice != 'C':
            print("Invalid choice. Exiting.")
            return
    
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
    
    # Pre-processing arguments
    parser.add_argument('--filter_flir_frames', action='store_true',
                       help='Filter FLIR frames for ML training (pre-processing step)')
    
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
    
    # Handle --filter_flir_frames flag (pre-processing mode)
    if args.filter_flir_frames:
        print("\n[PRE-PROCESSING] Filtering FLIR frames for ML training...")
        boards_filtered = filter_all_flir_frames()
        print(f"\nFiltered {boards_filtered} board(s) successfully.")
        print("Next step: Update ML dataset builder to use filtered folders")
        return
    
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
            
            # Phase 8: ML Training (NEW - U-Net CNN wrapper)
            # Note: Currently only configured for HBridge board
            if ML_AVAILABLE:
                print("\n>>> EXECUTING PHASE 8 (Machine Learning)")
                
                try:
                    # Call new phase8_ml_training wrapper
                    # This provides UI menu for U-Net CNN, Linear Regression, or both
                    # HARDCODED to HBridge for now - future: support multi-board training
                    results_8 = phase8_ml_training.run_phase8_ml_training(
                        session_dir=Path(config['output_dir']),
                        config=config,
                        board_name='HBridge'  # TODO: Make configurable for multi-board ML
                    )
                    
                    if not results_8.get('skipped'):
                        print("\n" + "="*80)
                        print(" PHASE 8 COMPLETE (Machine Learning)")
                        print("="*80)
                        if 'unet' in results_8:
                            print(f"✓ U-Net CNN: {results_8['unet'].get('model_file', 'N/A')}")
                        if 'linear' in results_8:
                            print(f"✓ Linear Regression: {results_8['linear'].get('status', 'N/A')}")
                        print("="*80)
                    
                except Exception as e:
                    print(f"\n⚠️ Warning: Phase 8 (ML Training) encountered an error: {e}")
                    if config.get('debug'):
                        import traceback
                        traceback.print_exc()
                    print("Continuing with pipeline...")
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
