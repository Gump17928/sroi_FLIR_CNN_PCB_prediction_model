"""
===============================================================================
RESEARCHIR THERMAL DATA POST-PROCESSOR
===============================================================================
Main orchestrator for thermal post-processing workflow.

Coordinates all 5 phases:
1. Data Loading - Load and parse ResearchIR exports
2. Filtering - Remove camera artifacts, analyze transients
3. Component Analysis - Statistical analysis and export
4. Potting Risk - Failure prediction for embedded systems  
5. Spatial Coupling - Thermal interaction analysis

Usage:
    python researchir_post_processor.py
    python researchir_post_processor.py --no_ui --input inputs/data --output outputs/results
===============================================================================
"""

import os
import sys
import argparse
from typing import Dict, Optional

# Import phase modules
import phase1_data_loading as phase1
import phase2_filtering as phase2
import phase3_component_analysis as phase3
import phase4_potting_risk as phase4
import phase5_spatial_coupling as phase5
import phase_visualizations as viz
import config_ui

# Import thermal modeling phases (6 & 7)
try:
    import phase6_thermal_calibration as phase6
    import phase7_thermal_prediction as phase7
    from parse_researchir_stats import ResearchIRStatsParser
    import json
    from datetime import datetime
    THERMAL_MODELING_AVAILABLE = True
except ImportError:
    THERMAL_MODELING_AVAILABLE = False
    print("Warning: Thermal modeling phases (6 & 7) not available")

# Component type definitions for consistent use across phases
COMPONENT_TYPES = phase1.DEFAULT_COMPONENT_TYPES


def process_researchir_data(input_folder: str, output_dir: str,
                           coordinates_file: Optional[str] = None,
                           proximity_threshold: float = 10.0,
                           filter_type: str = 'median',
                           spatial_enabled: bool = True,
                           debug: bool = False) -> Dict:
    """
    Complete ResearchIR thermal data post-processing workflow
    
    Executes all 5 phases of thermal analysis:
    1. Load data from ResearchIR exports
    2. Apply filtering and analyze transients
    3. Generate statistical summaries
    4. Assess potting/embedding risk
    5. Analyze spatial thermal coupling
    
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
    print(" RESEARCHIR THERMAL DATA POST-PROCESSING")
    print("="*60)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_dir}")
    print(f"Filter type:   {filter_type}")
    print(f"Spatial analysis: {'Enabled' if spatial_enabled and coordinates_file else 'Disabled'}")
    print("="*60)
    
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
    plot_files = viz.create_ieee_plots(filtered_component_data, grouped_components,
                                       analysis_results, output_dir, debug=debug)
    
    # =========================================================================
    # VISUALIZATION: FILTERING COMPARISON
    # =========================================================================
    comparison_files = []
    if filter_type != 'none':
        print(f"\n[VISUALIZATION] Creating {filter_type} filtering comparison plots...")
        comparison_files = viz.create_filtering_comparison_plots(
            component_data, filtered_component_data, grouped_components,
            output_dir, filter_type=filter_type, debug=debug)
        
        print(f"\n[VISUALIZATION] Creating multi-filter comparison grid...")
        multifilter_grid, comparison_df = viz.create_multifilter_component_grid(
            component_data, grouped_components, output_dir, 
            filter_params={}, debug=debug)
    
    # =========================================================================
    # MATLAB: USAGE SCRIPT
    # =========================================================================
    print("\n[MATLAB] Creating MATLAB usage script...")
    phase3.create_matlab_usage_script(output_dir, grouped_components, COMPONENT_TYPES)
    
    # =========================================================================
    # PHASE 5: SPATIAL COUPLING (if coordinates provided)
    # =========================================================================
    coupling_metrics = {}
    component_coordinates = {}
    proximity_matrix = {}
    coupling_plots = []
    
    if spatial_enabled and coordinates_file:
        print("\n[PHASE 5] Loading component coordinates...")
        component_coordinates = phase5.load_component_coordinates(coordinates_file, debug=debug)
        
        if component_coordinates:
            print(f"\n[PHASE 5] Building proximity matrix (threshold: {proximity_threshold}mm)...")
            proximity_matrix = phase5.build_proximity_matrix(
                component_data, component_coordinates, proximity_threshold)
            
            if proximity_matrix:
                print("\n[PHASE 5] Calculating thermal coupling metrics...")
                coupling_metrics = phase5.calculate_thermal_coupling_metrics(
                    filtered_component_data, proximity_matrix, analysis_results)
                
                print(f"  Analyzed thermal coupling for {len(coupling_metrics)} components")
                
                print("\n[PHASE 5] Creating thermal coupling visualizations...")
                coupling_plots = viz.create_thermal_coupling_visualizations(
                    filtered_component_data, coupling_metrics, analysis_results,
                    output_dir, component_coordinates, debug=debug)
    
    # =========================================================================
    # PHASE 4: POTTING RISK (requires coupling metrics from Phase 5)
    # =========================================================================
    risk_csv = None
    risk_df = None
    
    if coupling_metrics:
        print("\n[PHASE 4] Analyzing potted condition failure risk...")
        risk_csv, risk_df = phase4.analyze_potted_condition_risk(
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
    print(" PROCESSING COMPLETE")
    print("="*60)
    print(f"Total components processed: {len(component_data)}")
    print(f"Component groups: {len(grouped_components)}")
    if coupling_metrics:
        print(f"Spatial coupling analysis: {len(coupling_metrics)} components")
    print(f"\nOutput files saved to: {output_dir}")
    print("="*60)
    
    return results


def process_thermal_modeling(calibration_pcb: str, prediction_pcb: str,
                           inputs_dir: str = "inputs",
                           outputs_dir: str = "outputs",
                           debug: bool = False) -> Dict:
    """
    Execute thermal modeling workflow (Phases 6 & 7).
    
    Phase 6: Calibration using PCB with both FLIR and thermistor data
    Phase 7: Prediction using calibration to estimate temperatures on FLIR-only PCB
    
    Args:
        calibration_pcb: Name of PCB to use for calibration (has thermistors)
        prediction_pcb: Name of PCB to predict temperatures for (FLIR only)
        inputs_dir: Input directory containing ResearchIR and thermistor data
        outputs_dir: Output directory for results
        debug: Enable debug output
    
    Returns:
        Dictionary with processing results and output file paths
    """
    if workflow_type == 'calibration':
        available = {k: v for k, v in pcb_configs.items() if v['type'] == 'calibration_source'}
        title = "SELECT CALIBRATION SOURCES"
    else:
        available = {k: v for k, v in pcb_configs.items() if v['type'] == 'prediction_target'}
        title = "SELECT PREDICTION TARGETS"
    
    if not available:
        print(f"\nNo {workflow_type} PCBs detected!")
        return {}
    
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    
    pcb_list = list(available.keys())
    for i, pcb_name in enumerate(pcb_list, 1):
        print(f"{i}. {pcb_name}")
    print(f"{len(pcb_list) + 1}. Select All")
    print(f"{len(pcb_list) + 2}. Cancel")
    
    choice = input(f"\nSelect PCB(s) [1-{len(pcb_list) + 2}] (comma-separated for multiple): ").strip()
    
    if choice == str(len(pcb_list) + 2):
        return {}
    
    selected = {}
    if choice == str(len(pcb_list) + 1):
        selected = available
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            for idx in indices:
                if 1 <= idx <= len(pcb_list):
                    pcb_name = pcb_list[idx - 1]
                    selected[pcb_name] = available[pcb_name]
        except ValueError:
            print("Invalid selection!")
            return {}
    
    return selected


def display_workflow_plan(option_num: int, calibration_sources: Dict, prediction_targets: Dict, 
                         outputs_dir: str):
    """
    Display detailed plan of which files will be used and created.
    
    Args:
        option_num: Menu option number (1-7)
        calibration_sources: Selected calibration PCBs
        prediction_targets: Selected prediction PCBs
        outputs_dir: Output directory path
    """
    from pathlib import Path
    
    print(f"\n{'='*80}")
    print("WORKFLOW EXECUTION PLAN")
    print(f"{'='*80}")
    
    if option_num == 1:
        print("\nFULL PIPELINE EXECUTION:")
        print("  Phase 4: parse_researchir_stats.py")
        print("  Phase 6: phase6_thermal_calibration.py")
        print("  Phase 7: phase7_thermal_prediction.py")
    elif option_num == 2:
        print("\nCALIBRATION ONLY:")
        print("  Phase 6: phase6_thermal_calibration.py")
    elif option_num == 3:
        print("\nPREDICTION ONLY:")
        print("  Phase 7: phase7_thermal_prediction.py")
    elif option_num == 4:
        print("\nPARSING ONLY:")
        print("  Phase 4: parse_researchir_stats.py")
    
    # Show calibration sources
    if calibration_sources:
        print(f"\n{'─'*80}")
        print("CALIBRATION SOURCES:")
        for pcb_name, config in calibration_sources.items():
            print(f"\n  [{pcb_name}]")
            print(f"    Input Files:")
            print(f"      • ResearchIR: {config['researchir_dir']}")
            for therm_file in config['thermistor_files']:
                print(f"      • Thermistor: {Path(therm_file).name}")
            if config['mapping_file']:
                print(f"      • Mapping: {Path(config['mapping_file']).name}")
            print(f"    Output Files:")
            print(f"      • {outputs_dir}/{pcb_name}/{pcb_name}_FLIR_AllComponents.csv")
            print(f"      • {outputs_dir}/calibration_database/thermal_calibration_points.csv")
            print(f"      • {outputs_dir}/calibration_database/thermal_calibration_by_type.csv")
    
    # Show prediction targets
    if prediction_targets:
        print(f"\n{'─'*80}")
        print("PREDICTION TARGETS:")
        for pcb_name, config in prediction_targets.items():
            print(f"\n  [{pcb_name}]")
            print(f"    Input Files:")
            print(f"      • ResearchIR: {config['researchir_dir']}")
            print(f"      • Calibration DB: {outputs_dir}/calibration_database/thermal_calibration_by_type.csv")
            print(f"    Output Files:")
            print(f"      • {outputs_dir}/{pcb_name}/thermal_prediction_{pcb_name}_air.csv")
            print(f"      • {outputs_dir}/{pcb_name}/thermal_prediction_{pcb_name}_sand.csv")
            print(f"      • {outputs_dir}/{pcb_name}/thermal_prediction_{pcb_name}_summary.png")
    
    print(f"\n{'='*80}")


def thermal_modeling_menu(inputs_dir: str = "inputs", outputs_dir: str = "outputs"):
    """
    Interactive menu for thermal modeling workflow (Phase 6 & 7).
    
    Options:
    1. Run Full Pipeline (Parse → Calibrate → Predict)
    2. Run Calibration Only (Phase 6)
    3. Run Prediction Only (Phase 7)
    4. Parse ResearchIR Stats Only
    5. Refresh PCB Detection
    6. View Calibration History
    7. Back to Main Menu
    """
    if not THERMAL_MODELING_AVAILABLE:
        print("\nError: Thermal modeling modules not available!")
        return
    
    from pathlib import Path
    
    # Initial PCB detection
    pcb_configs = detect_pcb_configurations(inputs_dir)
    
    while True:
        print(f"\n{'='*80}")
        print("THERMAL MODELING WORKFLOW (Phase 6 & 7)")
        print(f"{'='*80}")
        
        # Show detected PCBs summary
        cal_count = sum(1 for v in pcb_configs.values() if v['type'] == 'calibration_source')
        pred_count = sum(1 for v in pcb_configs.values() if v['type'] == 'prediction_target')
        print(f"Detected: {cal_count} calibration source(s), {pred_count} prediction target(s)")
        
        print(f"\n{'─'*80}")
        print("1. Run Full Pipeline (Parse → Calibrate → Predict)")
        print("   Files: parse_researchir_stats.py, phase6_thermal_calibration.py, phase7_thermal_prediction.py")
        print("\n2. Run Calibration Only (Phase 6)")
        print("   Files: phase6_thermal_calibration.py")
        print("\n3. Run Prediction Only (Phase 7)")
        print("   Files: phase7_thermal_prediction.py")
        print("\n4. Parse ResearchIR Stats Only")
        print("   Files: parse_researchir_stats.py")
        print("\n5. Refresh PCB Detection")
        print("   Rescan inputs/ directory for PCB configurations")
        print("\n6. View Calibration History")
        print("   Display previous calibration sessions")
        print("\n7. Back to Main Menu")
        print(f"{'='*80}")
        
        choice = input("\nSelect option [1-7]: ").strip()
        
        if choice == '7':
            break
        
        if choice == '1':
            # Full pipeline - select PCBs for calibration and prediction
            calibration_sources = select_pcbs_for_workflow(pcb_configs, 'calibration')
            if not calibration_sources:
                continue
            
            prediction_targets = select_pcbs_for_workflow(pcb_configs, 'prediction')
            if not prediction_targets:
                continue
            
            # Show execution plan
            display_workflow_plan(1, calibration_sources, prediction_targets, outputs_dir)
            
            confirm = input("\nProceed with execution? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                continue
            
            # Phase 4: Parse ResearchIR data
            print(f"\n{'='*80}")
            print("PHASE 4: PARSING RESEARCHIR DATA")
            print("Running: parse_researchir_stats.py")
            print(f"{'='*80}")
            all_pcbs = {**calibration_sources, **prediction_targets}
            for pcb_name, config in all_pcbs.items():
                print(f"\n  Processing: {pcb_name}")
                parse_researchir_stats(config['researchir_dir'], pcb_name, outputs_dir)
            
            # Phase 6: Run calibration
            print(f"\n{'='*80}")
            print("PHASE 6: THERMAL CALIBRATION")
            print("Running: phase6_thermal_calibration.py")
            print(f"{'='*80}")
            run_thermal_calibration(calibration_sources, outputs_dir)
            
            # Phase 7: Run prediction
            print(f"\n{'='*80}")
            print("PHASE 7: THERMAL PREDICTION")
            print("Running: phase7_thermal_prediction.py")
            print(f"{'='*80}")
            run_thermal_prediction(prediction_targets, outputs_dir)
            
            print(f"\n{'='*80}")
            print("FULL PIPELINE COMPLETE")
            print(f"{'='*80}")
        
        elif choice == '2':
            # Calibration only - select calibration sources
            calibration_sources = select_pcbs_for_workflow(pcb_configs, 'calibration')
            if not calibration_sources:
                continue
            
            # Show execution plan
            display_workflow_plan(2, calibration_sources, {}, outputs_dir)
            
            confirm = input("\nProceed with execution? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                continue
            
            print(f"\n{'='*80}")
            print("PHASE 6: THERMAL CALIBRATION")
            print("Running: phase6_thermal_calibration.py")
            print(f"{'='*80}")
            run_thermal_calibration(calibration_sources, outputs_dir)
            
            print(f"\n{'='*80}")
            print("CALIBRATION COMPLETE")
            print(f"{'='*80}")
        
        elif choice == '3':
            # Prediction only - select prediction targets
            prediction_targets = select_pcbs_for_workflow(pcb_configs, 'prediction')
            if not prediction_targets:
                continue
            
            # Show execution plan
            display_workflow_plan(3, {}, prediction_targets, outputs_dir)
            
            confirm = input("\nProceed with execution? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                continue
            
            print(f"\n{'='*80}")
            print("PHASE 7: THERMAL PREDICTION")
            print("Running: phase7_thermal_prediction.py")
            print(f"{'='*80}")
            run_thermal_prediction(prediction_targets, outputs_dir)
            
            print(f"\n{'='*80}")
            print("PREDICTION COMPLETE")
            print(f"{'='*80}")
        
        elif choice == '4':
            # Parse only - select any PCBs
            print(f"\n{'='*80}")
            print("SELECT PCBs TO PARSE")
            print(f"{'='*80}")
            pcb_list = list(pcb_configs.keys())
            for i, pcb_name in enumerate(pcb_list, 1):
                print(f"{i}. {pcb_name}")
            print(f"{len(pcb_list) + 1}. Select All")
            print(f"{len(pcb_list) + 2}. Cancel")
            
            choice_parse = input(f"\nSelect PCB(s) [1-{len(pcb_list) + 2}] (comma-separated): ").strip()
            
            if choice_parse == str(len(pcb_list) + 2):
                continue
            
            selected_pcbs = {}
            if choice_parse == str(len(pcb_list) + 1):
                selected_pcbs = pcb_configs
            else:
                try:
                    indices = [int(x.strip()) for x in choice_parse.split(',')]
                    for idx in indices:
                        if 1 <= idx <= len(pcb_list):
                            pcb_name = pcb_list[idx - 1]
                            selected_pcbs[pcb_name] = pcb_configs[pcb_name]
                except ValueError:
                    print("Invalid selection!")
                    continue
            
            if not selected_pcbs:
                continue
            
            print(f"\n{'='*80}")
            print("PHASE 4: PARSING RESEARCHIR DATA")
            print("Running: parse_researchir_stats.py")
            print(f"{'='*80}")
            for pcb_name, config in selected_pcbs.items():
                print(f"\n  Processing: {pcb_name}")
                parse_researchir_stats(config['researchir_dir'], pcb_name, outputs_dir)
            
            print(f"\n{'='*80}")
            print("PARSING COMPLETE")
            print(f"{'='*80}")
        
        elif choice == '5':
            # Refresh detection
            pcb_configs = detect_pcb_configurations(inputs_dir)
        
        elif choice == '6':
            # View history
            view_calibration_history(outputs_dir)
        
        else:
            print("Invalid option, please try again.")


def parse_researchir_stats(researchir_dir: str, pcb_name: str, outputs_dir: str):
    """Parse ResearchIR Stats.txt files to CSV."""
    from pathlib import Path
    
    print(f"\n  → Parsing ResearchIR Stats.txt files from: {researchir_dir}")
    parser = ResearchIRStatsParser(researchir_dir)
    
    # Parse all frames
    df = parser.parse_all_frames()
    
    if df is not None and not df.empty:
        # Save to outputs/{pcb_name}/
        pcb_output_dir = Path(outputs_dir) / pcb_name
        pcb_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = pcb_output_dir / f"{pcb_name}_FLIR_AllComponents.csv"
        df.to_csv(output_file, index=False)
        print(f"  ✓ Created: {output_file}")
        print(f"    ({len(df)} frames, {df.shape[1]-1} components)")
    else:
        print(f"  ✗ Error: Failed to parse {researchir_dir}")


def run_thermal_calibration(calibration_sources: Dict, outputs_dir: str):
    """Run Phase 6 calibration on all calibration sources."""
    from pathlib import Path
    
    cal_db_dir = Path(outputs_dir) / "calibration_database"
    calibrator = phase6.ThermalCalibrator(output_dir=cal_db_dir)
    
    for pcb_name, config in calibration_sources.items():
        print(f"\n  → Processing calibration source: {pcb_name}")
        
        # Load FLIR data
        flir_file = Path(outputs_dir) / pcb_name / f"{pcb_name}_FLIR_AllComponents.csv"
        if not flir_file.exists():
            print(f"  Error: FLIR file not found: {flir_file}")
            continue
        
        # Load mapping
        mapping_file = Path(config['mapping_file'])
        if not mapping_file.exists():
            print(f"  Error: Mapping file not found: {mapping_file}")
            continue
        
        with open(mapping_file, 'r') as f:
            mapping_config = json.load(f)
        
        # Build measurement pairs
        pairs = []
        for therm_channel, mapping_data in mapping_config["thermistor_to_flir_mapping"].items():
            flir_comp = mapping_data["flir_component"]
            comp_type = mapping_data["component_type"]
            
            # Skip ambient measurements
            if "ambient" in mapping_data.get("notes", "").lower():
                continue
            
            pairs.append((flir_comp, therm_channel, f"{flir_comp}_{comp_type}", comp_type))
        
        # Find thermistor files
        therm_files = config['thermistor_files']
        air_file = next((f for f in therm_files if 'AIR' in f.upper()), None)
        sand_file = next((f for f in therm_files if 'SAND' in f.upper()), None)
        
        if not air_file:
            print(f"  Error: No AIR thermistor file found")
            continue
        
        # Add measurement pair
        print(f"    Using {len(pairs)} measurement pairs")
        calibrator.add_measurement_pair(
            flir_file=flir_file,
            therm_air_file=Path(air_file),
            therm_sand_file=Path(sand_file) if sand_file else None,
            pairs=pairs,
            pcb_name=pcb_name
        )
    
    # Compute and export calibrations
    print(f"\n  → Computing component type calibrations...")
    calibrator.compute_component_type_calibrations()
    
    print(f"\n  → Exporting calibration database...")
    calibrator.export_calibration_database(prefix="thermal_calibration")
    
    print(f"\n  ✓ Created: {cal_db_dir / 'thermal_calibration_points.csv'}")
    print(f"  ✓ Created: {cal_db_dir / 'thermal_calibration_by_type.csv'}")
    print(f"  ✓ Created: {cal_db_dir / 'thermal_calibration_metadata.json'}")
    print(f"  ✓ Created: {cal_db_dir / 'thermal_calibration_summary.png'}")
    
    # Update calibration history
    update_calibration_history(calibration_sources, outputs_dir)
    print(f"  ✓ Updated: {cal_db_dir / 'calibration_history.json'}")


def run_thermal_prediction(prediction_targets: Dict, outputs_dir: str):
    """Run Phase 7 prediction on all prediction targets."""
    from pathlib import Path
    
    # Load calibration database
    calibration_file = Path(outputs_dir) / "calibration_database" / "thermal_calibration_by_type.csv"
    if not calibration_file.exists():
        print(f"\n  ✗ Error: Calibration database not found: {calibration_file}")
        print(f"     Please run calibration first (Option 2)")
        return
    
    print(f"\n  → Loading calibration database: {calibration_file}")
    
    # Load component type patterns
    patterns_file = Path(__file__).parent / "component_type_patterns.json"
    if patterns_file.exists():
        with open(patterns_file, 'r') as f:
            patterns_config = json.load(f)
        component_patterns = patterns_config["patterns"]
    else:
        # Default patterns
        component_patterns = {
            "PowerSupply": [r"^PS\d+$", r"^CONV$"],
            "IC": [r"^U\d+$", r"^IC\d+$"],
            "Resistor": [r"^R\d+$"],
            "LED": [r"^DL\d+$", r"^CR\d+$"]
        }
    
    for pcb_name, config in prediction_targets.items():
        print(f"\n  → Processing prediction target: {pcb_name}")
        
        # Load FLIR data
        flir_file = Path(outputs_dir) / pcb_name / f"{pcb_name}_FLIR_AllComponents.csv"
        if not flir_file.exists():
            print(f"    ✗ Error: FLIR file not found: {flir_file}")
            continue
        
        print(f"    Using FLIR data: {flir_file}")
        
        # Run prediction
        predictor = phase7.ThermalPredictor(calibration_csv=calibration_file)
        predictor.load_flir_data(flir_file)
        
        print(f"    Inferring component types...")
        predictor.infer_component_types(component_patterns)
        
        # Predict for both air and sand
        print(f"    Predicting temperatures (air cooling)...")
        predictor.predict_temperatures(cooling_mode="air")
        print(f"    Predicting temperatures (sand cooling)...")
        predictor.predict_temperatures(cooling_mode="sand")
        
        # Export results
        pcb_output_dir = Path(outputs_dir) / pcb_name
        print(f"    Exporting results...")
        predictor.export_predictions(output_dir=pcb_output_dir, prefix=f"thermal_prediction_{pcb_name}")
        predictor.create_prediction_visualizations(output_dir=pcb_output_dir, prefix=f"thermal_prediction_{pcb_name}")
        
        print(f"\n  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_air.csv'}")
        print(f"  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_sand.csv'}")
        print(f"  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_summary.json'}")
        print(f"  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_air.png'}")
        print(f"  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_sand.png'}")
        print(f"  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_hottest_air.png'}")
        print(f"  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_hottest_sand.png'}")
        print(f"  ✓ Created: {pcb_output_dir / f'thermal_prediction_{pcb_name}_cooling_benefit.png'}")


def update_calibration_history(calibration_sources: Dict, outputs_dir: str):
    """Update calibration history JSON."""
    from pathlib import Path
    
    history_file = Path(outputs_dir) / "calibration_database" / "calibration_history.json"
    
    # Load existing history
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {"sessions": []}
    
    # Add new session
    session = {
        "timestamp": datetime.now().isoformat(),
        "pcbs": list(calibration_sources.keys()),
        "num_pcbs": len(calibration_sources)
    }
    history["sessions"].append(session)
    
    # Save history
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def view_calibration_history(outputs_dir: str):
    """Display calibration history."""
    from pathlib import Path
    
    history_file = Path(outputs_dir) / "calibration_database" / "calibration_history.json"
    
    if not history_file.exists():
        print("\nNo calibration history found.")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print(f"\n{'='*80}")
    print("CALIBRATION HISTORY")
    print(f"{'='*80}")
    
    for i, session in enumerate(history["sessions"], 1):
        print(f"\nSession {i}:")
        print(f"  Timestamp: {session['timestamp']}")
        print(f"  PCBs: {', '.join(session['pcbs'])}")
        print(f"  Count: {session['num_pcbs']}")


def main():
    """Main entry point with argument parsing and configuration UI"""
    
    parser = argparse.ArgumentParser(
        description='Post-process ResearchIR thermal data exports'
    )
    
    parser.add_argument('--input', type=str, default='inputs/ResearchIR_Outputs_HBridge_15s',
                       help='Input folder with ResearchIR CSV/TXT exports')
    parser.add_argument('--output', type=str, default='outputs/thermal_analysis_results',
                       help='Output directory for results')
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
    parser.add_argument('--thermal_modeling', action='store_true',
                       help='Launch thermal modeling workflow (Phase 6 & 7)')
    
    args = parser.parse_args()
    
    # Check if thermal modeling menu requested
    if args.thermal_modeling and THERMAL_MODELING_AVAILABLE:
        thermal_modeling_menu()
        return
    
    # Configuration dictionary
    config = {
        'input_folder': args.input,
        'output_dir': args.output,
        'coordinates_file': args.coordinates,
        'proximity_threshold': args.proximity_threshold,
        'filter_type': args.filter,
        'spatial_enabled': not args.no_spatial,
        'debug': args.debug
    }
    
    # Show interactive UI unless disabled
    if not args.no_ui:
        config = config_ui.display_configuration_ui(config)
        if config is None:
            print("Processing cancelled by user.")
            return
    
    # Run processing
    try:
        results = process_researchir_data(
            input_folder=config['input_folder'],
            output_dir=config['output_dir'],
            coordinates_file=config['coordinates_file'],
            proximity_threshold=config['proximity_threshold'],
            filter_type=config['filter_type'],
            spatial_enabled=config['spatial_enabled'],
            debug=config['debug']
        )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        if config['debug']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
