"""
===============================================================================
INTERACTIVE CONFIGURATION UI
===============================================================================
Provides interactive menu system for verifying and modifying analysis settings
before running thermal post-processing.
===============================================================================
"""

import os
from typing import Dict


def display_configuration_ui(config: Dict) -> Dict:
    """
    Interactive configuration UI for thermal post-processing
    
    Allows user to verify and modify settings before processing:
    - Workflow selection (Phases 1-5, 6-7, or 1-7)
    - Input/output directories
    - Coordinates file
    - Proximity threshold
    - Spatial coupling enable/disable
    - Thermal modeling options
    - Debug mode
    
    Args:
        config: Configuration dictionary with keys:
            - workflow: 'phases_1_5', 'phases_6_7', or 'full_pipeline'
            - input_folder: ResearchIR data directory
            - output_dir: Results output directory
            - coordinates_file: CSV with component coordinates
            - proximity_threshold: Distance threshold in mm
            - spatial_enabled: Enable spatial coupling analysis
            - calibration_pcb: PCB name for calibration (Phase 6)
            - prediction_pcb: PCB name for prediction (Phase 7)
            - mapping_file: Thermistor mapping JSON file
            - debug: Enable debug output
    
    Returns:
        Updated configuration dictionary
    """
    
    workflow_names = {
        'phases_1_5': 'Phases 1-5 (Standard Analysis)',
        'phases_6_7': 'Phases 6-7 (Thermal Modeling Only)',
        'full_pipeline': 'Phases 1-7 (Complete Pipeline)'
    }
    
    while True:
        print("\n" + "="*80)
        print(" THERMAL POST-PROCESSING CONFIGURATION")
        print("="*80)
        
        print("\nWorkflow Selection:")
        print(f"  W. Workflow: {workflow_names.get(config.get('workflow', 'phases_1_5'), 'Unknown')}")
        
        print("\nStandard Configuration (Phases 1-5):")
        print(f"  1. Input Folder:        {config['input_folder']}")
        
        # Show auto-generated output folder
        if config.get('auto_output', False):
            print(f"  2. Output Directory:    [AUTO] {config['output_dir']}")
        else:
            print(f"  2. Output Directory:    {config['output_dir']}")
        
        print(f"  3. Coordinates File:    {config['coordinates_file']}")
        print(f"  4. Proximity Threshold: {config['proximity_threshold']} mm")
        print(f"  5. Spatial Analysis:    {'Enabled' if config['spatial_enabled'] else 'Disabled'}")
        
        print("\nThermal Modeling Configuration (Phases 6-7):")
        
        # Show multi-session config if specified
        multi_config = config.get('multi_session_config')
        if multi_config:
            print(f"  6. Multi-Session Config: {multi_config}")
            print(f"     [Using batch processing mode]")
        else:
            print(f"  6. Calibration PCB:      {config.get('calibration_pcb', 'Load_Shedding')}")
            print(f"  7. Prediction PCB:       {config.get('prediction_pcb', 'HBridge_15s')}")
            print(f"  8. Mapping File:         {config.get('mapping_file', 'loadshedding_thermistor_mapping.json')}")
        
        print("\nAdvanced Thermal Options:")
        print(f"  M. Multi-Session Config: {'[Configured]' if multi_config else '[Not set - single session mode]'}")
        print(f"  A. Append Calibration:   {'Yes' if config.get('append_calibration', False) else 'No (replace existing)'}")
        print(f"  C. Convergence Analysis: {'Enabled' if config.get('convergence_analysis', False) else 'Disabled'}")
        print(f"  T. Test Session Name:    {config.get('test_session', '[Auto-generated]')}")
        
        print("\nGeneral Settings:")
        print(f"  9. Debug Mode:           {'On' if config['debug'] else 'Off'}")
        
        # File validation
        print("\nFile Validation:")
        input_exists = os.path.exists(config['input_folder'])
        coords_exists = os.path.exists(config['coordinates_file']) if config['coordinates_file'] else False
        multi_config_exists = os.path.exists(config['multi_session_config']) if config.get('multi_session_config') else False
        
        print(f"  Input folder:     {'[OK]' if input_exists else '[WARNING] Not found'}")
        if config['coordinates_file']:
            print(f"  Coordinates file: {'[OK]' if coords_exists else '[WARNING] Not found'}")
        else:
            print(f"  Coordinates file: [Not specified - spatial analysis disabled]")
        
        if config.get('multi_session_config'):
            print(f"  Multi-session config: {'[OK]' if multi_config_exists else '[WARNING] Not found'}")
        
        print("\nOptions:")
        print("  W:     Change workflow (Phases 1-5, 6-7, or 1-7)")
        print("  1-9:   Modify standard settings")
        print("  M:     Configure multi-session batch processing")
        print("  A:     Toggle append calibration mode")
        print("  C:     Toggle convergence analysis")
        print("  T:     Set test session name")
        print("  V:     Validate configuration (dry-run)")
        print("  Enter: Continue with processing")
        print("  0:     Exit without processing")
        
        choice = input("\nSelect option: ").strip().upper()
        
        if choice == 'W':
            print("\nSelect Workflow:")
            print("  1. Phases 1-5 (Standard Analysis)")
            print("  2. Phases 6-7 (Thermal Modeling Only)")
            print("  3. Phases 1-7 (Complete Pipeline)")
            workflow_choice = input("\nSelect [1-3]: ").strip()
            
            if workflow_choice == '1':
                config['workflow'] = 'phases_1_5'
            elif workflow_choice == '2':
                config['workflow'] = 'phases_6_7'
            elif workflow_choice == '3':
                config['workflow'] = 'full_pipeline'
            else:
                print("[ERROR] Invalid workflow selection")
                continue
            
            # Regenerate output folder if auto-generated
            if config.get('auto_output', False):
                from datetime import datetime
                from pathlib import Path
                now = datetime.now()
                timestamp = now.strftime("%m%d_%H%M")
                workflow_abbrev = {'phases_1_5': 'P1-5', 'phases_6_7': 'P6-7', 'full_pipeline': 'P1-7'}
                abbrev = workflow_abbrev.get(config['workflow'], 'P1-7')
                folder_name = f"{timestamp}_{abbrev}"
                if config.get('debug', False):
                    folder_name += "_debug"
                config['output_dir'] = str(Path("outputs") / folder_name)
        
        elif choice == '1':
            new_input = input(f"Enter input folder [{config['input_folder']}]: ").strip()
            if new_input:
                config['input_folder'] = new_input
        
        elif choice == '2':
            new_output = input(f"Enter output directory [{config['output_dir']}]: ").strip()
            if new_output:
                config['output_dir'] = new_output
        
        elif choice == '3':
            current = config['coordinates_file'] if config['coordinates_file'] else 'None'
            new_coords = input(f"Enter coordinates file [{current}]: ").strip()
            if new_coords:
                config['coordinates_file'] = new_coords
                config['spatial_enabled'] = True
        
        elif choice == '4':
            new_threshold = input(f"Enter proximity threshold in mm [{config['proximity_threshold']}]: ").strip()
            if new_threshold:
                try:
                    config['proximity_threshold'] = float(new_threshold)
                except ValueError:
                    print("[ERROR] Invalid number. Threshold not changed.")
        
        elif choice == '5':
            config['spatial_enabled'] = not config['spatial_enabled']
            print(f"Spatial analysis {'enabled' if config['spatial_enabled'] else 'disabled'}")
        
        elif choice == '6':
            if config.get('multi_session_config'):
                print("\n[NOTE] Multi-session mode active. Use 'M' to change config file.")
            else:
                new_cal_pcb = input(f"Enter calibration PCB name [{config.get('calibration_pcb', 'Load_Shedding')}]: ").strip()
                if new_cal_pcb:
                    config['calibration_pcb'] = new_cal_pcb
        
        elif choice == '7':
            if config.get('multi_session_config'):
                print("\n[NOTE] Multi-session mode active. Use 'M' to change config file.")
            else:
                new_pred_pcb = input(f"Enter prediction PCB name [{config.get('prediction_pcb', 'HBridge_15s')}]: ").strip()
                if new_pred_pcb:
                    config['prediction_pcb'] = new_pred_pcb
        
        elif choice == '8':
            if config.get('multi_session_config'):
                print("\n[NOTE] Multi-session mode active. Use 'M' to change config file.")
            else:
                new_mapping = input(f"Enter mapping file [{config.get('mapping_file', 'loadshedding_thermistor_mapping.json')}]: ").strip()
                if new_mapping:
                    config['mapping_file'] = new_mapping
        
        elif choice == '9':
            config['debug'] = not config['debug']
            print(f"Debug mode {'enabled' if config['debug'] else 'disabled'}")
        
        elif choice == 'M':
            print("\n" + "="*80)
            print(" MULTI-SESSION CONFIGURATION")
            print("="*80)
            print("\nMulti-session mode allows batch processing of multiple calibration")
            print("sessions from a single JSON configuration file.")
            print("\nCurrent mode: " + ("Multi-session (batch)" if config.get('multi_session_config') else "Single-session"))
            
            if config.get('multi_session_config'):
                print(f"Current config: {config['multi_session_config']}")
                print("\nOptions:")
                print("  1. Change config file")
                print("  2. View config details")
                print("  3. Validate config")
                print("  4. Clear (switch to single-session mode)")
                print("  0. Back to main menu")
                
                sub_choice = input("\nSelect [0-4]: ").strip()
                
                if sub_choice == '1':
                    selected_config = _select_config_file()
                    if selected_config:
                        config['multi_session_config'] = selected_config
                        print(f"\n[OK] Config file set: {selected_config}")
                        input("Press Enter to continue...")
                
                elif sub_choice == '2':
                    _display_config_summary(config['multi_session_config'])
                    input("\nPress Enter to continue...")
                
                elif sub_choice == '3':
                    _validate_multi_session_config(config['multi_session_config'])
                    input("\nPress Enter to continue...")
                
                elif sub_choice == '4':
                    config['multi_session_config'] = None
                    print("[OK] Switched to single-session mode")
                    input("Press Enter to continue...")
            
            else:
                # List and select from available config files
                selected_config = _select_config_file()
                if selected_config:
                    config['multi_session_config'] = selected_config
                    print(f"\n[OK] Multi-session mode enabled: {selected_config}")
                    input("Press Enter to continue...")
        
        elif choice == 'A':
            config['append_calibration'] = not config.get('append_calibration', False)
            if config['append_calibration']:
                print("\n[OK] Append mode enabled - new measurements will be added to existing calibration database")
            else:
                print("\n[OK] Replace mode - existing calibration database will be replaced")
            input("Press Enter to continue...")
        
        elif choice == 'C':
            config['convergence_analysis'] = not config.get('convergence_analysis', False)
            if config['convergence_analysis']:
                print("\n[OK] Convergence analysis enabled - quality metrics will be generated after calibration")
            else:
                print("\n[OK] Convergence analysis disabled")
            input("Press Enter to continue...")
        
        elif choice == 'T':
            current_session = config.get('test_session', '')
            new_session = input(f"Enter test session name (e.g., 'LoadShedding_Week1') [{current_session or 'Auto'}]: ").strip()
            if new_session:
                config['test_session'] = new_session
                print(f"[OK] Test session name set: {new_session}")
            else:
                config['test_session'] = None
                print("[OK] Using auto-generated session name")
            input("Press Enter to continue...")
        
        elif choice == 'V':
            print("\n" + "="*80)
            print(" CONFIGURATION VALIDATION")
            print("="*80)
            
            # Validate standard file paths
            print("\nStandard Configuration:")
            input_exists = os.path.exists(config['input_folder'])
            coords_exists = os.path.exists(config['coordinates_file']) if config['coordinates_file'] else False
            
            print(f"  Input folder:     {config['input_folder']}")
            print(f"    Status: {'[OK] Found' if input_exists else '[ERROR] Not found'}")
            
            if config['coordinates_file']:
                print(f"  Coordinates file: {config['coordinates_file']}")
                print(f"    Status: {'[OK] Found' if coords_exists else '[ERROR] Not found'}")
            
            # Validate thermal modeling config
            if config.get('multi_session_config'):
                print("\nThermal Modeling (Multi-Session):")
                _validate_multi_session_config(config['multi_session_config'], verbose=True)
            elif config.get('workflow') in ['phases_6_7', 'full_pipeline']:
                print("\nThermal Modeling (Single-Session):")
                print(f"  Calibration PCB:  {config.get('calibration_pcb', 'Load_Shedding')}")
                print(f"  Prediction PCB:   {config.get('prediction_pcb', 'HBridge_15s')}")
                print(f"  Mapping file:     {config.get('mapping_file', 'loadshedding_thermistor_mapping.json')}")
                
                mapping_exists = os.path.exists(config.get('mapping_file', 'loadshedding_thermistor_mapping.json'))
                print(f"    Status: {'[OK] Found' if mapping_exists else '[WARNING] Not found'}")
            
            # Show warnings
            print("\nValidation Summary:")
            has_errors = False
            if not input_exists:
                print("  [WARNING] Input folder not found. Processing will fail.")
                has_errors = True
            if config['spatial_enabled'] and not coords_exists:
                print("  [WARNING] Coordinates file not found. Spatial analysis will be skipped.")
            if not has_errors:
                print("  [OK] All critical files found")
            
            input("\nPress Enter to continue...")
        
        elif choice == '' or choice == 'ENTER':
            print("\nProceeding with processing...")
            return config
        
        elif choice == '0':
            print("\nExiting without processing.")
            return None
        
        else:
            print("[ERROR] Invalid option.")


def _display_config_summary(config_path: str) -> None:
    """Display summary of multi-session configuration file."""
    import json
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("\n" + "="*80)
        print(f" CONFIG FILE: {config_path}")
        print("="*80)
        
        if 'sessions' in config:
            print(f"\nTotal sessions: {len(config['sessions'])}")
            
            for i, session in enumerate(config['sessions'], 1):
                print(f"\n[Session {i}] {session.get('name', 'Unnamed')}")
                print(f"  PCB:        {session.get('pcb', 'Not specified')}")
                print(f"  FLIR file:  {session.get('flir_file', 'Not specified')}")
                
                # Show thermistor files
                air_files = session.get('therm_air_file', [])
                if isinstance(air_files, list):
                    print(f"  Air files:  {len(air_files)} devices")
                    for j, f in enumerate(air_files):
                        print(f"    Device {j}: {os.path.basename(f)}")
                else:
                    print(f"  Air file:   {os.path.basename(air_files) if air_files else 'Not specified'}")
                
                sand_files = session.get('therm_sand_file', [])
                if isinstance(sand_files, list):
                    print(f"  Sand files: {len(sand_files)} devices")
                else:
                    print(f"  Sand file:  {os.path.basename(sand_files) if sand_files else 'Not specified'}")
                
                # Show component pairs
                pairs = session.get('pairs', [])
                print(f"  Components: {len(pairs)}")
        else:
            print("\n[ERROR] Invalid config format - 'sessions' key not found")
    
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Invalid JSON: {e}")
    except Exception as e:
        print(f"\n[ERROR] Could not read config: {e}")


def _validate_multi_session_config(config_path: str, verbose: bool = False) -> None:
    """Validate multi-session configuration using built-in validation."""
    
    try:
        # Import validation function
        from viz_phase6_validation import validate_multi_session_config
        
        print(f"\nValidating: {config_path}")
        print("-"*80)
        
        result = validate_multi_session_config(
            config_path=config_path,
            check_files_exist=True,
            verbose=verbose
        )
        
        if result['valid']:
            print("\n✓ Configuration is valid!")
            print(f"  Total sessions: {result['stats']['total_sessions']}")
            print(f"  Total components: {result['stats']['total_pairs']}")
            
            if verbose:
                print("\n  Session breakdown:")
                for i, session_result in enumerate(result['session_results'], 1):
                    session_name = result['config']['sessions'][i-1]['name']
                    num_pairs = session_result['stats']['num_pairs']
                    is_multi = session_result['stats'].get('is_multi_device', False)
                    device_str = "(multi-device)" if is_multi else "(single-device)"
                    status = "✓" if session_result['valid'] else "✗"
                    print(f"    {status} {i}. {session_name}: {num_pairs} components {device_str}")
        else:
            print("\n✗ Validation failed!")
            print(f"  Errors: {len(result['errors'])}")
            for err in result['errors'][:5]:  # Show first 5 errors
                print(f"    • {err}")
            
            if len(result['errors']) > 5:
                print(f"    ... and {len(result['errors'])-5} more errors")
        
        if result['warnings']:
            print(f"\n  ⚠ Warnings: {len(result['warnings'])}")
            for warn in result['warnings'][:3]:  # Show first 3 warnings
                print(f"    • {warn}")
    
    except ImportError:
        print("\n[WARNING] Validation module not available")
        print("Run 'python researchir_post_processor.py --validate' for full validation")
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")


def _select_config_file() -> str:
    """
    List available JSON config files and let user select by number.
    
    Returns:
        Selected config file path, or None if cancelled
    """
    import glob
    
    print("\n" + "="*80)
    print(" SELECT CONFIG FILE")
    print("="*80)
    
    json_files = sorted(glob.glob("*.json"))
    
    if not json_files:
        print("\nNo JSON files found in current directory")
        print("Please create a config file or specify a custom path.")
        custom = input("\nEnter custom path (or press Enter to cancel): ").strip()
        return custom if custom and os.path.exists(custom) else None
    
    print(f"\nFound {len(json_files)} config file(s):")
    for i, f in enumerate(json_files, 1):
        file_size = os.path.getsize(f)
        # Try to show a preview of the config
        try:
            import json
            with open(f, 'r') as file:
                data = json.load(file)
                num_sessions = len(data.get('sessions', []))
                print(f"  {i}. {f:<40} ({num_sessions} sessions, {file_size} bytes)")
        except:
            print(f"  {i}. {f:<40} ({file_size} bytes)")
    
    print(f"  C. Enter custom path")
    print(f"  0. Cancel")
    
    choice = input(f"\nSelect [1-{len(json_files)}, C, 0]: ").strip().upper()
    
    if choice == '0' or choice == '':
        return None
    elif choice == 'C':
        custom = input("Enter custom config file path: ").strip()
        if custom and os.path.exists(custom):
            return custom
        elif custom:
            print(f"[WARNING] File not found: {custom}")
        return None
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(json_files):
                return json_files[idx]
            else:
                print(f"[ERROR] Invalid selection. Choose 1-{len(json_files)}")
                return None
        except ValueError:
            print(f"[ERROR] Invalid input. Enter a number 1-{len(json_files)}")
            return None

