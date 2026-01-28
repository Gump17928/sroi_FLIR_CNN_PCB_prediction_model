"""
===============================================================================
PHASE 8: MACHINE LEARNING TRAINING
===============================================================================
Orchestrates thermal prediction ML model training using U-Net CNN.

This module provides:
- U-Net CNN wrapper (calls ml_model/cnn_thermal_modeling/)
- Model evaluation interface
- User interface for training and evaluation

Created: January 15, 2026
Updated: January 28, 2026 - Removed legacy linear regression
===============================================================================
"""

import sys
from pathlib import Path
import shutil
import json
from datetime import datetime
from typing import Dict, Optional


def run_phase8_ml_training(session_dir: Path, config: Dict, 
                          board_name: str = "HBridge") -> Dict:
    """
    Main entry point for Phase 8 ML training.
    
    Provides UI menu for model selection and orchestrates training.
    
    Args:
        session_dir: Current session output directory (e.g., outputs/0115_1430_P1-7/)
        config: Configuration dictionary from JSON config file
        board_name: Board identifier ('HBridge', 'LoadShedding', etc.)
    
    Returns:
        Dictionary with model results and paths
    """
    print("\n" + "="*80)
    print("PHASE 8: MACHINE LEARNING MODEL TRAINING")
    print("="*80)
    print("\nAvailable Options:")
    print("  [1] Train U-Net CNN (spatial thermal prediction)")
    print("  [2] Evaluate existing model (quick test - no training)")
    print("  [3] Skip Phase 8")
    print("\nNote: Training U-Net CNN requires ~30-60 minutes for initial training")
    print("      Evaluation runs in ~1-2 minutes on existing model")
    
    choice = input("\nYour choice (1-3) [default: 3]: ").strip() or '3'
    
    results = {}
    
    if choice == '1':
        print("\n▶ Training U-Net CNN...")
        results['unet'] = run_unet_cnn(session_dir, config, board_name)
    
    elif choice == '2':
        print("\n▶ Evaluating existing model...")
        results['evaluation'] = evaluate_existing_model(session_dir, config, board_name)
    
    else:
        print("\n⏭️  Skipping Phase 8 (ML training)")
        return {'skipped': True}
    
    return results


def archive_old_models(results_dir: Path) -> None:
    """
    Archive old model files before new training to keep results/ clean.
    
    Moves .keras model files and training history to archive subfolder.
    
    Args:
        results_dir: Path to results directory
    """
    from datetime import datetime
    
    if not results_dir.exists():
        return
    
    # Create archive folder with timestamp
    archive_date = datetime.now().strftime("%Y%m%d")
    archive_dir = results_dir / f"archive_{archive_date}"
    
    # Find old model files
    model_files = list(results_dir.glob("*.keras"))
    history_files = list(results_dir.glob("training_history_*.npz"))
    curve_files = list(results_dir.glob("training_curves_*.png"))
    
    files_to_archive = model_files + history_files + curve_files
    
    if files_to_archive:
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        archived_count = 0
        for file in files_to_archive:
            dest = archive_dir / file.name
            if not dest.exists():  # Don't overwrite if already archived
                shutil.move(str(file), str(dest))
                archived_count += 1
        
        if archived_count > 0:
            print(f"  ℹ️  Archived {archived_count} old model files to: {archive_dir.name}")


def validate_dataset(dataset_file: Path) -> Dict:
    """
    Validate HDF5 dataset quality before training.
    
    Checks:
    - File exists and is readable
    - Shape consistency (FLIR frames vs thermistor samples)
    - Temperature range sanity (20-150°C)
    - ROI mask coverage
    - Time alignment quality
    
    Args:
        dataset_file: Path to HDF5 dataset
    
    Returns:
        Dictionary with 'valid' flag and list of 'issues'
    """
    issues = []
    
    try:
        import h5py
        import numpy as np
        
        with h5py.File(dataset_file, 'r') as f:
            # Check required keys exist
            required_keys = ['flir_frames', 'sand_temps', 'roi_masks', 'timestamps']
            for key in required_keys:
                if key not in f:
                    issues.append(f"Missing required dataset key: {key}")
            
            if issues:
                return {'valid': False, 'issues': issues}
            
            # Get shapes
            flir_shape = f['flir_frames'].shape
            temps_shape = f['sand_temps'].shape
            masks_shape = f['roi_masks'].shape
            times_shape = f['timestamps'].shape
            
            print(f"    FLIR frames: {flir_shape}")
            print(f"    Thermistor temps: {temps_shape}")
            print(f"    ROI masks: {masks_shape}")
            print(f"    Timestamps: {times_shape}")
            
            # Check shape consistency
            n_frames = flir_shape[0]
            
            # Check if dataset uses frame_indices (allows fewer frames than timestamps)
            uses_frame_indices = False
            if 'metadata' in f and 'uses_frame_indices' in f['metadata'].attrs:
                uses_frame_indices = f['metadata'].attrs['uses_frame_indices']
            
            # Only flag shape mismatch if NOT using frame_indices
            if not uses_frame_indices:
                if temps_shape[0] != n_frames:
                    issues.append(f"Shape mismatch: {n_frames} FLIR frames but {temps_shape[0]} thermistor samples")
                
                if times_shape[0] != n_frames:
                    issues.append(f"Shape mismatch: {n_frames} FLIR frames but {times_shape[0]} timestamps")
            else:
                # Using frame_indices - this is expected for extended steady-state datasets
                print(f"    Using frame_indices: {n_frames} frames mapped to {times_shape[0]} timestamps")
            
            # Check temperature ranges (sample first frame)
            if n_frames > 0:
                sample_flir = f['flir_frames'][0]
                sample_temps = f['sand_temps'][0]
                
                flir_min, flir_max = np.min(sample_flir), np.max(sample_flir)
                temps_min, temps_max = np.min(sample_temps), np.max(sample_temps)
                
                print(f"    FLIR temp range: {flir_min:.1f}°C to {flir_max:.1f}°C")
                print(f"    Thermistor range: {temps_min:.1f}°C to {temps_max:.1f}°C")
                
                # Sanity checks (typical PCB temps: 20-150°C)
                if flir_min < 0 or flir_max > 200:
                    issues.append(f"FLIR temperatures out of range: {flir_min:.1f} to {flir_max:.1f}°C")
                
                if temps_min < 0 or temps_max > 200:
                    issues.append(f"Thermistor temperatures out of range: {temps_min:.1f} to {temps_max:.1f}°C")
            
            # Check ROI mask coverage (check ALL components, not just first)
            if masks_shape[0] > 0:
                all_masks = f['roi_masks'][:]
                n_roi_pixels = np.sum(all_masks > 0)
                total_pixels = all_masks.shape[1] * all_masks.shape[2]
                coverage = 100 * n_roi_pixels / total_pixels
                
                print(f"    ROI coverage: {n_roi_pixels}/{total_pixels} pixels ({coverage:.2f}%)")
                
                if n_roi_pixels == 0:
                    issues.append("ROI masks have zero coverage (no pixels marked)")
                elif coverage < 0.05:
                    issues.append(f"Very low ROI coverage ({coverage:.2f}%) - check pixel map")
            
            # Check metadata if available
            if 'metadata' in f:
                print(f"    Metadata keys: {list(f['metadata'].attrs.keys())})")
        
        return {'valid': len(issues) == 0, 'issues': issues}
        
    except Exception as e:
        return {'valid': False, 'issues': [f"Failed to read dataset: {e}"]}


def run_unet_cnn(session_dir: Path, config: Dict, board_name: str) -> Dict:
    """
    Run U-Net CNN training using ml_model/cnn_thermal_modeling/.
    
    Steps:
    1. Verify required inputs (FLIR frames, thermistor CSV, ROI map)
    2. Build HDF5 dataset (with median filtering if enabled)
    3. Train model or use existing
    4. Generate predictions
    5. Import results back to session outputs/
    
    Args:
        session_dir: Current session output directory
        config: Configuration dictionary
        board_name: Board identifier
    
    Returns:
        Dictionary with model results
    """
    # Get paths
    project_root = Path(__file__).parent
    
    # Try to find the FLIR folder - check multiple patterns
    # Priority: Prefer _15s_filtered > _filtered > _15s > raw
    # Pattern 1: ResearchIR_Outputs_{board_name}_15s_filtered (BEST - filtered with 15s suffix)
    # Pattern 2: ResearchIR_Outputs_{board_name}_filtered (filtered without suffix)
    # Pattern 3: ResearchIR_Outputs_{board_name}_15s (raw with 15s suffix)
    # Pattern 4: ResearchIR_Outputs_{board_name} (raw)
    
    flir_folder = None
    potential_folders = [
        project_root / "inputs" / f"ResearchIR_Outputs_{board_name}_15s_filtered",
        project_root / "inputs" / f"ResearchIR_Outputs_{board_name}_filtered",
        project_root / "inputs" / f"ResearchIR_Outputs_{board_name}_15s",
        project_root / "inputs" / f"ResearchIR_Outputs_{board_name}",
    ]
    
    for folder in potential_folders:
        if folder.exists():
            flir_folder = folder
            if "_filtered" in folder.name:
                print(f"  ✓ Using filtered FLIR frames: {flir_folder.name}")
            else:
                print(f"  ⚠️ Using raw FLIR frames: {flir_folder.name}")
                print(f"    TIP: Run 'Filter FLIR frames' from pre-processing menu for better results")
            break
    
    if flir_folder is None:
        raise FileNotFoundError(f"FLIR folder not found for board '{board_name}'. Checked: {[f.name for f in potential_folders]}")
    
    # Check for thermistor data - USE SAND (full 36-hour dataset for prediction)
    thermistor_csv = session_dir / f"{board_name}_15s_thermistor_timeseries.csv"
    if not thermistor_csv.exists():
        raise FileNotFoundError(f"Thermistor CSV not found: {thermistor_csv}")
    
    # Check for ROI pixel map - PRIORITY: Use SROI-generated canonical version first
    canonical_roi_map = project_root / "outputs" / "roi_pixel_maps" / f"{board_name}_roi_pixel_map.csv"
    session_roi_map = session_dir / f"{board_name}_15s_roi_pixel_map.csv"
    
    if canonical_roi_map.exists():
        roi_map = canonical_roi_map
        print(f"  ✓ Using canonical ROI pixel map from SROI pipeline: {canonical_roi_map.name}")
    elif session_roi_map.exists():
        roi_map = session_roi_map
        print(f"  ⚠️  Using session-specific ROI pixel map: {session_roi_map.name}")
        print(f"     (Canonical version not found at: {canonical_roi_map})")
    else:
        print(f"  ⚠️  ROI pixel map not found in canonical location: {canonical_roi_map.name}")
        print(f"  ⚠️  ROI pixel map not found in session location: {session_roi_map.name}")
        print(f"  ▶  Generating ROI pixel map from component coordinates (may have incorrect PCB corners)...")
        
        # Generate it using the standalone script logic
        component_csv = project_root / "inputs" / f"{board_name.lower()}_pcb_components_enhanced.csv"
        if not component_csv.exists():
            raise FileNotFoundError(
                f"Component CSV not found: {component_csv}\n"
                f"  Cannot generate ROI pixel map"
            )
        
        # Import and run pixel map generator
        sys.path.insert(0, str(project_root))
        from generate_roi_pixel_map import generate_roi_pixel_map
        
        generate_roi_pixel_map(
            str(component_csv),
            str(roi_map),
            pcb_bounds=(0, 0, 100, 80),
            image_shape=(480, 640),
            roi_radius=5
        )
    
    print("\n  ✓ All required inputs found")
    print(f"    FLIR frames: {flir_folder}")
    print(f"    Thermistor CSV: {thermistor_csv}")
    print(f"    ROI map: {roi_map}")
    
    # Add ml_model/cnn_thermal_modeling to path
    ml_path = project_root / "ml_model" / "cnn_thermal_modeling"
    sys.path.insert(0, str(ml_path))
    
    # Import CNN modules
    try:
        import build_hbridge_dataset
        import train_hbridge_model
        # import generate_predictions  # If available
    except ImportError as e:
        raise ImportError(
            f"Failed to import CNN modules: {e}\n"
            f"  Check ml_model/cnn_thermal_modeling/ exists"
        )
    
    # Build dataset (if needed)
    dataset_file = ml_path / "datasets" / "HBridge_cnn_dataset.h5"
    if dataset_file.exists():
        print(f"\n  ✓ Found existing dataset: {dataset_file.name}")
        print("\n    [1] Use existing dataset")
        print("    [2] Rebuild dataset")
        rebuild_choice = input("\n    Select option [1-2] (default: 1): ").strip()
        rebuild = (rebuild_choice == '2')
    else:
        rebuild = True
    
    if rebuild:
        print("\n  ▶ Building CNN dataset...")
        print("    (This loads all FLIR frames into memory - may take a few minutes)")
        
        # Pass current session paths to dataset builder
        build_hbridge_dataset.build_hbridge_dataset(
            flir_folder=str(flir_folder),
            thermistor_csv=str(thermistor_csv),
            pixel_map_csv=str(roi_map),
            output_h5=str(dataset_file)
        )
        
        # Validate dataset after building
        print("\n  ▶ Validating dataset...")
        validation = validate_dataset(dataset_file)
        
        if not validation['valid']:
            print("\n  ⚠️ WARNING: Dataset validation issues detected:")
            for issue in validation['issues']:
                print(f"    - {issue}")
            
            proceed = input("    Continue anyway? (y/n) [default: n]: ").strip().lower()
            if proceed != 'y':
                raise RuntimeError("Dataset validation failed. Aborting training.")
        else:
            print("  ✓ Dataset validation passed!")
    
    # Train or use existing model
    model_file = ml_path / "models" / "thermal_unet_model.keras"
    if model_file.exists():
        print(f"\n  ✓ Found existing model: {model_file.name}")
        retrain = input("    Retrain model? (y/n) [default: n]: ").strip().lower() == 'y'
    else:
        retrain = True
    
    if retrain:
        print("\n  ▶ Training U-Net model...")
        print("    ⏱️  This may take 30-60 minutes depending on hardware")
        print("    ⏸️  You can press Ctrl+C to stop and use existing model later")
        
        # Archive old models before training (Priority 3)
        archive_old_models(ml_path / "results")
        
        # Save training manifest (Priority 2)
        training_start = datetime.now()
        manifest = {
            'training_timestamp': training_start.isoformat(),
            'session_dir': str(session_dir),
            'flir_folder': str(flir_folder),
            'thermistor_csv': str(thermistor_csv),
            'roi_map': str(roi_map),
            'dataset_file': str(dataset_file),
            'board_name': board_name,
            'config': {k: str(v) for k, v in config.items() if k != 'debug'}
        }
        
        manifest_file = ml_path / "results" / "training_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  ✓ Saved training manifest: {manifest_file.name}")
        
        try:
            train_hbridge_model.main()
            training_end = datetime.now()
        except KeyboardInterrupt:
            print("\n  ⏸️  Training interrupted. Using existing model (if available)")
            training_end = None
            if not model_file.exists():
                raise RuntimeError("No existing model found. Cannot continue without training.")
    
    # Generate predictions (if module available)
    print("\n  ▶ Generating predictions...")
    # try:
    #     import generate_predictions
    #     generate_predictions.main()
    # except ImportError:
    #     print("    ⚠️ generate_predictions.py not found. Skipping prediction generation.")
    
    # Import results to main pipeline outputs
    print("\n  ▶ Importing results to session outputs...")
    results_dir = import_cnn_results(session_dir, ml_path, training_start if retrain else None)
    
    return {
        'model_type': 'unet_cnn',
        'model_file': str(model_file),
        'dataset_file': str(dataset_file),
        'results_dir': str(results_dir),
        'board_name': board_name
    }


def import_cnn_results(session_dir: Path, ml_path: Path, training_timestamp=None) -> Path:
    """
    Copy ML model results to main pipeline outputs (smart filtering).
    
    Only copies files from the current training run, not old results.
    
    Args:
        session_dir: Current session output directory
        ml_path: Path to ml_model/cnn_thermal_modeling/
        training_timestamp: Timestamp when training started (if None, copy all recent files)
    
    Returns:
        Path to imported results directory
    """
    from datetime import datetime, timedelta
    import os
    
    ml_results = ml_path / "results"
    
    # Create timestamped results subfolder
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase8_dir = session_dir / "phase8_ml_results" / f"run_{timestamp_str}"
    phase8_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine cutoff time for "recent" files
    if training_timestamp:
        # Only copy files created after training started
        cutoff_time = training_timestamp
        print(f"  ℹ️  Copying files created after {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        # Copy files from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        print(f"  ℹ️  Copying files from last 24 hours")
    
    # Copy result files if they exist
    if ml_results.exists():
        copied_count = 0
        skipped_count = 0
        copied_files = []
        
        for file in ml_results.glob("*"):
            if file.is_file():
                # Get file modification time
                file_mtime = datetime.fromtimestamp(file.stat().st_mtime)
                
                # Only copy recent files
                if file_mtime >= cutoff_time:
                    shutil.copy2(file, phase8_dir / file.name)
                    copied_files.append(file.name)
                    copied_count += 1
                else:
                    skipped_count += 1
        
        if copied_count > 0:
            print(f"  ✓ Imported {copied_count} result files (skipped {skipped_count} old files)")
            print(f"    Location: {phase8_dir.relative_to(session_dir.parent.parent)}")
            
            # Create manifest of what was copied
            manifest_path = phase8_dir / "results_manifest.txt"
            with open(manifest_path, 'w') as f:
                f.write(f"Phase 8 ML Results - Run {timestamp_str}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Training timestamp: {training_timestamp or 'N/A'}\n")
                f.write(f"Import timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Files copied: {copied_count}\n")
                f.write(f"Files skipped (old): {skipped_count}\n\n")
                f.write(f"Copied files:\n")
                for fname in sorted(copied_files):
                    f.write(f"  - {fname}\n")
            
            print(f"  ✓ Created results manifest: results_manifest.txt")
        else:
            print(f"  ⚠️ No recent result files found in {ml_results}")
    else:
        print(f"  ⚠️ Results directory not found: {ml_results}")
    
    return phase8_dir


def evaluate_existing_model(session_dir: Path, config: Dict, board_name: str) -> Dict:
    """
    Evaluate existing trained model without retraining.
    
    Quick evaluation on existing model - useful for testing or re-validation.
    
    Args:
        session_dir: Current session output directory
        config: Configuration dictionary
        board_name: Board identifier
    
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*80)
    print("EVALUATING EXISTING MODEL")
    print("="*80)
    
    # Get paths
    project_root = Path(__file__).parent
    ml_path = project_root / "ml_model" / "cnn_thermal_modeling"
    
    # Look for most recent model in results/analysis_* folders
    results_dir = ml_path / "results"
    analysis_folders = sorted(results_dir.glob("analysis_*"), key=lambda p: p.name, reverse=True)
    
    model_file = None
    for folder in analysis_folders:
        candidate = folder / "unet_hbridge.keras"
        if candidate.exists():
            model_file = candidate
            break
    
    dataset_file = ml_path / "datasets" / "HBridge_cnn_dataset.h5"
    
    if model_file is None:
        print(f"\n❌ No trained model found in {results_dir / 'analysis_*'}")
        print("   Run option 1 to train a model first.")
        return {'status': 'failed', 'reason': 'no_model'}
    
    if not dataset_file.exists():
        print(f"\n❌ No dataset found: {dataset_file}")
        print("   Run option 1 to build dataset and train model.")
        return {'status': 'failed', 'reason': 'no_dataset'}
    
    print(f"\n  ✓ Found model: {model_file.relative_to(ml_path)}")
    print(f"  ✓ Found dataset: {dataset_file.name}")
    
    # Import evaluation module
    sys.path.insert(0, str(ml_path))
    try:
        import train_hbridge_model
    except ImportError as e:
        print(f"\n❌ Failed to import evaluation module: {e}")
        return {'status': 'failed', 'reason': 'import_error'}
    
    # Load model
    print("\n  ▶ Loading model...")
    sys.path.insert(0, str(project_root))
    from phase8c_spatial_cnn import SpatialCNNTrainer
    
    trainer = SpatialCNNTrainer(verbose=True)
    trainer.load_dataset(str(dataset_file))
    
    # Build model architecture (needed before loading weights)
    trainer.build_model(learning_rate=0.001)
    
    # Load trained weights
    trainer.model.load_weights(str(model_file))
    print(f"  ✓ Loaded weights from: {model_file.name}")
    
    # Run evaluation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = ml_path / "results" / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  ▶ Running evaluation (this may take 1-2 minutes)...")
    r2, rmse, mae = train_hbridge_model.evaluate_model(trainer, dataset_file, output_dir)
    
    if r2 is None:
        print("\n❌ Evaluation failed (likely empty ROI masks)")
        return {'status': 'failed', 'reason': 'evaluation_error'}
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"\nResults saved to: {output_dir.relative_to(project_root)}")
    
    # Import results
    results_dir = import_cnn_results(session_dir, ml_path, None)
    
    return {
        'status': 'success',
        'model_type': 'unet_cnn',
        'model_file': str(model_file),
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'results_dir': str(results_dir)
    }


def run_linear_regression(session_dir: Path, config: Dict, board_name: str) -> Dict:
    """
    Run legacy linear regression model.
    
    Args:
        session_dir: Current session output directory
        config: Configuration dictionary
        board_name: Board identifier
    
    Returns:
        Dictionary with model results
    """
    try:
        import phase8a_linear_regression as phase8a
        
        # Call legacy model
        # Note: This assumes phase8a has a compatible interface
        # May need to adapt based on actual phase8a_linear_regression.py structure
        print("  ⚠️ Legacy linear regression not fully integrated yet")
        print("    File renamed to phase8a_linear_regression.py but needs interface updates")
        
        return {
            'model_type': 'linear_regression',
            'status': 'legacy_model',
            'note': 'Not fully integrated - needs refactoring'
        }
    
    except ImportError as e:
        print(f"  ✗ Failed to import phase8a_linear_regression: {e}")
        return {'error': str(e)}


def compare_models(session_dir: Path, results: Dict) -> Dict:
    """
    Compare U-Net CNN vs Linear Regression performance.
    
    Args:
        session_dir: Current session output directory
        results: Dictionary with results from both models
    
    Returns:
        Comparison metrics dictionary
    """
    print("\n  ▶ Comparing models...")
    
    comparison_dir = session_dir / "phase8_model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement actual comparison logic
    # - Load metrics from both models
    # - Calculate R², RMSE, MAE
    # - Generate comparison plots
    # - Save to comparison_dir
    
    print(f"  ⚠️ Model comparison not yet implemented")
    print(f"    Placeholder directory created: {comparison_dir}")
    
    return {
        'comparison_dir': str(comparison_dir),
        'status': 'placeholder'
    }


if __name__ == "__main__":
    # Standalone test
    print("Phase 8 ML Training Module")
    print("This module is intended to be called from researchir_post_processor.py")
    print("\nFor standalone testing, use:")
    print("  python -c \"from phase8_ml_training import *; run_unet_cnn(Path('outputs/test'), {}, 'HBridge')\"")
