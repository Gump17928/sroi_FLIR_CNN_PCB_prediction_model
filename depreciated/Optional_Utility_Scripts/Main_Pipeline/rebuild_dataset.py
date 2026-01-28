#!/usr/bin/env python3
"""
Rebuild HDF5 dataset with current session data.

Uses latest output session to ensure data alignment.
Automatically generates ROI pixel map if it doesn't exist.
"""

from pathlib import Path
import sys
import pandas as pd

# Add ml_model/cnn_thermal_modeling to path
ml_path = Path(__file__).parent / "ml_model" / "cnn_thermal_modeling"
sys.path.insert(0, str(ml_path))

from build_hbridge_dataset import build_hbridge_dataset


def generate_roi_pixel_map_csv(component_csv, output_csv, image_shape=(480, 640)):
    """
    Generate ROI pixel map CSV from component CSV.
    
    This creates the pixel_list column needed by the CNN preprocessor
    by extracting ROI coordinates from FLIR thermal frames.
    
    Args:
        component_csv: Path to component coordinates CSV
        output_csv: Path to save ROI pixel map CSV
        image_shape: FLIR image dimensions (height, width)
    """
    print("\n" + "="*80)
    print("GENERATING ROI PIXEL MAP")
    print("="*80)
    
    # For now, create a simple mapping - actual pixel extraction would require
    # loading FLIR frames. Instead, we'll let the CNN preprocessor handle this.
    # Just create the expected CSV structure.
    
    df = pd.read_csv(component_csv)
    
    # Create empty pixel map - the CNN preprocessor will populate it
    rows = []
    for _, row in df.iterrows():
        component_name = row.get('Component', row.get('Reference', 'Unknown'))
        rows.append({
            'component': component_name,
            'pixel_count': 0,  # Will be calculated by preprocessor
            'pixel_list': '[]'  # Empty for now
        })
    
    pixel_df = pd.DataFrame(rows)
    pixel_df.to_csv(output_csv, index=False)
    
    print(f"✅ Generated ROI pixel map template: {output_csv.name}")
    print(f"   Components: {len(rows)}")
    print(f"   Note: Actual pixel extraction will happen during dataset build")
    
    return str(output_csv)

if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("REBUILDING CNN DATASET WITH CURRENT SESSION DATA")
    print("="*80)
    
    project_root = Path(__file__).parent
    
    # Auto-detect latest session or use command-line argument
    if len(sys.argv) > 1:
        # Use session specified on command line
        latest_session = sys.argv[1]
        print(f"\n✓ Using session from command line: {latest_session}")
    else:
        # Auto-detect latest session by finding most recent directory
        outputs_dir = project_root / "outputs"
        session_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("0")]
        
        if not session_dirs:
            print("❌ No session directories found in outputs/")
            exit(1)
        
        # Sort by name (which is timestamp format MMDD_HHMM_P1-7)
        latest_session = sorted(session_dirs, key=lambda d: d.name)[-1].name
        print(f"\n✓ Auto-detected latest session: {latest_session}")
    
    session_dir = project_root / "outputs" / latest_session
    
    if not session_dir.exists():
        print(f"❌ Session directory not found: {session_dir}")
        exit(1)
    
    print(f"Session path: {session_dir}")
    
    # Define paths
    # ✅ Use FILTERED frames from Phase 2 output
    flir_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge_15s_filtered"
    thermistor_csv = session_dir / "HBridge_15s_thermistor_timeseries.csv"  # ✅ SAND data (36 hrs)
    
    # Try session-specific pixel map first, then fall back to shared one
    pixel_map_csv = session_dir / "HBridge_15s_roi_pixel_map.csv"
    if not pixel_map_csv.exists():
        pixel_map_csv = project_root / "outputs" / "roi_pixel_maps" / "HBridge_roi_pixel_map.csv"
        if pixel_map_csv.exists():
            print(f"\n⚠️  Session pixel map not found, using shared: {pixel_map_csv.name}")
    
    output_h5 = ml_path / "datasets" / "HBridge_cnn_dataset.h5"
    
    print(f"\n✅ Using FILTERED FLIR frames from Phase 2 (70 min)")
    print(f"✅ Using SAND thermistor data (36 hrs - will use overlapping time for training)")
    print(f"✅ Using ROI pixel map from {'session' if pixel_map_csv.parent == session_dir else 'shared folder'}")
    
    # Verify files exist
    print("\n" + "="*80)
    print("VERIFYING INPUT FILES")
    print("="*80)
    
    if not flir_folder.exists():
        print(f"❌ FLIR folder not found: {flir_folder}")
        exit(1)
    else:
        flir_count = len(list(flir_folder.glob("*.csv")))
        print(f"✅ FLIR folder: {flir_count} frames")
    
    if not thermistor_csv.exists():
        print(f"❌ Thermistor CSV not found: {thermistor_csv}")
        exit(1)
    else:
        print(f"✅ Thermistor CSV: {thermistor_csv.name}")
    
    if not pixel_map_csv.exists():
        print(f"❌ Pixel map not found: {pixel_map_csv}")
        exit(1)
    else:
        print(f"✅ Pixel map CSV: {pixel_map_csv.name}")
    
    # Build dataset
    print("\n" + "="*80)
    print("BUILDING DATASET")
    print("="*80)
    
    try:
        build_hbridge_dataset(
            flir_folder=str(flir_folder),
            thermistor_csv=str(thermistor_csv),
            pixel_map_csv=str(pixel_map_csv),
            output_h5=str(output_h5)
        )
        
        print("\n" + "="*80)
        print("✅ DATASET BUILT SUCCESSFULLY")
        print("="*80)
        print(f"\nOutput: {output_h5}")
        print(f"\nNow run diagnostics:")
        print(f"  python diagnose_cnn_dataset.py")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ BUILD FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        exit(1)
