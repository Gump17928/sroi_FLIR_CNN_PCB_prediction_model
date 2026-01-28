# Pipeline File Dependencies - Exact Import Chain

This document shows **exactly** which files are loaded when you run each pipeline entry point.

## Entry Point 1: `researchir_post_processor.py`

### Direct Imports (Always Loaded)
```
researchir_post_processor.py
├── phase1_data_loading.py
├── phase2_filtering.py
├── phase3_component_analysis.py
├── phase4_spatial_coupling.py
├── phase5_potting_risk.py
├── viz_phase2_filtering.py
├── viz_phase3_statistics.py
├── viz_phase4_coupling.py
└── config_ui.py
```

### Conditional Imports (Phase 6-7)
```
If thermal_modeling selected:
├── phase6_thermal_calibration.py
│   └── loader_thermistor.py
├── phase7_thermal_prediction.py
│   └── loader_thermistor.py
└── loader_researchir.py
```

### Conditional Imports (Phase 8 - ML)
```
If full_pipeline selected:
├── phase8_ml_training.py
│   └── ml_model/cnn_thermal_modeling/train_hbridge_model.py
│       ├── phase8c_spatial_cnn.py
│       ├── viz_phase8c_spatial.py
│       └── (at runtime) cnn_data_preprocessor.py
├── phase8a_linear_regression.py (legacy option)
└── viz_phase8_ml_results.py
```

### Total Files Loaded (Full Pipeline):
- **Core:** 20 Python files
- **Size:** ~5-10 MB

---

## Entry Point 2: `test_phase8.py`

### Direct Imports
```
test_phase8.py
└── phase8_ml_training.py
    └── ml_model/cnn_thermal_modeling/train_hbridge_model.py
        ├── phase8c_spatial_cnn.py
        ├── viz_phase8c_spatial.py
        └── (at runtime) cnn_data_preprocessor.py
```

### With `--use_cnn_pipeline` Flag
```
test_phase8.py
└── ml_model/cnn_thermal_modeling/train_hbridge_model.py
    ├── phase8c_spatial_cnn.py
    ├── viz_phase8c_spatial.py
    ├── build_dataset.py (if --build_dataset)
    │   └── cnn_data_preprocessor.py
    └── flir_frame_loader.py (via data generator)
```

### Total Files Loaded:
- **Direct mode:** 5 Python files
- **CNN pipeline mode:** 6-7 Python files
- **Size:** ~2-3 MB

---

## Entry Point 3: `ml_model/cnn_thermal_modeling/train_hbridge_model.py` (Direct)

### Imports
```
train_hbridge_model.py
├── phase8c_spatial_cnn.py (SpatialCNNTrainer)
├── viz_phase8c_spatial.py (Phase8cVisualizer)
└── (at runtime, if --build_dataset)
    └── build_dataset.py
        └── cnn_data_preprocessor.py
            ├── flir_frame_loader.py
            └── loader_thermistor.py
```

### Total Files Loaded:
- **Without dataset build:** 3 Python files
- **With dataset build:** 6 Python files
- **Size:** ~1-2 MB

---

## Data Files Required (Not Python Code)

### Configuration Files (Always Needed)
```
├── HBridge_multi_test_config.json (10 KB)
├── LoadShedding_multi_test_config.json (15 KB)
├── component_type_patterns.json (5 KB)
└── loadshedding_thermistor_mapping.json (8 KB)
```

### Runtime Data (Downloaded Separately)
```
inputs/
├── ResearchIR_Outputs_HBridge_15s_filtered/
│   └── Rec-*.csv (120 files, ~300 MB total)
├── ResearchIR_Outputs_Load_Shedding_filtered/
│   └── Rec-*.csv (120 files, ~300 MB total)
├── Test_Air_*/
│   └── *_usb_temp_DAQami.csv
└── Test_Sand_*/
    └── *_usb_temp_DAQami.csv
```

### ROI Pixel Maps (Generated or Provided)
```
sroi_generation_ResearchIR/outputs/
├── hbridge_full_enhanced_roi_pixel_map.csv (50 KB)
└── loadshedding_ac_switch_full_enhanced_roi_pixel_map.csv (60 KB)
```

### Datasets (Generated, NOT in Git)
```
ml_model/cnn_thermal_modeling/datasets/
├── HBridge_cnn_dataset.h5 (90 MB)
└── LoadShedding_cnn_dataset.h5 (20 MB)
```

### Models (Generated, NOT in Git)
```
ml_model/cnn_thermal_modeling/results/latest/
└── unet_hbridge.keras (8 GB)
```

---

## Files NOT Used by Any Pipeline

### Diagnostic Scripts (27 files - Safe to Delete or Move to utilities/)
```
├── diagnose_cnn_dataset.py
├── diagnose_filtering.py
├── diagnose_pixel_positions.py
├── diagnose_roi_alignment.py
├── diagnose_thermistor.py
├── diagnose_time_alignment_comprehensive.py
├── diagnose_training_performance.py
├── debug_time_alignment_comprehensive.py
├── debug_training_performance.py
├── trace_coordinate_bug.py
├── verify_flir_filtering.py
├── verify_hdf5_masks.py
├── verify_roi_pixel_map.py
├── verify_roi_pixel_map_old.py
├── fix_roi_coordinates.py
├── calibrate_pcb_corners.py
├── generate_roi_pixel_map.py
├── import_roi_pixel_maps.py
├── manual_time_alignment.py
├── migrate_to_depreciated.py
├── rebuild_dataset.py
├── run_smart_pipeline.py
├── test_phase8_standalone.py
└── ml_model/cnn_thermal_modeling/
    ├── check_dataset_split.py
    ├── verify_dataset.py
    ├── verify_denormalization.py
    └── generate_predictions.py (old)
```

### Legacy ML Scripts (3 files - Superseded)
```
├── phase8a_linear_regression.py (superseded by U-Net)
├── phase8e_sensor_fusion.py (experimental)
└── validate_leave_one_component_out.py (old validation method)
```

### Documentation/Reports (Can keep for reference)
```
├── MIGRATION_REPORT_*.txt
├── GITLAB_CLEANUP_SUMMARY.md
├── ML_INTEGRATION_PLAN.txt
├── ML_INTEGRATION_REVIEW.md
├── PHASE8_FIXES_SUMMARY.md
├── PHASE8_VISUALIZATION_INTEGRATION.md
└── SUBPLOT_TRUNCATION_GUIDE.md
```

---

## Recommended File Organization

### Keep (Required for Pipeline - 32 files)
```
Core Pipeline (13):
- researchir_post_processor.py
- test_phase8.py
- config_ui.py
- phase1-7 (7 files)
- phase8_ml_training.py
- cnn_data_preprocessor.py
- flir_frame_loader.py

Visualization (5):
- viz_phase2-4 (3 files)
- viz_phase8_ml_results.py
- viz_phase8c_spatial.py

Loaders (2):
- loader_researchir.py
- loader_thermistor.py

ML Core (7):
- ml_model/cnn_thermal_modeling/train_hbridge_model.py
- ml_model/cnn_thermal_modeling/build_dataset.py
- ml_model/cnn_thermal_modeling/re_evaluate_model.py
- ml_model/cnn_thermal_modeling/create_thermal_animation.py
- ml_model/cnn_thermal_modeling/diagnose_model_predictions.py
- viz_phase8_compare_csv_vs_hdf5.py
- ml_model/cnn_thermal_modeling/phase8c_spatial_cnn.py

Configuration (4):
- *.json config files

Documentation (5):
- README.md
- ACTIVE_FILE_STRUCTURE.md
- CROSS_PCB_VALIDATION_ANALYSIS.txt
- CROSS_PCB_TRAINING_GUIDE.md
- CORNER_CALIBRATION_GUIDE.md
```

### Move to utilities/ (30 files)
- All diagnose_*.py
- All verify_*.py
- All debug_*.py
- All fix_*.py
- Legacy scripts

### Delete (Safe to Remove - 5 files)
- MIGRATION_REPORT_*.txt (after verified)
- Old backup configs
- Duplicate scripts (*_old.py)

---

## Git Repository Size Breakdown

### With All Code (No Data)
```
Python files (48):           ~8 MB
Configuration (4):           ~1 MB
Documentation (10):          ~1 MB
------------------------------------
Total:                      ~10 MB ✓
```

### With Minimal Code (After Cleanup)
```
Python files (32):           ~5 MB
Configuration (4):           ~1 MB
Documentation (5):           ~500 KB
------------------------------------
Total:                      ~7 MB ✓✓
```

### Current (Before Cleanup)
```
Code:                       ~10 MB
Outputs:                     ~5 GB
Models:                     ~8 GB
Datasets:                   ~110 MB
Inputs:                     ~1 GB
Logs/temps:                 ~12 GB
------------------------------------
Total:                      ~26 GB ✗
```

**Recommendation:** Clean up 27 diagnostic files + exclude outputs/models/datasets = **~7 MB Git repo** ✓

