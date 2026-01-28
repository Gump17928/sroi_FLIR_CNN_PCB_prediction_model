# Active File Structure - Main Pipeline

This document shows **ONLY** the files actively used by the main pipeline (`researchir_post_processor.py`) and ML training (`test_phase8.py`).

## Core Pipeline Files (Required)

### Main Entry Points
```
thermal_post_processing/
├── researchir_post_processor.py    # Main pipeline entry point
├── test_phase8.py                  # Standalone ML training entry
└── config_ui.py                    # Interactive configuration UI
```

### Phase Modules (Core Pipeline)
```
├── phase1_data_loading.py          # Load FLIR + thermistor data
├── phase2_filtering.py             # Temporal median filtering
├── phase3_component_analysis.py    # Component statistics
├── phase4_spatial_coupling.py      # Thermal coupling analysis
├── phase5_potting_risk.py          # Potting compound risk assessment
├── phase6_thermal_calibration.py   # Calibrate thermistor offsets
├── phase7_thermal_prediction.py    # Generate timeseries predictions
└── phase8_ml_training.py           # ML model training wrapper
```

### Visualization Modules (Core)
```
├── viz_phase2_filtering.py         # Filtering verification plots
├── viz_phase3_statistics.py        # Component statistics plots
├── viz_phase4_coupling.py          # Coupling heatmaps
└── viz_phase8_ml_results.py        # ML training results plots
```

### Data Loaders
```
├── loader_researchir.py            # Parse ResearchIR FLIR stats
├── loader_thermistor.py            # Load thermistor CSV data
└── flir_frame_loader.py            # Load FLIR frame sequences
```

### ML Training Infrastructure
```
ml_model/
└── cnn_thermal_modeling/
    ├── train_hbridge_model.py      # Main training script (interactive + CLI)
    ├── build_dataset.py            # Generic dataset builder (HBridge/LoadShedding)
    ├── re_evaluate_model.py        # Re-run evaluation on saved model
    ├── create_thermal_animation.py # Generate GIF/MP4 animations
    ├── diagnose_model_predictions.py  # Diagnostic analysis
    └── viz_phase8_compare_csv_vs_hdf5.py      # ROI alignment verification
```

### Core ML Modules (Imported by training)
```
ml_model/cnn_thermal_modeling/
├── phase8c_spatial_cnn.py          # U-Net model definition + trainer
├── viz_phase8c_spatial.py          # Spatial CNN visualization
└── cnn_data_preprocessor.py        # Dataset building (in parent dir)
```

### Configuration Files
```
├── HBridge_multi_test_config.json         # HBridge test configuration
├── LoadShedding_multi_test_config.json    # LoadShedding test configuration
├── component_type_patterns.json           # Component type classification
└── loadshedding_thermistor_mapping.json   # LoadShedding thermistor map
```

### Documentation (Active)
```
├── README.md                                      # Main documentation
├── CROSS_PCB_VALIDATION_ANALYSIS.txt             # Cross-PCB analysis plan
└── ml_model/cnn_thermal_modeling/
    ├── CROSS_PCB_TRAINING_GUIDE.md                # Training guide
    └── test_interactive_prompts.py                # UI demo
```

---

## Optional/Utility Files (Can be Removed for GitHub)

### Legacy ML Scripts (Not in pipeline)
```
├── phase8a_linear_regression.py    # Old linear regression (superseded by U-Net)
├── phase8e_sensor_fusion.py        # Experimental sensor fusion
├── rebuild_dataset.py              # Old dataset builder
└── run_smart_pipeline.py           # Deprecated pipeline runner
```

### Diagnostic/Debug Scripts (Useful but not required)
```
├── diagnose_*.py                   # Various diagnostic scripts (13 files)
├── verify_*.py                     # Verification scripts (6 files)
├── debug_*.py                      # Debug scripts (3 files)
├── trace_*.py                      # Tracing scripts
├── fix_*.py                        # One-time fix scripts
└── compare_*.py                    # Comparison utilities
```

### Test Scripts (Development only)
```
├── test_phase8_standalone.py       # Standalone Phase 8 test
├── test_phase8.py                  # (THIS ONE IS ACTIVE - keep!)
└── ml_model/cnn_thermal_modeling/
    ├── check_dataset_split.py
    ├── verify_dataset.py
    └── verify_denormalization.py
```

### Visualization Utilities (Optional extras)
```
├── viz_helpers.py                  # Shared viz utilities
├── viz_phase6_validation.py        # Phase 6 validation plots
└── viz_phase8_dataset_validation.py  # Dataset validation plots
```

### Migration/Maintenance Scripts (Can delete)
```
├── migrate_to_depreciated.py
├── import_roi_pixel_maps.py
├── calibrate_pcb_corners.py
├── generate_roi_pixel_map.py
├── manual_time_alignment.py
└── MIGRATION_REPORT_*.txt
```

---

## Large Files to Exclude from GitHub

### Model Files (Too large - 8GB each)
```
ml_model/cnn_thermal_modeling/results/
├── latest/unet_hbridge.keras          # 8GB - exclude
├── analysis_*/unet_hbridge.keras      # 8GB each - exclude
└── */training_history.npz             # Keep (small)
```

### Datasets (Large - 90-110MB total, could keep compressed)
```
ml_model/cnn_thermal_modeling/datasets/
├── HBridge_cnn_dataset.h5             # 90MB
└── LoadShedding_cnn_dataset.h5        # 20MB
```

### Output Directories (Exclude all - regenerated on run)
```
outputs/                               # All test outputs (large)
├── *_P1-7/                           # Session outputs
├── latest/                            # Latest run
└── *.png, *.csv, *.txt               # All generated files
```

### Input Data (Large - should be downloaded separately)
```
inputs/
├── ResearchIR_Outputs_*/             # FLIR frames (100s of MB each)
├── Test_*_Load_Shedding/             # Thermistor CSVs
└── Test_*_HBridge/                    # Thermistor CSVs
```

---

## Git Repository Size Reduction Strategy

### .gitignore Additions (Exclude these)
```gitignore
# Model files (too large)
*.keras
*.h5

# Outputs (regenerated)
outputs/
results/

# Input data (provide download link)
inputs/ResearchIR_Outputs_*/
inputs/Test_*/

# Python cache
__pycache__/
*.pyc
.venv/
venv/

# Deprecated
depreciated/
deprecated/
```

### Files to Keep in Git (<100MB total)
- All `.py` core pipeline files (48 files ≈ 5MB)
- Configuration JSON files (≈ 1MB)
- Documentation (README.md, guides)
- Small example outputs (optional, <10MB)

### Files to Provide via External Link
- Trained models (8GB) → Google Drive/Zenodo
- Datasets (110MB) → Could include compressed or external
- Input data (1GB+) → Provide download instructions

---

## Recommended File Cleanup

### Safe to Delete (Not in pipeline)
1. All `diagnose_*.py` except `diagnose_model_predictions.py`
2. All `verify_*.py` except those in ml_model/
3. All `debug_*.py`
4. `migrate_to_depreciated.py`
5. `rebuild_dataset.py`
6. `phase8a_linear_regression.py` (superseded)
7. `phase8e_sensor_fusion.py` (experimental)
8. All `MIGRATION_REPORT_*.txt`
9. `run_smart_pipeline.py`

### Move to `utilities/` folder (Optional tools)
1. `viz_phase8_compare_csv_vs_hdf5.py`
2. `calibrate_pcb_corners.py`
3. `generate_roi_pixel_map.py`
4. `manual_time_alignment.py`
5. `import_roi_pixel_maps.py`

### Keep (Active in pipeline)
- All `phase*.py` files
- All `viz_phase*.py` files
- All `loader_*.py` files
- `config_ui.py`
- `test_phase8.py`
- ML training scripts in `ml_model/cnn_thermal_modeling/`

---

## Final Repository Structure (Git-friendly, <100MB)

```
thermal_post_processing/
├── README.md
├── ACTIVE_FILE_STRUCTURE.md
├── .gitignore
│
├── Core Pipeline (13 files)
│   ├── researchir_post_processor.py
│   ├── test_phase8.py
│   ├── config_ui.py
│   ├── phase1_data_loading.py
│   ├── phase2_filtering.py
│   ├── phase3_component_analysis.py
│   ├── phase4_spatial_coupling.py
│   ├── phase5_potting_risk.py
│   ├── phase6_thermal_calibration.py
│   ├── phase7_thermal_prediction.py
│   ├── phase8_ml_training.py
│   ├── cnn_data_preprocessor.py
│   └── flir_frame_loader.py
│
├── Visualization (5 files)
│   ├── viz_phase2_filtering.py
│   ├── viz_phase3_statistics.py
│   ├── viz_phase4_coupling.py
│   ├── viz_phase8_ml_results.py
│   └── viz_phase8c_spatial.py
│
├── Data Loaders (2 files)
│   ├── loader_researchir.py
│   └── loader_thermistor.py
│
├── Configuration (4 files)
│   ├── HBridge_multi_test_config.json
│   ├── LoadShedding_multi_test_config.json
│   ├── component_type_patterns.json
│   └── loadshedding_thermistor_mapping.json
│
├── ml_model/cnn_thermal_modeling/ (7 core files)
│   ├── train_hbridge_model.py
│   ├── build_dataset.py
│   ├── re_evaluate_model.py
│   ├── create_thermal_animation.py
│   ├── diagnose_model_predictions.py
│   ├── viz_phase8_compare_csv_vs_hdf5.py
│   ├── phase8c_spatial_cnn.py
│   ├── CROSS_PCB_TRAINING_GUIDE.md
│   └── CROSS_PCB_VALIDATION_ANALYSIS.txt
│
├── utilities/ (optional tools)
│   └── (moved diagnostic/utility scripts)
│
└── docs/ (documentation)
    └── (migration reports, old guides)
```

**Total Size:** ~30-50MB (without outputs, models, or input data)
**GitHub Compatible:** ✓ Yes
