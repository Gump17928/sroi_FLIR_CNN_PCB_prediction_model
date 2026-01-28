# SROI FLIR CNN PCB Prediction Model

**Complete 8-Phase Thermal Analysis Pipeline with U-Net CNN Machine Learning**  
*ResearchIR Data Analysis • Thermistor Validation • Thermal Modeling • Spatial CNN Prediction*

The sroi_FLIR_CNN_PCB_prediction_model is a comprehensive thermal post-processing system for learning about and reverse engineering the .sroi files that exist for FLIR cameras, and utilizing advanced U-Net CNN models to predict embedded thermal temperatures of PCBs using full spatial thermal imaging.

**📁 Repository Structure:** See [ACTIVE_FILE_STRUCTURE.md](ACTIVE_FILE_STRUCTURE.md) for detailed file organization and GitHub size management.

---

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Analysis Phases](#analysis-phases-1-5)
- [Thermal Modeling](#thermal-modeling-phases-6-7)
- [Machine Learning](#machine-learning-phase-8)
- [File Management](#file-management-for-github)

---

## Overview

Comprehensive thermal post-processing for PCB analysis using FLIR thermal camera and thermistor measurements. Provides 8 phases from raw data processing through U-Net CNN spatial thermal prediction.

**Key Capabilities:**
- **Phases 1-5**: Thermal analysis (filtering, statistics, spatial coupling, risk assessment)
- **Phase 6**: Calibration using thermistor-validated measurements
- **Phase 7**: Temperature prediction for FLIR-only PCBs (statistical model)
- **Phase 8 (NEW)**: U-Net CNN spatial thermal prediction with full-frame filtering

**Folder Structure:**

```
thermal_post_processing/
├── inputs/                          # Input data
│   ├── ResearchIR_Outputs_*/       # FLIR CSV exports (raw)
│   ├── ResearchIR_Outputs_*_filtered/ # (NEW) Filtered FLIR frames for ML
│   ├── *_usb_temp_DAQami.csv       # Thermistor data
│   ├── *_components_enhanced.csv   # Coordinates
│   └── *_thermistor_mapping.json   # Channel mapping
├── outputs/                         # Results (auto-timestamped)
│   ├── MMDD_HHMM_P1-5/             # Phases 1-5 + CSV exports
│   ├── MMDD_HHMM_P6-7/             # Phases 6-7
│   ├── MMDD_HHMM_P1-8/             # (NEW) Complete with ML
│   └── roi_pixel_maps/             # ROI masks for CNN
├── ml_model/                        # (NEW) U-Net CNN module
│   └── cnn_thermal_modeling/
│       ├── build_hbridge_dataset.py
│       ├── train_hbridge_model.py
│       ├── datasets/               # HDF5 datasets
│       ├── models/                 # Trained models
│       └── results/                # Predictions
├── researchir_post_processor.py    # Main entry
├── phase[1-7]_*.py                 # Analysis modules
├── phase8_ml_training.py           # (NEW) U-Net CNN wrapper
├── phase8a_linear_regression.py    # Legacy linear model
└── README.md                        # This file
```

---

## Quick Start

**Pre-Processing (Recommended for ML):**
```powershell
python researchir_post_processor.py
# Select: [F] Filter FLIR frames for ML training
# This creates filtered frames with temporal median noise reduction
```

**Interactive Mode (Recommended):**
```powershell
python researchir_post_processor.py
```
- Select workflow (Phases 1-5, 6-7, 1-7, or 1-8), validate paths, toggle debug mode

**Command-Line Mode:**
```powershell
python researchir_post_processor.py --no_ui              # Phases 1-5
python researchir_post_processor.py --thermal_modeling   # Phases 6-7
python researchir_post_processor.py --full_pipeline      # Phases 1-8 (with ML)
python researchir_post_processor.py --full_pipeline --debug  # Verbose output
```

**Output Folders:** Auto-generated as `MMDD_HHMM_<workflow>[_debug]` (e.g., `0115_1430_P1-8`). Override with `--output` flag.

## System Architecture

**Four Workflows:**
1. **Phases 1-5**: Standard analysis (loading, filtering, statistics, spatial coupling, risk)
2. **Phases 6-7**: Thermal modeling (calibration & prediction)
3. **Full Pipeline (1-7)**: Complete thermal analysis without ML
4. **Full Pipeline (1-8) [NEW]**: Complete analysis + U-Net CNN spatial prediction

## Analysis Phases (1-5)

| Phase | Purpose | Key Features | Outputs |
|-------|---------|--------------|---------|
| **1. Data Loading** | Load ResearchIR exports | Auto-detect CSV/TXT, validate data, classify components | Validated dataset |
| **2. Filtering** | Compare filter methods & prepare for ML | Component-level median filtering, **full-frame filtering for CNN** | `*_phase2_filtered_temperatures.csv`, `*_phase2_filter_comparison.csv`, filtered FLIR frames |
| **3. Component Analysis** | Statistical profiling | Group by type, calc stats, identify hotspots, IEEE plots | `*_phase3_component_statistics.csv`, `*_phase3_component_metadata.csv`, MATLAB export |
| **4. Spatial Coupling** | Thermal interactions | Proximity detection (10mm), coupling strength, heat sources | `*_phase4_spatial_coupling.csv`, coupling plots |
| **5. Potting Risk** | Failure prediction | Risk scoring for potted systems (requires Phase 4) | `potted_condition_risk_analysis.csv` |

**NEW in Phase 2: FLIR Frame Filtering for ML**
- **Purpose:** Remove camera refocusing artifacts and noise from full 480×640 thermal frames
- **Algorithm:** Pixel-by-pixel temporal median filter (kernel_size=5)
- **Input:** Raw FLIR frames (300 frames × 480×640 pixels = 1.4 GB)
- **Output:** Filtered frames (799 MB, 30% smaller with cleaner data)
- **Storage:** `inputs/ResearchIR_Outputs_*_filtered/` (automatically detected by Phase 8)
- **Processing:** ~4 minutes per board, 368 MB RAM usage
- **Usage:** Pre-processing menu → [F] Filter FLIR frames for ML training

**Verification:**
After filtering, verify quality with:
```bash
python verify_flir_filtering.py
```

This generates a comprehensive 2x2 comparison plot showing:
- Raw and filtered frames side-by-side
- **Absolute difference map** (highlights where filtering changed temperatures)
- Temperature distribution histograms
- Statistical metrics (range preservation, noise reduction, mean/max differences)

The verification is also available programmatically:
```python
import viz_phase2_filtering as viz

results = viz.verify_flir_filtering_quality(
    raw_folder='inputs/ResearchIR_Outputs_HBridge_15s',
    filtered_folder='inputs/ResearchIR_Outputs_HBridge_15s_filtered',
    num_samples=20,
    output_dir='outputs'
)
```

**NEW: CSV Exports**
All phases now export intermediate results to CSV for standalone analysis and visualization:
- **Phase 2:** Filtered component temperatures, raw vs filtered comparison
- **Phase 3:** Component statistics (mean, max, min, std, temp rise), metadata (ROI pixels, area, centroid)
- **Phase 4:** Spatial coupling matrix (pairwise distance, correlation, p-value)

## Thermal Modeling (Phases 6-7)

### Phase 6: Calibration
**Purpose:** Learn FLIR correction offsets from thermistor-validated measurements

**Process:** Compare FLIR to thermistor → Calculate type-specific offsets → Model air/sand cooling → Export database

**Inputs:** Calibration PCB ResearchIR Stats, thermistor air/sand CSVs, mapping JSON

**Calibration Modes:**
- **Standard:** Single test session, replace existing calibration
- **Incremental:** Append new measurements to existing database (week-by-week data collection)
- **Multi-Session:** Batch process multiple test sessions in single run

**Key Results:**
| Type | FLIR Offset | Air SS | Cooling Benefit | Ratio |
|------|-------------|--------|-----------------|-------|
| LDO | +2.89°C | 24.61°C | -1.04°C | 1.0423 |
| PowerSupply | +3.37°C | 29.49°C | +1.73°C | 0.9413 |
| Resistor | +2.00°C | 24.57°C | +0.04°C | 0.9985 |
| LED | +2.60°C | 24.49°C | -0.56°C | 1.0228 |

**Outputs:** `thermal_calibration_*.csv/json/png`, `calibration_database/detailed_plots/` (PNG+PDF), `convergence_analysis/` (optional)

### Phase 7: Prediction
**Purpose:** Apply calibration to predict temperatures on FLIR-only PCBs

**Workflow:**
```
1. Corrected_Air = FLIR_Raw - FLIR_Offset[type]
2. Predicted_Sand = Corrected_Air × Cooling_Ratio[type]
3. Confidence = High (σ<2°C, multiple samples) | Medium (σ≥2°C) | Low (no cal)
```

**Inputs:** Prediction PCB ResearchIR Stats, calibration database

**Outputs:** `thermal_prediction_<PCB>.csv/json/png`

### Phase 8: U-Net CNN Sensor Fusion Thermal Prediction (NEW)
**Purpose:** Predict full PCB thermal field by fusing FLIR surface measurements with sparse embedded thermistor readings

---

**THE PROBLEM:**
- FLIR camera sees surface temps, but we need to know embedded component temps under potting
- Thermistors measure embedded temps but only for 22 components (we want all components)
- Surface temps alone don't predict embedded temps well (correlation only 0.12)

**THE SOLUTION:**
Use BOTH data sources together:
1. FLIR gives surface temps for ALL 307,200 pixels (complete spatial coverage)
2. Thermistors give embedded temps for 22 ROI locations (sparse but accurate)
3. U-Net learns: "Given surface pattern X and nearby embedded measurements Y, predict embedded temp Z"
4. Model fills in unmeasured regions by learning thermal coupling patterns

**ANALOGY:**
Like predicting the temperature inside a wall by combining:
- Infrared camera showing surface (FLIR)
- A few thermometers inserted through holes (thermistors)
- Learning how surface patterns relate to internal temps from training data

---

**DATA FLOW DIAGRAM:**
```
TRAINING:
┌─────────────────────────────────────────────────────────────┐
│ Input Frame #1                                              │
│                                                             │
│  Channel 0: FLIR Surface [480×640]                         │
│  ┌──────────────────┐                                      │
│  │ 25.3  26.1  27.5 │ ← All pixels measured                │
│  │ 26.8  28.2  29.1 │                                       │
│  └──────────────────┘                                       │
│                                                             │
│  Channel 1: Thermistor Embedded [480×640, sparse]         │
│  ┌──────────────────┐                                      │
│  │  0.0  34.2   0.0 │ ← Only ROI pixels non-zero           │
│  │  0.0   0.0  41.8 │                                       │
│  └──────────────────┘                                       │
│         ↓                                                   │
│    U-Net CNN (learns spatial patterns)                     │
│         ↓                                                   │
│  Output: Predicted Embedded Temps [480×640]                │
│  ┌──────────────────┐                                      │
│  │ 32.1  34.2  38.5 │ ← Predicted for ALL pixels           │
│  │ 35.7  39.2  41.8 │                                       │
│  └──────────────────┘                                       │
│                                                             │
│  Loss: Compare prediction vs thermistor at ROI pixels only │
└─────────────────────────────────────────────────────────────┘

INFERENCE (Future Boards):
- Same FLIR + thermistor inputs
- Model predicts embedded temps everywhere
- Can reduce number of thermistors needed for validation
```

---

**Core Concept:** Sensor Fusion
- Combine **complete** FLIR surface temperature data (480×640 pixels, all components visible)
- With **sparse** embedded thermistor data (22 components measured, painted onto ROI pixels)
- To predict **full** embedded thermal field (all 307,200 pixels)

**Architecture:** U-Net CNN with dual-channel input for sensor fusion

```
INPUT (2 Channels):
  Channel 0: FLIR Surface Temperatures [480×640]
            - Complete thermal image from camera
            - Range: 18.9°C - 63.3°C
            
  Channel 1: Thermistor Embedded Temperatures [480×640, sparse]
            - Thermistor readings painted onto ROI pixels
            - Zero elsewhere (no measurement available)
            - Range: 0.0°C - 48.8°C at measured locations
            
OUTPUT (1 Channel):
  Predicted Thermal Map [480×640]
            - Full embedded temperature field
            - Leverages thermal coupling between components
            - Fills in unmeasured regions using learned spatial patterns

U-NET ARCHITECTURE:
  Encoder: 4 blocks (Conv→Conv→MaxPool) - learns spatial features
  Bottleneck: 512 filters - compressed representation
  Decoder: 4 blocks (UpConv→Concat→Conv) - reconstructs spatial field
  Loss: Masked MSE (penalizes only at ROI pixels where ground truth exists)
```

**Key Innovation:**
The model learns that components are thermally coupled (correlation 0.95-0.99). By knowing:
- Surface temps of ALL components (from FLIR)
- Embedded temps of SOME components (from thermistors)

It can predict embedded temps of components WITHOUT thermistors by learning the spatial thermal coupling patterns.

**Training Strategy:**
- Component-level split: 80% train / 20% validation (random, seed=42)
- Training components: Thermistor data in input Channel 1
- Validation components: Excluded from input, predicted from FLIR + other thermistors
- Masked loss: only penalize predictions at ROI locations
- Tests if model learns thermal coupling vs copying thermistor inputs

**Validation Philosophy:**
Unlike traditional temporal holdout, this uses **component-level validation**:
- **Old approach (temporal):** Train on early timesteps, validate on late timesteps
  - Problem: Model sees all components, just at different times
  - Doesn't test if model learns thermal physics
  
- **New approach (component-level):** Train with some components, validate on held-out components
  - Input: FLIR (all components) + thermistors (training components only)
  - Target: Predict temperatures of validation components
  - Tests: Can model infer unmeasured component temps from thermal coupling?
  - Use case: Reduce thermistor count on future boards

**Workflow:**
```
1. Pre-process: Filter FLIR frames (Phase 2 pre-processing menu)
2. Build Dataset: Load frames + thermistor data + ROI masks → HDF5
3. Train Model: U-Net CNN learns FLIR+Thermistor→Thermal_Field mapping
4. Predict: Generate full spatial thermal predictions
5. Import: Copy results to session outputs/phase8_ml_results/
```

**Interactive Menu:**
```
================================================================================
PHASE 8: MACHINE LEARNING MODEL TRAINING
================================================================================

Available Models:
  [1] U-Net CNN (spatial thermal prediction) [RECOMMENDED]
  [2] Linear Regression (delta-T method - legacy)
  [3] Both (comparison mode)
  [4] Skip ML training

Your choice (1-4) [default: 4]:
```

**Inputs:**
- **FLIR Frames:** `inputs/ResearchIR_Outputs_*_filtered/` (auto-detected, 300 frames × 480×640)
  - Each pixel: surface temperature measured by camera
  - Pre-filtered to remove camera artifacts
  
- **Thermistor Data:** `outputs/TIMESTAMP/*_thermistor_timeseries.csv` (300 frames × 22 components)
  - Embedded temperatures measured by sensors under potting compound
  - Sparse: only 22 out of 307,200 pixels have ground truth
  
- **ROI Pixel Map:** `outputs/roi_pixel_maps/*_roi_pixel_map.csv`
  - Defines which pixels belong to each component
  - Used to paint thermistor temps onto spatial map

**How Inputs Are Combined:**
```python
# For each training sample (frame):
X[frame, :, :, 0] = FLIR_surface_temps        # Channel 0: all 307,200 pixels
X[frame, :, :, 1] = Thermistor_embedded_temps # Channel 1: sparse (only ROI pixels non-zero)

# Target output:
y[frame, :, :, 0] = Full_embedded_thermal_map # What we want to predict
```

**Training Process:**
1. Model sees FLIR surface + sparse embedded temps as input
2. Learns spatial patterns: "when surface shows X and nearby thermistors show Y, embedded temp is Z"
3. Loss function penalizes errors only at ROI locations (where we have ground truth)
4. Model learns to fill in unmeasured regions using learned thermal coupling

**Evaluation Metrics:**
- **R² Score:** How well predictions match thermistor ground truth (goal: > 0.7)
  - Computed at ROI pixels where thermistors exist
  - R² = 1.0 means perfect prediction
  - R² < 0 means model worse than just guessing average temp
  
- **MAE (Mean Absolute Error):** Average temperature error in °C (goal: < 3°C)
  - Only computed at ROI locations with thermistor ground truth
  - Example: MAE = 2.5°C means predictions are off by ±2.5°C on average
  
- **RMSE:** Root mean squared error for outlier sensitivity
  - Penalizes large errors more than MAE
  
- **Masked Loss:** Custom loss function used during training
  - Only penalizes errors at ROI pixels (where we have ground truth)
  - Allows model to learn full-field patterns while being supervised only at sparse locations
  
**How The Model Learns the Relationship:**
1. **Spatial Convolutions:** U-Net learns local thermal patterns
   - "Hot surface + high thermal mass → slower embedded heating"
   - "Component clusters → thermal coupling effects"
   
2. **Multi-Scale Features:** Encoder-decoder captures both:
   - Fine details (individual component behavior)
   - Global context (board-level thermal distribution)
   
3. **Skip Connections:** Preserve spatial information during upsampling
   - Ensures predictions align with component locations
   
4. **Masked Loss:** Focus learning on ROI locations
   - Model learns to predict embedded temps where thermistors exist
   - Then generalizes to unmeasured regions using learned patterns

**Why This Works:**
- Components are thermally coupled (measured correlation: 0.95-0.99)
- Surface temps + nearby embedded temps → good predictor for unmeasured embedded temps
- U-Net learns complex non-linear thermal transfer relationships
- More training data (across boards) improves generalization

**Outputs:**
- HDF5 Dataset: `ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5` (87.99 MB)
- Trained Model: `ml_model/cnn_thermal_modeling/models/thermal_unet_model.keras`
- Predictions: `outputs/TIMESTAMP/phase8_ml_results/` (imported from ml_model/results/)
- Visualizations: Training curves, prediction comparisons, spatial heatmaps

**Performance:**
- Dataset build: ~4 minutes (loads 300 frames, 368 MB RAM)
- Model training: ~30-60 minutes (first time), reusable for future sessions
- Prediction: ~1 minute

**Advantages over Phase 7:**
- Captures spatial thermal coupling between components
- Learns complex non-linear relationships
- Uses full thermal context (not just component-level statistics)
- Improves accuracy with more training data

---

#### Memory Requirements

**Training Mode Comparison:**

| Training Mode | RAM Usage | Speed | Use Case |
|--------------|-----------|-------|----------|
| **Generator** (default) | ~300-500 MB | Moderate | Large datasets, limited RAM, laptops |
| **In-Memory** | ~31.5 GB | Fast | Small datasets, high-RAM servers |

**Generator breakdown:**
- Per-batch data: ~10 MB (8 samples × 2 channels × 240×320 pixels)
- U-Net model: ~20-40 MB (parameters + gradients)
- TensorFlow overhead: ~100-200 MB (optimizer states, graph compilation)
- HDF5 metadata: ~50-100 MB (file mapping, caching)

**Recommended system specs:**
- Minimum: 2 GB RAM (will work but tight)
- Comfortable: 4 GB RAM (leaves room for OS)
- Optimal: 8+ GB RAM (smooth operation)

The generator achieves **~60-100x memory reduction** by loading only 8 samples at a time.

---

#### Hybrid Batch Sampling: Optimizing Training Time vs Accuracy

**Purpose:** Reduce training time by sampling more densely during transient periods (early data) and sparsely during steady-state periods.

**Configuration uses time-based parameters (in seconds)** for clarity and flexibility across different datasets.

**The Challenge:**
When extending FLIR data to match thermistor duration (300 frames → 8719 timestamps using steady-state extension):
- **All frames used:** 8719 batches per epoch → very long training time
- **Skip too many:** Miss important transient behavior → poor accuracy

**The Solution: Hybrid Sampling**
Sample densely where data changes rapidly (transients), sparsely where it's stable (steady-state).

**⚠️ CRITICAL: Component-Level Train/Val Split (NEW)**

The implementation uses **component-level validation** instead of temporal validation to properly test if the model learns thermal physics:

**Why Component-Level Split?**
- **Temporal split problem:** Model sees all thermistor inputs, learns to copy them at different times
- **Component split solution:** Model must predict held-out components from FLIR + thermal coupling

**How It Works:**
1. **Random component split (seed=42):** 80% training components, 20% validation components
2. **Training:**
   - Input Channel 1: Thermistor data from TRAINING components only (e.g., 18/22 components)
   - Target: Predict temperatures at training component ROIs
   - Uses ALL timesteps (no temporal split)
3. **Validation:**
   - Input Channel 1: Same TRAINING components (validation components excluded!)
   - Target: Predict temperatures at VALIDATION component ROIs (e.g., 4/22 components)
   - Model must infer held-out component temps from FLIR + thermal coupling
   - Uses ALL timesteps (same as training, different components)

**Hybrid Sampling (OPTIONAL):**
- Can still use hybrid timestep sampling for efficiency (dense/sparse)
- Or use ALL timesteps for maximum data (recommended for powerful machines)
- Sampling choice is independent of component split

**Why This Matters:**
- ✅ **Tests thermal understanding:** Can model predict components it hasn't seen thermistor data for?
- ✅ **Real-world use case:** Reduce thermistor count on future boards
- ✅ **Prevents cheating:** Model can't just copy thermistor inputs
- ✅ **Validation has proportional timestep coverage** if hybrid sampling used
- ✅ **Deterministic:** Using seed=42 ensures reproducible component splits across runs

**Old Temporal Split (LEGACY, not recommended):**
- Still available via `temporal_split=True` parameter
- Splits by time: train on early timesteps, validate on late timesteps  
- Problem: Model sees all components, doesn't test thermal learning
- Use only for backward compatibility

**Data Flow with Component Split:**
```
Training Batch:
  Input Ch0: FLIR [480×640] - All components visible
  Input Ch1: Thermistors [480×640, sparse] - Only 18/22 components embedded
  Target: Temps at 18 training component ROIs
  Loss: Masked MSE at 18 ROI locations

Validation Batch:
  Input Ch0: FLIR [480×640] - All components visible  
  Input Ch1: Thermistors [480×640, sparse] - Same 18/22 training components
  Target: Temps at 4 validation component ROIs (held-out!)
  Loss: Masked MSE at 4 validation ROI locations
  
→ Model must learn thermal coupling to predict held-out components
```

**Additional Training Best Practices:**
1. **Batch Shuffling:** Batches are shuffled at the end of each epoch (train only) for better generalization
2. **Data Normalization:** Both FLIR and thermistor temperatures normalized to [0, 1] using min-max scaling
   - **Applied to BOTH generator and in-memory training paths** for consistency
   - Computed from entire dataset (all FLIR frames and thermistor readings)
   - Example ranges: FLIR [18.9, 63.3]°C → [0, 1], Thermistor [22.1, 67.3]°C → [0, 1]
   - **Automatic denormalization:** Predictions are automatically converted back to °C during evaluation
   - **Normalization stats saved with model:** Ensures consistent inference behavior
3. **Random Seed:** All random operations use seed=42 for reproducible experiments
4. **No Shuffle for Validation:** Validation batches maintain order for consistent evaluation

**⚠️ IMPORTANT: Model Compatibility**
- **All models use normalized training data** regardless of training method (generator or in-memory)
- Models trained with normalization **cannot** be used with unnormalized data inputs
- Normalization stats are saved alongside the model as `*_normalization.json`
- When loading a model, normalization stats are automatically loaded if present
- Old models trained before normalization was added will not have these stats and are incompatible

**Configuration Parameters:**

All parameters are specified in **seconds** for clarity. They are automatically converted to timestamp indices internally based on the dataset's time resolution.

1. **`dense_limit_time`** (default: 4500 seconds)
   - **What it does:** Time cutoff between "dense" and "sparse" sampling regions
   - **Units:** Seconds from test start
   - **Example:** `dense_limit_time=4500` means 0-4500s sampled densely, 4500s+ sampled sparsely
   - **For HBridge:** First 300 FLIR frames = 4500s (300 frames × 15s/frame) covers the transient region with unique thermal images. After 4500s, the averaged steady-state FLIR frame is used.
   - **How to choose:** 
     - For transient-focused training: Use time when temperatures stabilize (e.g., 4500s for HBridge)
     - For steady-state focus: Set lower (e.g., 1500s = first 100 FLIR frames)
   - **Impact:** Higher → more dense samples → longer training but better transient accuracy

2. **`dense_step_time`** (default: 15 seconds)
   - **What it does:** Sampling interval for times 0 to `dense_limit_time`
   - **Units:** Seconds between samples
   - **Example:** 
     - `dense_step_time=15` → sample every 15s (every FLIR frame for HBridge)
     - `dense_step_time=30` → sample every 30s (every 2 FLIR frames)
     - `dense_step_time=60` → sample every 60s (every 4 FLIR frames)
   - **How to choose:**
     - **For HBridge (15s FLIR):** Use 15 to match native FLIR rate (recommended)
     - For faster prototyping: Use 30-60s (skip frames in dense region)
     - For maximum detail: Use 15s or match your FLIR sampling rate
   - **Impact:** Higher → fewer samples → faster training; Lower → more detail in transient region

3. **`sparse_step_time`** (default: 300 seconds)
   - **What it does:** Sampling interval for times after `dense_limit_time`
   - **Units:** Seconds between samples
   - **Example:**
     - `sparse_step_time=300` → sample every 300s = every 20 FLIR frames (for 15s FLIR)
     - `sparse_step_time=600` → sample every 600s = every 40 FLIR frames
     - `sparse_step_time=150` → sample every 150s = every 10 FLIR frames
   - **How to choose:**
     - For steady-state: Use large values (300-600s) since temps don't change (uses averaged frame)
     - For slow drift: Use moderate values (150-300s)
   - **Impact:** Higher → fewer steady-state samples → faster training but less steady-state validation data

**Training Time Impact:**

Total batches per epoch = (dense_samples + sparse_samples) / batch_size

**Example Calculations** (for HBridge dataset: 8719 timestamps, 130770s total, batch_size=8):

| Config | Dense Samples | Sparse Samples | Total Batches | Time per Epoch | Use Case |
|--------|---------------|----------------|---------------|----------------|----------|
| **`dense_limit=4500s, dense_step=15s, sparse_step=300s`** | **300** | **421** | **90** | **~2 min** | **🌟 Recommended for HBridge (all unique FLIR frames + sparse steady-state)** |
| `dense_limit=4500s, dense_step=30s, sparse_step=300s` | 150 | 421 | 71 | ~1.5 min | **Faster (skips half of transient frames)** |
| `dense_limit=4500s, dense_step=15s, sparse_step=600s` | 300 | 210 | 64 | ~1.5 min | **Less steady-state data** |
| `dense_limit=2250s, dense_step=15s, sparse_step=300s` | 150 | 428 | 72 | ~1.5 min | **Shorter transient window (first 150 frames)** |
| All timestamps (step=15s everywhere) | 8719 | 0 | 1090 | ~25 min | Maximum data but mostly redundant (steady-state uses same averaged frame) |

**Interactive Prompts:**
```
================================================================================
HYBRID BATCH SAMPLING CONFIGURATION
================================================================================
Configure dense vs sparse sampling for time-efficient training.
Specify all times in seconds (will be automatically converted to frame indices).

Dataset info:
  - HBridge FLIR: 15 second intervals
  - First 300 FLIR frames (0-4500s): unique thermal images (transient region)
  - After 4500s: uses averaged steady-state FLIR frame (extended for thermistor duration)

  Dense limit (seconds) - e.g., 4500 = first 300 FLIR frames [default: 4500]: 
  Dense step (seconds) - e.g., 15 = every FLIR frame, 30 = every 2 frames [default: 15]: 
  Sparse step (seconds) - e.g., 300 = every 20 FLIR frames [default: 300]: 

  Configuration: dense_limit=4500s, dense_step=15s, sparse_step=300s
```

**Recommendations:**
- **For HBridge (15s FLIR):** Use `dense_limit=4500s, dense_step=15s, sparse_step=300s` - samples all 300 unique FLIR frames plus sparse steady-state
- **For faster prototyping:** Use `dense_limit=2250s, dense_step=30s, sparse_step=600s` - trains in ~1 min
- **For long datasets (>12 hours):** Increase sparse_step to 600-1200s to reduce redundant steady-state samples
- **For different FLIR rates:** Set dense_step = your FLIR interval (e.g., 10s FLIR → dense_step=10s)

**Trade-offs:**
- ✅ **More dense samples:** Better transient accuracy, longer training
- ✅ **More sparse samples:** Better steady-state representation, longer training  
- ⚠️ **Too aggressive sampling (large steps):** May miss important thermal events, poor accuracy
- ⚠️ **Too conservative sampling (small steps):** Redundant data (steady-state uses same averaged frame), wasted training time

---

#### Complete Training Workflow Verification

**End-to-End Data Flow (Ensures Non-Negative RMSE in °C):**

1. **Data Loading:**
   ```
   HDF5 Dataset:
   - FLIR frames: [18.87°C to 63.28°C] (301 frames, 480×640 pixels)
   - Thermistor temps: [22.12°C to 67.33°C] (8719 timestamps, 22 components)
   ```

2. **Normalization (Applied Automatically):**
   ```
   FLIR channel:     (data - 18.87) / (63.28 - 18.87) → [0, 1]
   Thermistor channel: (data - 22.12) / (67.33 - 22.12) → [0, 1]
   Ground truth:     (data - 22.12) / (67.33 - 22.12) → [0, 1]
   
   Normalization stats saved in self.normalization_stats
   ```

3. **Training:**
   ```
   Model Input:  [batch, 480, 640, 2] with values in [0, 1]
                 Channel 0: Normalized FLIR temps
                 Channel 1: Normalized thermistor temps (sparse, at ROI pixels)
   
   Model Output: [batch, 480, 640, 1] with values in [0, 1]
                 Predicted normalized temps at all pixels
   
   Loss:         Computed on normalized predictions vs normalized ground truth
   ```

4. **Evaluation (Automatic Denormalization):**
   ```python
   # Predict (returns normalized values [0, 1])
   y_pred_norm = model.predict(X_val)  
   
   # Denormalize to °C
   y_pred_degC = y_pred_norm * (67.33 - 22.12) + 22.12
   y_true_degC = y_true_norm * (67.33 - 22.12) + 22.12
   
   # Compute metrics in °C
   RMSE = sqrt(mean((y_true_degC - y_pred_degC)^2))  # Always ≥ 0
   MAE  = mean(abs(y_true_degC - y_pred_degC))       # Always ≥ 0
   ```

5. **Expected Results:**
   ```
   ✅ RMSE: 0.5-5.0°C (typical for well-trained thermal model)
   ✅ MAE:  0.3-3.0°C (mean absolute error)
   ✅ R²:   0.85-0.99 (high correlation)
   
   ❌ If RMSE < 0: Impossible (square root always positive)
   ❌ If RMSE > 20°C: Model poorly trained or data issues
   ```

**Common Issues Prevented:**

| Issue | Old Behavior | Fixed Behavior | Verification |
|-------|-------------|----------------|--------------|
| **No normalization** | Raw temps (unstable training) | Normalized [0,1] | ✅ Both paths normalize |
| **Inconsistent paths** | Generator≠In-memory | Both use same normalization | ✅ Consistent stats |
| **Wrong metrics** | Normalized RMSE reported as °C | Denormalized before metrics | ✅ Auto-denormalize |
| **No stratified split** | Val has no transients | 80/20 split within dense+sparse | ✅ Stratified split |
| **No shuffling** | Same batch order every epoch | Shuffled at epoch end | ✅ Random batches |

**Workflow Checklist:**
- [x] Dataset has 301 FLIR frames + averaged steady-state frame
- [x] Normalization computed from full dataset (18.87-63.28°C FLIR, 22.12-67.33°C thermistor)
- [x] Stratified 80/20 split maintains dense/sparse proportions
- [x] Training batches shuffled each epoch for better generalization
- [x] Validation predictions denormalized before computing RMSE
- [x] Normalization stats saved with model for inference consistency
- [x] RMSE computed in °C (always non-negative by definition)



**Files:**
- `phase8_ml_training.py` - Main wrapper (orchestrates ML workflows)
- `phase8a_linear_regression.py` - Legacy linear model (optional)
- `phase8c_spatial_cnn.py` - U-Net CNN implementation with HDF5DataGenerator
- `ml_model/cnn_thermal_modeling/` - U-Net CNN module

---

## Incremental Calibration Workflow

**NEW:** Build calibration database gradually across multiple test sessions with automatic convergence tracking.

### Usage Modes

**Mode 1: Incremental (Week-by-Week)**
```powershell
# Week 1: Initial calibration (6 components, air only)
python researchir_post_processor.py --thermal_modeling `
    --test_session "LoadShedding_Week1" `
    --convergence_analysis

# Week 2: Add more components (append mode)
python researchir_post_processor.py --thermal_modeling `
    --append_calibration `
    --test_session "LoadShedding_Week2" `
    --convergence_analysis

# Week 3: Add sand cooling data
python researchir_post_processor.py --thermal_modeling `
    --append_calibration `
    --test_session "LoadShedding_Week3_Sand" `
    --convergence_analysis
```

**Mode 2: Multi-Session (Batch Processing)**
```powershell
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config multi_session_config_example.json `
    --convergence_analysis
```

### Multi-Session Config Format

Create `multi_session_config.json`:
```json
{
  "sessions": [
    {
      "name": "LoadShedding_Week1",
      "pcb": "LoadShedding_RevA",
      "flir_file": "outputs/Load_Shedding/Load_Shedding_FLIR_AllComponents.csv",
      "therm_air_file": "inputs/Week1_AIR_usb_temp_001.csv",
      "therm_sand_file": null,
      "pairs": [
        ["R1", "AI1", "R1", "Resistor"],
        ["R5", "AI2", "R5", "Resistor"]
      ]
    },
    {
      "name": "LoadShedding_Week2",
      "pcb": "LoadShedding_RevA",
      "flir_file": "outputs/Load_Shedding/Load_Shedding_FLIR_AllComponents.csv",
      "therm_air_file": "inputs/Week2_AIR_usb_temp_001.csv",
      "therm_sand_file": "inputs/Week2_SAND_usb_temp_001.csv",
      "pairs": [
        ["R8", "AI1", "R8", "Resistor"]
      ]
    }
  ]
}
```

See `multi_session_config_example.json` for complete example.

### Configuration Validation

**NEW:** Validate your configuration before processing to catch errors early!

**Basic Validation:**
```powershell
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config your_config.json `
    --validate
```

**Verbose Validation (Detailed Output):**
```powershell
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config your_config.json `
    --validate --verbose_validation
```

**What Gets Validated:**
- ✅ JSON syntax and structure
- ✅ Required fields present (`name`, `pcb`, `flir_file`, `therm_air_file`, `pairs`)
- ✅ File paths exist (FLIR, thermistor files)
- ✅ Channel naming conventions (Device prefix for multi-device)
- ✅ Component pair definitions
- ✅ Minimum pair count per session

**Example Output (Success):**
```
================================================================================
 VALIDATION SUMMARY
================================================================================

✓ ALL VALIDATION CHECKS PASSED

  Total sessions: 2
  Total component pairs: 13

  Session Breakdown:
    1. Batch1_Test3_LoadShedding: 5 components (single-device)
    2. Batch2_MultiDevice_LoadShedding: 8 components (multi-device)

  Ready to process!
  Run without --validate to execute full workflow.

================================================================================
```

**Example Output (Failure):**
```
================================================================================
 VALIDATION SUMMARY
================================================================================

✗ VALIDATION FAILED

  Found 1 error(s)

  Errors:
    • Session 1 validation failed

  ⚠ Warnings (3):
    • Session 1: FLIR file not found: nonexistent_file.csv
    • Session 1: Pair 0 (U1): Multi-device setup requires 'Device{N}_' prefix
    • Session 1: Pair 1 (R1): Multi-device setup requires 'Device{N}_' prefix

================================================================================
```

**Workflow Best Practice:**
```powershell
# Step 1: Validate configuration
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config my_config.json --validate --verbose_validation

# Step 2: If validation passes, run processing
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config my_config.json --convergence_analysis
```

**Common Validation Errors:**

| Error | Solution |
|-------|----------|
| Missing required field | Add `name`, `pcb`, `flir_file`, `therm_air_file`, or `pairs` to session |
| File not found | Check file paths are correct and files exist |
| Device prefix required | For multi-device: use `Device0_AI0`, `Device1_AI4`, etc. |
| Device prefix unexpected | For single-device: use `AI0`, `AI1`, etc. (no Device prefix) |
| Invalid pair format | Use `[flir_roi, therm_chan, component, type]` or dict format |
| Empty pairs list | Add at least one component pair per session |

### Convergence Analysis

**Automatic Quality Assessment:**

| Quality | Criteria | Interpretation |
|---------|----------|----------------|
| **HIGH** | n ≥ 5 samples and σ < 2.0°C | Well-calibrated, high confidence |
| **MEDIUM** | n ≥ 3 samples and σ < 3.0°C | Limited data, moderate confidence |
| **LOW** | n ≥ 2 samples and σ < 4.0°C | Minimal data, low confidence |
| **VERY_LOW** | Otherwise | Insufficient for reliable predictions |

**Generated Outputs:**
- `calibration_convergence_by_type.png/pdf` - Offset evolution per component type
- `calibration_confidence_evolution.png/pdf` - Uncertainty reduction with more data
- `calibration_sessions_heatmap.png/pdf` - Sample distribution across sessions
- `calibration_convergence_report.json` - Quality metrics and recommendations

**Standalone Analysis:**
```powershell
python calibration_convergence_analysis.py `
    --calibration_file outputs/calibration_database/thermal_calibration_points.csv `
    --output_dir outputs/convergence_analysis
```

### Best Practices

1. **Session Naming:** Use descriptive names with dates/weeks
   - ✅ Good: `LoadShedding_Week1`, `LoadShedding_20250115`
   - ❌ Bad: `test1`, `data`

2. **Component Selection:** Start with high-power/critical components
   - Power supplies, voltage regulators, high-power resistors, MOSFETs
   - Then expand to LEDs, ICs, passives

3. **Air-Only Sessions:** OK to collect air data without sand
   - FLIR offset calibration works with air data only
   - Sand cooling model requires `therm_sand_file`
   - Set `therm_sand_file: null` if not available

4. **Quality Targets:**
   - Aim for n ≥ 5 samples per component type
   - Target σ < 2.0°C for high-confidence predictions
   - Use convergence plots to identify types needing more data

5. **Reuse FLIR Data:** Same FLIR file can be used across sessions
   - Different thermistor channels measure different components

### Multi-Device Thermistor Support

**NEW:** Support for multiple USB thermistor devices in a single test session (e.g., when measuring more than 8 components).

#### When to Use Multi-Device Mode

Use multi-device mode when:
- Measuring more than 8 components (single USB-TEMP device limit)
- Using multiple USB-TEMP devices simultaneously (Device 0 + Device 1)
- Channel numbers overlap between devices (both have AI0-AI7)

#### Device-Prefixed Channel Naming

**Problem:** Two devices both have "AI4" → conflict when merging data

**Solution:** Automatic device-prefixed labels
```
Device 0 channels: Device0_AI0, Device0_AI1, Device0_AI4, Device0_AI5, Device0_AI6, Device0_AI7
Device 1 channels: Device1_AI4, Device1_AI5
```

**Pattern Matching:** Use device-prefixed patterns in configuration
```json
{
  "pairs": [
    ["Box 1", "Device0_AI0", "U3", "IC"],
    ["Box 3", "Device1_AI4", "R8", "Resistor"]
  ]
}
```

**Note:** Do NOT include "(°C)" in patterns - the loader adds this automatically. Use `"Device0_AI0"` not `"Device0_AI0 (°C)"`.

#### Multi-Device Configuration Format

**Single-Device (Backward Compatible):**
```json
{
  "name": "Batch1_SingleDevice",
  "pcb": "LoadShedding_AC_Switch",
  "flir_file": "inputs/Test_3_FLIR_Camera_Results.csv",
  "therm_air_file": "inputs/Test_3_AIR_usb_temp_DAQami.csv",
  "therm_sand_file": "inputs/Test_3_SAND_usb_temp_DAQami.csv",
  "pairs": [
    ["Box 1", "AI0", "U3", "IC"],
    ["Box 3", "AI1", "U1", "PowerSupply"]
  ]
}
```

**Multi-Device (New Capability):**
```json
{
  "name": "Batch2_MultiDevice",
  "pcb": "LoadShedding_AC_Switch",
  "flir_file": "inputs/Test_3_FLIR_Camera_Results.csv",
  "therm_air_file": [
    "inputs/Test_Air/USB-TEMP (Device 0) - Analog.csv",
    "inputs/Test_Air/USB-TEMP-AI (Device 1) - Analog.csv"
  ],
  "therm_sand_file": [
    "inputs/Test_Sand/USB-TEMP (Device 0) - Analog.csv",
    "inputs/Test_Sand/USB-TEMP-AI (Device 1) - Analog.csv"
  ],
  "pairs": [
    ["Box 1", "Device0_AI0", "U2", "IC"],
    ["Box 3", "Device0_AI1", "R5", "Resistor"],
    ["SMD RES1", "Device1_AI4", "C3", "Capacitor"]
  ]
}
```

**Key Differences:**
- `therm_air_file`: String → Array of strings
- `therm_sand_file`: String → Array of strings (or null)
- `pairs`: Thermistor patterns use `Device0_*` or `Device1_*` prefixes

#### Device Auto-Detection

The loader automatically detects device numbers from filenames:
```
"USB-TEMP (Device 0) - Analog - 12-4-2025 12-13-19.6 PM.csv"  → Device0
"USB-TEMP-AI (Device 1) - Analog - 12-4-2025 12-13-19.6 PM.csv" → Device1
```

If filenames don't contain device numbers, devices are numbered sequentially (0, 1, 2...).

#### Data Merging Process

1. **Load each device file separately** using standard CSV parser
2. **Prefix channel labels** with device identifier (Device0_AI0, Device1_AI4)
3. **Synchronize time vectors** (uses first file as reference, truncates to minimum length)
4. **Merge horizontally** using numpy.hstack() to combine all channels
5. **Return merged dataset** with 8+ channels available for calibration

**Example Merge:**
```
Device 0: 6 channels (AI0, AI1, AI4, AI5, AI6, AI7) → Device0_AI0...Device0_AI7
Device 1: 2 channels (AI4, AI5)                     → Device1_AI4, Device1_AI5
────────────────────────────────────────────────────────────────────────────
Merged:   8 total channels (no conflicts)
```

#### Complete Multi-Device Workflow

**Step 1: Prepare Test Data**
```
inputs/
├── Test_Air_Load_Shedding/
│   ├── USB-TEMP (Device 0) - Analog.csv      # 6 channels
│   └── USB-TEMP-AI (Device 1) - Analog.csv   # 2 channels
├── Test_Sand_Load_Shedding/
│   ├── USB-TEMP (Device 0) - Analog.csv      # 6 channels
│   └── USB-TEMP-AI (Device 1) - Analog.csv   # 2 channels
└── Test_3_FLIR_Camera_Results_5_of_7.csv     # FLIR data
```

**Step 2: Create Multi-Batch Config**

See `multi_batch_loadshedding_config.json` for complete example with:
- Batch 1: Single-device session (5 components)
- Batch 2: Multi-device session (8 channels)

**Step 3: Run Calibration**
```powershell
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config multi_batch_loadshedding_config.json `
    --append_calibration `
    --convergence_analysis
```

**Step 4: Verify Results**
Check calibration output for all components from both batches:
```
Calibration points: 10 (5 from Batch1 + 5 from Batch2)
Device merging: 8 total channels (Device0: 6, Device1: 2)
```

#### Troubleshooting Multi-Device Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Pattern 'Device0_AI0' not found" | Wrong pattern format | Remove "(°C)" from pattern - use "Device0_AI0" not "Device0_AI0 (°C)" |
| "Channel conflict detected" | Channel numbers overlap | Verify device prefixes are different (Device0 vs Device1) |
| "Time vector mismatch" | Different sample counts | Loader auto-truncates to minimum length - check logs |
| "Only X channels found" | Missing device file | Verify all files in array exist and are readable |
| JSON parsing error | Syntax error in array | Validate JSON: arrays use `["file1", "file2"]` format |

**Debug Tips:**
- Add `--debug` flag to see device detection and merging details
- Check console output for "Loading DeviceX: filename" messages
- Verify merged channel count matches expected (Device0 channels + Device1 channels)
- Use test scripts (test_batch1.py, test_quick_multidevice.py) to validate before full run

#### Mapping Component Guide

When creating multi-device mappings:

1. **Identify measured components** - Which physical components did each thermistor measure?
2. **Match to FLIR ROIs** - Find corresponding ROI box names in FLIR CSV headers
3. **Use correct device prefix** - Device0_AI* for first device, Device1_AI* for second
4. **Avoid duplicates** - Each component should appear in only one batch
5. **Verify component types** - Use consistent type names (IC, Resistor, LED, etc.)

**Available Components List:**
Check `multi_batch_loadshedding_config.json` → `mapping_guidance` → `batch2_available_components` for components not yet measured.

### Thermistor Validation
**Purpose:** Ground-truth comparison plots (FLIR vs thermistor)

**Features:** Side-by-side overlays, Delta T visualization, steady-state detection, air vs sand comparison

**Outputs:** High-res comparison plots (PNG+PDF, 600 DPI)

---

## Board-Based Multi-Test Configuration

**NEW:** Hierarchical configuration structure for managing multiple tests per board where the same thermistor channels measure different components across tests.

### Overview

The board-based configuration format provides:
- **Hierarchical structure**: `boards` → `tests` → `pairs`
- **Multiple tests per board**: Different component mappings per test
- **Channel reuse**: Same thermistor channel (e.g., AI0) can map to different components in different tests
- **Mixed file handling**: Single files or multi-device arrays
- **Backward compatibility**: Legacy session-based configs still supported

### When to Use Board-Based Configs

Use board-based configs when:
- Running multiple test sessions on the same PCB with different component sets
- Same thermistor channels measure different components across tests
- Need organized tracking of board→test→measurement hierarchy
- Building comprehensive calibration database for a single PCB design

Use legacy session-based configs for:
- Simple single-test scenarios
- Batch processing unrelated PCBs
- Quick one-off calibrations

### Configuration Structure

**Hierarchical Format:**
```json
{
  "boards": [
    {
      "name": "LoadShedding",
      "pcb": "Load_Shedding_Rev_E.brd",
      "tests": [
        {
          "test_id": "Test1_Air",
          "therm_air_file": "inputs/Test_Air_Load_Shedding/Test 1/Test_3_AIR_usb_temp_DAQami.csv",
          "therm_sand_file": "inputs/Test_Sand_Load_Shedding/Test 1/Test_3_SAND_usb_temp_DAQami.csv",
          "pairs": [
            {"flir_roi": "U3", "therm_channel": "AI0", "component": "U3", "type": "IC"},
            {"flir_roi": "U1", "therm_channel": "AI1", "component": "U1", "type": "PowerSupply"},
            {"flir_roi": "J1", "therm_channel": "AI4", "component": "J1", "type": "Connector"}
          ]
        },
        {
          "test_id": "Test2_Air",
          "therm_air_file": [
            "inputs/Test_Air_Load_Shedding/Test 2/Analog - 12-2-2025 11-21-18.5 AM.csv",
            "inputs/Test_Air_Load_Shedding/Test 2/Analog - 12-2-2025 2-12-16.8 PM.csv"
          ],
          "pairs": [
            {"flir_roi": "R15", "therm_channel": "Device0_AI0", "component": "R15", "type": "Resistor"},
            {"flir_roi": "L3", "therm_channel": "Device1_AI1", "component": "L3", "type": "Inductor"}
          ]
        }
      ]
    }
  ]
}
```

**Note:** `flir_file` is optional - system automatically uses parsed ResearchIR_Outputs data if not specified.

### Key Features

**1. Channel Remapping Across Tests:**
```json
// Test1: AI0 measures U3 (IC)
{"flir_roi": "U3", "therm_channel": "AI0", "component": "U3", "type": "IC"}

// Test2: AI0 measures R15 (Resistor) - same channel, different component
{"flir_roi": "R15", "therm_channel": "Device0_AI0", "component": "R15", "type": "Resistor"}
```

**2. Mixed Single/Multi-Device Files:**
```json
// Single file
"therm_air_file": "path/to/file.csv"

// Multi-device array
"therm_air_file": ["device0.csv", "device1.csv"]
```

**3. Automatic FLIR Data Detection:**
- System parses ResearchIR_Outputs folders automatically
- No need to specify `flir_file` if using standard folder structure
- Parsed CSV files auto-detected from `outputs/{timestamp}_P6-7/{PCB}/{PCB}_FLIR_AllComponents.csv`

### Creating Board Configs

**Step 1: Discover Files**
```powershell
python config_file_finder.py --format json
```

This generates a template with all detected files organized by board/test.

**Step 2: Map Components**
Edit the template to add correct FLIR ROI names and component types:
```json
"pairs": [
  {"flir_roi": "U3", "therm_channel": "AI0", "component": "U3", "type": "IC"},
  {"flir_roi": "U1", "therm_channel": "AI1", "component": "U1", "type": "PowerSupply"}
]
```

**Component Types:** IC, PowerSupply, LDO, Resistor, LED, Inductor, Capacitor, Diode, Connector, Switch, FET

**Step 3: Validate Configuration**
```powershell
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config your_board_config.json --validate
```

**Step 4: Run Processing**
```powershell
python researchir_post_processor.py --thermal_modeling `
    --multi_session_config your_board_config.json
```

### Output Structure

Board-based configs create hierarchical output folders:

```
outputs/MMDD_HHMM_P6-7/
└── calibration_database/
    ├── thermal_calibration_points.csv          # Includes board_name & test_id columns
    ├── thermal_calibration_by_type.csv
    ├── thermal_calibration_metadata.json
    └── calibration_database/
        ├── LoadShedding_Test1_Air/
        │   ├── raw_inputs_overview.png
        │   └── detailed_plots/
        │       ├── *_FLIR_Air_vs_Sand_*.png
        │       └── *_FLIR_Air_vs_Sand_*.pdf
        ├── LoadShedding_Test2_Air/
        │   └── ...
        └── LoadShedding_Test3_Air/
            └── ...
```

### Database Export Enhancements

**Calibration points CSV includes:**
- `board_name` - Board identifier from config
- `test_id` - Test session identifier
- All standard calibration fields (offsets, cooling ratios, etc.)

**Example:**
| board_name | test_id | component_name | component_type | flir_offset | cooling_ratio |
|------------|---------|----------------|----------------|-------------|---------------|
| LoadShedding | Test1_Air | U3 | IC | +0.09 | 1.0423 |
| LoadShedding | Test2_Air | R15 | Resistor | -0.67 | 0.9985 |

### Helper Tool: config_file_finder.py

Scans your inputs directory and generates configuration templates:

**Text Format (Review):**
```powershell
python config_file_finder.py
```

**JSON Format (Ready to Use):**
```powershell
python config_file_finder.py --format json > my_board_config.json
```

**Features:**
- Auto-detects board folders
- Identifies FLIR and thermistor files (air vs sand)
- Groups files by test number
- Generates skeleton config with placeholder component mappings

### Example Configurations

**Single-Device Test:**
```json
{
  "test_id": "Test1_Air",
  "therm_air_file": "inputs/Test 1/AIR_usb_temp.csv",
  "therm_sand_file": "inputs/Test 1/SAND_usb_temp.csv",
  "pairs": [
    {"flir_roi": "U3", "therm_channel": "AI0", "component": "U3", "type": "IC"}
  ]
}
```

**Multi-Device Test:**
```json
{
  "test_id": "Test2_Air",
  "therm_air_file": [
    "inputs/Test 2/USB-TEMP (Device 0) - Analog.csv",
    "inputs/Test 2/USB-TEMP-AI (Device 1) - Analog.csv"
  ],
  "pairs": [
    {"flir_roi": "R15", "therm_channel": "Device0_AI0", "component": "R15", "type": "Resistor"},
    {"flir_roi": "L3", "therm_channel": "Device1_AI1", "component": "L3", "type": "Inductor"}
  ]
}
```

See `LoadShedding_multi_test_config.json` and `HBridge_multi_test_config.json` for complete examples.

### Validation Rules

Board-based configs are validated for:
- ✅ Required fields: `name`, `pcb`, `tests` at board level
- ✅ Required fields: `test_id`, `therm_air_file`, `pairs` at test level
- ✅ File existence (with warnings for missing files)
- ✅ Channel naming (Device prefix for multi-device)
- ✅ Component pair structure (dict or array format)
- ✅ Minimum pair count (at least 1 per test)

### Backward Compatibility

Legacy session-based configs automatically detected and processed:

```json
{
  "sessions": [
    {
      "name": "Batch1",
      "pcb": "LoadShedding",
      "flir_file": "...",
      "therm_air_file": "...",
      "pairs": [...]
    }
  ]
}
```

System detects format based on top-level key (`boards` vs `sessions`).

---

## Command-Line Reference

**Basic:**
```powershell
python researchir_post_processor.py                    # Interactive UI
python researchir_post_processor.py --no_ui            # Phases 1-5
python researchir_post_processor.py --thermal_modeling # Phases 6-7
python researchir_post_processor.py --full_pipeline    # Phases 1-8 (with ML)
```

**Pre-Processing:**
```powershell
# Step 1: Generate ROI pixel maps (in sroi_generation_ResearchIR/)
cd ../sroi_generation_ResearchIR
python interactive_pipeline.py
# Enter correct PCB corner coordinates when prompted
# Creates: outputs/*_roi_pixel_map.csv

# Step 2: Import pixel maps to thermal pipeline
cd ../thermal_post_processing
python import_roi_pixel_maps.py
# Copies pixel maps to: outputs/roi_pixel_maps/

# Step 3: Filter FLIR frames for ML (run before Phase 8)
python researchir_post_processor.py
# Select: [F] Filter FLIR frames for ML training
# Creates: inputs/ResearchIR_Outputs_*_filtered/
```

**Options:**
```powershell
--input <folder>              # Input folder (default: inputs/ResearchIR_Outputs_*)
--output <folder>             # Output folder (overrides auto-timestamp)
--debug                       # Verbose output
--coordinates <csv>           # Component coordinates (X,Y in mm)
--proximity_threshold <mm>    # Coupling distance (default: 10.0)
--no_spatial                  # Disable spatial analysis
--custom_groups <json>        # Custom component grouping
--calibration_pcb <name>      # Calibration PCB name
--prediction_pcb <name>       # Prediction PCB name
--calibration_mapping <json>  # FLIR-to-thermistor mapping
--append_calibration          # Append to existing calibration database
--test_session <name>         # Test session name/ID for tracking
--multi_session_config <json> # Multi-session batch processing config
--convergence_analysis        # Run convergence analysis after calibration
--filter_flir_frames          # (NEW) Pre-process FLIR frames for ML
```

## Input/Output Files

**Inputs:** 
- `inputs/ResearchIR_Outputs_<PCB>/*.csv` - Raw FLIR frames
- `inputs/ResearchIR_Outputs_<PCB>_filtered/*.csv` - (NEW) Filtered FLIR frames
- `*_components_enhanced.csv` - Component coordinates
- `*_usb_temp_DAQami.csv` - Thermistor data
- `*_thermistor_mapping.json` - FLIR-to-thermistor mapping

**Outputs:** Auto-timestamped folders (`MMDD_HHMM_P*`) containing:
- **P1-5**: 
  - Analysis summaries (PNG/CSV/PDF)
  - Filtering comparisons
  - (NEW) `*_phase2_filtered_temperatures.csv` - Component temperatures
  - (NEW) `*_phase2_filter_comparison.csv` - Raw vs filtered
  - (NEW) `*_phase3_component_statistics.csv` - Thermal stats
  - (NEW) `*_phase3_component_metadata.csv` - ROI info
  - (NEW) `*_phase4_spatial_coupling.csv` - Coupling matrix
  - Coupling maps, risk assessment, MATLAB export
- **P6-7**: 
  - Calibration database (CSV/JSON/PNG)
  - Prediction results
  - Detailed comparison plots (PNG+PDF)
- **P1-8** (NEW):
  - Complete combined outputs
  - `phase8_ml_results/` - U-Net CNN predictions and visualizations
  - `ml_model/cnn_thermal_modeling/datasets/` - HDF5 dataset (87.99 MB)
  - `ml_model/cnn_thermal_modeling/models/` - Trained models

## ROI Pixel Map Generation (CRITICAL for Phase 8 ML)

**What is an ROI Pixel Map?**
The ROI (Region of Interest) pixel map defines which pixels in the 640×480 FLIR thermal image correspond to each component on the PCB. This mapping is essential for CNN training because it tells the model where to find thermistor ground truth data in the spatial image.

**File Structure:**
```csv
component_name,roi_type,center_x_px,center_y_px,width_px,height_px,pixel_count,pixel_list
PS3,Cursor_3x3,179.7,204.6,1,1,9,"[(178,204),(179,204),(180,204),...]"
DL11,Cursor_3x3,241.8,205.3,1,1,9,"[(240,204),(241,204),(242,204),...]"
```

**Generation Source:**
ROI pixel maps are generated by the **SROI pipeline** (`sroi_generation_ResearchIR/enhanced_scalable_csv_to_sroi.py`), which uses **calibrated PCB corner coordinates** to transform component mm positions to FLIR pixel positions.

**Workflow:**
1. **Generate SROI file with ROI pixel map** (in `sroi_generation_ResearchIR/`):
   ```powershell
   cd ../sroi_generation_ResearchIR
   python interactive_pipeline.py
   # When prompted for PCB corners, enter calibrated values:
   #   Bottom-left: (102, 400)
   #   Top-right: (500, 150)
   # Output: outputs/<session>/HBridge_15s_roi_pixel_map.csv
   ```

2. **Import to thermal pipeline** (in `thermal_post_processing/`):
   ```powershell
   cd ../thermal_post_processing
   python import_roi_pixel_maps.py
   # Copies: <session>/HBridge_15s_roi_pixel_map.csv 
   #      → outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv
   ```

3. **Verify pixel positions**:
   ```powershell
   python viz_phase8_compare_csv_vs_hdf5.py
   # Generates: outputs/CSV_vs_HDF5_MISMATCH.png
   # Shows overlay of CSV positions vs HDF5 masks
   # Zero mismatch = correct pixel map!
   ```

**CRITICAL: PCB Corner Calibration**
The pixel map accuracy depends entirely on correct PCB corner calibration:

| Corner | Correct Values | Wrong Values | Result |
|--------|---------------|--------------|--------|
| Bottom-left | `(102, 400)` | `(0, 479)` | ✗ Components mapped outside PCB |
| Top-right | `(500, 150)` | `(639, 0)` | ✗ Positional errors up to 379px |

**Common Issues:**
- **12/22 components have 0 pixels**: Wrong PCB corners used (assumes PCB fills entire image)
- **Large position mismatch (>50px)**: Check `viz_phase8_compare_csv_vs_hdf5.py` output, regenerate with correct corners
- **HDF5 dataset corrupted**: Delete `ml_model/cnn_thermal_modeling/datasets/*.h5` and rebuild

**Files Involved:**
```
sroi_generation_ResearchIR/
  ├── enhanced_scalable_csv_to_sroi.py  # Generates pixel map (line 550)
  └── outputs/<session>/
      └── HBridge_15s_roi_pixel_map.csv  # Session-specific pixel map

thermal_post_processing/
  ├── import_roi_pixel_maps.py          # Copies pixel map
  ├── viz_phase8_compare_csv_vs_hdf5.py            # Verification tool
  ├── cnn_data_preprocessor.py          # Uses pixel map to build HDF5
  └── outputs/roi_pixel_maps/
      └── HBridge_roi_pixel_map.csv     # Canonical pixel map for CNN
```

**Verification Checklist:**
- [ ] PCB corners calibrated in SROI pipeline
- [ ] All 22 components have `pixel_count > 0`
- [ ] `viz_phase8_compare_csv_vs_hdf5.py` shows position mismatch < 5px
- [ ] HDF5 dataset built with correct pixel map

## Configuration

**Component Grouping** (`component_grouping_config.json`):
```json
{"groups": {"Power_Management": ["VR*", "PS*", "LDO*"], "Custom": ["U1", "U2"]}}
```

**Thermistor Mapping** (`*_thermistor_mapping.json`):
```json
{
  "pcb_name": "LoadShedding", "test_name": "Test_3",
  "mappings": {
    "LDO2": {"flir_component": "Box 3 [C]:mean", "thermistor_air_channel": "AI0",
             "thermistor_sand_channel": "AI0", "component_type": "LDO"}
  }
}
```

**Plot Customization:** Edit `thermistor_validation.py` → `ThermistorValidator()` parameters (fig_size, dpi, smooth_win, etc.)

## Dependencies

**Required (Phases 1-7):** 
- Python 3.7+
- numpy, pandas, matplotlib, scipy

**Additional for Phase 8 (ML):**
- h5py (HDF5 dataset storage)
- tensorflow/keras (U-Net CNN model)
- scikit-learn (optional, for metrics)

**Install:**
```bash
# Basic dependencies (Phases 1-7)
pip install numpy pandas matplotlib scipy

# ML dependencies (Phase 8)
pip install h5py tensorflow scikit-learn

# Or install everything at once
pip install numpy pandas matplotlib scipy h5py tensorflow scikit-learn
```

**Optional:** 
- LaTeX (for publication-quality rendering)
- CUDA/cuDNN (for GPU-accelerated training, recommended for Phase 8)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Use UI option V to validate paths |
| No spatial coupling | Ensure coordinates CSV has Component, X, Y columns |
| Import errors | `pip install numpy pandas matplotlib scipy` |
| Cluttered plots | Adjust component grouping or use custom config |
| Calibration not found | Run Phase 6 before Phase 7 |
| Unknown component type | Add to thermistor mapping (uses average if missing) |
| Negative cooling benefit | Review for measurement artifacts |
| Convergence plots high variance | Check test conditions consistency, thermistor attachment |
| Multi-session config error | Validate JSON syntax, ensure required fields present |
| **Phase 8: Filtered frames not detected** | Run pre-processing menu [F] to create filtered frames |
| **Phase 8:SROI pixel maps generated with correct corners (from sroi_generation_ResearchIR/)
- [ ] (NEW) PROI pixel map not found** | Run `import_roi_pixel_maps.py` to copy from SROI generation |
| **Phase 8: Wrong pixel locations** | Regenerate SROI with correct corners, then re-import pixel map |
| **Phase 8: ixel maps imported to `outputs/roi_pixel_maps/` (use `import_roi_pixel_maps.py`)
- [ ] (NEW) Filtered FLIR frames for Phase 8 ML (optional but recommendedkes 30-60 min first time |
| **Phase 8: Out of memory** | Reduce batch size or use fewer frames (modify config) |
| **Phase 8: Dimension mismatch** | Ensure filtered frames match raw format (480×640 + 5-line header) |

**Debug Mode:** Add `--debug` flag for verbose per-component output

**Validation Checklist:**
- [ ] ResearchIR CSV exports in `inputs/`
- [ ] Coordinates CSV (for spatial), thermistor CSVs (for Phase 6)
- [ ] Thermistor mapping JSON (for Phase 6)
- [ ] (NEW) Filtered FLIR frames for Phase 8 ML (optional but recommended)
- [ ] (NEW) ROI pixel maps generated (auto-created by Phase 3)
- [ ] Dependencies installed

**Advanced:**
- **Batch processing:** Loop through input folders with `--no_ui`
- **MATLAB export:** Load `thermal_data_export.mat` from outputs
- **Calibration expansion:** Add PCBs to Phase 6, database auto-updates
- **Portability:** Fully portable with relative paths
- **(NEW) ML model reuse:** Train once, reuse model for future boards
- **(NEW) Filtered frame caching:** Filter once, reuse for multiple training runs

---

## Additional Resources

- **Incremental Calibration Guide:** See `INCREMENTAL_CALIBRATION_GUIDE.md` for detailed workflows and troubleshooting
- **Multi-Session Example:** See `multi_session_config_example.json` for complete configuration template
- **Phase Documentation:** See individual `phase*.py` files for detailed module documentation
- **(NEW) ML Integration Plan:** See `ML_INTEGRATION_PLAN.txt` for U-Net CNN architecture and implementation details
- **(NEW) CNN Data Preprocessing:** See `ml_model/cnn_thermal_modeling/cnn_data_preprocessor.py` for dataset structure

---

**Version:** v3.0 (Jan 15, 2026) - Added Phase 8 U-Net CNN spatial thermal prediction, FLIR frame filtering, and CSV exports for all phases  
**Previous:** v2.2 (Dec 5, 2025) - Added multi-device thermistor support for sessions with multiple USB-TEMP devices  
**Previous:** v2.1 (Dec 2, 2025) - Added incremental calibration, multi-session processing, and convergence analysis  
**Previous:** v2.0 (Nov 26, 2025) - Complete 7-phase pipeline with auto-timestamped outputs

## What's New in v3.0

**Major Features:**
1. **Phase 8: U-Net CNN Spatial Thermal Prediction**
   - Full-frame deep learning model for component temperature prediction
   - Captures spatial thermal coupling between components
   - Interactive UI for model selection (CNN vs linear regression vs both)
   - Automatic filtered frame detection and usage

2. **FLIR Frame Filtering for ML**
   - Pre-processing menu option to filter full thermal frames
   - Pixel-by-pixel temporal median filter (kernel_size=5)
   - Removes camera refocusing artifacts and noise
   - 30% file size reduction (1.4GB → 799MB for 300 frames)
   - Stored in `inputs/ResearchIR_Outputs_*_filtered/`

3. **CSV Exports for All Phases**
   - **Phase 2:** Filtered temperatures, raw vs filtered comparison
   - **Phase 3:** Component statistics, ROI metadata
   - **Phase 4:** Spatial coupling matrix with correlations
   - Enables standalone visualization and analysis

4. **ML Model Infrastructure**
   - HDF5 dataset format (87.99 MB for 300 frames)
   - Reusable trained models
   - Automatic result import to session outputs
   - Graceful error handling with helpful messages

**Breaking Changes:**
- `phase8_ml_training.py` renamed to `phase8a_linear_regression.py` (legacy model)
- New `phase8_ml_training.py` is now the U-Net CNN wrapper
- Full pipeline now includes Phase 8 (use `--full_pipeline` for Phases 1-8)

**Migration Guide:**
- Existing workflows (Phases 1-7) continue to work unchanged
- To use new ML features: Filter FLIR frames first, then run Phase 8
- Legacy linear regression still available via Phase 8 menu option [2]

---

## File Management for GitHub

**Current Repository Size:** ~26GB (too large for GitHub)  
**Target Size:** <15MB for GitHub compatibility

### What's INCLUDED in Git Repository (~12 MB)

**Code (32 core files, ~7 MB):**
- **Pipeline scripts:** 13 phase modules + main entry points
- **ML training:** 7 core scripts in `ml_model/cnn_thermal_modeling/`
- **Visualization:** 5 viz modules
- **Loaders & preprocessors:** 2 loader modules + 1 preprocessor
- **Configuration UI:** config_ui.py
- **Documentation:** README, guides, analysis docs

**Essential Configuration Files (~3 MB):**
- ✅ **JSON configs (30 KB):** `*_multi_test_config.json`, `component_type_patterns.json`, `loadshedding_thermistor_mapping.json`
- ✅ **Enhanced component CSVs (50 KB):** `hbridge_pcb_components_enhanced.csv`, `loadshedding_ac_switch_pcb_components_enhanced.csv`
- ✅ **Pick & Place files (27 KB):** `Pick_Place_for_Hbridge_*.csv`, `Pick_Place_for_LoadShedding_*.csv`
- ✅ **ROI pixel maps (1.6 MB):** `outputs/roi_pixel_maps/HBridge_*.csv`, `outputs/roi_pixel_maps/LoadShedding_*.csv`

**See [ACTIVE_FILE_STRUCTURE.md](ACTIVE_FILE_STRUCTURE.md) for complete file organization.**

### What's EXCLUDED (Download Separately)

**Large FLIR Data (3 GB)** - Not in Git, provide download link:
- ❌ `inputs/ResearchIR_Outputs_HBridge_15s/` (1.4 GB)
- ❌ `inputs/ResearchIR_Outputs_Load_Shedding/` (531 MB)
- ❌ Filtered versions (803 MB + 321 MB, regenerated by pipeline)

**Large Thermistor Test Data (110 MB)** - Not in Git, provide download link:
- ❌ `inputs/Test_Air_HBridge_Sensing/` (4.2 MB)
- ❌ `inputs/Test_Sand_HBridge_Sensing/` (74 MB)
- ❌ `inputs/Test_Air_Load_Shedding/` (7.1 MB)
- ❌ `inputs/Test_Sand_Load_Shedding/` (25 MB)
- ❌ Large CSV files:
  - `Test_3_FLIR_Camera_Results_5_of_7.csv` (2.7 MB)
  - `Test_3_AIR_usb_temp_DAQami.csv` (346 KB)
  - `Test_3_SAND_usb_temp_DAQami.csv` (2.4 MB)

**Pre-trained Models (8 GB each, optional)** - Not in Git:
- ❌ `ml_model/cnn_thermal_modeling/models/*.keras`
- Users can train their own or download from external link

**Pre-built Datasets (110 MB, optional)** - Not in Git:
- ❌ `ml_model/cnn_thermal_modeling/datasets/HBridge_dataset.h5` (90 MB)
- ❌ `ml_model/cnn_thermal_modeling/datasets/LoadShedding_dataset.h5` (20 MB)
- Users rebuild locally with `python build_dataset.py [board_name]`

**Generated Outputs (5 GB, regenerated)** - Not in Git:
- ❌ `outputs/` (except roi_pixel_maps which IS included)
- ❌ `ml_model/cnn_thermal_modeling/results/`

### Setup Instructions for New Users

1. **Clone repository** (~12 MB):
   ```bash
   git clone https://github.com/Gump17928/sroi_FLIR_CNN_PCB_prediction_model.git
   cd sroi_FLIR_CNN_PCB_prediction_model/thermal_post_processing
   ```

2. **Download large data files** (3.1 GB required):
   - Download FLIR data: [Provide external link - Google Drive/Zenodo]
   - Download thermistor test data: [Provide external link]
   - Extract to `inputs/` directory

3. **Build datasets locally** (or download pre-built):
   ```bash
   python ml_model/cnn_thermal_modeling/build_dataset.py HBridge
   python ml_model/cnn_thermal_modeling/build_dataset.py LoadShedding
   ```

4. **Train model** (or download pre-trained):
   ```bash
   python test_phase8.py --board HBridge --validation_mode cross_pcb
   ```

### Recommended Actions
1. ✅ **.gitignore updated:** Large files already excluded
2. ❌ **Delete unused files:** 27 diagnostic/debug scripts (see PIPELINE_DEPENDENCIES.md)
3. ❌ **External storage:** Upload FLIR/thermistor data to Google Drive/Zenodo
4. ❌ **Add download links:** Update README with external data links

**Result:** Repository size reduced from 26GB → ~12MB (GitHub compatible)

---

## Leave-One-Component-Out Validation (NEW)

**Purpose:**
Validates the generalization ability of the U-Net CNN model by excluding one component's thermistor data during training and predicting its temperature using FLIR and other components' thermistor data. This tests whether the model can infer thermal responses for components it hasn't directly seen.

**Script:**
`validate_leave_one_component_out.py` (located in `thermal_post_processing/`)

**Usage:**
```bash
python validate_leave_one_component_out.py
```
This will loop through all components, train the model with each left out, and save per-component metrics (RMSE, R²) to `leave_one_out_results.txt` in the appropriate analysis output folder (e.g., `ml_model/cnn_thermal_modeling/results/analysis_<date>/`).

**Use Case:**
- Assess model's ability to predict unseen components
- Detect overfitting to thermistor data
- Quantify generalization across board regions

**Output:**
- Results file: `leave_one_out_results.txt` (per-component metrics)
- Can be used to compare model performance and guide further improvements

---
