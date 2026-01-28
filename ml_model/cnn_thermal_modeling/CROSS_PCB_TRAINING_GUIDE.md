# Cross-PCB Training Guide

## Quick Start

### Option 1: Using test_phase8.py (Recommended for integrated pipeline)

```bash
cd /home/forrest/home/modeling/thermal_post_processing

# Component holdout validation (traditional - train/validate on same PCB)
python test_phase8.py --use_cnn_pipeline --board HBridge --validation_mode component_holdout

# Cross-PCB validation (NEW - train HBridge, validate LoadShedding)
python test_phase8.py --use_cnn_pipeline --validation_mode cross_pcb

# Build datasets before training
python test_phase8.py --use_cnn_pipeline --build_dataset --validation_mode cross_pcb
```

### Option 2: Using train_hbridge_model.py directly

```bash
cd /home/forrest/home/modeling/thermal_post_processing/ml_model/cnn_thermal_modeling

# Activate virtual environment
cd /home/forrest/home/modeling && source .venv/bin/activate && cd thermal_post_processing/ml_model/cnn_thermal_modeling

# Component holdout - HBridge
python train_hbridge_model.py --board HBridge --validation_mode component_holdout

# Component holdout - LoadShedding
python train_hbridge_model.py --board LoadShedding --validation_mode component_holdout

# Cross-PCB validation (train HBridge → validate LoadShedding)
python train_hbridge_model.py --validation_mode cross_pcb

# Build datasets first, then train
python train_hbridge_model.py --build_dataset --board HBridge --validation_mode component_holdout
python train_hbridge_model.py --build_dataset --validation_mode cross_pcb
```

### Option 3: Build datasets manually

```bash
cd /home/forrest/home/modeling/thermal_post_processing/ml_model/cnn_thermal_modeling

# Build HBridge dataset
python build_dataset.py HBridge --component_val_split 0.2    # Component holdout
python build_dataset.py HBridge --component_val_split 0.0    # Cross-PCB mode

# Build LoadShedding dataset
python build_dataset.py LoadShedding --component_val_split 0.2    # Component holdout
python build_dataset.py LoadShedding --component_val_split 0.0    # Cross-PCB mode
```

## Validation Modes Explained

### Component Holdout (Traditional)
- **Use case**: Test if model learns thermal coupling between components
- **Method**: Randomly split components 80/20, train on 80%, validate on 20%
- **Expected results**: R² = -1 to -3 (model copies FLIR, doesn't learn coupling)
- **Command**: `--validation_mode component_holdout --board HBridge`

### Cross-PCB (NEW - Recommended for deployment testing)
- **Use case**: Test if model can generalize to NEW PCB designs
- **Method**: Train on HBridge (22 components), validate on LoadShedding (20 components)
- **Expected results**: R² = -10 to -50 (proves location-specific learning)
- **Command**: `--validation_mode cross_pcb`
- **Purpose**: Establishes baseline for future architectural improvements

## Dataset Overview

### HBridge
- **Frames**: 301 (120 physical + 181 steady-state extended)
- **Timestamps**: 8719 (36.3 hours)
- **Components**: 22
- **FLIR range**: 18.9-63.3°C
- **Embedded range**: 22.1-67.3°C
- **File**: `datasets/HBridge_cnn_dataset.h5` (90 MB)

### LoadShedding
- **Frames**: 121 (120 physical + 1 steady-state extended)
- **Timestamps**: 4219 (17.6 hours)
- **Components**: 20
- **FLIR range**: 19.6-33.5°C
- **Embedded range**: 20.6-29.6°C
- **Time offset**: +20s (FLIR leads thermistor)
- **ROI alignment**: <0.5px verified ✓
- **File**: `datasets/LoadShedding_cnn_dataset.h5` (20 MB)

## Expected Results

### Component Holdout (Current Baseline)
**HBridge:**
- R² ≈ -3.5
- RMSE ≈ 10°C
- Diagnostic: r(pred, FLIR) = 0.993 (model copying FLIR)

### Cross-PCB (Expected Poor Performance)
**Train: HBridge → Validate: LoadShedding:**
- **Expected R² < -10** (much worse than component holdout)
- **Why**: Model learns HBridge pixel positions, not physics
- **Evidence needed**: This proves need for architectural changes

## Troubleshooting

### Dataset Not Found
```bash
# Build the missing dataset
python build_dataset.py HBridge
python build_dataset.py LoadShedding
```

### ROI Misalignment
```bash
# Verify ROI masks align with pixel maps
cd /home/forrest/home/modeling/thermal_post_processing
python ../../viz_phase8_compare_csv_vs_hdf5.py HBridge
python ../../viz_phase8_compare_csv_vs_hdf5.py LoadShedding
```

### Memory Issues
- Use `batch_size=8` (recommended for CPU)
- Avoid `batch_size > 8` (causes hanging/OOM)
- Check memory with: `top` or `htop`

## Next Steps After Baseline

1. **Phase 2**: Temporal validation (sanity check)
2. **Phase 3**: Patch-based architecture (position-independent)
3. **Phase 4**: Multi-PCB training
4. **Phase 5**: Advanced fixes (attention, PINN, dense supervision)

See `CROSS_PCB_VALIDATION_ANALYSIS.txt` for full implementation strategy.
