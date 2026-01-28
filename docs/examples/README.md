# Test Script Examples

This folder contains example test scripts that demonstrate various features of the thermal post-processing workflow. These scripts were used during development to validate functionality and can serve as reference examples for users.

## ⚠️ Important Note

**These test scripts are now superseded by the integrated validation workflow.**

Instead of running individual test scripts, use the main entry point with validation:

```bash
# Validate your configuration before processing
python researchir_post_processor.py --thermal_modeling \
    --multi_session_config your_config.json --validate

# Run full workflow after validation passes
python researchir_post_processor.py --thermal_modeling \
    --multi_session_config your_config.json --convergence_analysis
```

## Test Scripts Overview

### `test_batch1.py`
**Purpose:** Validates single-device thermistor calibration processing.

**What it tests:**
- Single USB-TEMP device with 6 channels
- Basic calibration workflow
- Output folder structure creation

**Replaced by:** Standard single-session mode in `researchir_post_processor.py`

**Config equivalent:**
```json
{
  "sessions": [{
    "name": "Batch1_Test",
    "therm_air_file": "inputs/AIR_usb_temp.csv",  // Single file (not array)
    "pairs": [...],  // Standard channel names (AI0, AI1, etc.)
  }]
}
```

---

### `test_batch2_plotting.py`
**Purpose:** Validates multi-device thermistor merging and comprehensive plotting features.

**What it tests:**
- Multi-device setup (USB-TEMP + USB-TEMP-AI = 8 channels)
- Device{N}_ channel prefix handling
- Raw input overview plot generation
- Detailed comparison plot generation
- Session-specific output folders

**Replaced by:** Multi-device mode with automatic plotting in `researchir_post_processor.py`

**Config equivalent:**
```json
{
  "sessions": [{
    "name": "Batch2_MultiDevice",
    "therm_air_file": [  // Array of files
      "inputs/AIR_device0.csv",
      "inputs/AIR_device1.csv"
    ],
    "pairs": [
      {"flir_roi": "U2", "therm_air_chan": "Device0_AI0", ...},  // Device prefix!
      {"flir_roi": "R5", "therm_air_chan": "Device1_AI4", ...}
    ]
  }]
}
```

---

### `test_final_multibatch.py`
**Purpose:** Tests complete multi-batch workflow with aggregate database export.

**What it tests:**
- Processing multiple sessions in sequence
- Batch 1 (5 components, single device)
- Batch 2 (8 components, multi-device)
- Aggregate calibration database export
- Per-session output folder structure
- Calibration quality summary

**Replaced by:** Multi-session config mode in `researchir_post_processor.py`

**CLI equivalent:**
```bash
python researchir_post_processor.py --thermal_modeling \
    --multi_session_config multi_batch_loadshedding_config.json \
    --convergence_analysis
```

---

### `test_multibatch_complete.py`
**Purpose:** Validates complete multi-batch output structure and file generation.

**What it tests:**
- Session folder creation for each batch
- Raw input overview plots for each session
- Detailed comparison plots for each component
- PNG and PDF export for all plots
- Aggregate output file generation

**Replaced by:** Validation mode + standard processing

**CLI equivalent:**
```bash
# Validate first
python researchir_post_processor.py --thermal_modeling \
    --multi_session_config config.json --validate --verbose_validation

# Run after validation passes
python researchir_post_processor.py --thermal_modeling \
    --multi_session_config config.json
```

---

## How to Use These Examples

### Option 1: Reference (Recommended)
Use these scripts as **code examples** to understand how the workflow functions:
- See how `ThermalCalibrator` is initialized
- Understand `add_measurement_sessions()` usage
- Learn output verification techniques

### Option 2: Direct Execution (Not Recommended)
You can still run these scripts directly for testing:

```bash
cd thermal_post_processing/docs/examples
python test_batch1.py
```

**However,** this approach is **not recommended for production use**. Instead:
1. Create a proper multi-session config JSON file
2. Use the validation workflow (`--validate`)
3. Run through the main entry point (`researchir_post_processor.py`)

## Migration Guide

If you have existing test scripts like these, migrate to the integrated workflow:

### Old Approach (Test Scripts)
```python
# test_my_calibration.py
calibrator = ThermalCalibrator(output_dir=Path('outputs/test'))
calibrator.add_measurement_sessions([session_dict])
print('✓ Processing complete')
```

### New Approach (Integrated Workflow)

**Step 1:** Create config file (`my_calibration_config.json`):
```json
{
  "sessions": [
    {
      "name": "MyCalibrationSession",
      "pcb": "MyPCB",
      "flir_file": "...",
      "therm_air_file": "...",
      "pairs": [...]
    }
  ]
}
```

**Step 2:** Validate configuration:
```bash
python researchir_post_processor.py --thermal_modeling \
    --multi_session_config my_calibration_config.json \
    --validate --verbose_validation
```

**Step 3:** Run workflow:
```bash
python researchir_post_processor.py --thermal_modeling \
    --multi_session_config my_calibration_config.json \
    --convergence_analysis
```

## Benefits of Integrated Workflow

1. **Validation First:** Catch configuration errors before processing
2. **Auto-timestamped Outputs:** No manual output directory management
3. **Progress Tracking:** Clear session-by-session processing feedback
4. **Consistent Interface:** Single entry point for all users
5. **Documentation:** Comprehensive help via `--help` and docstrings

## Questions?

See the main README or run:
```bash
python researchir_post_processor.py --help
```

For thermal modeling configuration examples, check the docstring in `researchir_post_processor.py` function `process_thermal_modeling()`.
