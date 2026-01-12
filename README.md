# SROI FLIR CNN PCB Prediction Model

**Complete 7-Phase Thermal Analysis Pipeline with Machine Learning**  
*ResearchIR Data Analysis • Thermistor Validation • Thermal Modeling & CNN Prediction*

The sroi_FLIR_CNN_PCB_prediction_model is a comprehensive thermal post-processing system for learning about and reverse engineering the .sroi files that exist for FLIR cameras, and utilizing ML models to predict embedded thermal temperatures of PCBs.

---

## Overview

Comprehensive thermal post-processing for PCB analysis using FLIR thermal camera and thermistor measurements. Provides 7 phases from raw data processing through thermal modeling and prediction.

**Key Capabilities:**
- **Phases 1-5**: Thermal analysis (filtering, statistics, spatial coupling, risk assessment)
- **Phase 6**: Calibration using thermistor-validated measurements
- **Phase 7**: Temperature prediction for FLIR-only PCBs

**Folder Structure:**

```
thermal_post_processing/
├── inputs/                          # Input data
│   ├── ResearchIR_Outputs_*/       # FLIR CSV exports
│   ├── *_usb_temp_DAQami.csv       # Thermistor data
│   ├── *_components_enhanced.csv   # Coordinates
│   └── *_thermistor_mapping.json   # Channel mapping
├── outputs/                         # Results (auto-timestamped)
│   ├── MMDD_HHMM_P1-5/             # Phases 1-5
│   ├── MMDD_HHMM_P6-7/             # Phases 6-7
│   └── MMDD_HHMM_P1-7/             # Complete
├── researchir_post_processor.py    # Main entry
├── phase[1-7]_*.py                 # Analysis modules
└── README.md                        # This file
```

---

## Quick Start

**Interactive Mode (Recommended):**
```powershell
python researchir_post_processor.py
```
- Select workflow (Phases 1-5, 6-7, or 1-7), validate paths, toggle debug mode

**Command-Line Mode:**
```powershell
python researchir_post_processor.py --no_ui              # Phases 1-5
python researchir_post_processor.py --thermal_modeling   # Phases 6-7
python researchir_post_processor.py --full_pipeline      # Phases 1-7
python researchir_post_processor.py --full_pipeline --debug  # Verbose output
```

**Output Folders:** Auto-generated as `MMDD_HHMM_<workflow>[_debug]` (e.g., `1126_1430_P1-7`). Override with `--output` flag.

## System Architecture

**Three Workflows:**
1. **Phases 1-5**: Standard analysis (loading, filtering, statistics, spatial coupling, risk)
2. **Phases 6-7**: Thermal modeling (calibration & prediction)
3. **Full Pipeline**: Complete Phases 1-7

## Analysis Phases (1-5)

| Phase | Purpose | Key Features | Outputs |
|-------|---------|--------------|---------|
| **1. Data Loading** | Load ResearchIR exports | Auto-detect CSV/TXT, validate data, classify components | Validated dataset |
| **2. Filtering** | Compare filter methods | No filter, moving avg, median (recommended), Savitzky-Golay | `filtering_comparison_*.png`, CSV |
| **3. Component Analysis** | Statistical profiling | Group by type, calc stats, identify hotspots, IEEE plots | `thermal_analysis_summary.*`, MATLAB export |
| **4. Spatial Coupling** | Thermal interactions | Proximity detection (10mm), coupling strength, heat sources | `thermal_coupling_*.pdf` |
| **5. Potting Risk** | Failure prediction | Risk scoring for potted systems (requires Phase 4) | `potted_condition_risk_analysis.csv` |

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
python researchir_post_processor.py --full_pipeline    # Phases 1-7
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
```

## Input/Output Files

**Inputs:** `inputs/ResearchIR_Outputs_<PCB>/*.csv`, `*_components_enhanced.csv`, `*_usb_temp_DAQami.csv`, `*_thermistor_mapping.json`

**Outputs:** Auto-timestamped folders (`MMDD_HHMM_P*`) containing:
- **P1-5**: Analysis summaries (PNG/CSV/PDF), filtering comparisons, coupling maps, risk assessment, MATLAB export
- **P6-7**: Calibration database (CSV/JSON/PNG), prediction results, detailed comparison plots (PNG+PDF)
- **P1-7**: Complete combined outputs

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

**Required:** Python 3.7+, numpy, pandas, matplotlib, scipy

**Install:** `pip install numpy pandas matplotlib scipy`

**Optional:** LaTeX (for publication-quality rendering)

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

**Debug Mode:** Add `--debug` flag for verbose per-component output

**Validation Checklist:**
- [ ] ResearchIR CSV exports in `inputs/`
- [ ] Coordinates CSV (for spatial), thermistor CSVs (for Phase 6)
- [ ] Thermistor mapping JSON (for Phase 6)
- [ ] Dependencies installed

**Advanced:**
- **Batch processing:** Loop through input folders with `--no_ui`
- **MATLAB export:** Load `thermal_data_export.mat` from outputs
- **Calibration expansion:** Add PCBs to Phase 6, database auto-updates
- **Portability:** Fully portable with relative paths

---

## Additional Resources

- **Incremental Calibration Guide:** See `INCREMENTAL_CALIBRATION_GUIDE.md` for detailed workflows and troubleshooting
- **Multi-Session Example:** See `multi_session_config_example.json` for complete configuration template
- **Phase Documentation:** See individual `phase*.py` files for detailed module documentation

---


**Version:** v2.2 (Dec 5, 2025) - Added multi-device thermistor support for sessions with multiple USB-TEMP devices  
**Previous:** v2.1 (Dec 2, 2025) - Added incremental calibration, multi-session processing, and convergence analysis  
**Previous:** v2.0 (Nov 26, 2025) - Complete 7-phase pipeline with auto-timestamped outputs
