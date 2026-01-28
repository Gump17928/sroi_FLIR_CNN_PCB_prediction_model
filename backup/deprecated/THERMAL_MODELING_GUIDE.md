# Thermal Calibration & Prediction System - Usage Guide

## Overview

You now have a complete thermal modeling pipeline with **Phase 6 (Calibration)** and **Phase 7 (Prediction)** that learns thermal relationships from thermistor-validated PCBs and applies them to FLIR-only measurements.

---

## Phase 6: Thermal Calibration (COMPLETE)

### What It Does
- Compares FLIR camera measurements to ground-truth thermistor data
- Calculates component-type-specific FLIR correction offsets
- Models air vs sand cooling effectiveness by component type
- Exports calibration database for future predictions

### Calibration Results (LoadShedding_AC_Switch)

| Component Type | FLIR Offset | Air SS Temp | Cooling Benefit | Cooling Ratio |
|----------------|-------------|-------------|-----------------|---------------|
| **LDO**        | +2.89°C     | 24.61°C     | -1.04°C         | 1.0423        |
| **PowerSupply**| +3.37°C     | 29.49°C     | +1.73°C         | 0.9413        |
| **Resistor**   | +2.00°C     | 24.57°C     | +0.04°C         | 0.9985        |
| **LED**        | +2.60°C     | 24.49°C     | -0.56°C         | 1.0228        |

**Key Insights:**
- FLIR camera reads **+2.0°C to +3.4°C higher** than actual thermistor measurements
- This offset varies by component type (power supplies have larger offset)
- Sand cooling shows minimal benefit for most components (±0-2°C difference)
- Note: Sand cooling showed unusual results for LDO2 (-1.04°C worse) - likely measurement artifact

### Output Files
```
outputs/
├── thermal_calibration_points.csv       # Individual calibration measurements
├── thermal_calibration_by_type.csv      # Component type statistics
├── thermal_calibration_metadata.json    # Complete calibration info
└── thermal_calibration_summary.png      # Visualization of offsets
```

### Usage
```bash
python phase6_thermal_calibration.py
```

To add more calibration PCBs, edit `phase6_thermal_calibration.py` and add more `calibrator.add_measurement_pair()` calls.

---

## Phase 7: Thermal Prediction (READY FOR H-BRIDGE DATA)

### What It Will Do
1. Load H-Bridge FLIR camera measurements (air cooling only)
2. Apply learned FLIR correction offsets by component type
3. Predict true air temperatures for each component
4. Predict sand cooling temperatures using cooling models
5. Export temperature predictions and confidence metrics

### What You Need to Provide

#### 1. **H-Bridge FLIR Thermal Data** (REQUIRED)
- **Format**: ResearchIR CSV export (same format as LoadShedding Test_3 file)
- **Must contain**: `reltime` column + temperature data columns (Cursor/Box ROIs)
- **Condition**: Air cooling measurements
- **Where to place**: `thermal_post_processing/inputs/H_Bridge_FLIR_Camera_Results.csv`

#### 2. **Component Type Mapping** (REQUIRED)
You need to map H-Bridge FLIR ROI names to component types. Edit `phase7_thermal_prediction.py` line ~260:

```python
component_mapping = {
    # Map FLIR patterns to component types from calibration
    r'Cursor 1': 'Resistor',       # Example: Cursor 1 is a resistor
    r'Cursor 2': 'Resistor',       # Cursor 2 is also a resistor
    r'Cursor 4': 'IC',             # Cursor 4 is an IC
    r'Box.*PS1': 'PowerSupply',    # Any box with PS1 is power supply
    r'Box.*LDO': 'LDO',            # Any box with LDO is linear regulator
    # Add mappings for ALL your H-Bridge components
}
```

**Available Component Types from Calibration:**
- `'LDO'` - Linear regulators
- `'PowerSupply'` - Power supply modules
- `'Resistor'` - SMD resistors
- `'LED'` - LED indicators

**How to determine mapping:**
1. Look at your H-Bridge SROI file or CSV to see ROI names
2. Cross-reference with H-Bridge pick-and-place to identify component types
3. Create regex patterns matching ROI names to types

---

## Workflow Diagram

PHASE 6: CALIBRATION (LoadShedding with Thermistors)

Input:
- Test_3_FLIR_Camera_Results_5_of_7.csv  (FLIR air)
- Test_3_AIR_usb_temp_DAQami.csv         (Thermistor air)
- Test_3_SAND_usb_temp_DAQami.csv        (Thermistor sand)

Process:
1. Compare FLIR SS temps vs Thermistor SS temps
2. Calculate offset: FLIR_temp - Thermistor_temp
3. Group offsets by component type (LDO, PS, R, LED)
4. Calculate air to sand cooling factors

Output:
- thermal_calibration_by_type.csv (CALIBRATION DATABASE)

---

PHASE 7: PREDICTION (H-Bridge FLIR-only)

Input:
- H_Bridge_FLIR_Camera_Results.csv  (FLIR air)
- thermal_calibration_by_type.csv   (from Phase 6)
- component_mapping dict            (user-defined)

Process:
1. Load H-Bridge FLIR measurements
2. For each component:
   a. Determine component type (from mapping)
   b. Apply FLIR offset correction
      Corrected_Air = FLIR_raw - Offset
   c. Predict sand cooling temperature
      Predicted_Sand = Corrected_Air * Cooling_Ratio
3. Assign confidence based on calibration quality

Output:
- thermal_prediction_H_Bridge_Sensing.csv
- thermal_prediction_*_visualization.png
- thermal_prediction_*_summary.json

---

## Example: H-Bridge Prediction Output

Once you provide the H-Bridge FLIR data, Phase 7 will generate:

### CSV Output (`thermal_prediction_H_Bridge_Sensing.csv`)
```csv
component_name,component_type,flir_raw_ss,flir_offset_applied,predicted_air_ss,predicted_sand_ss,predicted_temp_drop,confidence
Cursor 1,Resistor,26.5,2.00,24.5,24.46,0.04,High
Cursor 2,Resistor,28.3,2.00,26.3,26.26,0.04,High
Cursor 4,IC,42.1,2.72,39.38,38.15,1.23,Medium
Box PS1,PowerSupply,55.2,3.37,51.83,48.79,3.04,High
Box LDO1,LDO,38.4,2.89,35.51,37.01,-1.50,High
...
```

### Visualization
- **Plot 1**: FLIR Raw vs Corrected temperatures (scatter)
- **Plot 2**: Calibration correction applied (before/after)
- **Plot 3**: Air vs Sand predictions
- **Plot 4**: Temperature distribution boxplots

---

## Next Steps

### Immediate Actions Needed:

1. **Capture or locate H-Bridge FLIR thermal data**
   - Use ResearchIR to export temperature statistics
   - Should have same format as LoadShedding Test_3 file
   - Place in: `thermal_post_processing/inputs/H_Bridge_FLIR_Camera_Results.csv`

2. **Create component type mapping**
   - Review your H-Bridge SROI ROI names
   - Map each ROI pattern to component type
   - Edit `phase7_thermal_prediction.py` `component_mapping` dict

3. **Run Phase 7**
   ```bash
   python phase7_thermal_prediction.py
   ```

### Optional Enhancements:

- **Add more calibration PCBs** to Phase 6 for better statistics
- **Validate predictions** if you have H-Bridge thermistor data
- **Refine component mapping** based on prediction confidence
- **Export to MATLAB** for further analysis

---

## Calibration Database Schema

### `thermal_calibration_by_type.csv` Structure:
```
component_type: string - Component category (LDO, PowerSupply, etc.)
count: int - Number of calibration samples for this type
flir_offset_mean: float - Average FLIR correction offset (°C)
flir_offset_std: float - Standard deviation of offset
flir_offset_median: float - Median offset (more robust)
cooling_benefit_mean: float - Average temp drop from air to sand (°C)
cooling_benefit_std: float - Standard deviation of cooling
cooling_ratio_mean: float - Sand_temp / Air_temp ratio
cooling_ratio_std: float - Standard deviation of ratio
air_ss_mean: float - Average air steady-state temperature
air_ss_std: float - Standard deviation of air temperature
```

### How Prediction Works:

**Step 1: Correct FLIR measurement**
```
Corrected_Air_Temp = FLIR_Raw_Temp - FLIR_Offset[component_type]
```

**Step 2: Predict sand cooling**
```
Method 1 (Delta): Predicted_Sand = Corrected_Air - Cooling_Benefit[component_type]
Method 2 (Ratio): Predicted_Sand = Corrected_Air × Cooling_Ratio[component_type]
```

**Step 3: Assign confidence**
- **High**: Offset std dev < 2.0°C, multiple calibration samples
- **Medium**: Offset std dev ≥ 2.0°C or single calibration sample
- **Low**: No calibration for component type (uses average)

---

## Troubleshooting

### Issue: "Calibration file not found"
**Solution**: Run Phase 6 first to generate calibration database

### Issue: "Unknown component type"
**Solution**: Add component type to `component_mapping` or calibration will use average offset

### Issue: "FLIR file not found"
**Solution**: Place H-Bridge FLIR CSV in `inputs/` folder with correct filename

### Issue: Negative cooling benefit
**Explanation**: LoadShedding LDO2 showed anomalous result (-1.04°C). This could be:
- Thermistor placement artifact (2.5mm tip averaging nearby components)
- Measurement timing differences
- Actual thermal coupling effects
Consider this when interpreting predictions for LDO components.

---

## Files Summary

### Created:
- `phase6_thermal_calibration.py` - Calibration pipeline
- `phase7_thermal_prediction.py` - Prediction pipeline
- `thermal_calibration_points.csv` - Raw calibration data
- `thermal_calibration_by_type.csv` - Aggregated by component type
- `thermal_calibration_metadata.json` - Complete metadata
- `thermal_calibration_summary.png` - Calibration visualization

### Required:
- `inputs/H_Bridge_FLIR_Camera_Results.csv` - **YOU NEED TO PROVIDE THIS**

### Will Generate (after Phase 7):
- `thermal_prediction_H_Bridge_Sensing.csv`
- `thermal_prediction_H_Bridge_Sensing_summary.json`
- `thermal_prediction_H_Bridge_Sensing_visualization.png`

---

## Contact

For questions about thermal modeling or prediction confidence, review the calibration summary visualization and component-type statistics in the output files.
