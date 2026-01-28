# Thermistor Validation Module

## Overview

The `thermistor_validation.py` module compares FLIR thermal camera measurements with USB thermistor data to validate thermal measurement accuracy. This is a Python port of the MATLAB script `plot_flir_cursors_usb_temp.m`.

## Features

- **Side-by-side temperature comparison plots**
  - Overlay of FLIR camera and thermistor measurements
  - Delta T (temperature difference) visualization
  - Steady-state value calculation and display

- **Multi-condition support**
  - Air cooling vs. Sand cooling comparison
  - Automatic pairing of matching measurement channels

- **Robust data processing**
  - Automatic file format detection (FLIR vs. USB thermistor)
  - Time synchronization and interpolation
  - Smoothing filters to reduce measurement noise
  - Steady-state detection algorithm

## Input Files

The module expects CSV files in the `inputs/` directory:

### FLIR Camera Data Format
- Contains `reltime` column for time axis
- Temperature data in cursor/box columns (e.g., "Box 1 [C]:mean")
- Example: `Test_3_FLIR_Camera_Results_5_of_7.csv`

### USB Thermistor Data Format
- Contains `Sample,Date/Time` header line
- Scan rate metadata (optional, for time calculation)
- Analog input channels: AI0, AI1, AI4, AI6, AI7, etc.
- Examples:
  - `Test_3_AIR_usb_temp_DAQami.csv` (air cooling)
  - `Test_3_SAND_usb_temp_DAQami.csv` (sand cooling)

## Output Files

All outputs are saved to the `outputs/` directory:

### Comparison Plots (PNG format, 600 DPI)
- **FLIR vs Sand**: Compares FLIR camera with sand-cooled thermistors
  - `Test_3_AIR_usb_temp_DAQami_FLIR_Air_vs_Sand_LDO2_pair_compare.png`
  - `Test_3_AIR_usb_temp_DAQami_FLIR_Air_vs_Sand_PS2_pair_compare.png`
  - `Test_3_AIR_usb_temp_DAQami_FLIR_Air_vs_Sand_SMD_RES1_pair_compare.png`
  - `Test_3_AIR_usb_temp_DAQami_FLIR_Air_vs_Sand_LED1_pair_compare.png`

- **Air vs Sand**: Compares air-cooled vs sand-cooled thermistors
  - `Test_3_SAND_usb_temp_DAQami_vsSand_LDO2_pair_compare.png`
  - `Test_3_SAND_usb_temp_DAQami_vsSand_PS2_pair_compare.png`
  - `Test_3_SAND_usb_temp_DAQami_vsSand_Ambient_pair_compare.png`
  - `Test_3_SAND_usb_temp_DAQami_vsSand_SMD_RES1_pair_compare.png`
  - `Test_3_SAND_usb_temp_DAQami_vsSand_LED1_pair_compare.png`

Each plot contains:
- Left panel: Temperature overlay with steady-state lines
- Right panel: Delta T with steady-state value

## Usage

### Standalone Execution

```bash
python thermistor_validation.py
```

The script will automatically:
1. Load data from `inputs/` directory
2. Process and compare matched channels
3. Generate comparison plots in `outputs/` directory
4. Print progress and summary to console

### Configuration

Edit the `main()` function in `thermistor_validation.py` to customize:

**Input files:**
```python
flir_file = input_dir / "Your_FLIR_File.csv"
air_file = input_dir / "Your_Air_Thermistor_File.csv"
sand_file = input_dir / "Your_Sand_Thermistor_File.csv"
```

**Plot parameters:**
```python
validator = ThermistorValidator(
    fig_size=(7.0, 3.5),     # Figure size (inches)
    line_width=2.0,           # Line width
    dpi=600,                  # Resolution
    smooth_win=20,            # Smoothing window (samples)
    y_tick_step=1.0,          # Y-axis tick spacing (°C)
    data_shift=10.0           # Skip initial transient (seconds)
)
```

**Comparison pairs:**
```python
pair_flir_comp = [
    ('FLIR_pattern', 'USB_pattern', 'Display_Name', 'air_label', 'sand_label'),
    # Example:
    ('Box 3', 'AI0', 'LDO2', 'therm-air', 'therm-sand'),
]
```

## Algorithm Details

### Steady-State Detection
The module uses a robust algorithm to find steady-state conditions:
1. **Smoothing**: Moving average filter to reduce noise
2. **Slope detection**: Identifies regions with low temperature gradient
3. **Flatness refinement**: Ensures temperature variation is minimal
4. **Duration check**: Requires minimum duration (default 120s)
5. **Center calculation**: Median or mean of steady-state region

### Data Processing Pipeline
1. **Load and parse** CSV files (auto-detect format)
2. **Time alignment** via interpolation to common timebase
3. **Smoothing** with moving average filter
4. **Steady-state calculation** for each series
5. **Delta computation** with hybrid tail substitution
6. **Plot generation** with IEEE-format styling

## Dependencies

```
numpy
pandas
matplotlib
scipy
pathlib (standard library)
datetime (standard library)
re (standard library)
```

Install via:
```bash
pip install numpy pandas matplotlib scipy
```

## Comparison with MATLAB Original

### Preserved Functionality ✓
- Data loading for both FLIR and USB thermistor formats
- Steady-state detection algorithm
- Pair comparison plots with overlay and delta
- Smoothing and interpolation
- Configurable plot parameters
- Export to high-resolution PNG

### Differences
- **GUI window arrangement**: MATLAB's `autoArrangeFigures_fancy()` not ported (VS Code limitation)
- **LaTeX rendering**: Optional in Python (requires LaTeX installation)
- **File selection**: Hardcoded paths instead of MATLAB's `uigetfile()`

### Python Advantages
- Cleaner object-oriented design
- Better integration with thermal_post_processing pipeline
- More maintainable and extensible code structure
- Familiar ecosystem for Python developers

## Future Enhancements

Potential additions for integration:
1. **CSV summary export** (steady-state values table)
2. **Ratio plots** (B/A temperature ratio)
3. **Statistical analysis** (correlation, RMSE, etc.)
4. **Integration with main_processor.py** as optional validation phase
5. **Interactive file selection** (GUI or CLI prompts)
6. **Batch processing** for multiple test runs

## Original MATLAB Source

This module is based on:
- **File**: `plot_flir_cursors_usb_temp.m`
- **Location**: `C:\Users\A02242301\Box\PRJ_A57941_Pilot_DWPT_Embedded\30_Hardware_Development\60_Hardware_Testing\IGBT_Board_PCB_Testing\`
- **Date**: November 17, 2025
- **Size**: ~36 KB

## Contact

For questions or issues with this module, refer to the main thermal_post_processing documentation or contact the development team.
