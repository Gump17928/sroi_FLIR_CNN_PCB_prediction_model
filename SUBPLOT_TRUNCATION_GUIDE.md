# Subplot Truncation Configuration Guide

## Overview
You can now independently control the time limits for each subplot (FLIR, Air, Sand) in your three-condition comparison plots using the JSON configuration file.

## Configuration Options

### In `All_Boards_multi_test_config.json`

Each board's `plot_settings` section supports the following truncation controls:

```json
"plot_settings": {
  "y_axis_min": 20,
  "y_axis_max": 70,
  "unified_y_axis": true,
  
  // Global time limits (applied across all tests)
  "max_sand_time_hours": 36,
  "max_air_time_hours": null,
  
  // Test-specific truncation (only for Test2 sand data)
  "test2_truncate_hours": null,
  
  // Per-subplot truncation limits (NEW!)
  "flir_subplot_max_hours": null,
  "air_subplot_max_hours": null,
  "sand_subplot_max_hours": null
}
```

## How Each Setting Works

### 1. `flir_subplot_max_hours`
Controls the maximum time shown in the FLIR camera subplot.
- **null**: Shows all FLIR data
- **number**: Truncates FLIR time axis to specified hours

**Example**: `"flir_subplot_max_hours": 3` → FLIR subplot shows 0-3 hours only

---

### 2. `air_subplot_max_hours`
Controls the maximum time shown in the Air thermocouple subplot.
- **null**: Uses `max_air_time_hours` if set, otherwise shows all data
- **number**: Uses the **stricter** of `air_subplot_max_hours` and `max_air_time_hours`

**Example**: 
```json
"max_air_time_hours": 5,
"air_subplot_max_hours": 3
```
→ Air subplot shows 0-3 hours (stricter limit)

---

### 3. `sand_subplot_max_hours`
Controls the maximum time shown in the Sand thermocouple subplot.
- **null**: Uses `max_sand_time_hours` if set, otherwise shows all data
- **number**: Uses the **stricter** of `sand_subplot_max_hours` and `max_sand_time_hours`

**Example**:
```json
"max_sand_time_hours": 36,
"sand_subplot_max_hours": 10
```
→ Sand subplot shows 0-10 hours (stricter limit)

**Note**: `test2_truncate_hours` still applies separately for Test2 sand data.

---

## Use Cases

### Use Case 1: Match FLIR Duration to Air/Sand
FLIR data might run longer than needed. Truncate it to match other measurements:

```json
"flir_subplot_max_hours": 5,
"air_subplot_max_hours": 5,
"sand_subplot_max_hours": 5
```
All subplots show 0-5 hours.

---

### Use Case 2: Different Boards, Different Scales

**LoadShedding Board** (fast thermal response):
```json
"plot_settings": {
  "flir_subplot_max_hours": 2,
  "air_subplot_max_hours": 2,
  "sand_subplot_max_hours": 7
}
```
FLIR and Air show 0-2 hours, Sand shows 0-7 hours.

**HBridge Board** (slow thermal response):
```json
"plot_settings": {
  "flir_subplot_max_hours": 10,
  "air_subplot_max_hours": 10,
  "sand_subplot_max_hours": 36
}
```
FLIR and Air show 0-10 hours, Sand shows full 36-hour duration.

---

### Use Case 3: Focus on Transient vs. Steady-State

Show transient region in FLIR, but full steady-state in embedded measurements:

```json
"flir_subplot_max_hours": 1,     // First hour only
"air_subplot_max_hours": null,   // Full duration
"sand_subplot_max_hours": null   // Full duration
```

---

## Priority Order

When multiple limits are set, the code uses the **stricter** (shorter) limit:

### Air Subplot:
```
Effective Limit = min(max_air_time_hours, air_subplot_max_hours)
```

### Sand Subplot:
```
Effective Limit = min(max_sand_time_hours, sand_subplot_max_hours)
```

### Test2 Sand Data:
```
If "Test2" in test name:
    Effective Limit = min(max_sand_time_hours, sand_subplot_max_hours, test2_truncate_hours)
```

---

## Example Configuration

```json
{
  "boards": [
    {
      "name": "HBridge_15s",
      "pcb": "H_Bridge_Sensing_Rev_C.brd",
      "plot_settings": {
        "y_axis_min": 20,
        "y_axis_max": 70,
        "unified_y_axis": true,
        "max_sand_time_hours": 36,
        "max_air_time_hours": null,
        "test2_truncate_hours": null,
        
        // Independent subplot limits
        "flir_subplot_max_hours": 8,
        "air_subplot_max_hours": 8,
        "sand_subplot_max_hours": 24
      },
      "tests": [ ... ]
    }
  ]
}
```

This configuration:
- FLIR subplot: Shows 0-8 hours
- Air subplot: Shows 0-8 hours  
- Sand subplot: Shows 0-24 hours (stricter than max_sand_time_hours=36)

---

## Testing Your Configuration

1. Edit `All_Boards_multi_test_config.json`
2. Set desired truncation values for each board
3. Run: `python researchir_post_processor.py`
4. Check plots in: `outputs/<timestamp>_P1-7/three_condition_comparison/`

The plots will automatically use your configured time limits for each subplot.
