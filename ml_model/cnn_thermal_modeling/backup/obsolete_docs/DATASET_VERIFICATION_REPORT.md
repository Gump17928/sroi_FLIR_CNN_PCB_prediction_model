# CNN Dataset Verification Report

**Generated:** January 13, 2026  
**Purpose:** Verify training data configuration before model training

---

## ❌ CRITICAL FINDING: Incomplete Dataset Configuration

### Current State: **ONLY HBRIDGE DATA CONFIGURED**

The pipeline is currently configured to train on **ONLY HBridge data**, missing the LoadShedding board entirely.

---

## 📊 Available Data Inventory

### ✅ FLIR Thermal Images (Both Boards Available)

| Board | Folder | Status |
|-------|--------|--------|
| **HBridge** | `inputs/ResearchIR_Outputs_HBridge_15s/` | ✅ **Currently Used** |
| **LoadShedding** | `inputs/ResearchIR_Outputs_Load_Shedding/` | ❌ **NOT Used** |

### ✅ Thermistor Ground Truth (Both Boards Available)

Found in: `outputs/0112_1126_P1-7/`

| Board | File | Status |
|-------|------|--------|
| **HBridge** | `HBridge_15s_thermistor_timeseries.csv` | ✅ **Currently Used** |
| **LoadShedding** | `Load_Shedding_thermistor_timeseries.csv` | ❌ **NOT Used** |

### ✅ ROI Pixel Maps (Both Boards Available)

Found in: `outputs/roi_pixel_maps/`

| Board | File | Status |
|-------|------|--------|
| **HBridge** | `HBridge_roi_pixel_map.csv` | ✅ **Currently Used** |
| **LoadShedding** | `LoadShedding_roi_pixel_map.csv` | ❌ **NOT Used** |

---

## 🔍 Current Pipeline Configuration

### File: `build_hbridge_dataset.py`

**Currently builds ONLY:**
```python
flir_folder = "inputs/ResearchIR_Outputs_HBridge_15s"
thermistor_csv = "outputs/0112_1126_P1-7/HBridge_15s_thermistor_timeseries.csv"
pixel_map_csv = "outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv"
output_h5 = "datasets/HBridge_cnn_dataset.h5"
```

### File: `run_smart_pipeline.py`

**Trains ONLY on:**
```python
import train_hbridge_model  # Only HBridge
```

---

## 🎯 What You Need

To train on BOTH boards (as you requested), you need:

### Option 1: Combined Dataset (Recommended)
**Single dataset with both boards mixed together**
- More training data (better model)
- Single model works on both board types
- Learns general thermal patterns

### Option 2: Separate Datasets
**Two separate datasets, one per board**
- Board-specific models
- Easier to compare performance
- Can identify board-specific issues

### Option 3: Sequential Training
**Train on HBridge first, then fine-tune on LoadShedding**
- Transfer learning approach
- Good if one board has more data

---

## ✅ Verification Checklist

Before training, verify:

- [ ] **FLIR Images:** Do both `ResearchIR_Outputs_HBridge_15s` and `ResearchIR_Outputs_Load_Shedding` contain thermal CSV files?
- [ ] **Thermistor Data:** Are the thermistor CSV files complete with all component columns?
- [ ] **ROI Pixel Maps:** Do pixel maps have correct component names matching thermistor columns?
- [ ] **Temporal Alignment:** Are FLIR and thermistor data from the same test runs?
- [ ] **Data Intervals:** Are both datasets at 15s intervals?

---

## 🚨 Action Required

**You have 3 options:**

### 1️⃣ Train on HBridge Only (Current Setup - No Changes)
- Run `python run_from_here.py` as-is
- Quick to test, but limited to one board

### 2️⃣ Add LoadShedding Dataset (Create Second Dataset)
- Create `build_loadshedding_dataset.py`
- Build second HDF5 file
- Modify pipeline to train on both

### 3️⃣ Build Combined Multi-Board Dataset (Recommended)
- Create `build_combined_dataset.py`
- Merge both boards into single HDF5
- Train one model on all data

---

## 📝 Recommended Next Steps

1. **Verify Data Quality First:**
   ```bash
   # Check HBridge FLIR folder
   ls inputs/ResearchIR_Outputs_HBridge_15s/*.csv | wc -l
   
   # Check LoadShedding FLIR folder
   ls inputs/ResearchIR_Outputs_Load_Shedding/*.csv | wc -l
   
   # Check thermistor files
   head -5 outputs/0112_1126_P1-7/HBridge_15s_thermistor_timeseries.csv
   head -5 outputs/0112_1126_P1-7/Load_Shedding_thermistor_timeseries.csv
   ```

2. **Decide on Training Strategy:**
   - Combined dataset? (Most training data)
   - Separate datasets? (Board-specific models)
   - HBridge only for testing? (Fastest to verify pipeline works)

3. **Build Dataset:**
   - Run appropriate build script
   - Verify HDF5 output

4. **Train Model:**
   - Run `python run_from_here.py`
   - Monitor training progress

---

## 🔧 Quick Fix: Add LoadShedding Support

If you want to train on BOTH boards with minimal changes:

1. **Create** `build_loadshedding_dataset.py` (copy from HBridge version)
2. **Modify** `run_smart_pipeline.py` to build both datasets
3. **Train** separate models or combine datasets

Would you like me to create these files for you?

---

## 📊 Expected Dataset Sizes

**HBridge_15s:**
- Frames: ~300 (15s intervals over ~75 minutes)
- Components: ~30-50 (depends on thermistor coverage)
- File size: ~200-400 MB

**LoadShedding:**
- Frames: ~TBD (check folder)
- Components: ~TBD (check thermistor CSV)
- File size: ~TBD

**Combined:**
- Frames: ~600+
- Components: ~60-100
- File size: ~400-800 MB

---

## ✅ Conclusion

**Current Status:** Pipeline is configured for **HBridge ONLY**

**Your Requirement:** Train on **BOTH HBridge AND LoadShedding**

**Recommendation:** 
1. Verify LoadShedding data quality first
2. Create combined dataset builder
3. Train single model on both boards

This will give you the most robust model with maximum training data.
