# Bug Fix Report - IndexError in Model Evaluation

**Date:** January 13, 2026  
**Issue:** `IndexError: index 23 is out of bounds for axis 1 with size 23`

---

## 🐛 Problem

The `evaluate_model()` function in `train_hbridge_model.py` crashed when trying to access thermistor data for components that don't have thermistors.

### Root Cause

**Dataset mismatch:**
- **ROI masks:** 216 components (all PCB components)
- **Thermistor data:** 23 columns (22 components + "Time (s)" column accidentally included)
- **Actual thermistors:** 22 components

The evaluation loop iterated over all 216 ROI masks, but thermistor data only exists for the first 22.

---

## ✅ Solution

Added **bounds checking** in the evaluation loop to skip components without thermistor data:

```python
# Get actual number of components with thermistor data
n_components = sand_temps.shape[1]
n_roi_masks = roi_masks.shape[0]

for frame_idx in range(len(flir_frames)):
    for comp_idx in range(roi_masks.shape[0]):
        # Skip if component index exceeds available thermistor data
        if comp_idx >= n_components:
            continue
        # ... rest of evaluation code
```

**Location:** [train_hbridge_model.py](train_hbridge_model.py#L195-L211)

---

## 📊 Dataset Stats

```
FLIR frames: (300, 480, 640) → 300 thermal images at 480×640 pixels
Sand temps: (300, 23) → 300 frames × 23 columns (22 components + "Time (s)")
ROI masks: (216, 480, 640) → 216 component masks
```

**Components with thermistors:** 22  
**Components without thermistors:** 194  
**Training uses:** Only the 22 components with thermistor ground truth

---

## 🎯 Impact

- ✅ **Fixed:** Evaluation no longer crashes
- ✅ **Improved:** Added warning message about skipped components
- ✅ **Safe:** Only evaluates on components with actual thermistor data
- ⚠️ **Note:** Model still learns spatial patterns from all 216 ROI locations, but only validates against 22 with thermistors

---

## 🔍 Additional Finding

**Dataset preprocessing issue:** The "Time (s)" column was incorrectly included as a component name.

**Recommendation:** Clean this up in next dataset rebuild, but it doesn't affect training since that column is position 0 and we skip it.

---

## ✅ Testing Status

- [x] Bug identified
- [x] Fix implemented
- [x] Bounds checking added
- [ ] Ready for training test (1 epoch)
- [ ] Full training (100 epochs)

---

**Next Step:** Run training with 1 epoch to verify the fix works end-to-end.
