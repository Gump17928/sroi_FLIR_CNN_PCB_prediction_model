# ML Model File Dependency Analysis
**Generated:** January 14, 2026  
**Purpose:** Identify which files are actively used by run_smart_pipeline.py

---

## SUMMARY

### ✅ ACTIVELY USED FILES (Required for ML Pipeline)

**Core Pipeline Scripts:**
1. `run_smart_pipeline.py` - Main orchestration script
2. `train_hbridge_model.py` - U-Net training wrapper
3. `generate_predictions.py` - Prediction generation and component extraction
4. `compare_phase8_vs_phase8c.py` - Method comparison
5. `viz_3d_thermal.py` - 3D thermal visualizations

**Dataset Building:**
6. `build_hbridge_dataset.py` - Creates HDF5 training dataset
7. `verify_dataset.py` - Validates dataset integrity

**External Dependencies (parent directory):**
8. `../../phase8c_spatial_cnn.py` - U-Net architecture & SpatialCNNTrainer class
9. `../../viz_phase8c_spatial.py` - Phase8cVisualizer for thermal plots
10. `../../cnn_data_preprocessor.py` - CNNDataPreprocessor for dataset building

**Data Files:**
11. `datasets/HBridge_cnn_dataset.h5` - Training dataset (required)

**Documentation (useful references):**
12. `VISUALIZATION_GUIDE.md` - Explains all 15 plot types
13. `README.md` - Project overview
14. `RESULTS_UPDATE_SUMMARY.md` - Recent changes documentation

---

### ❌ NOT USED / DEPRECATED FILES (Candidates for Cleanup)

**Obsolete Scripts:**
1. `run_complete_pipeline.py` - Replaced by `run_smart_pipeline.py`
2. `run_smart_pipeline_v2.py` - Rejected version, deleted by user request

**Documentation (can archive):**
3. `BUG_FIX_INDEXERROR.md` - Historical bug fix notes
4. `DATASET_VERIFICATION_REPORT.md` - One-time verification report

---

## DETAILED DEPENDENCY TREE

### run_smart_pipeline.py
```
run_smart_pipeline.py
├── train_hbridge_model.py
│   └── ../../phase8c_spatial_cnn.py (SpatialCNNTrainer class)
│       └── tensorflow.keras (U-Net architecture)
├── generate_predictions.py
│   ├── tensorflow (model loading)
│   └── ../../viz_phase8c_spatial.py (Phase8cVisualizer)
├── compare_phase8_vs_phase8c.py
│   └── matplotlib (plotting)
└── viz_3d_thermal.py (optional, for 3D plots)
    └── matplotlib (3D surface plots)
```

### Dataset Creation Pipeline
```
build_hbridge_dataset.py
└── ../../cnn_data_preprocessor.py (CNNDataPreprocessor)
    ├── flir_frame_loader.py
    ├── loader_thermistor.py
    └── numpy/pandas/h5py
```

---

## IMPORT ANALYSIS

### run_smart_pipeline.py imports:
- **Standard library:** os, sys, pathlib, datetime
- **Local modules:** train_hbridge_model, generate_predictions, compare_phase8_vs_phase8c
- **Optional:** viz_3d_thermal (for 3D visualizations)

### train_hbridge_model.py imports:
- **Standard library:** os, sys, pathlib, numpy, datetime
- **External:** h5py, matplotlib
- **Parent directory:** phase8c_spatial_cnn.SpatialCNNTrainer

### generate_predictions.py imports:
- **Standard library:** os, sys, pathlib, numpy, pandas, datetime
- **External:** h5py, matplotlib, tensorflow, sklearn
- **Parent directory:** viz_phase8c_spatial.Phase8cVisualizer

### compare_phase8_vs_phase8c.py imports:
- **Standard library:** os, sys, pathlib, numpy, pandas, datetime
- **External:** matplotlib

### viz_3d_thermal.py imports:
- **Standard library:** pathlib, numpy
- **External:** h5py, matplotlib (mpl_toolkits.mplot3d)

---

## FILES NOT DIRECTLY USED BY PIPELINE

### run_complete_pipeline.py
- **Status:** Likely obsolete
- **Reason:** run_smart_pipeline.py replaced this with smarter model detection
- **Recommendation:** Move to `backup/` or delete
- **Check first:** Does it have unique functionality?

### run_smart_pipeline_v2.py
- **Status:** Rejected by user
- **Reason:** User wanted single-file solution, not v2
- **Recommendation:** DELETE (already deleted?)

### Documentation files:
- `BUG_FIX_INDEXERROR.md` - Historical, can archive
- `DATASET_VERIFICATION_REPORT.md` - One-time report, can archive

---

## RECOMMENDATIONS

### 1. Keep (Essential)
```
ml_model/cnn_thermal_modeling/
├── run_smart_pipeline.py          ✅ Main script
├── train_hbridge_model.py         ✅ Training
├── generate_predictions.py        ✅ Prediction
├── compare_phase8_vs_phase8c.py   ✅ Comparison
├── viz_3d_thermal.py              ✅ 3D viz
├── build_hbridge_dataset.py       ✅ Dataset creation
├── verify_dataset.py              ✅ Dataset validation
├── datasets/                      ✅ Training data
├── results/                       ✅ Output folder
├── VISUALIZATION_GUIDE.md         ✅ Reference doc
├── README.md                      ✅ Overview
└── RESULTS_UPDATE_SUMMARY.md      ✅ Recent changes
```

### 2. Archive (Move to backup/)
```
├── run_complete_pipeline.py       ⚠️ Check if needed first
├── BUG_FIX_INDEXERROR.md         📝 Historical
└── DATASET_VERIFICATION_REPORT.md 📝 One-time report
```

### 3. Delete (Confirmed obsolete)
```
└── run_smart_pipeline_v2.py       ❌ User rejected
```

---

## VERIFICATION STEPS

To confirm run_complete_pipeline.py is safe to remove:

1. Check if it has unique functionality:
   ```bash
   diff run_complete_pipeline.py run_smart_pipeline.py
   ```

2. Search for references in other files:
   ```bash
   grep -r "run_complete_pipeline" .
   ```

3. Check git history for context:
   ```bash
   git log run_complete_pipeline.py
   ```

---

## PARENT DIRECTORY DEPENDENCIES

The ML pipeline depends on these files in `../../` (parent):

**Critical:**
- `phase8c_spatial_cnn.py` - U-Net model architecture
- `viz_phase8c_spatial.py` - Visualization classes
- `cnn_data_preprocessor.py` - Dataset preprocessing

**For dataset building:**
- `flir_frame_loader.py` - Load thermal images
- `loader_thermistor.py` - Load ground truth data

**Do NOT delete these from parent directory!**

---

## CONCLUSION

**Safe to keep:** 12 files (7 scripts + 1 dataset + 4 docs)  
**Safe to archive:** 3 files (1 script + 2 docs)  
**Safe to delete:** 1 file (run_smart_pipeline_v2.py)  

**Next step:** Review run_complete_pipeline.py to confirm if it's truly obsolete.
