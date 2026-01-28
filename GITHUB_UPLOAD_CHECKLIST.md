# GitHub Upload Checklist

## Repository Size Summary

### ✅ What's INCLUDED in Git (~12 MB total)

**1. Core Code Files (~7 MB):**
- 32 Python files (pipeline, ML, visualization, loaders)
- See `PIPELINE_DEPENDENCIES.md` for complete list

**2. Essential Configuration Files (~1.7 MB):**
- **Component mappings:**
  - `inputs/hbridge_pcb_components_enhanced.csv` (11 KB)
  - `inputs/loadshedding_ac_switch_pcb_components_enhanced.csv` (2.8 KB)
  
- **Pick & Place data:**
  - `inputs/Pick_Place_for_Hbridge_Sensing_GateDriver_Interface.csv` (22 KB)
  - `inputs/Pick_Place_for_LoadShedding_AC_Switch.csv` (5 KB)
  
- **ROI Pixel Maps (CRITICAL - required for dataset building):**
  - `outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv` (22 KB)
  - `outputs/roi_pixel_maps/LoadShedding_roi_pixel_map.csv` (6.7 KB)
  
- **Configuration JSONs:**
  - `All_Boards_multi_test_config.json` (13 KB)
  - `HBridge_multi_test_config.json` (6.1 KB)
  - `LoadShedding_multi_test_config.json` (5.3 KB)
  - `component_type_patterns.json` (1.8 KB)
  - `loadshedding_thermistor_mapping.json` (1.5 KB)
  - `multi_session_config_example.json` (2.3 KB)

**3. Documentation (~3 MB):**
- README.md, guides, analysis documents

**Total Git Repository Size: ~12 MB** ✅ (GitHub compatible)

---

## ❌ What's EXCLUDED (Download Separately)

### Large FLIR Data (3 GB) - **REQUIRED to run pipeline**
```
inputs/ResearchIR_Outputs_HBridge_15s/          (1.4 GB)
inputs/ResearchIR_Outputs_Load_Shedding/        (531 MB)
inputs/ResearchIR_Outputs_HBridge_15s_filtered/ (803 MB) - optional, regenerated
inputs/ResearchIR_Outputs_Load_Shedding_filtered/ (321 MB) - optional, regenerated
```

**External Download Link:** [TODO: Add Google Drive/Zenodo link]

---

### Large Thermistor Test Data (110 MB) - **REQUIRED to run pipeline**
```
inputs/Test_Air_HBridge_Sensing/    (4.2 MB)
inputs/Test_Sand_HBridge_Sensing/   (74 MB)
inputs/Test_Air_Load_Shedding/      (7.1 MB)
inputs/Test_Sand_Load_Shedding/     (25 MB)

Large CSV files:
inputs/Test_3_FLIR_Camera_Results_5_of_7.csv (2.7 MB)
inputs/Test_3_AIR_usb_temp_DAQami.csv        (346 KB)
inputs/Test_3_SAND_usb_temp_DAQami.csv       (2.4 MB)
```

**External Download Link:** [TODO: Add Google Drive/Zenodo link]

---

### Pre-trained Models (8 GB each) - **OPTIONAL**
```
ml_model/cnn_thermal_modeling/models/*.keras
```

Users can either:
- Download pre-trained models: [TODO: Add external link]
- Train their own: `python test_phase8.py --board HBridge --validation_mode cross_pcb`

---

### Pre-built Datasets (110 MB) - **OPTIONAL**
```
ml_model/cnn_thermal_modeling/datasets/HBridge_dataset.h5 (90 MB)
ml_model/cnn_thermal_modeling/datasets/LoadShedding_dataset.h5 (20 MB)
```

Users rebuild locally (requires FLIR + thermistor data):
```bash
python ml_model/cnn_thermal_modeling/build_dataset.py HBridge
python ml_model/cnn_thermal_modeling/build_dataset.py LoadShedding
```

---

### Generated Outputs (5 GB) - **REGENERATED**
```
outputs/* (except roi_pixel_maps/ which IS included)
ml_model/cnn_thermal_modeling/results/
```

These are created during pipeline execution.

---

## Setup Instructions for New Users

### 1. Clone Repository (~12 MB download)
```bash
git clone https://github.com/Gump17928/sroi_FLIR_CNN_PCB_prediction_model.git
cd sroi_FLIR_CNN_PCB_prediction_model/thermal_post_processing
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Required Data Files (3.1 GB)
Download and extract to project directory:
- **FLIR data** → `inputs/ResearchIR_Outputs_*/`
- **Thermistor test data** → `inputs/Test_*/` and CSV files

**Download links:**
- FLIR Data (3 GB): [TODO: Add link]
- Thermistor Data (110 MB): [TODO: Add link]

### 4. Verify Essential Config Files Present
These should already be in Git (no download needed):
```bash
ls -lh inputs/*enhanced.csv           # Component mappings
ls -lh inputs/Pick_Place_*.csv        # Pick & Place data
ls -lh outputs/roi_pixel_maps/*.csv   # ROI pixel maps
ls -lh *.json                         # Configuration files
```

### 5. Build Datasets Locally
```bash
cd ml_model/cnn_thermal_modeling
python build_dataset.py HBridge      # Creates 90 MB HDF5
python build_dataset.py LoadShedding # Creates 20 MB HDF5
```

### 6. Run Pipeline
```bash
cd ../..
python researchir_post_processor.py  # Full pipeline (Phases 1-8)
# OR
python test_phase8.py --board HBridge --validation_mode cross_pcb  # ML only
```

---

## Pre-Upload Checklist

### ✅ Completed
- [x] Updated `.gitignore` to exclude large files
- [x] Updated `.gitignore` to INCLUDE essential config files
- [x] Updated README.md with file management section
- [x] Created documentation (ACTIVE_FILE_STRUCTURE.md, PIPELINE_DEPENDENCIES.md)
- [x] Verified essential files will be tracked (~1.7 MB configs)

### ❌ TODO Before Upload
- [ ] Delete 27 unused diagnostic files (see `PIPELINE_DEPENDENCIES.md`)
- [ ] Upload FLIR data to external storage (Google Drive/Zenodo)
- [ ] Upload thermistor test data to external storage
- [ ] Add download links to README.md and this checklist
- [ ] Test `git add .` to verify correct files included
- [ ] Check final size: `du -sh .git/` (should be ~12-15 MB)
- [ ] Create `requirements.txt` with package versions
- [ ] Test repository clone on fresh system

### Optional Enhancements
- [ ] Upload pre-trained models to external storage
- [ ] Upload pre-built datasets to external storage
- [ ] Create setup script: `setup.sh` to automate data download
- [ ] Add badges to README (build status, license, etc.)
- [ ] Create CONTRIBUTING.md for open source collaboration

---

## Verification Commands

### Check what Git will track:
```bash
git status --short
git ls-files --cached  # Files already tracked
```

### Check what Git will ignore:
```bash
git status --ignored
git check-ignore inputs/* outputs/* ml_model/cnn_thermal_modeling/datasets/*
```

### Estimate repository size:
```bash
# Before commit
find . -type f -not -path './.git/*' -not -path './inputs/ResearchIR_*' \
  -not -path './inputs/Test_*' -not -path './ml_model/*/datasets/*' \
  -not -path './ml_model/*/results/*' -exec ls -lh {} \; | \
  awk '{s+=$5} END {print "Estimated size:", s/1024/1024, "MB"}'

# After commit
du -sh .git/
```

### Test essential files are tracked:
```bash
git ls-files | grep -E "(enhanced\.csv|Pick_Place|roi_pixel_map|\.json)"
```

Expected output:
```
All_Boards_multi_test_config.json
HBridge_multi_test_config.json
LoadShedding_multi_test_config.json
component_type_patterns.json
inputs/Pick_Place_for_Hbridge_Sensing_GateDriver_Interface.csv
inputs/Pick_Place_for_LoadShedding_AC_Switch.csv
inputs/hbridge_pcb_components_enhanced.csv
inputs/loadshedding_ac_switch_pcb_components_enhanced.csv
loadshedding_thermistor_mapping.json
outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv
outputs/roi_pixel_maps/LoadShedding_roi_pixel_map.csv
```

---

## Summary

**Current Status:**
- ✅ `.gitignore` properly configured
- ✅ Essential config files will be tracked (~1.7 MB)
- ✅ Large data files will be excluded (3.1 GB)
- ✅ Documentation updated
- ❌ Cleanup not yet performed (27 unused files still present)
- ❌ External data hosting not yet set up

**Next Steps:**
1. Delete unused diagnostic files (optional, saves ~3 MB)
2. Upload FLIR + thermistor data to Google Drive/Zenodo
3. Add download links to README.md
4. Test `git add .` and verify size
5. Push to GitHub!

**Expected Result:**
- Repository clone size: ~12 MB (down from 26 GB)
- Users download 3.1 GB data separately
- Users rebuild datasets locally (110 MB)
- Users train models or download pre-trained (8 GB)
