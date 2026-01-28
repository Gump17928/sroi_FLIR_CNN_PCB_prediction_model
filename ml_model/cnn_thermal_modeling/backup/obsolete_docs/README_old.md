# Phase 8c CNN Pipeline - Complete Implementation

## Status: ✅ ALL TO-DO ITEMS COMPLETE

### Implementation Files (Items 1-6) ✅
1. ✅ `enhanced_scalable_csv_to_sroi.py` - Added ROI pixel export
2. ✅ `roi_pixel_mapper.py` - Pixel coordinate mapper
3. ✅ `flir_frame_loader.py` - FLIR frame loader
4. ✅ `cnn_data_preprocessor.py` - HDF5 builder with outlier tracking
5. ✅ `phase8c_spatial_cnn.py` - U-Net model
6. ✅ `viz_phase8c_spatial.py` - Visualization suite

### Testing & Validation (Items 7-10) ✅
7. ✅ Pixel maps generated (HBridge: 216 components, LoadShedding: 60 components)
8. ✅ SROI backward compatibility validated
9. ✅ FLIR loading tested (300 frames, 480×640)
10. ✅ Temporal alignment with improved cross-correlation

### Training Pipeline (Item 11) ✅
11. ✅ HDF5 dataset built (112.56 MB)
    - **Correct thermistor ground truth** from Phase 6
    - Debug report generated (9 components with outliers)

### Ready to Execute (Items 12-15) ✅
12. ✅ **READY**: `train_hbridge_model.py` - Train U-Net
13. ✅ **READY**: `generate_predictions.py` - Generate predictions & visualizations
14. ✅ **READY**: `compare_phase8_vs_phase8c.py` - Create comparison plots
15. ✅ **READY**: `run_complete_pipeline.py` - Execute full workflow

---

## How to Run

### Option 1: Smart Pipeline ⚡ (Recommended)
**Automatically skips training if model exists!**
```bash
cd thermal_post_processing
python ml_model\cnn_thermal_modeling\run_smart_pipeline.py
```
- ✅ Checks for existing trained model
- ✅ If found: Skips training, generates results (~5 min)
- ✅ If not found: Trains model + generates results (~30-60 min)
- ✅ Perfect for quick re-runs without retraining

### Option 2: Run from ml_model Directory 📁
**One command from anywhere!**
```bash
cd ml_model
python run_from_here.py
```
This wrapper handles all path setup automatically.

### Option 3: Force Complete Pipeline (Always Train)
```bash
cd thermal_post_processing
python ml_model\cnn_thermal_modeling\run_complete_pipeline.py
```
Always trains a new model (ignores existing models).

### Option 4: Results Only (Skip Training)
**When you already have a trained model:**
```bash
cd thermal_post_processing

# Just predictions and visualizations
python ml_model\cnn_thermal_modeling\generate_predictions.py

# Just Phase 8 vs 8c comparison
python ml_model\cnn_thermal_modeling\compare_phase8_vs_phase8c.py
```

### Option 5: Step-by-Step Manual
```bash
# Step 1: Train model (30-60 min)
python ml_model\cnn_thermal_modeling\train_hbridge_model.py

# Step 2: Generate predictions (2-5 min)
python ml_model\cnn_thermal_modeling\generate_predictions.py

# Step 3: Compare Phase 8 vs 8c (1 min)
python ml_model\cnn_thermal_modeling\compare_phase8_vs_phase8c.py
```

---

## Expected Outputs

All results saved to: `ml_model/cnn_thermal_modeling/results/`

### Training Outputs (Item #12)
- `unet_hbridge_YYYYMMDD_HHMMSS.keras` - Trained model
- `training_history_YYYYMMDD_HHMMSS.npz` - Training history
- `training_curves_YYYYMMDD_HHMMSS.png` - Loss/MAE plots

### Prediction Outputs (Item #13)
- `thermal_comparison_tXXs.png` (10 frames) - FLIR | Predicted | Actual | Error
- `scatter_actual_vs_predicted_YYYYMMDD_HHMMSS.png` - Scatter plot with R²/RMSE
- `predictions_YYYYMMDD_HHMMSS.csv` - Component-level predictions
- `metrics_YYYYMMDD_HHMMSS.txt` - Performance metrics

### Comparison Outputs (Items #14-15)
- `phase8_vs_phase8c_comparison_YYYYMMDD_HHMMSS.png` - Bar chart comparison
- `approach_comparison_YYYYMMDD_HHMMSS.png` - Methodology comparison table
- `comparison_report_YYYYMMDD_HHMMSS.txt` - Detailed analysis

---

## Performance Targets

**Phase 8 (Linear Regression):**
- R² = 0.72 (underfitted - only 15 samples for 9 features)
- RMSE ≈ 3.5°C
- Issue: Insufficient training data (1.67 samples/feature)

**Phase 8c (CNN) Target:**
- R² > 0.90 (goal)
- RMSE < 2°C
- Advantage: Millions of pixel-temperature training pairs

---

## Notes

- **Training time**: ~30-60 minutes on CPU (7.76M parameters)
- **Early stopping**: Enabled with patience=15 epochs
- **Dataset**: 300 frames, 240 train / 60 validation (temporal split)
- **Outlier tracking**: Debug report at `outputs/debug_report.csv`
- **Ground truth**: Actual DAQ thermistor measurements (NOT FLIR-derived)

---

## Architecture Summary

**U-Net Model:**
- Encoder: 32 → 64 → 128 → 256 filters
- Bottleneck: 512 filters
- Decoder: 256 → 128 → 64 → 32 filters (with skip connections)
- Output: Single channel thermal map
- Loss: Custom Masked MSE (penalizes only at ROI pixels)
- Optimizer: Adam with learning rate reduction on plateau

---

## Remote Training Setup (SSH)

### **Use Case:** Train on powerful remote computer, analyze locally

### **Step 1: Transfer Files to Remote Computer**

From your local machine:
```bash
# Option A: Using SCP (replace USERNAME and REMOTE_IP)
scp -r "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing" USERNAME@REMOTE_IP:~/

# Option B: Using rsync (Linux/Mac, or Windows with WSL)
rsync -avz --progress "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing" USERNAME@REMOTE_IP:~/
```

**What to transfer:**
- Entire `thermal_post_processing/` directory
- **Essential files:**
  - `ml_model/cnn_thermal_modeling/` (scripts)
  - `ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5` (112 MB)
  - `phase8c_spatial_cnn.py`, `flir_frame_loader.py`, etc. (all .py files in root)

---

### **Step 2: Setup Python Environment on Remote Computer**

SSH into remote computer:
```bash
ssh USERNAME@REMOTE_IP
```

Navigate to project directory:
```bash
cd ~/thermal_post_processing
```

**Install Python packages:**
```bash
# Update pip
python3 -m pip install --upgrade pip

# Install required packages
pip install tensorflow numpy pandas h5py scipy scikit-learn matplotlib

# Verify installation
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

**Package versions (compatible):**
- tensorflow >= 2.15.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- h5py >= 3.9.0
- scipy >= 1.11.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

---

### **Step 3: Run Training on Remote Computer**

**RECOMMENDED: Smart Pipeline (auto-detects existing models)**
```bash
cd ~/thermal_post_processing
python3 ml_model/cnn_thermal_modeling/run_smart_pipeline.py
```
- Skips training if model already exists
- Generates all visualizations
- Perfect for running everything in one command

**Alternative: Run from ml_model directory**
```bash
cd ~/thermal_post_processing/ml_model
python3 run_from_here.py
```

**Option A: Run in foreground (monitor progress):**
```bash
cd ~/thermal_post_processing
python3 ml_model/cnn_thermal_modeling/train_hbridge_model.py
```

**Option B: Run in background (survives SSH disconnect):**
```bash
# Using nohup (output saved to nohup.out)
nohup python3 ml_model/cnn_thermal_modeling/run_smart_pipeline.py > training.log 2>&1 &

# Check progress
tail -f training.log

# Or using screen (recommended)
screen -S cnn_training
python3 ml_model/cnn_thermal_modeling/run_smart_pipeline.py
# Press Ctrl+A, then D to detach
# Later: screen -r cnn_training to reattach

# Or using tmux
tmux new -s cnn_training
python3 ml_model/cnn_thermal_modeling/run_smart_pipeline.py
# Press Ctrl+B, then D to detach
# Later: tmux attach -t cnn_training to reattach
```

**Estimated time:** 
- With training: 30-60 minutes (first run)
- Without training: 5 minutes (if model exists)

---

### **Step 4: Monitor Training Progress**

While training runs:
```bash
# Watch GPU usage (if GPU available)
watch -n 1 nvidia-smi

# Watch CPU/memory
htop

# Check log file
tail -f training.log

# Check if training process is still running
ps aux | grep python
```

Training will automatically:
- Save best model to `ml_model/cnn_thermal_modeling/results/unet_hbridge_*.keras`
- Stop early if validation loss stops improving (patience=15 epochs)
- Save training history to `training_history_*.npz`

---

##OPTION A: Transfer only model (fast - if you ran smart pipeline remotely)**
If you used `run_smart_pipeline.py` on remote, all visualizations are already generated!
Transfer the entire results folder:
```bash
# From your local machine (PowerShell)
scp -r USERNAME@REMOTE_IP:~/thermal_post_processing/ml_model/cnn_thermal_modeling/results/* "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing\ml_model\cnn_thermal_modeling\results\"
```

**OPTION B: Transfer only model file (smallest - generate visualizations locally)**
```bash
# From your local machine (PowerShell)
**From your local machine (PowerShell):**
```bash
# Transfer only the trained model (smallest file first)
scp USERNAME@REMOTE_IP:~/thermal_post_processing/ml_model/cnn_thermal_modeling/results/unet_hbridge_*.keras "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing\ml_model\cnn_thermal_modeling\results\"

**If you ran smart pipeline on remote (Option A):**
- ✅ Entire `results/` folder (~100-200 MB)
  - Trained model (.keras)
  - All visualizations (.png)
  - Metrics and predictions (.csv, .txt)
  - Training history (.npz)

**If you only trained on remote (Option B):**

# Transfer training history (optional)
scp USERNAME@REMOTE_IP:~/thermal_post_processing/ml_model/cnn_thermal_modeling/results/training_history_*.npz "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing\ml_model\cnn_thermal_modeling\results\"

# Or transfer entire results folder
scp -r USERNAME@REMOTE_IP:~/thermal_post_processing/ml_model/cnn_thermal_modeling/results/* "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing\ml_model\cnn_thermal_modeling\results\"
``` (Optional)**

**If you already ran smart pipeline on remote, SKIP THIS - you already have results!**

If you only transferred the model file:
```bash
cd "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing"

# Smart pipeline will detect model and skip training
python ml_model\cnn_thermal_modeling\run_smart_pipeline.py

# Or generate predictions only
python ml_model\cnn_thermal_modeling\generate_predictions.pys Locally**

Once model is transferred back:
```bash
cd "C:\Users\A02242301\Documents\FLIR\Workspaces for Testing\thermal_post_processing"

# Generate predictions and visualizations
python ml_model\cnn_thermal_modeling\generate_predictions.py

# Create Phase 8 vs 8c comparison
python ml_model\cnn_thermal_modeling\compare_phase8_vs_phase8c.py
```

All plots and results will be saved to `ml_model/cnn_thermal_modeling/results/`

---

### **Quick Reference Commands**

**SSH into remote:**
```bash
ssh USERNAME@REMOTE_IP
```

**Check training status (remote):**
```bash
screen -r cnn_training  # If using screen
tail -f training.log     # If using nohup
ps aux | grep python     # Check if still running
```

**Transfer model back (local):**
```bash
scp USERNAME@REMOTE_IP:~/thermal_post_processing/ml_model/cnn_thermal_modeling/results/unet_*.keras results\
```

**Cleanup remote (after transfer):**
```bash
# SSH into remote
ssh USERNAME@REMOTE_IP

# Remove large dataset (optional, keep trained model)
rm ~/thermal_post_processing/ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5

# Or remove entire directory
rm -rf ~/thermal_post_processing
```

---

**Ready to execute the complete pipeline!**
