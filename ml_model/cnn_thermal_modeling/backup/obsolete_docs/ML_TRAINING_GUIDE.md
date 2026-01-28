# ML Model Training Guide - Understanding & Tuning Your U-Net CNN

## 🎯 Quick Overview

Your model is a **U-Net Convolutional Neural Network** that learns to predict thermal temperatures from FLIR camera images.

**What it does:**
- **Input:** FLIR thermal image (480×640 pixels)
- **Output:** Predicted temperature field with thermistor-validated values
- **Training data:** ~1000+ thermal frames from HBridge tests
- **Validation:** Compares predictions against actual thermistor readings

---

## 📚 Understanding Key Concepts

### **1. FRAMES** 
- **What:** Individual thermal images captured by FLIR camera over time
- **Example:** If your test ran for 100 seconds at 10 fps, you have ~1000 frames
- **In code:** Shape is `[n_frames, height, width]` = `[1000, 480, 640]`
- **Purpose:** Each frame is ONE training example showing thermal state at a moment in time

### **2. EPOCHS**
- **What:** Complete pass through ALL training frames
- **Example:** 100 epochs = model sees all 1000 frames 100 times
- **Why:** Model learns patterns through repetition
- **Current setting:** `epochs=100` in [train_hbridge_model.py](train_hbridge_model.py#L292)

### **3. BATCH SIZE**
- **What:** Number of frames processed simultaneously before updating model weights
- **Example:** batch_size=8 means process 8 frames → update model → next 8 frames
- **Purpose:** Balances speed vs memory vs learning stability
- **Current setting:** `batch_size=8` in [train_hbridge_model.py](train_hbridge_model.py#L293)

### **4. LEARNING RATE**
- **What:** How big of adjustments the model makes when learning
- **Example:** 0.001 = small careful steps, 0.1 = large aggressive steps
- **Current setting:** `learning_rate=0.001` in [train_hbridge_model.py](train_hbridge_model.py#L104)

### **5. VALIDATION SPLIT**
- **What:** Percentage of data held back to test model on unseen frames
- **Example:** 0.2 = 80% training, 20% validation
- **Purpose:** Prevents overfitting (memorizing instead of learning)
- **Current setting:** `validation_split=0.2` in [train_hbridge_model.py](train_hbridge_model.py#L294)

### **6. EARLY STOPPING PATIENCE**
- **What:** How many epochs to wait without improvement before stopping
- **Example:** patience=15 means "stop if validation doesn't improve for 15 epochs"
- **Purpose:** Saves time, prevents overfitting
- **Current setting:** `early_stopping_patience=15` in [train_hbridge_model.py](train_hbridge_model.py#L118)

---

## 🎛️ TUNING DIALS - What to Adjust

### **For FASTER Training (Less Accurate)**

| Parameter | Current | Change To | Effect |
|-----------|---------|-----------|--------|
| **epochs** | 100 | 20-50 | ⚡ Much faster, less refined learning |
| **batch_size** | 8 | 16 or 32 | ⚡ Faster (if GPU has memory), less stable |
| **early_stopping_patience** | 15 | 5-10 | ⚡ Stops sooner if not improving |
| **validation_split** | 0.2 | 0.1 | ⚡ More training data, less validation |

**Example quick-test settings:**
```python
epochs=30
batch_size=16
early_stopping_patience=7
```

---

### **For HIGHER Accuracy (Slower)**

| Parameter | Current | Change To | Effect |
|-----------|---------|-----------|--------|
| **epochs** | 100 | 150-300 | 🎯 More learning time, diminishing returns >200 |
| **batch_size** | 8 | 4 | 🎯 More stable gradients, slower |
| **learning_rate** | 0.001 | 0.0005 | 🎯 Smaller steps, more precise, slower convergence |
| **early_stopping_patience** | 15 | 20-30 | 🎯 Gives more time to find improvements |

**Example high-accuracy settings:**
```python
epochs=200
batch_size=4
learning_rate=0.0005
early_stopping_patience=25
```

---

### **For BALANCED Performance (Recommended)**

Current settings are already well-balanced! But you can try:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **epochs** | 100 | Good sweet spot for convergence |
| **batch_size** | 8 | Fits most GPUs, stable learning |
| **learning_rate** | 0.001 | Standard Adam optimizer rate |
| **early_stopping_patience** | 15 | Prevents wasting time on plateaus |
| **validation_split** | 0.2 | Industry standard |

---

## 📍 Where to Make Changes

### **Main Training File**
[train_hbridge_model.py](train_hbridge_model.py#L288-L296)

**Current main() function:**
```python
def main():
    # ...
    trainer, history = train_model(
        dataset_path=dataset_file,
        output_dir=output_dir,
        epochs=100,              # ← CHANGE THIS
        batch_size=8,            # ← CHANGE THIS
        validation_split=0.2     # ← CHANGE THIS
    )
```

**To change learning rate:**
[train_hbridge_model.py](train_hbridge_model.py#L104)
```python
trainer.build_model(learning_rate=0.001)  # ← CHANGE THIS
```

**To change early stopping:**
[train_hbridge_model.py](train_hbridge_model.py#L118)
```python
early_stopping_patience=15  # ← CHANGE THIS
```

---

### **Core Model Architecture**
[phase8c_spatial_cnn.py](phase8c_spatial_cnn.py#L303-L317)

**Training function with all parameters:**
```python
def train(self, X_train, y_train, X_val, y_val, 
         epochs: int = 100,                    # ← Max training iterations
         batch_size: int = 8,                  # ← Frames per batch
         early_stopping_patience: int = 15):   # ← Stop if no improvement
```

---

## 🚀 How to Run Training

### **Option 1: Quick Run (Recommended)**
```bash
cd ml_model
python run_from_here.py
```
- Auto-detects if model exists
- Skips training if model found (fast predictions only)
- Runs full training if no model exists

### **Option 2: Force Retrain**
```bash
cd ml_model/cnn_thermal_modeling
python train_hbridge_model.py
```
- Always trains from scratch
- Saves new model with timestamp

### **Option 3: Complete Pipeline**
```bash
cd ml_model/cnn_thermal_modeling
python run_complete_pipeline.py
```
- Builds dataset → trains → predicts → compares → visualizes

---

## 📊 Understanding Training Output

### **During Training:**
```
Epoch 1/100
125/125 [==============================] - 45s 360ms/step
  loss: 12.5432 - mae: 2.8451 - val_loss: 10.2341 - val_mae: 2.3412
Epoch 2/100
125/125 [==============================] - 42s 335ms/step
  loss: 9.8765 - mae: 2.1234 - val_loss: 8.9012 - val_mae: 2.0123
```

**What this means:**
- **125/125:** Total batches (1000 frames ÷ 8 batch_size = 125 batches)
- **loss:** Training error (lower is better)
- **mae:** Mean Absolute Error in °C (how far off predictions are)
- **val_loss:** Validation error (most important - tests on unseen data)
- **val_mae:** Validation error in °C

**Good signs:**
- ✅ Loss decreasing over epochs
- ✅ val_loss close to loss (not overfitting)
- ✅ mae under 2-3°C (good temperature accuracy)

**Bad signs:**
- ❌ Loss stuck/not decreasing (increase learning_rate or epochs)
- ❌ val_loss much higher than loss (overfitting - increase validation_split)
- ❌ mae above 5°C (poor accuracy - need more data or different architecture)

---

## 🎯 Performance Metrics

**After training, you'll see:**

```
Performance Metrics:
  R² Score: 0.9523
  RMSE: 1.82°C
  MAE: 1.45°C
  Sample count: 12,450
```

**What they mean:**

| Metric | What It Is | Good Value | Your Goal |
|--------|-----------|------------|-----------|
| **R²** | How well model explains variance | 0.90-0.99 | >0.92 |
| **RMSE** | Root Mean Squared Error | <2.5°C | <2.0°C |
| **MAE** | Mean Absolute Error | <2.0°C | <1.5°C |

---

## 🔧 Advanced Tuning

### **Model Architecture (Advanced)**
[phase8c_spatial_cnn.py](phase8c_spatial_cnn.py#L91-L155)

Current U-Net has 5 levels. To make it:
- **Deeper (more accurate, slower):** Add more Conv/Pool blocks
- **Shallower (faster, less accurate):** Remove levels
- **Wider (more features):** Increase filter counts (32→64, 64→128, etc.)

### **Learning Rate Schedule**
[phase8c_spatial_cnn.py](phase8c_spatial_cnn.py#L330-L336)

Currently uses ReduceLROnPlateau:
```python
keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Reduce LR by 50% when stuck
    patience=5,        # Wait 5 epochs before reducing
    min_lr=1e-6,      # Don't go below this
)
```

**To be more aggressive:** Change `patience=5` to `patience=3`
**To be more conservative:** Change `factor=0.5` to `factor=0.7`

---

## 📁 Output Files Explained

After training, you'll find in `ml_model/cnn_thermal_modeling/results/`:

| File | What It Contains |
|------|------------------|
| `unet_hbridge_*.keras` | Trained model (load to make predictions) |
| `training_history_*.npz` | Loss/accuracy over epochs |
| `training_curves_*.png` | Graphs showing learning progress |
| `evaluation_metrics_*.txt` | Final R², RMSE, MAE |
| `scatter_plot_*.png` | Actual vs Predicted temperatures |

---

## 🐛 Troubleshooting

**Problem:** Training too slow
- ✅ Reduce epochs (100 → 50)
- ✅ Increase batch_size (8 → 16)
- ✅ Use GPU if available

**Problem:** Poor accuracy (MAE > 3°C)
- ✅ Increase epochs (100 → 150)
- ✅ Decrease learning_rate (0.001 → 0.0005)
- ✅ Check dataset quality (run `verify_dataset.py`)

**Problem:** Overfitting (val_loss >> training loss)
- ✅ Increase validation_split (0.2 → 0.3)
- ✅ Use more training data
- ✅ Add data augmentation

**Problem:** Out of memory
- ✅ Reduce batch_size (8 → 4)
- ✅ Reduce image size (requires dataset rebuild)

---

## 🎓 Quick Reference Card

```python
# SPEED SETTINGS (Fast, less accurate)
epochs=30, batch_size=16, learning_rate=0.001, patience=7

# BALANCED SETTINGS (Recommended - CURRENT)
epochs=100, batch_size=8, learning_rate=0.001, patience=15

# ACCURACY SETTINGS (Slow, very accurate)
epochs=200, batch_size=4, learning_rate=0.0005, patience=25
```

---

## 📞 Where Things Are Run

All training happens in **[phase8c_spatial_cnn.py](phase8c_spatial_cnn.py)**:
- **Class:** `SpatialCNNTrainer` (line 159)
- **Training function:** `train()` (line 303)
- **Model building:** `build_unet()` (line 78)

Called by **[train_hbridge_model.py](train_hbridge_model.py)**:
- **Main training logic:** `train_model()` (line 71)
- **Entry point:** `main()` (line 280)

Orchestrated by **[run_smart_pipeline.py](run_smart_pipeline.py)**:
- **Smart detection** of existing models
- **Auto-skip** if already trained

---

## ✅ Next Steps

1. **Activate virtual environment:**
   ```bash
   source ml_model_venv/bin/activate
   ```

2. **Run training:**
   ```bash
   cd ml_model
   python run_from_here.py
   ```

3. **Watch the output** to understand how frames/epochs work

4. **Try different settings** from the Quick Reference Card above

5. **Compare results** using metrics (R², RMSE, MAE)

---

**Questions? Check these files:**
- Training parameters: [train_hbridge_model.py](train_hbridge_model.py#L71)
- Model architecture: [phase8c_spatial_cnn.py](phase8c_spatial_cnn.py#L78)
- Quick start: [run_from_here.py](run_from_here.py)
