# Phase 8c CNN Visualization Guide

Generated: January 14, 2026

This guide explains each visualization created by the CNN training and prediction pipeline.

---

## 📊 Overview of Generated Plots

The pipeline generates **4 types** of visualizations:
1. **Training Diagnostics** (1 plot)
2. **Prediction Quality** (2 plots) 
3. **Thermal Field Comparisons** (10 plots)
4. **Method Comparison** (2 plots)

**Total: 15 PNG files**

---

## 1️⃣ TRAINING DIAGNOSTICS

### `training_curves_YYYYMMDD_HHMMSS.png`

**Purpose:** Monitor model learning progress during training

**Layout:** 2 side-by-side plots
- **Left panel:** Training Loss vs Validation Loss over epochs
- **Right panel:** Training MAE vs Validation MAE over epochs

**What to look for:**
- ✅ **Good training:** Both loss and val_loss decrease smoothly
- ✅ **Convergence:** Curves flatten out at low values
- ⚠️ **Overfitting:** Training loss continues decreasing while val_loss increases
- ⚠️ **Underfitting:** Both losses remain high and don't improve much
- ❌ **Failure:** NaN values or loss exploding upward

**Your latest result (5 epochs):**
- Loss decreased: 63.1 → 7.4 ✅
- Validation loss: 27.8 → 9.1 ✅
- No overfitting observed (training stopped at epoch 4 due to early stopping)
- Model is still learning - could benefit from more epochs

**Key Metrics:**
- **Loss (Masked MSE):** Mean Squared Error computed only at ROI component locations
- **MAE (Mean Absolute Error):** Average temperature prediction error in °C

---

## 2️⃣ PREDICTION QUALITY PLOTS

### `scatter_plot_YYYYMMDD_HHMMSS.png` (from evaluation)

**Purpose:** Show overall model accuracy at ROI component locations during validation

**Layout:** Single scatter plot
- **X-axis:** Actual thermistor temperatures (°C)
- **Y-axis:** Predicted temperatures by CNN (°C)
- **Red dashed line:** Perfect prediction (y=x)
- **Blue points:** Individual predictions (1,100 total from validation set)

**What to look for:**
- ✅ **Perfect model:** All points on red line
- ✅ **Good model:** Points clustered tightly around red line
- ⚠️ **Bias:** Points systematically above/below line (over/under-predicting)
- ⚠️ **High variance:** Points scattered far from line
- ❌ **Poor model:** No clear relationship to red line

**Your result (R² = 0.17):**
- Moderate scatter around the line
- Some systematic under-prediction visible
- Shows room for improvement with more training

---

### `scatter_actual_vs_predicted_YYYYMMDD_HHMMSS.png` (from predictions)

**Purpose:** Show model accuracy across ALL component-frame pairs (full dataset)

**Layout:** Same as scatter_plot but with 6,600 points (all 22 components × 300 frames)

**Differences from evaluation scatter:**
- **More data:** 6,600 predictions vs 1,100 (validation only)
- **Full range:** Includes training and validation data
- **Better statistics:** More reliable R², RMSE, MAE estimates

**What to look for:**
- Same interpretation as evaluation scatter plot
- Denser point cloud due to more data
- Should have similar or better metrics than validation-only plot

**Your result (R² = 0.38):**
- Better than validation set alone (0.38 vs 0.17)
- Points show clear positive correlation but with significant scatter
- RMSE = 2.62°C, MAE = 2.04°C (close to target but can improve)

---

## 3️⃣ THERMAL FIELD COMPARISON PLOTS (10 plots)

### `thermal_comparison_tXXX.Xs.png` (10 different timestamps)

**Purpose:** Visualize spatial thermal predictions vs FLIR input and ground truth

**Layout:** 4 side-by-side panels showing same PCB at one moment in time
1. **Panel 1 - FLIR Input:** Raw thermal camera image (480×640 pixels)
2. **Panel 2 - CNN Predicted:** CNN's prediction of sand/substrate temperature
3. **Panel 3 - Ground Truth:** Actual thermistor measurements (sparse, only 22 ROI locations)
4. **Panel 4 - Absolute Error:** |Predicted - Actual| at ROI locations only

**Color scales:**
- Panels 1-3: 'hot' colormap (dark=cold, bright=hot), same temperature range
- Panel 4: 'coolwarm' colormap (blue=low error, red=high error), 0-5°C range

**What to look for:**
- ✅ **Good spatial prediction:** Panel 2 closely matches Panel 1's spatial patterns
- ✅ **Accurate component temps:** Panel 2 values at ROI locations match Panel 3
- ✅ **Low error:** Panel 4 shows mostly blue/green (errors <2°C)
- ⚠️ **Spatial mismatch:** Panel 2 has hot spots in wrong locations vs Panel 1
- ⚠️ **Component errors:** Panel 4 shows red regions (errors >3°C)

**Timestamps generated (10 total):**
- t=0.0s (initial state)
- t=38.0s, t=68.0s, t=99.0s (heating phase)
- t=128.0s, t=158.0s, t=188.0s (mid-test)
- t=217.0s, t=248.0s, t=278.0s (peak/steady-state)

**Why 10 frames?**
- Spread evenly across 300-frame test (every ~30 frames)
- Shows model performance at different thermal states
- Captures both transient heating and steady-state

**Your results:**
- These show how well the CNN reconstructs the full thermal field
- Check if hot components in FLIR (Panel 1) appear hot in predictions (Panel 2)
- Error map (Panel 4) shows where CNN struggles most

---

## 4️⃣ METHOD COMPARISON PLOTS (2 plots)

### `phase8_vs_phase8c_comparison_YYYYMMDD_HHMMSS.png`

**Purpose:** Compare Phase 8 (linear regression) vs Phase 8c (CNN) performance metrics

**Layout:** 3 side-by-side bar charts
1. **R² Score:** Higher is better (max = 1.0)
2. **RMSE:** Lower is better (target: <2.0°C)
3. **MAE:** Lower is better (target: <1.5°C)

**Color coding:**
- Blue bars: Phase 8 (Linear Regression)
- Red bars: Phase 8c (U-Net CNN)

**Annotations:**
- Green boxes show % improvement or reduction
- Numbers on top of bars show exact values

**What to look for:**
- ✅ **CNN wins:** Phase 8c bars taller (R²) or shorter (RMSE/MAE)
- ⚠️ **Mixed results:** CNN better on some metrics, worse on others
- ❌ **CNN worse:** Phase 8c consistently underperforms Phase 8

**Your result:**
- **R²:** Phase 8 = 0.72, Phase 8c = 0.38 ❌ (CNN worse by 47%)
- **RMSE:** Phase 8 = 3.50°C, Phase 8c = 2.62°C ✅ (CNN better by 25%)
- **MAE:** Phase 8 = 2.80°C, Phase 8c = 2.04°C ✅ (CNN better by 27%)

**Interpretation:**
- CNN has lower absolute errors (RMSE/MAE better)
- But lower R² means CNN explains less variance than linear model
- **Paradox explanation:** Phase 8 was underfitted (only 15 samples for 9 features), giving artificially high R² despite higher errors
- Phase 8c needs more training to improve R²

---

### `approach_comparison_YYYYMMDD_HHMMSS.png`

**Purpose:** Show fundamental differences between Phase 8 and Phase 8c modeling approaches

**Layout:** Comparison table with 3 columns
- **Column 1:** Metric/property name
- **Column 2:** Phase 8 (Linear Regression) values
- **Column 3:** Phase 8c (U-Net CNN) values

**Color coding:**
- **Light gray:** Header row
- **Light green:** Better performing method for each metric
- **Light yellow:** Worse performing method

**Sections in table:**
1. **Approach characteristics:**
   - Model Type: Linear Regression vs U-Net CNN
   - Approach: Point-based (averaged temps) vs Spatial (full field)
   - Training Data: 15 samples vs 6,600 pixel-temp pairs
   - Features/Parameters: 9 features vs 7.7M parameters
   - Samples per Feature: 1.67 vs N/A (spatial model)

2. **Performance metrics:**
   - R² Score
   - RMSE (°C)
   - MAE (°C)

**What to look for:**
- **Fundamental difference:** Phase 8 averages spatial data to points, Phase 8c preserves spatial structure
- **Data efficiency:** Phase 8c uses 440× more training data (6,600 vs 15)
- **Model complexity:** Phase 8c has 863,000× more parameters
- **Tradeoff:** More complex model should eventually outperform simple linear model with sufficient training

**Your result:**
- Shows CNN approach is fundamentally different and more sophisticated
- But needs more training to leverage its spatial modeling capability
- Current underperformance on R² is likely due to insufficient training (only 5 epochs)

---

## 📈 INTERPRETING YOUR RESULTS

### Current Model Performance Summary

| Metric | Validation (1,100 pts) | Full Dataset (6,600 pts) | Target | Status |
|--------|----------------------|-------------------------|--------|---------|
| **R²** | 0.17 | 0.38 | >0.90 | ❌ Needs improvement |
| **RMSE** | 2.62°C | 2.62°C | <2.0°C | ⚠️ Close to target |
| **MAE** | 1.99°C | 2.04°C | <1.5°C | ⚠️ Close to target |

### Best Performing Components (R² > 0.5)
1. **U13:** R² = 0.57, MAE = 1.10°C ✅
2. **U19:** R² = 0.51, MAE = 0.95°C ✅
3. **U6:** R² = 0.54, MAE = 0.94°C ✅

### Worst Performing Components (R² < -5)
1. **R5:** R² = -37.9, MAE = 1.45°C ❌
2. **R88:** R² = -19.1, MAE = 1.77°C ❌
3. **U34:** R² = -13.5, MAE = 1.31°C ❌
4. **DL11:** R² = -11.5, MAE = 4.46°C ❌

**Note:** Negative R² means model performs worse than just predicting the mean temperature for that component.

---

## 🔍 RECOMMENDATIONS FOR IMPROVEMENT

Based on visualization analysis:

### 1. **Train Longer** (PRIORITY 1)
- **Current:** 5 epochs, training still improving (loss 63.1 → 7.4)
- **Recommendation:** Train for 20-50 epochs
- **Expected:** R² improvement from 0.38 to 0.70-0.85
- **Time cost:** ~2-8 hours on CPU (or 10-40 min on GPU)

**Evidence from training_curves:**
- Validation loss still decreasing at epoch 5
- No overfitting observed
- Model has capacity to learn more

### 2. **Add More Training Data** (PRIORITY 2)
- **Current:** 300 frames from HBridge board only
- **Available:** LoadShedding board has 120 more frames, 20 components
- **Recommendation:** Create combined dataset (420 frames, 42 components)
- **Expected:** Better generalization, higher R²

**Evidence from metrics:**
- Some components (R5, R88, U34) have very poor R²
- May be undertrained due to limited data
- More varied thermal conditions would help

### 3. **Investigate Poor Components** (PRIORITY 3)
- **Action:** Check thermal_comparison plots for components with R² < 0
- **Look for:**
  - ROI mask misalignment (component location wrong?)
  - Thermistor placement errors (not actually measuring sand temp?)
  - Extreme temperature ranges (outside training distribution?)
  - Small ROI size (too few pixels for accurate prediction?)

**Evidence from per-component metrics:**
- 9 out of 22 components have negative R²
- DL11 has extremely high MAE (4.46°C)
- Suggests systematic issues with specific components

### 4. **Tune Hyperparameters** (PRIORITY 4)
- **Current:** lr=0.001, batch_size=8, no data augmentation
- **Recommendations:**
  - Try learning rate schedule (reduce on plateau)
  - Experiment with batch sizes: 4, 16, 32
  - Add data augmentation (flips, small rotations)
  - Adjust early stopping patience (currently 15)

### 5. **Consider GPU Training** (PRIORITY 5)
- **Current:** 8 min/epoch on CPU
- **With GPU:** ~30 sec/epoch (16× faster)
- **Benefits:** Faster iteration for hyperparameter tuning
- **Cost:** ~$350 for RTX 3060, saves ~2.5 hours per training run

---

## 📁 File Organization

All visualizations saved to:
```
ml_model/cnn_thermal_modeling/results/
```

**Naming convention:**
- `{plot_type}_{timestamp}.png`
- Timestamp format: `YYYYMMDD_HHMMSS`
- Latest files have newest timestamp

**File sizes:**
- Training curves: ~50-100 KB (line plots)
- Scatter plots: ~100-350 KB (many points)
- Thermal comparisons: ~800 KB-1 MB (high-res heatmaps)
- Comparison plots: ~100 KB (tables/bar charts)

---

## 🎯 QUICK DIAGNOSTIC CHECKLIST

When reviewing results, check in this order:

1. ✅ **training_curves:** Loss decreasing? No NaN?
2. ✅ **scatter_actual_vs_predicted:** Points near red line? R² >0.3?
3. ✅ **thermal_comparison_t0.0s:** Spatial patterns look reasonable?
4. ⚠️ **thermal_comparison_t278.0s:** Predictions good at peak temperature?
5. ⚠️ **Error panels (panel 4):** Mostly blue/green (errors <2°C)?
6. ⚠️ **phase8_vs_phase8c_comparison:** CNN improving over linear model?
7. ⚠️ **Per-component metrics:** Any components with R² < -5?

**If all ✅:** Model is healthy, proceed with longer training
**If multiple ⚠️:** Review recommendations and investigate issues before scaling up

---

**End of Visualization Guide**
