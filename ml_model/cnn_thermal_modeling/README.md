# Phase 8c CNN Thermal Modeling

**U-Net CNN for PCB thermal field prediction from FLIR images**

---

## 🚀 Quick Start

```bash
python run_smart_pipeline.py
```

**See [QUICK_START.md](QUICK_START.md) for detailed instructions.**

---

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** ⭐ Start here! How to run the pipeline
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** 📊 Understand the 15+ plots generated
- **[RESULTS_UPDATE_SUMMARY.md](RESULTS_UPDATE_SUMMARY.md)** 🎨 3D visualizations & folder structure
- **[TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)** 🔧 Architecture, dependencies, troubleshooting

---

## 🎯 What This Does

1. **Train** a U-Net CNN to predict PCB thermal fields
2. **Generate** spatial thermal predictions from FLIR input
3. **Compare** CNN (Phase 8c) vs Linear Regression (Phase 8)
4. **Visualize** results with 15+ publication-quality plots
5. **Export** predictions and metrics to timestamped analysis folders

---

## 📊 Current Performance

| Metric | Value | Target |
|--------|-------|--------|
| R² Score | 0.38 | >0.90 |
| RMSE | 2.62°C | <2.0°C |
| MAE | 2.04°C | <1.5°C |

**Note:** Trained for only 5 epochs. Performance improves significantly with 20-50 epochs.

---

## 🏗️ Architecture

- **Model:** U-Net CNN (7.76M parameters)
- **Input:** FLIR thermal image (480×640)
- **Output:** Predicted thermal field (480×640)
- **Loss:** Masked MSE (penalize only at thermistor ROI locations)
- **Dataset:** 300 frames, 22 components, 112 MB HDF5

---

## ⚡ Features

✅ Smart model detection (reuse or retrain)  
✅ Timestamped analysis folders  
✅ 15+ visualization types  
✅ Optional 3D thermal visualizations  
✅ Backwards compatible with old model formats  
✅ Per-component performance metrics  
✅ Phase 8 vs 8c comparison plots  

---

## 📁 Generated Outputs

All results saved to: `results/analysis_YYYYMMDD_HHMMSS/`

- Trained model (`.keras`)
- Training curves
- 10 thermal field comparisons
- Scatter plots (actual vs predicted)
- Method comparison charts
- CSV predictions
- Detailed metrics
- Optional 3D visualizations

---

**Ready to start?** See [QUICK_START.md](QUICK_START.md)
