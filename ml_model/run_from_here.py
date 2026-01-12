#!/usr/bin/env python3
"""
One-Command Wrapper - Run from ml_model directory
==================================================

This wrapper allows you to run the pipeline from the ml_model directory
with a single command. It handles path setup automatically.

Usage:
    cd ml_model
    python run_from_here.py

Author: CNN Pipeline  
Date: 2025-01-12
"""

import os
import sys
from pathlib import Path

# Setup paths
current_dir = Path(__file__).resolve().parent
ml_model_dir = current_dir.parent  # ml_model/
workspace_dir = ml_model_dir.parent  # thermal_post_processing/

# Add to path
sys.path.append(str(workspace_dir))
sys.path.append(str(current_dir))

# Change to workspace directory for relative imports
os.chdir(workspace_dir)

# Import and run the smart pipeline
from cnn_thermal_modeling.run_smart_pipeline import main

if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path includes: {workspace_dir}\n")
    main()
