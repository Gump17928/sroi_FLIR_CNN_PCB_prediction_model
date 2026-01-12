"""
Setup script for sroi_FLIR_CNN_PCB_prediction_model
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sroi_flir_cnn_pcb",
    version="0.1.0",
    author="Your Name",
    description="CNN model for predicting PCB thermal temperatures from FLIR images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gump17928/sroi_FLIR_CNN_PCB_prediction_model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "tensorflow>=2.6.0",
        "keras>=2.6.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "opencv-python>=4.5.0",
        "pillow>=8.3.0",
    ],
)
