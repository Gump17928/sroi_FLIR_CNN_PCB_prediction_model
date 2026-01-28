"""
Test script for FLIR frame loader - load first 10 frames
"""

import sys
sys.path.append('.')

from flir_frame_loader import FLIRFrameLoader

# Test with first 10 frames from HBridge
loader = FLIRFrameLoader(verbose=True)

frames, timestamps = loader.load_sequence(
    "inputs/ResearchIR_Outputs_HBridge_15s",
    max_frames=10
)

print(f"\nTest Results:")
print(f"  Loaded frames: {frames.shape}")
print(f"  Timestamps: {timestamps}")
print(f"  Temperature range: {loader.get_temperature_range()}")

# Show frame 0 statistics
stats = loader.get_frame_statistics(0)
print(f"\nFrame 0 Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")
