"""
===============================================================================
ROI PIXEL MAPPER - CNN Training Data Generator
===============================================================================
Standalone utility to generate ROI-to-pixel mappings from SROI generation data.

This module provides a convenient interface to generate pixel maps for CNN
thermal field reconstruction without modifying the SROI generation workflow.

Purpose:
    - Extract pixel coordinates for each component ROI
    - Map component names to their constituent pixels in thermal images
    - Generate CSV files for CNN training data preprocessing

Usage:
    from roi_pixel_mapper import ROIPixelMapper
    
    mapper = ROIPixelMapper()
    mapper.generate_pixel_map_from_csv(
        component_csv="hbridge_pcb_components_enhanced.csv",
        output_file="HBridge_roi_pixel_map.csv",
        pcb_width_mm=227.381,
        pcb_height_mm=164.937,
        pcb_bottom_left_px=(102, 400),
        pcb_top_right_px=(500, 150)
    )

Integration:
    This module wraps the enhanced_scalable_csv_to_sroi functionality
    specifically for pixel map generation, making it easy to integrate
    with the CNN training pipeline.

Created: January 12, 2026
===============================================================================
"""

import sys
from pathlib import Path

# Add parent directory to path to import enhanced_scalable_csv_to_sroi
sys.path.insert(0, str(Path(__file__).parent.parent / "sroi_generation_ResearchIR"))

from enhanced_scalable_csv_to_sroi import EnhancedScalableCSVToSROI


class ROIPixelMapper:
    """
    Utility for generating ROI pixel maps for CNN training.
    
    Wraps the SROI generator to provide a clean interface for pixel map
    generation without requiring SROI file creation.
    """
    
    def __init__(self, image_width: int = 640, image_height: int = 480):
        """
        Initialize ROI pixel mapper.
        
        Args:
            image_width: FLIR thermal image width in pixels
            image_height: FLIR thermal image height in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
        self.generator = EnhancedScalableCSVToSROI()
    
    def generate_pixel_map_from_csv(self, 
                                    component_csv: str,
                                    output_file: str,
                                    pcb_width_mm: float,
                                    pcb_height_mm: float,
                                    pcb_bottom_left_px: tuple,
                                    pcb_top_right_px: tuple) -> str:
        """
        Generate ROI pixel map from component CSV file.
        
        Args:
            component_csv: Path to enhanced component CSV (from Altium export)
            output_file: Path to output pixel map CSV
            pcb_width_mm: PCB width in millimeters
            pcb_height_mm: PCB height in millimeters
            pcb_bottom_left_px: Bottom-left corner in FLIR image (x, y) pixels
            pcb_top_right_px: Top-right corner in FLIR image (x, y) pixels
        
        Returns:
            Path to generated pixel map CSV file
        """
        print(f"\n{'='*80}")
        print(f"  ROI PIXEL MAP GENERATION")
        print(f"{'='*80}")
        print(f"  Component CSV: {component_csv}")
        print(f"  Output File:   {output_file}")
        print(f"  PCB Size:      {pcb_width_mm:.1f} x {pcb_height_mm:.1f} mm")
        print(f"  Image Size:    {self.image_width} x {self.image_height} px")
        print(f"{'='*80}\n")
        
        # Calculate transformation using corner coordinates
        self.generator.auto_calculate_transformation(
            pcb_width_mm, pcb_height_mm,
            pcb_bottom_left_px, pcb_top_right_px
        )
        
        # Read and transform CSV components
        components = self.generator.read_csv_components_with_transform(component_csv)
        
        if not components:
            print(f"  ✗ Failed to read components from {component_csv}")
            return None
        
        # Export pixel map
        result = self.generator.export_roi_pixel_map(
            components, 
            output_file,
            image_width=self.image_width,
            image_height=self.image_height
        )
        
        return result
    
    def generate_pixel_maps_for_boards(self, board_configs: dict, output_dir: str = "."):
        """
        Generate pixel maps for multiple boards.
        
        Args:
            board_configs: Dictionary of board configurations:
                {
                    "HBridge": {
                        "component_csv": "hbridge_pcb_components_enhanced.csv",
                        "dimensions": (227.381, 164.937),  # mm
                        "corners": {
                            "bottom_left": (102, 400),
                            "top_right": (500, 150)
                        }
                    },
                    ...
                }
            output_dir: Directory for output pixel map files
        
        Returns:
            Dictionary mapping board names to pixel map file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for board_name, config in board_configs.items():
            output_file = output_path / f"{board_name}_roi_pixel_map.csv"
            
            result = self.generate_pixel_map_from_csv(
                component_csv=config['component_csv'],
                output_file=str(output_file),
                pcb_width_mm=config['dimensions'][0],
                pcb_height_mm=config['dimensions'][1],
                pcb_bottom_left_px=config['corners']['bottom_left'],
                pcb_top_right_px=config['corners']['top_right']
            )
            
            results[board_name] = result
        
        return results


def main():
    """
    Example usage: Generate pixel maps for HBridge and LoadShedding boards.
    """
    # Initialize mapper
    mapper = ROIPixelMapper(image_width=640, image_height=480)
    
    # Board configurations
    board_configs = {
        "HBridge": {
            "component_csv": "inputs/hbridge_pcb_components_enhanced.csv",
            "dimensions": (227.381, 164.937),  # mm (width, height)
            "corners": {
                "bottom_left": (102, 400),  # (x, y) in FLIR image pixels
                "top_right": (500, 150)
            }
        },
        "LoadShedding": {
            "component_csv": "inputs/loadshedding_ac_switch_pcb_components_enhanced.csv",
            "dimensions": (167.0, 78.4),  # mm (width, height)
            "corners": {
                "bottom_left": (105, 405),  # (x, y) in FLIR image pixels
                "top_right": (495, 145)
            }
        }
    }
    
    # Generate pixel maps
    print("\n" + "="*80)
    print("  BATCH PIXEL MAP GENERATION")
    print("="*80 + "\n")
    
    results = mapper.generate_pixel_maps_for_boards(
        board_configs,
        output_dir="outputs/roi_pixel_maps"
    )
    
    # Summary
    print("\n" + "="*80)
    print("  GENERATION COMPLETE")
    print("="*80)
    for board_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {board_name}: {result}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
