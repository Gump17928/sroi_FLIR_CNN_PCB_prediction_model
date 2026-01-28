"""
ResearchIR Stats Parser
========================
Parses ResearchIR Stats.txt files to extract time-series FLIR data for all ROI components.

This module converts the multi-file ResearchIR output (one Stats.txt per frame) into a 
single pandas DataFrame suitable for thermal analysis and calibration.

Author: Thermal Analysis Pipeline
Date: November 25, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import re
from datetime import datetime, timedelta


class ResearchIRStatsParser:
    """Parses ResearchIR Stats.txt files to extract component temperature time-series."""
    
    def __init__(self, stats_directory: Path):
        """
        Initialize parser with directory containing Stats.txt files.
        
        Args:
            stats_directory: Path to directory with Rec-XXXXXX_N - Stats.txt files
        """
        self.stats_dir = Path(stats_directory)
        self.component_names = []
        self.parsed_data = None
        
    def parse_all_frames(self) -> pd.DataFrame:
        """
        Parse all Stats.txt files in directory and return combined DataFrame.
        
        Returns:
            DataFrame with columns: frame, reltime, [component1], [component2], ...
        """
        # Find all Stats.txt files and sort by frame number
        stats_files = sorted(
            self.stats_dir.glob("*_* - Stats.txt"),
            key=lambda x: self._extract_frame_number(x.name)
        )
        
        if not stats_files:
            raise FileNotFoundError(f"No Stats.txt files found in {self.stats_dir}")
        
        print(f"Found {len(stats_files)} Stats.txt files to parse...")
        
        # Parse first file to get component names
        first_data = self._parse_single_stats_file(stats_files[0])
        self.component_names = list(first_data.keys())
        
        # Initialize data storage
        all_frames = []
        
        # Parse all files
        for i, stats_file in enumerate(stats_files):
            frame_data = self._parse_single_stats_file(stats_file)
            frame_number = self._extract_frame_number(stats_file.name)
            
            # Add frame number
            frame_data['frame'] = frame_number
            all_frames.append(frame_data)
            
            if (i + 1) % 50 == 0:
                print(f"  Parsed {i + 1}/{len(stats_files)} frames...")
        
        # Combine into DataFrame
        df = pd.DataFrame(all_frames)
        
        # Sort by frame number
        df = df.sort_values('frame').reset_index(drop=True)
        
        # Add relative time (frames are 15 seconds apart)
        df['reltime'] = df['frame'] * 15.0  # 15 seconds between frames
        
        # Reorder columns: frame, reltime, then all components
        col_order = ['frame', 'reltime'] + self.component_names
        df = df[col_order]
        
        self.parsed_data = df
        print(f"\nParsing complete! {len(df)} frames, {len(self.component_names)} components")
        
        return df
    
    def _parse_single_stats_file(self, stats_file: Path) -> Dict[str, float]:
        """
        Parse a single Stats.txt file and extract Mean temperature for each component.
        
        Args:
            stats_file: Path to Stats.txt file
            
        Returns:
            Dictionary mapping component name to mean temperature
        """
        with open(stats_file, 'r') as f:
            lines = f.readlines()
        
        # Find header line (contains component names)
        header_line = None
        mean_line = None
        
        for i, line in enumerate(lines):
            if 'Statistic [units]' in line:
                header_line = line
            elif 'Mean [C]' in line:
                mean_line = line
                break
        
        if not header_line or not mean_line:
            raise ValueError(f"Could not find header or mean line in {stats_file}")
        
        # Parse component names from header
        # Split by multiple spaces and filter out empty strings
        header_parts = [p.strip() for p in re.split(r'\s{2,}', header_line) if p.strip()]
        
        # Parse mean temperatures
        mean_parts = [p.strip() for p in re.split(r'\s{2,}', mean_line) if p.strip()]
        
        # First column is "Statistic [units]" / "Mean [C]", rest are component names/values
        component_names = header_parts[1:]  # Skip first column
        mean_temps = mean_parts[1:]  # Skip first column
        
        # Build dictionary
        result = {}
        for comp_name, temp_str in zip(component_names, mean_temps):
            try:
                temp = float(temp_str)
                result[comp_name] = temp
            except ValueError:
                # Handle N/A or invalid values
                result[comp_name] = np.nan
        
        return result
    
    def _extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename like 'Rec-000011_42 - Stats.txt'."""
        match = re.search(r'_(\d+)\s*-\s*Stats', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def get_component_list(self) -> List[str]:
        """Return list of all component names found in FLIR data."""
        return self.component_names.copy()
    
    def export_to_csv(self, output_file: Path):
        """Export parsed data to CSV file."""
        if self.parsed_data is None:
            raise ValueError("No data parsed yet. Call parse_all_frames() first.")
        
        self.parsed_data.to_csv(output_file, index=False)
        print(f"Exported FLIR data to {output_file}")


def main():
    """Parse LoadShedding ResearchIR Stats files and export to CSV."""
    
    # Input directory
    stats_dir = Path(__file__).parent / "inputs" / "ResearchIR_Outputs_Load_Shedding"
    
    # Output file
    output_file = Path(__file__).parent / "outputs" / "LoadShedding_FLIR_AllComponents.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse
    parser = ResearchIRStatsParser(stats_dir)
    df = parser.parse_all_frames()
    
    # Display component list
    print("\n" + "="*80)
    print("FLIR Components Found:")
    print("="*80)
    components = parser.get_component_list()
    for i, comp in enumerate(components, 1):
        print(f"{i:3d}. {comp}")
    
    # Show data preview
    print("\n" + "="*80)
    print("Data Preview (first 5 rows):")
    print("="*80)
    print(df.head())
    
    print("\n" + "="*80)
    print("Data Preview (last 5 rows):")
    print("="*80)
    print(df.tail())
    
    # Export
    parser.export_to_csv(output_file)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the component list above")
    print("2. Identify which components have thermistor measurements:")
    print("   - AI0, AI1, AI4, AI6, AI7 from USB-TEMP device")
    print("3. Create component mapping for calibration")
    print(f"\nParsed data saved to: {output_file}")


if __name__ == "__main__":
    main()
