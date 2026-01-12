"""
SROI File Parser for FLIR Thermal Images

This module provides utilities to parse and extract thermal data from .sroi files.
"""

import struct
import numpy as np


class SROIParser:
    """Parser for FLIR .sroi files."""
    
    def __init__(self, filepath):
        """
        Initialize the SROI parser.
        
        Args:
            filepath (str): Path to the .sroi file
        """
        self.filepath = filepath
        self.data = None
        self.metadata = {}
        
    def read_file(self):
        """
        Read and parse the .sroi file.
        
        Returns:
            dict: Dictionary containing thermal data and metadata
        """
        try:
            with open(self.filepath, 'rb') as f:
                # Read file header
                header = f.read(128)
                self._parse_header(header)
                
                # Read thermal data
                thermal_data = f.read()
                self.data = self._parse_thermal_data(thermal_data)
                
            return {
                'data': self.data,
                'metadata': self.metadata
            }
        except Exception as e:
            raise ValueError(f"Error reading SROI file: {str(e)}")
    
    def _parse_header(self, header):
        """Parse the file header to extract metadata."""
        # Basic header parsing (to be refined based on actual SROI format)
        self.metadata['file_version'] = header[:4]
        self.metadata['image_width'] = struct.unpack('I', header[8:12])[0] if len(header) >= 12 else 640
        self.metadata['image_height'] = struct.unpack('I', header[12:16])[0] if len(header) >= 16 else 480
        
    def _parse_thermal_data(self, data):
        """Parse thermal data from the file."""
        # Convert binary data to numpy array
        # Assuming 16-bit thermal values
        width = self.metadata.get('image_width', 640)
        height = self.metadata.get('image_height', 480)
        
        try:
            thermal_array = np.frombuffer(data, dtype=np.uint16)
            if thermal_array.size >= width * height:
                thermal_array = thermal_array[:width * height].reshape((height, width))
            else:
                # If not enough data, create placeholder
                thermal_array = np.zeros((height, width), dtype=np.uint16)
        except Exception:
            thermal_array = np.zeros((height, width), dtype=np.uint16)
            
        return thermal_array
    
    def get_temperature_array(self):
        """
        Convert raw thermal data to temperature values in Celsius.
        
        Returns:
            numpy.ndarray: Temperature array in Celsius
        """
        if self.data is None:
            self.read_file()
        
        # Convert raw values to temperature (simplified conversion)
        # Actual conversion depends on FLIR calibration parameters
        temp_celsius = (self.data.astype(float) / 100.0) - 273.15
        
        return temp_celsius


def load_sroi_file(filepath):
    """
    Convenience function to load a .sroi file.
    
    Args:
        filepath (str): Path to the .sroi file
        
    Returns:
        numpy.ndarray: Temperature array in Celsius
    """
    parser = SROIParser(filepath)
    parser.read_file()
    return parser.get_temperature_array()
