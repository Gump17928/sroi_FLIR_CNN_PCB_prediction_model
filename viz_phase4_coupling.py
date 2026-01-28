"""
===============================================================================
PHASE 4 VISUALIZATION - Thermal Coupling & Spatial Maps
===============================================================================
Visualization functions for Phase 4 (Spatial Thermal Coupling Analysis):
- Heat source decomposition (self-heating vs proximity heating)
- Spatial thermal coupling heatmaps
- Component proximity connection visualizations

Used by: phase4_spatial_coupling.py
Data source: FLIR thermal measurements + PCB component coordinates
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List
import os

# Import visualization helpers and phase modules
import viz_helpers as viz
import phase4_spatial_coupling as phase4


def create_thermal_coupling_visualizations(component_data: Dict, coupling_metrics: Dict, 
                                          analysis_results: Dict, output_dir: str,
                                          component_coordinates: Dict,
                                          debug: bool = False) -> List[str]:
    """
    Create thermal coupling visualization plots:
    1. Stacked bar chart: self-heating vs proximity heating
    2. Spatial thermal coupling heatmap
    
    Args:
        component_data: Dict of temperature DataFrames
        coupling_metrics: Dict of thermal coupling metrics
        analysis_results: Dict of transient analysis results
        output_dir: Output directory for files
        component_coordinates: Dict of component X,Y coordinates
        debug: Enable debug output
        
    Returns:
        List of output file paths
    """
    
    output_files = []
    
    if not coupling_metrics:
        print("Warning: No coupling metrics available for visualization")
        return output_files
    
    viz.setup_ieee_plot_style()
    
    # --- Plot 1: Stacked Bar Chart ---
    sorted_components = sorted(
        [(comp, analysis_results[comp]['delta_temp']) for comp in coupling_metrics.keys() if comp in analysis_results],
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    if sorted_components:
        fig, ax = plt.subplots(figsize=(7, 5))
        
        component_names = []
        self_heating_vals = []
        proximity_heating_vals = []
        
        for comp, _ in sorted_components:
            heating_sources = phase4.estimate_heating_sources(comp, coupling_metrics, analysis_results)
            
            if heating_sources:
                component_names.append(comp)
                self_heating_vals.append(heating_sources['self_heating_C'])
                proximity_heating_vals.append(heating_sources['proximity_heating_C'])
        
        x_pos = np.arange(len(component_names))
        
        ax.bar(x_pos, self_heating_vals, label='Self-Heating', color='#ff7f0e', alpha=0.8)
        ax.bar(x_pos, proximity_heating_vals, bottom=self_heating_vals, 
              label='Proximity Heating', color='#2ca02c', alpha=0.8)
        
        ax.set_xlabel('Component', fontsize=8)
        ax.set_ylabel('Temperature Rise (°C)', fontsize=8)
        ax.set_title('Heat Source Decomposition: Self-Heating vs Proximity Heating', fontsize=8, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(component_names, fontsize=7)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'thermal_coupling_heat_sources')
        png_file = viz.save_plot(filename, 'png')
        pdf_file = viz.save_plot(filename, 'pdf')
        output_files.extend([png_file, pdf_file])
        plt.close()
    
    # --- Plot 2: Spatial Thermal Coupling Map ---
    if component_coordinates:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot all components
        for comp in component_data.keys():
            if comp not in component_coordinates:
                continue
            
            coords = component_coordinates[comp]
            activity = phase4.classify_thermal_activity(comp, analysis_results)
            delta_temp = analysis_results[comp]['delta_temp'] if comp in analysis_results else 0
            
            marker_size = 30 + delta_temp * 10
            
            if activity == 'active':
                color = plt.cm.Reds(min(delta_temp / 30.0, 1.0))
                marker = 's'
            elif activity == 'passive_high_heat':
                color = plt.cm.Oranges(min(delta_temp / 20.0, 1.0))
                marker = '^'
            else:
                color = plt.cm.Blues(min(delta_temp / 10.0, 1.0))
                marker = 'o'
            
            ax.scatter(coords['x'], coords['y'], s=marker_size, c=[color], 
                      marker=marker, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            if delta_temp > 10.0:
                ax.annotate(comp, (coords['x'], coords['y']), 
                          fontsize=6, ha='center', va='bottom')
        
        # Draw proximity connections
        for comp, metrics in coupling_metrics.items():
            if comp not in component_coordinates:
                continue
            
            comp_coords = component_coordinates[comp]
            
            for neighbor_info in metrics['neighbors'][:3]:
                neighbor = neighbor_info['neighbor']
                if neighbor not in component_coordinates:
                    continue
                
                coupling_strength = neighbor_info['coupling_strength']
                
                if coupling_strength > 0.3:
                    neighbor_coords = component_coordinates[neighbor]
                    linewidth = coupling_strength * 2
                    alpha = min(coupling_strength, 0.7)
                    
                    ax.plot([comp_coords['x'], neighbor_coords['x']], 
                           [comp_coords['y'], neighbor_coords['y']], 
                           'r-', linewidth=linewidth, alpha=alpha)
        
        ax.set_xlabel('X Position (mm)', fontsize=8)
        ax.set_ylabel('Y Position (mm)', fontsize=8)
        ax.set_title('Spatial Thermal Coupling Map', fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='Active IC'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='High-Power Passive'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Passive'),
            Line2D([0], [0], color='red', linewidth=2, label='Strong Thermal Coupling')
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='best')
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'spatial_thermal_coupling_map')
        png_file = viz.save_plot(filename, 'png')
        pdf_file = viz.save_plot(filename, 'pdf')
        output_files.extend([png_file, pdf_file])
        plt.close()
    
    print(f"Created thermal coupling visualizations: {len(output_files)} files")
    return output_files
