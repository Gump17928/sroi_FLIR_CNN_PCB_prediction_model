"""
===============================================================================
PHASE 4: SPATIAL THERMAL COUPLING ANALYSIS
===============================================================================
Analyzes how nearby components thermally influence each other based on PCB
layout proximity.

This module provides:
- Component coordinate loading from PCB layout
- Proximity matrix calculation (Euclidean distance)
- Thermal coupling strength analysis
- Heat source identification (self-heating vs proximity heating)
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional


def load_component_coordinates(coordinates_file: str, debug: bool = False) -> Dict:
    """
    Load component X,Y coordinates from CSV file
    
    Expected CSV format:
    - Component: Component designator (e.g., R1, U2, C5)
    - X: X coordinate in mm
    - Y: Y coordinate in mm
    
    Alternative: 'Reference' column instead of 'Component'
    
    Args:
        coordinates_file: Path to CSV file with component coordinates
        debug: Enable verbose output
    
    Returns:
        Dictionary mapping component names to {'x': float, 'y': float}
    """
    
    if not os.path.exists(coordinates_file):
        print(f"Warning: Coordinates file not found: {coordinates_file}")
        return {}
    
    try:
        df = pd.read_csv(coordinates_file)
        
        # Expected columns: Component, X, Y (in mm)
        if 'Component' not in df.columns:
            # Try alternative column names
            if 'Reference' in df.columns:
                df['Component'] = df['Reference']
            else:
                print(f"Warning: Could not find Component/Reference column")
                return {}
        
        if 'X' not in df.columns or 'Y' not in df.columns:
            print(f"Warning: Missing X or Y columns in coordinates file")
            return {}
        
        coordinates = {}
        for _, row in df.iterrows():
            comp_name = str(row['Component']).strip()
            x = float(row['X'])
            y = float(row['Y'])
            coordinates[comp_name] = {'x': x, 'y': y}
        
        print(f"Loaded coordinates for {len(coordinates)} components")
        return coordinates
        
    except Exception as e:
        print(f"Error loading coordinates file: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return {}


def calculate_euclidean_distance(comp1_coords: Dict, comp2_coords: Dict) -> float:
    """
    Calculate Euclidean distance between two components
    
    Args:
        comp1_coords: {'x': float, 'y': float}
        comp2_coords: {'x': float, 'y': float}
    
    Returns:
        Distance in mm
    """
    
    distance = np.sqrt((comp1_coords['x'] - comp2_coords['x'])**2 +
                      (comp1_coords['y'] - comp2_coords['y'])**2)
    return distance


def build_proximity_matrix(component_data: Dict, component_coordinates: Dict,
                          proximity_threshold: float = 10.0) -> Dict:
    """
    Build proximity matrix identifying thermal neighbors for each component
    
    A thermal neighbor is a component within the proximity threshold distance.
    These neighbors can thermally influence each other through PCB conduction
    and localized air heating.
    
    Args:
        component_data: Dictionary of component DataFrames
        component_coordinates: Dictionary of component coordinates
        proximity_threshold: Distance threshold in mm (default: 10mm)
    
    Returns:
        Dictionary mapping components to lists of neighbor info:
        {'component': str, 'distance_mm': float}
    """
    
    if not component_coordinates:
        print("Warning: No coordinate data available for proximity analysis")
        return {}
    
    proximity_matrix = {}
    
    for comp1 in component_data.keys():
        if comp1 not in component_coordinates:
            continue
        
        neighbors = []
        for comp2 in component_data.keys():
            if comp1 == comp2:
                continue
            
            if comp2 not in component_coordinates:
                continue
            
            distance = calculate_euclidean_distance(component_coordinates[comp1],
                                                   component_coordinates[comp2])
            
            if distance <= proximity_threshold:
                neighbors.append({
                    'component': comp2,
                    'distance_mm': distance
                })
        
        # Sort by distance (closest first)
        neighbors.sort(key=lambda x: x['distance_mm'])
        proximity_matrix[comp1] = neighbors
    
    # Print statistics
    total_proximities = sum(len(n) for n in proximity_matrix.values())
    avg_neighbors = total_proximities / len(proximity_matrix) if proximity_matrix else 0
    
    print(f"Proximity analysis: {len(proximity_matrix)} components, "
          f"{total_proximities} thermal neighbor pairs")
    print(f"Average neighbors per component: {avg_neighbors:.1f} (within {proximity_threshold}mm)")
    
    return proximity_matrix


def classify_thermal_activity(component_name: str, analysis_results: Dict,
                              active_temp_threshold: float = 5.0) -> str:
    """
    Classify component as active heat generator or passive
    
    Classification based on:
    - Component type (U, VR, Q, D = active power components)
    - Temperature rise (>5°C indicates significant self-heating)
    
    Returns:
    - 'active': Active power component with significant heating
    - 'passive_high_heat': Passive component with high self-heating (e.g., power resistor)
    - 'passive': Low-power passive component
    - 'unknown': Insufficient data
    
    Args:
        component_name: Component designator
        analysis_results: Temperature analysis results
        active_temp_threshold: Temperature rise threshold (Celsius)
    
    Returns:
        Activity classification string
    """
    
    if component_name not in analysis_results:
        return 'unknown'
    
    stats = analysis_results[component_name]
    delta_temp = stats['delta_temp']
    
    # Active component type prefixes
    active_prefixes = ['U', 'IC', 'VR', 'Q', 'D', 'DL']
    
    is_active_type = any(component_name.upper().startswith(prefix)
                        for prefix in active_prefixes)
    
    if is_active_type and delta_temp > active_temp_threshold:
        return 'active'
    elif delta_temp > active_temp_threshold:
        return 'passive_high_heat'
    else:
        return 'passive'


def calculate_thermal_coupling_metrics(component_data: Dict, proximity_matrix: Dict,
                                      analysis_results: Dict) -> Dict:
    """
    Calculate comprehensive thermal coupling metrics between nearby components
    
    For each component and its neighbors, calculates:
    - Temperature correlation (how synchronized are temperature changes)
    - Thermal gradient (steady-state temperature difference)
    - Coupling strength (correlation weighted by distance)
    - Thermal influence (heating effect from hot neighbors)
    
    Args:
        component_data: Filtered component DataFrames
        proximity_matrix: Neighbor lists from build_proximity_matrix()
        analysis_results: Temperature statistics from Phase 2
    
    Returns:
        Dictionary mapping components to coupling metrics
    """
    
    if not proximity_matrix:
        print("Warning: No proximity matrix available for thermal coupling")
        return {}
    
    coupling_metrics = {}
    
    for component, neighbors in proximity_matrix.items():
        if component not in component_data or not neighbors:
            continue
        
        comp_temps = component_data[component]['Temperature'].values
        
        neighbor_coupling = []
        
        for neighbor_info in neighbors:
            neighbor = neighbor_info['component']
            distance = neighbor_info['distance_mm']
            
            if neighbor not in component_data:
                continue
            
            neighbor_temps = component_data[neighbor]['Temperature'].values
            
            # Ensure same length time series
            if len(comp_temps) != len(neighbor_temps):
                continue
            
            # Calculate coupling metrics
            
            # 1. Temperature Correlation (how synchronized)
            correlation = np.corrcoef(comp_temps, neighbor_temps)[0, 1]
            
            # 2. Thermal Gradient (steady-state temp difference)
            comp_ss = np.mean(comp_temps[-10:]) if len(comp_temps) >= 10 else comp_temps[-1]
            neighbor_ss = np.mean(neighbor_temps[-10:]) if len(neighbor_temps) >= 10 else neighbor_temps[-1]
            temp_gradient = neighbor_ss - comp_ss
            
            # 3. Coupling Strength (distance-weighted correlation)
            coupling_strength = correlation / (1 + distance/10.0)
            
            # 4. Thermal Influence (heating effect from hot neighbors)
            thermal_influence = coupling_strength * temp_gradient if temp_gradient > 0 else 0.0
            
            # Classify neighbor activity
            neighbor_activity = classify_thermal_activity(neighbor, analysis_results)
            
            neighbor_coupling.append({
                'neighbor': neighbor,
                'distance_mm': distance,
                'correlation': correlation,
                'temp_gradient_C': temp_gradient,
                'coupling_strength': coupling_strength,
                'thermal_influence': thermal_influence,
                'neighbor_activity': neighbor_activity,
                'neighbor_steady_state_C': neighbor_ss
            })
        
        # Sort by thermal influence (strongest first)
        neighbor_coupling.sort(key=lambda x: x['thermal_influence'], reverse=True)
        
        # Calculate totals
        total_external_influence = sum(nc['thermal_influence']
                                      for nc in neighbor_coupling
                                      if nc['thermal_influence'] > 0)
        
        num_hot_neighbors = sum(1 for nc in neighbor_coupling
                               if nc['temp_gradient_C'] > 2.0)
        
        # Component's own activity
        comp_activity = classify_thermal_activity(component, analysis_results)
        comp_delta_temp = analysis_results[component]['delta_temp'] if component in analysis_results else 0
        
        coupling_metrics[component] = {
            'neighbors': neighbor_coupling,
            'total_external_influence': total_external_influence,
            'num_hot_neighbors': num_hot_neighbors,
            'component_activity': comp_activity,
            'self_heating_C': comp_delta_temp,
            'estimated_proximity_heating_C': total_external_influence
        }
    
    return coupling_metrics


def estimate_heating_sources(component: str, coupling_metrics: Dict,
                            analysis_results: Dict) -> Optional[Dict]:
    """
    Separate self-heating from proximity heating for a component
    
    Estimates what fraction of temperature rise is due to:
    - Self-heating (internal power dissipation)
    - Proximity heating (heat from nearby hot components)
    
    Args:
        component: Component name
        coupling_metrics: Coupling data from calculate_thermal_coupling_metrics()
        analysis_results: Temperature statistics from Phase 2
    
    Returns:
        Dictionary with heating source breakdown or None if data unavailable
    """
    
    if component not in coupling_metrics or component not in analysis_results:
        return None
    
    metrics = coupling_metrics[component]
    stats = analysis_results[component]
    
    total_temp_rise = stats['delta_temp']
    comp_activity = metrics['component_activity']
    
    # Estimate self-heating fraction based on component activity type
    if comp_activity == 'active':
        self_heating_fraction = 0.7  # Active ICs: mostly self-heated
    elif comp_activity == 'passive_high_heat':
        self_heating_fraction = 0.6  # Power resistors: significant self-heating
    else:
        self_heating_fraction = 0.2  # Passive: mostly heated by surroundings
    
    # Calculate heating source estimates
    estimated_self_heating = total_temp_rise * self_heating_fraction
    estimated_proximity_heating = total_temp_rise * (1 - self_heating_fraction)
    
    # Identify top heat contributors
    top_contributors = []
    if 'neighbors' in metrics:
        for neighbor in metrics['neighbors'][:5]:  # Top 5
            if neighbor['thermal_influence'] > 0.1:
                top_contributors.append({
                    'component': neighbor['neighbor'],
                    'distance_mm': neighbor['distance_mm'],
                    'contribution_C': neighbor['thermal_influence'],
                    'neighbor_temp_C': neighbor['neighbor_steady_state_C']
                })
    
    return {
        'component': component,
        'total_temp_rise_C': total_temp_rise,
        'self_heating_C': estimated_self_heating,
        'proximity_heating_C': estimated_proximity_heating,
        'self_heating_fraction': self_heating_fraction,
        'proximity_heating_fraction': 1 - self_heating_fraction,
        'component_activity': comp_activity,
        'num_hot_neighbors': metrics['num_hot_neighbors'],
        'top_heat_contributors': top_contributors
    }
