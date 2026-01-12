"""
===============================================================================
PHASE 5: POTTING RISK ANALYSIS
===============================================================================
Analyzes component failure risk in potted/embedded systems where convective
cooling is eliminated.

This module provides:
- Potted condition temperature estimation
- Risk scoring based on temperature and thermal coupling
- Failure prediction for embedded systems
===============================================================================
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional


def analyze_potted_condition_risk(coupling_metrics: Dict, analysis_results: Dict,
                                  output_dir: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Analyze component failure risk in potted/embedded conditions
    
    In potted systems (epoxy, sand, potting compounds):
    - No convective cooling means temperatures rise significantly
    - Thermal coupling dominates - hot components affect neighbors more
    - Cumulative heating effects are critical for failure prediction
    
    Risk scoring factors:
    - Estimated potted temperature (2.5x multiplier for no convection)
    - Proximity heating from nearby hot components
    - Number of hot neighbors
    
    Args:
        coupling_metrics: Thermal coupling data from Phase 4
        analysis_results: Component statistics from Phase 2
        output_dir: Output directory for CSV file
    
    Returns:
        Tuple of (csv_filename, risk_dataframe) or (None, None) if no coupling data
    """
    
    if not coupling_metrics:
        print("No thermal coupling data available for potting risk analysis")
        return None, None
    
    risk_analysis = []
    
    for component, metrics in coupling_metrics.items():
        if component not in analysis_results:
            continue
        
        stats = analysis_results[component]
        
        # Extract risk factors
        baseline_temp = stats['final_temp']
        delta_temp = stats['delta_temp']
        proximity_heating = metrics['estimated_proximity_heating_C']
        num_hot_neighbors = metrics['num_hot_neighbors']
        activity_type = metrics['component_activity']
        
        # Estimate temperature rise in potted condition
        # In potting compound: no air cooling, only conduction through PCB/compound
        # Empirical multiplier: 2.5x for systems going from forced air to potting
        potted_temp_multiplier = 2.5
        estimated_potted_temp = baseline_temp + (delta_temp * (potted_temp_multiplier - 1))
        
        # Risk score calculation (0-10 scale)
        risk_score = 0
        
        # Factor 1: Absolute temperature risk
        if estimated_potted_temp > 85:  # Standard industrial temp limit
            risk_score += 5
        elif estimated_potted_temp > 70:
            risk_score += 3
        elif estimated_potted_temp > 60:
            risk_score += 1
        
        # Factor 2: Proximity heating risk
        if proximity_heating > 5.0:  # Significant external heating
            risk_score += 3
        elif proximity_heating > 2.0:
            risk_score += 1
        
        # Factor 3: Hot neighbor count risk
        if num_hot_neighbors >= 3:
            risk_score += 2
        elif num_hot_neighbors >= 2:
            risk_score += 1
        
        # Risk classification
        if risk_score >= 7:
            risk_level = 'HIGH'
        elif risk_score >= 4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        risk_analysis.append({
            'Component': component,
            'Activity_Type': activity_type,
            'Baseline_Temp_C': f'{baseline_temp:.1f}',
            'Delta_Temp_C': f'{delta_temp:.1f}',
            'Self_Heating_C': f'{metrics["self_heating_C"]:.1f}',
            'Proximity_Heating_C': f'{proximity_heating:.1f}',
            'Hot_Neighbors': num_hot_neighbors,
            'Estimated_Potted_Temp_C': f'{estimated_potted_temp:.1f}',
            'Risk_Score': risk_score,
            'Risk_Level': risk_level
        })
    
    # Sort by risk score (highest first)
    risk_analysis.sort(key=lambda x: x['Risk_Score'], reverse=True)
    
    # Create DataFrame
    df_risk = pd.DataFrame(risk_analysis)
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, 'potted_condition_risk_analysis.csv')
    df_risk.to_csv(csv_filename, index=False)
    
    # Print risk summary
    print(f"\n=== Potted Condition Risk Analysis ===")
    high_risk = sum(1 for r in risk_analysis if r['Risk_Level'] == 'HIGH')
    medium_risk = sum(1 for r in risk_analysis if r['Risk_Level'] == 'MEDIUM')
    low_risk = sum(1 for r in risk_analysis if r['Risk_Level'] == 'LOW')
    
    print(f"HIGH RISK components:   {high_risk}")
    print(f"MEDIUM RISK components: {medium_risk}")
    print(f"LOW RISK components:    {low_risk}")
    
    if high_risk > 0:
        print(f"\nTop HIGH RISK components:")
        for r in risk_analysis[:min(5, high_risk)]:
            if r['Risk_Level'] == 'HIGH':
                print(f"  {r['Component']}: Est. potted temp {r['Estimated_Potted_Temp_C']}°C, "
                      f"Risk score {r['Risk_Score']}")
    
    print(f"\nRisk analysis saved to: {csv_filename}")
    
    return csv_filename, df_risk


def get_high_risk_components(risk_df: pd.DataFrame, risk_level: str = 'HIGH') -> list:
    """
    Extract list of components at specified risk level
    
    Args:
        risk_df: Risk analysis DataFrame from analyze_potted_condition_risk()
        risk_level: 'HIGH', 'MEDIUM', or 'LOW'
    
    Returns:
        List of component names at the specified risk level
    """
    
    if risk_df is None or risk_df.empty:
        return []
    
    filtered = risk_df[risk_df['Risk_Level'] == risk_level]
    return filtered['Component'].tolist()


def summarize_risk_by_activity_type(risk_df: pd.DataFrame) -> Dict:
    """
    Summarize risk distribution by component activity type
    
    Groups risk analysis by thermal activity (active, passive, etc.)
    to identify patterns.
    
    Args:
        risk_df: Risk analysis DataFrame
    
    Returns:
        Dictionary mapping activity types to risk summaries
    """
    
    if risk_df is None or risk_df.empty:
        return {}
    
    summary = {}
    
    for activity_type in risk_df['Activity_Type'].unique():
        subset = risk_df[risk_df['Activity_Type'] == activity_type]
        
        summary[activity_type] = {
            'total_components': len(subset),
            'high_risk': sum(subset['Risk_Level'] == 'HIGH'),
            'medium_risk': sum(subset['Risk_Level'] == 'MEDIUM'),
            'low_risk': sum(subset['Risk_Level'] == 'LOW'),
            'avg_risk_score': subset['Risk_Score'].mean()
        }
    
    return summary
