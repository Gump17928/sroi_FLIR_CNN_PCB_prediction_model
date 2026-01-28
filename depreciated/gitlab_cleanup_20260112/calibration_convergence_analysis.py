"""
===============================================================================
CALIBRATION CONVERGENCE ANALYSIS
===============================================================================
Analyzes how thermal calibration accuracy improves as more test data is added.

This module tracks:
- FLIR offset convergence (how mean/std evolve with sample count)
- Cooling ratio stability (air vs sand predictions)
- Component-type calibration quality metrics
- Statistical confidence intervals

Generates:
- Convergence plots showing offset evolution over test sessions
- Confidence interval plots (shrinking uncertainty with more data)
- Component-type comparison (which types need more calibration data)
- Per-component calibration history

Author: Thermal Analysis Pipeline
Date: December 2, 2025
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json


class CalibrationConvergenceAnalyzer:
    """Analyzes calibration data convergence as test sessions accumulate."""
    
    def __init__(self, calibration_points_file: Path, output_dir: Optional[Path] = None):
        """
        Initialize convergence analyzer.
        
        Args:
            calibration_points_file: Path to thermal_calibration_points.csv
            output_dir: Output directory for convergence plots
        """
        self.calibration_points_file = Path(calibration_points_file)
        self.output_dir = Path(output_dir) if output_dir else self.calibration_points_file.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load calibration data
        self.df = pd.read_csv(calibration_points_file)
        
        # Ensure required columns exist
        required_cols = ['component_name', 'component_type', 'flir_offset', 'test_session']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by test session (chronological order if sessions have timestamps)
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df = self.df.sort_values('timestamp')
        elif 'test_session' in self.df.columns:
            self.df = self.df.sort_values('test_session')
        
        print(f"Loaded {len(self.df)} calibration points from {len(self.df['test_session'].unique())} test sessions")
    
    def analyze_component_type_convergence(self) -> pd.DataFrame:
        """
        Analyze how component-type calibration converges with sample count.
        
        Returns:
            DataFrame with convergence metrics per component type
        """
        convergence_data = []
        
        for comp_type in self.df['component_type'].unique():
            type_df = self.df[self.df['component_type'] == comp_type].copy()
            
            # Calculate cumulative statistics as samples accumulate
            n_samples = len(type_df)
            
            for i in range(1, n_samples + 1):
                subset = type_df.iloc[:i]
                
                convergence_data.append({
                    'component_type': comp_type,
                    'sample_count': i,
                    'mean_offset': subset['flir_offset'].mean(),
                    'std_offset': subset['flir_offset'].std() if i > 1 else np.nan,
                    'min_offset': subset['flir_offset'].min(),
                    'max_offset': subset['flir_offset'].max(),
                    'confidence_interval_95': 1.96 * subset['flir_offset'].std() / np.sqrt(i) if i > 1 else np.nan,
                    'test_sessions': subset['test_session'].nunique()
                })
        
        return pd.DataFrame(convergence_data)
    
    def plot_offset_convergence(self, save_prefix: str = "calibration_convergence") -> List[str]:
        """
        Plot FLIR offset convergence for each component type.
        
        Shows how mean offset and confidence intervals evolve with sample count.
        
        Args:
            save_prefix: Prefix for output files
            
        Returns:
            List of saved plot file paths
        """
        convergence_df = self.analyze_component_type_convergence()
        
        output_files = []
        
        # Get unique component types
        component_types = convergence_df['component_type'].unique()
        
        # Create figure with subplots (one per component type)
        n_types = len(component_types)
        n_cols = min(3, n_types)
        n_rows = int(np.ceil(n_types / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_types == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, comp_type in enumerate(component_types):
            ax = axes[idx]
            type_data = convergence_df[convergence_df['component_type'] == comp_type]
            
            # Plot mean offset with confidence interval
            sample_counts = type_data['sample_count']
            mean_offsets = type_data['mean_offset']
            ci_95 = type_data['confidence_interval_95']
            
            # Main convergence line
            ax.plot(sample_counts, mean_offsets, 'o-', linewidth=2, markersize=8, 
                   label='Mean Offset', color='#2E86AB')
            
            # Confidence interval shading
            if not ci_95.isna().all():
                ax.fill_between(sample_counts, 
                               mean_offsets - ci_95,
                               mean_offsets + ci_95,
                               alpha=0.3, color='#2E86AB', label='95% CI')
            
            # Min/max range
            ax.plot(sample_counts, type_data['min_offset'], '--', color='gray', 
                   alpha=0.5, label='Min/Max Range')
            ax.plot(sample_counts, type_data['max_offset'], '--', color='gray', alpha=0.5)
            
            ax.set_xlabel('Sample Count', fontsize=11, fontweight='bold')
            ax.set_ylabel('FLIR Offset (°C)', fontsize=11, fontweight='bold')
            ax.set_title(f'{comp_type}\n({len(type_data)} measurements)', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Add final value annotation
            final_mean = mean_offsets.iloc[-1]
            final_std = type_data['std_offset'].iloc[-1]
            final_n = sample_counts.iloc[-1]
            ax.text(0.98, 0.98, f'Final: {final_mean:.2f}±{final_std:.2f}°C\nn={final_n}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(n_types, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        png_file = self.output_dir / f"{save_prefix}_by_type.png"
        pdf_file = self.output_dir / f"{save_prefix}_by_type.pdf"
        fig.savefig(png_file, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_file, bbox_inches='tight')
        plt.close(fig)
        
        output_files.extend([str(png_file), str(pdf_file)])
        print(f"\nSaved offset convergence plots:")
        print(f"  {png_file}")
        print(f"  {pdf_file}")
        
        return output_files
    
    def plot_confidence_evolution(self, save_prefix: str = "calibration_confidence") -> List[str]:
        """
        Plot how calibration confidence improves with sample count.
        
        Shows standard deviation and confidence interval shrinking.
        
        Args:
            save_prefix: Prefix for output files
            
        Returns:
            List of saved plot file paths
        """
        convergence_df = self.analyze_component_type_convergence()
        
        output_files = []
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Standard Deviation vs Sample Count
        for comp_type in convergence_df['component_type'].unique():
            type_data = convergence_df[convergence_df['component_type'] == comp_type]
            # Filter out NaN values (first sample has no std)
            type_data_clean = type_data.dropna(subset=['std_offset'])
            if len(type_data_clean) > 0:
                ax1.plot(type_data_clean['sample_count'], type_data_clean['std_offset'],
                        'o-', linewidth=2, markersize=6, label=comp_type)
        
        ax1.set_xlabel('Sample Count', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Standard Deviation (°C)', fontsize=12, fontweight='bold')
        ax1.set_title('Calibration Uncertainty vs Sample Count', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Add reference line for target uncertainty
        ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                   label='Target (σ<2.0°C)')
        
        # Plot 2: 95% Confidence Interval vs Sample Count
        for comp_type in convergence_df['component_type'].unique():
            type_data = convergence_df[convergence_df['component_type'] == comp_type]
            type_data_clean = type_data.dropna(subset=['confidence_interval_95'])
            if len(type_data_clean) > 0:
                ax2.plot(type_data_clean['sample_count'], type_data_clean['confidence_interval_95'],
                        'o-', linewidth=2, markersize=6, label=comp_type)
        
        ax2.set_xlabel('Sample Count', fontsize=12, fontweight='bold')
        ax2.set_ylabel('95% Confidence Interval (±°C)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Confidence vs Sample Count', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Add reference line
        ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5,
                   label='High Confidence (CI<1.0°C)')
        
        plt.tight_layout()
        
        # Save plot
        png_file = self.output_dir / f"{save_prefix}_evolution.png"
        pdf_file = self.output_dir / f"{save_prefix}_evolution.pdf"
        fig.savefig(png_file, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_file, bbox_inches='tight')
        plt.close(fig)
        
        output_files.extend([str(png_file), str(pdf_file)])
        print(f"\nSaved confidence evolution plots:")
        print(f"  {png_file}")
        print(f"  {pdf_file}")
        
        return output_files
    
    def plot_test_session_contribution(self, save_prefix: str = "calibration_sessions") -> List[str]:
        """
        Plot contribution of each test session to calibration database.
        
        Shows which test sessions added data for which component types.
        
        Args:
            save_prefix: Prefix for output files
            
        Returns:
            List of saved plot file paths
        """
        output_files = []
        
        # Create pivot table: test_session vs component_type
        session_counts = self.df.groupby(['test_session', 'component_type']).size().reset_index(name='count')
        pivot = session_counts.pivot(index='test_session', columns='component_type', values='count').fillna(0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))
        
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Sample Count'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Component Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Session', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Sample Distribution Across Test Sessions', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        png_file = self.output_dir / f"{save_prefix}_heatmap.png"
        pdf_file = self.output_dir / f"{save_prefix}_heatmap.pdf"
        fig.savefig(png_file, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_file, bbox_inches='tight')
        plt.close(fig)
        
        output_files.extend([str(png_file), str(pdf_file)])
        print(f"\nSaved test session contribution plots:")
        print(f"  {png_file}")
        print(f"  {pdf_file}")
        
        return output_files
    
    def generate_convergence_report(self, save_file: str = "calibration_convergence_report.json") -> Dict:
        """
        Generate comprehensive convergence analysis report.
        
        Args:
            save_file: Output JSON file name
            
        Returns:
            Dictionary with convergence metrics
        """
        convergence_df = self.analyze_component_type_convergence()
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_samples': len(self.df),
            'test_sessions': self.df['test_session'].unique().tolist(),
            'n_test_sessions': self.df['test_session'].nunique(),
            'component_types': {},
            'overall_statistics': {}
        }
        
        # Per component type statistics
        for comp_type in self.df['component_type'].unique():
            type_df = self.df[self.df['component_type'] == comp_type]
            type_conv = convergence_df[convergence_df['component_type'] == comp_type]
            
            final_stats = type_conv[type_conv['sample_count'] == type_conv['sample_count'].max()].iloc[0]
            
            report['component_types'][comp_type] = {
                'sample_count': int(final_stats['sample_count']),
                'mean_offset': float(final_stats['mean_offset']),
                'std_offset': float(final_stats['std_offset']) if not pd.isna(final_stats['std_offset']) else None,
                'confidence_interval_95': float(final_stats['confidence_interval_95']) if not pd.isna(final_stats['confidence_interval_95']) else None,
                'calibration_quality': self._assess_calibration_quality(final_stats),
                'test_sessions_contributing': type_df['test_session'].unique().tolist()
            }
        
        # Overall statistics
        report['overall_statistics'] = {
            'mean_samples_per_type': float(self.df.groupby('component_type').size().mean()),
            'min_samples_per_type': int(self.df.groupby('component_type').size().min()),
            'max_samples_per_type': int(self.df.groupby('component_type').size().max()),
            'types_needing_more_data': [
                comp_type for comp_type, stats in report['component_types'].items()
                if stats['sample_count'] < 3 or (stats['std_offset'] and stats['std_offset'] > 2.0)
            ]
        }
        
        # Save report
        report_file = self.output_dir / save_file
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSaved convergence report: {report_file}")
        
        return report
    
    def _assess_calibration_quality(self, stats: pd.Series) -> str:
        """
        Assess calibration quality based on sample count and uncertainty.
        
        Args:
            stats: Row from convergence DataFrame
            
        Returns:
            Quality assessment string
        """
        n = stats['sample_count']
        std = stats['std_offset']
        
        if pd.isna(std):
            return 'INSUFFICIENT_DATA'
        
        if n >= 5 and std < 2.0:
            return 'HIGH'
        elif n >= 3 and std < 3.0:
            return 'MEDIUM'
        elif n >= 2 and std < 4.0:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def run_complete_analysis(self) -> Dict:
        """
        Run complete convergence analysis with all plots.
        
        Returns:
            Dictionary with report and output file paths
        """
        print("="*80)
        print(" CALIBRATION CONVERGENCE ANALYSIS")
        print("="*80)
        print(f"\nAnalyzing: {self.calibration_points_file}")
        print(f"Output directory: {self.output_dir}")
        
        output_files = []
        
        # Generate all plots
        print("\n[1/3] Plotting offset convergence...")
        output_files.extend(self.plot_offset_convergence())
        
        print("\n[2/3] Plotting confidence evolution...")
        output_files.extend(self.plot_confidence_evolution())
        
        print("\n[3/3] Plotting test session contributions...")
        output_files.extend(self.plot_test_session_contribution())
        
        # Generate report
        print("\n[4/4] Generating convergence report...")
        report = self.generate_convergence_report()
        
        print("\n" + "="*80)
        print(" CONVERGENCE ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nGenerated {len(output_files)} plots")
        print(f"Convergence report: {self.output_dir / 'calibration_convergence_report.json'}")
        
        # Print summary
        print("\n=== CALIBRATION QUALITY SUMMARY ===")
        for comp_type, stats in report['component_types'].items():
            quality = stats['calibration_quality']
            n = stats['sample_count']
            mean_offset = stats['mean_offset']
            std_offset = stats['std_offset'] if stats['std_offset'] else 'N/A'
            print(f"  {comp_type:15s}: {quality:15s} (n={n:2d}, offset={mean_offset:+.2f}°C, σ={std_offset})")
        
        if report['overall_statistics']['types_needing_more_data']:
            print("\n⚠ Component types needing more calibration data:")
            for comp_type in report['overall_statistics']['types_needing_more_data']:
                print(f"  - {comp_type}")
        
        return {
            'report': report,
            'output_files': output_files
        }


def main():
    """Standalone execution for convergence analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze thermal calibration convergence'
    )
    parser.add_argument('--calibration_file', type=str, 
                       default='outputs/calibration_database/thermal_calibration_points.csv',
                       help='Path to thermal_calibration_points.csv')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as calibration file)')
    
    args = parser.parse_args()
    
    # Run analysis
    cal_file = Path(args.calibration_file)
    if not cal_file.exists():
        print(f"Error: Calibration file not found: {cal_file}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else cal_file.parent
    
    analyzer = CalibrationConvergenceAnalyzer(cal_file, output_dir)
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
