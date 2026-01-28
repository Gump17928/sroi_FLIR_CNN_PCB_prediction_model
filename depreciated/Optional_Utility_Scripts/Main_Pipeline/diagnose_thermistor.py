"""
DIAGNOSTIC 3: Check Thermistor Data Quality
Find if outliers/corruption causing bad training
"""
import pandas as pd
import numpy as np

print('='*80)
print('DIAGNOSTIC 3: Thermistor Data Quality Check')
print('='*80)

# Load thermistor CSV
therm_csv = pd.read_csv('outputs/0116_1730_P1-7/HBridge_15s_thermistor_timeseries.csv')

print(f"\nThermistor CSV shape: {therm_csv.shape}")
print(f"Columns: {list(therm_csv.columns[:5])}... ({len(therm_csv.columns)} total)")

print(f"\nData statistics:")
print(therm_csv.describe())

print(f"\n" + "="*80)
print("OUTLIER DETECTION")
print("="*80)

outliers_found = False

for col in therm_csv.columns[1:]:  # Skip Time column
    max_temp = therm_csv[col].max()
    min_temp = therm_csv[col].min()
    mean_temp = therm_csv[col].mean()
    
    if max_temp > 100 or min_temp < 0:
        print(f"\n⚠️  {col}:")
        print(f"   Range: {min_temp:.1f}°C to {max_temp:.1f}°C (SUSPICIOUS!)")
        print(f"   Mean: {mean_temp:.1f}°C")
        
        # Count outliers
        outlier_high = (therm_csv[col] > 100).sum()
        outlier_low = (therm_csv[col] < 0).sum()
        if outlier_high > 0:
            print(f"   Outliers >100°C: {outlier_high}/{len(therm_csv)} samples")
        if outlier_low > 0:
            print(f"   Outliers <0°C: {outlier_low}/{len(therm_csv)} samples")
        
        outliers_found = True

if not outliers_found:
    print("\n✅ All thermistor temperatures in reasonable range (0-100°C)")

# Check for NaN values
nan_counts = therm_csv.isna().sum()
if nan_counts.sum() > 0:
    print(f"\n⚠️  NaN values found:")
    for col, count in nan_counts[nan_counts > 0].items():
        print(f"   {col}: {count} NaN values")
else:
    print("\n✅ No NaN values found")
