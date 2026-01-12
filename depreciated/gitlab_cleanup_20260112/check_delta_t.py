import pandas as pd

df = pd.read_csv('outputs/0107_1439_P1-7/calibration_database/thermal_calibration_points.csv')

print('=== HBridge Data ===')
hb = df[df['pcb'].str.contains('HBridge', na=False)].head(5)
print(hb[['component_name', 'component_type', 'air_ss', 'air_initial', 'sand_ss', 'sand_initial']].to_string())

print('\n=== Delta T Calculations ===')
for idx, row in hb.iterrows():
    air_delta = row['air_ss'] - row['air_initial']
    sand_delta = row['sand_ss'] - row['sand_initial']
    print(f"  {row['component_name']:6s}: air_ΔT = {air_delta:6.2f}°C, sand_ΔT = {sand_delta:6.2f}°C")

print('\n=== Load_Shedding Data ===')
ls = df[df['pcb'].str.contains('Load_Shedding', na=False)].head(5)
print(ls[['component_name', 'component_type', 'air_ss', 'air_initial', 'sand_ss', 'sand_initial']].to_string())

print('\n=== Delta T Calculations ===')
for idx, row in ls.iterrows():
    air_delta = row['air_ss'] - row['air_initial']
    sand_delta = row['sand_ss'] - row['sand_initial']
    print(f"  {row['component_name']:6s}: air_ΔT = {air_delta:6.2f}°C, sand_ΔT = {sand_delta:6.2f}°C")

print('\n=== Comparison ===')
print(f"HBridge mean sand ΔT: {(df[df['pcb'].str.contains('HBridge', na=False)]['sand_ss'] - df[df['pcb'].str.contains('HBridge', na=False)]['sand_initial']).mean():.2f}°C")
print(f"Load_Shedding mean sand ΔT: {(df[df['pcb'].str.contains('Load_Shedding', na=False)]['sand_ss'] - df[df['pcb'].str.contains('Load_Shedding', na=False)]['sand_initial']).mean():.2f}°C")
