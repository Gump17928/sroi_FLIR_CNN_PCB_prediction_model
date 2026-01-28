import pandas as pd

df = pd.read_csv('outputs/0107_1439_P1-7/calibration_database/thermal_calibration_points.csv')

print('=== H_Bridge_Sensing Data ===')
hb = df[df['pcb'] == 'H_Bridge_Sensing_Rev_C.brd'].head(5)
print(hb[['component_name', 'component_type', 'air_ss', 'air_initial', 'sand_ss', 'sand_initial']].to_string())

print('\n=== Delta T Calculations (H_Bridge_Sensing) ===')
for idx, row in hb.iterrows():
    air_delta = row['air_ss'] - row['air_initial']
    sand_delta = row['sand_ss'] - row['sand_initial']
    print(f"  {row['component_name']:6s}: air_ΔT = {air_delta:6.2f}°C, sand_ΔT = {sand_delta:6.2f}°C")

print('\n=== Load_Shedding Data ===')
ls = df[df['pcb'] == 'Load_Shedding_Rev_E.brd'].head(5)
print(ls[['component_name', 'component_type', 'air_ss', 'air_initial', 'sand_ss', 'sand_initial']].to_string())

print('\n=== Delta T Calculations (Load_Shedding) ===')
for idx, row in ls.iterrows():
    air_delta = row['air_ss'] - row['air_initial']
    sand_delta = row['sand_ss'] - row['sand_initial']
    print(f"  {row['component_name']:6s}: air_ΔT = {air_delta:6.2f}°C, sand_ΔT = {sand_delta:6.2f}°C")

print('\n=== Comparison ===')
hb_all = df[df['pcb'] == 'H_Bridge_Sensing_Rev_C.brd']
ls_all = df[df['pcb'] == 'Load_Shedding_Rev_E.brd']

hb_sand_delta = hb_all['sand_ss'] - hb_all['sand_initial']
ls_sand_delta = ls_all['sand_ss'] - ls_all['sand_initial']

print(f"H_Bridge mean sand ΔT: {hb_sand_delta.mean():.2f}°C (n={len(hb_sand_delta)})")
print(f"Load_Shedding mean sand ΔT: {ls_sand_delta.mean():.2f}°C (n={len(ls_sand_delta)})")

# Check for any NaN or zero values
print(f"\nH_Bridge NaN count: {hb_sand_delta.isna().sum()}")
print(f"Load_Shedding NaN count: {ls_sand_delta.isna().sum()}")
