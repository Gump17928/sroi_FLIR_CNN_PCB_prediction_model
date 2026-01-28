import pandas as pd

df = pd.read_csv('outputs/0107_1439_P1-7/calibration_database/thermal_calibration_points.csv')

print('=== PCB Names in Database ===')
print(df['pcb'].unique())

print(f'\n=== Row Counts ===')
print(f'Total rows: {len(df)}')

for pcb in df['pcb'].unique():
    count = len(df[df['pcb'] == pcb])
    print(f'  {pcb}: {count} rows')

print('\n=== Sample from each PCB ===')
for pcb in df['pcb'].unique():
    pcb_data = df[df['pcb'] == pcb].head(2)
    print(f'\n{pcb}:')
    print(pcb_data[['component_name', 'component_type', 'test_session']].to_string())
