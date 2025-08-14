import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# === CONFIGURATION ===
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
csv_file = script_dir.parent / "big_script" / "checkpoints" / "multi_parameter_benchmark_results.csv"
system_col = 'system_type'
gf_exp_col = 'gf_exp'
x_col = 'vec_len'
y_col = 'avg_execution_time_ms'

# === FILTERPARAMETER ===
systems_to_show = ['RSID', 'SHA1ID']
required_num_tags = 1
required_test_type = 'execution_time'

# === LOAD DATA ===
try:
    df = pd.read_csv(csv_file)
    print("CSV loaded successfully.")
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit()

# === FILTERING ===
df = df[
    (df[system_col].isin(systems_to_show)) &
    (df['num_tags'] == required_num_tags) &
    (df['test_type'] == required_test_type)
]

# === CHECK EMPTY DATA ===
if df.empty:
    print("Keine Daten nach Filterung übrig!")
    exit()

# === UNIQUE VALUES & MAPPINGS ===
unique_systems = sorted(df[system_col].unique())
unique_gf_exps = sorted(df[gf_exp_col].unique())

color_cycle = plt.cm.tab10.colors
marker_cycle = ['o', 's', 'D', '^', 'v', '*', 'x', 'P', 'H', '+']

system_to_color = {system: color_cycle[i % len(color_cycle)] for i, system in enumerate(unique_systems)}
gf_exp_to_marker = {gf: marker_cycle[i % len(marker_cycle)] for i, gf in enumerate(unique_gf_exps)}

# === PLOT SETUP ===
plt.figure(figsize=(10, 7))

# Gesamtdaten zur Regression vorbereiten
all_x = []
all_y = []

for (category, group), group_df in df.groupby([system_col, gf_exp_col]):
    # Mittelwert für gleiche vec_len-Werte berechnen
    mean_df = group_df.groupby(x_col, as_index=False)[y_col].mean()
    
    # Speichern für Regression
    all_x.extend(mean_df[x_col])
    all_y.extend(mean_df[y_col])

    # Plotte die Punkte
    plt.loglog(
        mean_df[x_col],
        mean_df[y_col],
        marker=gf_exp_to_marker[group],
        linestyle='-',
        label=f'{category} - GF_exp: {group}',
        color=system_to_color[category]
    )

# === QUADRATISCHE REGRESSION IM LOG-LOG-RAUM ===
log_x = np.log10(all_x)
log_y = np.log10(all_y)

# Fit quadratische Funktion: log_y = a*log_x^2 + b*log_x + c
coeffs = np.polyfit(log_x, log_y, deg=2)
a, b, c = coeffs

# Erzeuge glatte x-Werte für die Linie
x_vals = np.logspace(np.log10(min(all_x)), np.log10(max(all_x)), 200)
log_x_vals = np.log10(x_vals)
log_y_vals = a * log_x_vals**2 + b * log_x_vals + c
y_vals = 10 ** log_y_vals

# === REGRESSIONSLINIE PLOTTEN ===
plt.loglog(
    x_vals,
    y_vals,
    'r--',
    linewidth=2,
    label=f'quadratic regression\n (a={a:.3g}, b={b:.3g}, c={c:.3g})'
)

# === FINAL TOUCHES ===
plt.xlabel("Vector Length (vec_len)")
plt.ylabel("Average Execution Time (ms)")
plt.title(f'Execution Time of RSID and SHA1ID with quadratic regression', fontsize=14, fontweight='bold')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig('execution_time_quadratic_regression.png', dpi=300, bbox_inches='tight')
plt.show()