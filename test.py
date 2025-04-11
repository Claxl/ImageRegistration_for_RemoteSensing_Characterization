import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# -------------------------------
# Presentation Style Settings
# -------------------------------
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# -------------------------------
# Data Preparation
# -------------------------------
methods   = ['SIFT', 'AKAZE', 'ORB', 'BRISK', 'RIFT', 'LGHD', 'MINIMA_sp_lg', 'MINIMA_loftr']
rmse      = [141.094317, np.nan, np.nan, np.nan, 29.478175, 128.852933, 35.955993, 35.308154]
time_sec  = [3.2955, 0.9142, 0.2389, 5.9314, 726.3694, 263.5279, 40.8981, 67.3505]
power_w   = [3.4089, 3.5174, 3.3483, 3.3395, 3.4308, 3.4487, 4.0778, 4.1416]

df = pd.DataFrame({
    'Method': methods,
    'RMSE': rmse,
    'Time (s)': time_sec,
    'Power (W)': power_w
})

# A method is "Success" if RMSE exists; otherwise, it's "Failed"
df['Test_Status'] = np.where(df['RMSE'].isna(), 'Failed', 'Success')
df_success = df[df['Test_Status'] == 'Success']
df_failed  = df[df['Test_Status'] == 'Failed']

# -------------------------------
# Marker Assignment
# -------------------------------
# Each method is assigned a unique marker.
markers_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
method_to_marker = dict(zip(methods, markers_list))

# -------------------------------
# Color Mapping for Successful Methods
# -------------------------------
# Successful methods are colored based on power consumption.
norm = plt.Normalize(vmin=df['Power (W)'].min(), vmax=df['Power (W)'].max())
cmap = plt.cm.cividis  # colorblind-friendly

# -------------------------------
# Figure and Axes Setup
# -------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

# -------------------------------
# Plot Successful Methods (without individual annotations)
# -------------------------------
for _, row in df_success.iterrows():
    method = row['Method']
    marker = method_to_marker[method]
    color = cmap(norm(row['Power (W)']))
    ax.scatter(
        row['Time (s)'], row['RMSE'],
        s=500,
        color=color,
        marker=marker,
        edgecolor='black',
        linewidth=1.5,
        zorder=10
    )

# -------------------------------
# Plot Failed Methods (fixed at RMSE = -50, with red "X" markers)
# -------------------------------
failed_y = -50  # Fixed y-value for failed methods.
for _, row in df_failed.iterrows():
    ax.scatter(
        row['Time (s)'], failed_y,
        s=500,
        color='red',
        marker='X',
        edgecolor='black',
        linewidth=1.5,
        zorder=10
    )

# -------------------------------
# RMSE Axis Grid Settings
# -------------------------------
ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
ax.minorticks_on()
ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5, color='gray')
ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='lightgray')

# -------------------------------
# Axes Labels and Limits
# -------------------------------
ax.set_xscale('log')
ax.set_xlabel('Execution Time (s, Log Scale, Lower is Better)')
ax.set_ylabel('RMSE (Lower is Better)')
ax.set_title('CrossSeason Comparison', fontsize=18)
max_rmse = df_success['RMSE'].max() if not df_success.empty else 150
ax.set_ylim(-75, max_rmse * 1.1)

# -------------------------------
# Overall Legend for Methods (placed in the upper left)
# -------------------------------
handles = []
for method in methods:
    if method in df_success['Method'].values:
        row = df_success[df_success['Method'] == method].iloc[0]
        color = cmap(norm(row['Power (W)']))
        marker = method_to_marker[method]
    else:
        color = 'red'
        marker = 'X'
    handle = Line2D([0], [0],
                    marker=marker,
                    color='w',
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=12,
                    label=method)
    handles.append(handle)

legend1 = ax.legend(
    handles=handles,
    title="Methods",
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98)
)
ax.add_artist(legend1)

# -------------------------------
# Additional Legend Entry: Red indicates Failed (placed below the overall legend in the upper left)
# -------------------------------
failed_handle = Line2D([0], [0],
                       marker='X',
                       color='w',
                       markerfacecolor='red',
                       markeredgecolor='black',
                       markersize=12,
                       label='Red indicates Failed')
legend2 = ax.legend(
    handles=[failed_handle],
    loc='upper left',
    bbox_to_anchor=(0.02, 0.55)
)
ax.add_artist(legend2)

# -------------------------------
# Colorbar for Power Consumption
# -------------------------------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Power Consumption (W)', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
