import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
csv_path = "/Users/lewisarnold1/Desktop/eye_tracking_data.csv"
output_image_path = "/Users/lewisarnold1/Desktop/Eye_Tracking_Heatmap.png"

# Load the CSV file
df = pd.read_csv(csv_path)

# Extract truth and predicted coordinates
truth_x = df["Truth_X"]
truth_y = df["Truth_Y"]
pred_x = df["Predicted_X"]
pred_y = df["Predicted_Y"]

# Fixed axes limits
x_min, x_max = 0, 720
y_min, y_max = 0, 355

# Define bins for heatmap resolution
x_bins = np.linspace(x_min, x_max, 20)  # 50 bins along x-axis
y_bins = np.linspace(y_min, y_max, 20)  # 50 bins along y-axis

# Compute heatmaps as 2D histograms
heatmap_truth, _, _ = np.histogram2d(truth_y, truth_x, bins=[y_bins, x_bins])  # Swap X-Y to match plotting
heatmap_pred, _, _ = np.histogram2d(pred_y, pred_x, bins=[y_bins, x_bins])  # Swap X-Y to match plotting

# Set global font size
plt.rcParams.update({'font.size': 14})

# Plot heatmaps
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap for truth values
sns.heatmap(heatmap_truth, cmap="Blues", ax=ax[0], cbar=True, xticklabels=True, yticklabels=True)
ax[0].set_title("Ground Truth Heatmap")
ax[0].set_xlabel("X Position")
ax[0].set_ylabel("Y Position")
ax[0].set_xticks(np.linspace(0, 20, num=7))  # 7 tick marks for readability
ax[0].set_yticks(np.linspace(0, 20, num=7))
ax[0].set_xticklabels(np.round(np.linspace(x_min, x_max, num=7)).astype(int))
ax[0].set_yticklabels(np.round(np.linspace(y_min, y_max, num=7)).astype(int))

# Heatmap for predicted values
sns.heatmap(heatmap_pred, cmap="Reds", ax=ax[1], cbar=True, xticklabels=True, yticklabels=True)
ax[1].set_title("Predicted Values Heatmap")
ax[1].set_xlabel("X Position")
ax[1].set_ylabel("Y Position")
ax[1].set_xticks(np.linspace(0, 20, num=7))
ax[1].set_yticks(np.linspace(0, 20, num=7))
ax[1].set_xticklabels(np.round(np.linspace(x_min, x_max, num=7)).astype(int))
ax[1].set_yticklabels(np.round(np.linspace(y_min, y_max, num=7)).astype(int))

# Save the heatmap
plt.tight_layout()
plt.savefig(output_image_path, dpi=300)
plt.show()

print(f"Heatmap saved at: {output_image_path}")
