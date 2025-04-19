import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

# === Load the CSV ===
file_path = 'pareto_points.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# === Ensure there are at least 4 numerical columns ===
if df.shape[1] < 4:
    raise ValueError("CSV must contain at least 4 numerical columns.")

# === Select the first 4 numerical columns ===
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4]
x_col, y_col, z_col, color_col = numerical_cols

# === Extract values ===
x = df[x_col]
y = df[y_col]
z = df[z_col]
c = df[color_col]

# === Normalize the color values ===
norm = colors.Normalize(vmin=c.min(), vmax=c.max())
colormap = cm.viridis  # You can use 'plasma', 'inferno', 'coolwarm', etc.
color_values = colormap(norm(c))

# === Create 3D scatter plot ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=color_values, s=50, edgecolors='k')

# === Add labels and title ===
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_zlabel(z_col)
ax.set_title(f"4D Scatter Plot ({x_col}, {y_col}, {z_col}, color: {color_col})")

# === Add color bar ===
mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
mappable.set_array(c)
cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
cbar.set_label(color_col)

plt.tight_layout()
plt.show()