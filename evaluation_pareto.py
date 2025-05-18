import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("full_pareto_front.csv")

precision = -df['precision'].to_numpy()  # Negate for minimization
recall = -df['recall'].to_numpy()        # Negate for minimization
bias = df['bias'].to_numpy()             # Already in minimization direction

objectives = np.stack([precision, recall, bias], axis=1)

ref_point = np.max(objectives, axis=0) + 0.1  # Because we minimize now

hv = HV(ref_point=ref_point)
hypervolume = hv(objectives)

print(f"3D Hypervolume (↑ Precision, ↑ Recall, ↓ Bias): {hypervolume}")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(-precision, -recall, bias, c='blue', label="Pareto Points")  # Undo negation for display

ax.set_xlabel("Precision")
ax.set_ylabel("Recall")
ax.set_zlabel("Bias")
ax.set_title("3D Pareto Front (↑ Precision, ↑ Recall, ↓ Bias)")
plt.legend()
plt.show()