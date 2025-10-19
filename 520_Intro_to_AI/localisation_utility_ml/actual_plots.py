import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
df = pd.read_csv("train_test/test_pi_0.csv")
# df = pd.read_csv("train_test/test_pi_1.csv")

# Bin the |L| values
bin_size = 10
max_len = df["len(L_i)"].max()
# max_len = df["len(L_0)"].max()
bins = np.arange(0, max_len + bin_size, bin_size)
labels = [(f"{bins[i]}–{bins[i+1]-1}") for i in range(len(bins)-1)]
df["L_bin"] = pd.cut(df["len(L_i)"], bins=bins, labels=labels, right=False)
# df["L_bin"] = pd.cut(df["len(L_0)"], bins=bins, labels=labels, right=False)

# Group by bin and compute average of (t_n - t_i) or (t_n)
grouped = df.groupby("L_bin")["t_n - t_i"].mean().reset_index()
# grouped = df.groupby("L_bin")["t_n"].mean().reset_index()
grouped.columns = ["L_bin", "AverageNumberOfMoves"]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(grouped["L_bin"], grouped["AverageNumberOfMoves"], marker='o')
plt.xticks(rotation=45)
plt.xlabel("|L| (Binned)")
plt.ylabel("Average Number of Moves Needed")
plt.title("Average Moves vs. Size of L (Binned by 10)")
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("plots/actual_moves_on_test_pi_0_csv.png")
# plt.savefig("plots/actual_moves_on_test_pi_1_csv.png")



