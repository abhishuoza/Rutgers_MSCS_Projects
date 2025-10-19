import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import Pi0Predictor
from localization_functions import encode_input

# Constants
BLOCKED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "models/pi0_predictor.pt"
MODEL_PATH = "models/pi1_predictor.pt"
SHIP_PATH = "ship.npy"
# TEST_CSV = "train_test/test_pi_0.csv"
TEST_CSV = "train_test/test_pi_1.csv"
# OUTPUT_PLOT = "plots/pi0_predictions_by_L_on_test_pi_0_csv.png"
OUTPUT_PLOT = "plots/pi1_predictions_by_L_on_test_pi_1_csv.png"
BIN_SIZE = 10

# Load ship map and model
ship = np.load(SHIP_PATH)
model = Pi0Predictor(input_size=ship.shape[0])
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load test CSV
df = pd.read_csv(TEST_CSV)

# Predict using the model
predictions = []
for _, row in df.iterrows():
    # L = eval(row["L_i"])
    L = eval(row["L_0"])
    input_tensor = encode_input(ship, L, DEVICE)
    with torch.no_grad():
        output = model(input_tensor).item()
    predictions.append(output)

print("Predictions done")

df["predicted_moves"] = predictions

# Bin the |L| values
# max_len = df["len(L_i)"].max()
max_len = df["len(L_0)"].max()
bins = np.arange(0, max_len + BIN_SIZE, BIN_SIZE)
labels = [f"{bins[i]}–{bins[i+1]-1}" for i in range(len(bins)-1)]
# df["L_bin"] = pd.cut(df["len(L_i)"], bins=bins, labels=labels, right=False)
df["L_bin"] = pd.cut(df["len(L_0)"], bins=bins, labels=labels, right=False)

# Group by bin and average predicted values
grouped = df.groupby("L_bin")["predicted_moves"].mean().reset_index()
grouped.columns = ["L_bin", "AveragePredictedMoves"]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(grouped["L_bin"], grouped["AveragePredictedMoves"], marker='o')
plt.xticks(rotation=45)
plt.xlabel("|L| (Binned)")
plt.ylabel("Average Predicted Number of Moves")
plt.title("Predicted Moves vs. Size of L (Binned by 10)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
# plt.show()  # Optional
