import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from models import Pi0Predictor, Pi1Predictor

BLOCKED, OPEN, BOT = 0, 1, 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load ship
ship = np.load("ship.npy")
print("Ship size:", ship.shape)

# Load CSV and split
# df = pd.read_csv("datasets/localisation_dataset.csv")  # pi_0 data
df = pd.read_csv("datasets/localisation_dataset_pi_1.csv") # pi_1 data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# train_df.to_csv("train_test/train_pi_0.csv", index=False)  # pi_0 data
# test_df.to_csv("train_test/test_pi_0.csv", index=False)  # pi_0 data
train_df.to_csv("train_test/train_pi_1.csv", index=False) # pi_1 data
test_df.to_csv("train_test/test_pi_1.csv", index=False) # pi_1 data
print("CSV split saved")

# Lazy loading dataset using mmap
class LazyLocalizationDataset(Dataset):
    def __init__(self, csv_path, ship):
        self.df = pd.read_csv(csv_path)
        self.ship = np.where(ship == BLOCKED, 0, 1).astype(np.float32)
        self.size = ship.shape[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # L = set(eval(row["L_i"])) # pi_0 data
        L = set(eval(row["L_0"])) # pi_1 data
        L_layer = np.zeros((self.size, self.size), dtype=np.float32)
        for i, j in L:
            if 0 <= i < self.size and 0 <= j < self.size:
                L_layer[i, j] = 1
        input_tensor = np.stack([self.ship, L_layer], axis=0)
        # label = np.float32(row["t_n - t_i"]) # pi_0 data
        label = np.float32(row["t_n"]) # pi_1 data
        return torch.tensor(input_tensor), torch.tensor([label])

# train_loader = DataLoader(LazyLocalizationDataset("train_test/train_pi_0.csv", ship), batch_size=32, shuffle=True) # pi_0 data
# test_loader = DataLoader(LazyLocalizationDataset("train_test/test_pi_0.csv", ship), batch_size=32) # pi_0 data
train_loader = DataLoader(LazyLocalizationDataset("train_test/train_pi_1.csv", ship), batch_size=32, shuffle=True) # pi_1 data
test_loader = DataLoader(LazyLocalizationDataset("train_test/test_pi_1.csv", ship), batch_size=32) # pi_1 data


print("DataLoaders initialized")

# Initialize model
# model = Pi0Predictor(ship.shape[0]).to(device) # pi_0 model
model = Pi1Predictor(ship.shape[0]).to(device) # pi_1 model
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model initialized")

# Training loop
train_losses, test_losses = [], []
for epoch in range(20):
    model.train()
    train_loss = 0
    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        # print(f"Epoch {epoch+1} Batch {batch_idx+1}: xb shape = {xb.shape}, yb shape = {yb.shape}")
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader.dataset))

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            test_loss += loss_fn(model(xb), yb).item()
    test_losses.append(test_loss / len(test_loader.dataset))

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_losses[-1]:.4f}")

# Save model
os.makedirs("models", exist_ok=True)

# torch.save(model.state_dict(), "models/pi0_predictor.pt") # pi_0 model
# print("Model saved to models/pi0_predictor.pt") # pi_0 model
torch.save(model.state_dict(), "models/pi1_predictor.pt") # pi_1 model
print("Model saved to models/pi1_predictor.pt") # pi_1 model


# Plot
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
# plt.show()
# plt.savefig("plots/pi0_predictor.png") # pi_0 model
plt.savefig("plots/pi1_predictor.png") # pi_1 model
