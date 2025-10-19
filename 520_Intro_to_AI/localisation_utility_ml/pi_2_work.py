import torch
import numpy as np
from models import Pi1Predictor
import csv
import random
import matplotlib.pyplot as plt
from localization_functions import (
    build_ship, place_bot, BLOCKED, OPEN, BOT,
    random_L0_generator, cmap, norm,
    pi_0, pi_1, pi_2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Pi1Predictor(30)
model.load_state_dict(torch.load("models/pi1_predictor.pt", map_location=device))
model.to(device)
model.eval()

ship = build_ship(30)
bot_pos, ship = place_bot(ship)

# Optional plot
plt.imshow(ship, cmap=cmap, norm=norm)
plt.title("Ship Configuration for pi_2")
plt.show()

# Optional initial state logging header
# with open("datasets/localisation_dataset_pi_2.csv", mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["L_0", "len(L_0)", "t_n"])

for test in range(50000):
    print(f"\nTest {test}")
    ship1 = ship.copy()
    bot_pos1 = bot_pos
    L0, _ = random_L0_generator(ship1, bot_pos1)

    bot_pos1, move_sequence = pi_2(ship1, bot_pos1, L0, model, device=device)
    t_n = len(move_sequence)

    # Optional initial state logging
    # with open("datasets/localisation_dataset_pi_2.csv", mode="a", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow([sorted(L0), len(L0), t_n])

    print(f"Moves: {t_n}, Final bot_pos: {bot_pos1}")
