import csv
import random
import matplotlib.pyplot as plt
from localization_functions import (
    build_ship, place_bot,
    cmap, norm, random_L0_generator,
    pi_0
)

size = 30
ship = build_ship(size)
bot_pos, ship = place_bot(ship)

# Optional plot
plt.imshow(ship, cmap=cmap, norm=norm)
plt.title("Ship Configuration")
plt.show()

# Optional logging header
with open("datasets/localisation_dataset_pi_0.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["L_i", "len(L_i)", "t_n - t_i"])

for test in range(1000):
    print(f"\nTest {test}")
    ship1 = ship.copy()
    bot_pos1 = bot_pos
    L0, _ = random_L0_generator(ship1, bot_pos1)
    L_data_log = [] # Optional logging
    bot_pos1, move_sequence = pi_0(ship1, bot_pos1, L0, L_data_log)
    t_n = len(move_sequence)
    print(L_data_log)

    # Optional logging
    with open("datasets/localisation_dataset_pi_0.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        for t_i, L_i in enumerate(L_data_log):
            row = [sorted(L_i), len(L_i), t_n - t_i]
            writer.writerow(row)

    print(f"Total moves: {t_n}, Final bot_pos: {bot_pos1}")