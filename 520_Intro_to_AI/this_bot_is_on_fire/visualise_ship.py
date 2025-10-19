import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import ast

BLOCKED, OPEN, BOT, FIRE, BUTTON = 0, 1, 2, 3, 4

# For coloring grid plot
cmap = ListedColormap(["black", "grey", "blue", "orange", "green"])
bounds = [BLOCKED - 0.5, OPEN - 0.5, BOT - 0.5, FIRE - 0.5, BUTTON - 0.5, BUTTON + 0.5]
norm = BoundaryNorm(bounds, cmap.N)

results = pd.read_csv('results.csv')

# Enter test number to find ship and penultimate states for each bot
test = 1946
row = results.loc[test]

t1, t2, t3, t4 = int(row[" t1"]), int(row[" t2"]), int(row[" t3"]), int(row[" t4"])

print(t1, t2, t3, t4)

path = "ship_arrays/"

ship = np.load(path + str(test) + " - t0.npy")
ship1 = np.load(path+ str(test) + " - t1 - " + str(t1) + ".npy")
ship2 = np.load(path+ str(test) + " - t2 - " + str(t2) + ".npy")
ship3 = np.load(path+ str(test) + " - t3 - " + str(t3) + ".npy")
ship4 = np.load(path+ str(test) + " - t4 - " + str(t4) + ".npy")

# Comment to print without path
final_path1 = ast.literal_eval(row[" final_path1"])
for i, j in final_path1:
    ship1[i, j] = BOT
final_path2 = ast.literal_eval(row[" final_path2"])
for i, j in final_path2:
    ship2[i, j] = BOT
final_path3 = ast.literal_eval(row[" final_path3"])
for i, j in final_path3:
    ship3[i, j] = BOT
final_path4 = ast.literal_eval(row[" final_path4"])
for i, j in final_path4:
    ship4[i, j] = BOT

plt.imshow(ship, cmap=cmap, norm=norm)
plt.title("Ship config")
plt.savefig("ship_plots/" + str(test) + " - t0")
plt.imshow(ship1, cmap=cmap, norm=norm)
plt.title("Bot 1 path")
plt.savefig("ship_plots/" + str(test) + " - t1 - " + str(t1) + ".png")
plt.imshow(ship2, cmap=cmap, norm=norm)
plt.title("Bot 2 path")
plt.savefig("ship_plots/" + str(test) + " - t2 - " + str(t2) + ".png")
plt.imshow(ship3, cmap=cmap, norm=norm)
plt.title("Bot 3 path")
plt.savefig("ship_plots/" + str(test) + " - t3 - " + str(t3) + ".png")
plt.imshow(ship4, cmap=cmap, norm=norm)
plt.title("Bot 4 path")
plt.savefig("ship_plots/" + str(test) + " - t4 - " + str(t4) + ".png")

print("Done!")

