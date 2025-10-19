import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
import random

BLOCKED, OPEN, BOT, FIRE, BUTTON = 0, 1, 2, 3, 4

# For coloring grid plot
cmap = ListedColormap(["black", "grey", "blue", "orange", "green"])
bounds = [BLOCKED - 0.5, OPEN - 0.5, BOT - 0.5, FIRE - 0.5, BUTTON - 0.5, BUTTON + 0.5]
norm = BoundaryNorm(bounds, cmap.N)

def build_ship(size):
    # Create square grid with all blocked cells
    grid = np.zeros((size, size))
    # Open random cell
    randx = np.random.randint(0, size)
    randy = np.random.randint(0, size)
    grid[randx, randy] = OPEN

    # Iteratively open blocked cells
    while True:
        candidates = []
        for i in range(size):
            for j in range(size):
                counter = 0
                if grid[i, j] == BLOCKED:
                    if i != 0 and grid[i-1, j] == OPEN:
                        counter += 1
                    if j != 0 and grid[i, j-1] == OPEN:
                        counter += 1
                    if i != size-1 and grid[i+1, j] == OPEN:
                        counter += 1
                    if j != size-1 and grid[i, j+1] == OPEN:
                        counter += 1
                if counter == 1:
                    candidates.append((i, j))
        if len(candidates) == 0:
            break # End while loop if no candidates left
        cell_to_open = random.choice(candidates)
        grid[cell_to_open[0], cell_to_open[1]] = OPEN

    # Identifying dead ends
    dead_ends = []
    for i in range(size):
        for j in range(size):
            counter = 0
            if grid[i, j] == OPEN:
                if i != 0 and grid[i - 1, j] == OPEN:
                    counter += 1
                if j != 0 and grid[i, j - 1] == OPEN:
                    counter += 1
                if i != size - 1 and grid[i + 1, j] == OPEN:
                    counter += 1
                if j != size - 1 and grid[i, j + 1] == OPEN:
                    counter += 1
            if counter == 1:
                dead_ends.append((i, j))

    # Randomly picking one blocked neighbor
    ends_to_open = []
    for cell in dead_ends:
        if random.random() < 0.5: # For approximately half dead ends
            i = cell[0]
            j = cell[1]
            neighbors = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]
            neighbors_copy = neighbors.copy()
            # remove illegal neighbors to create random pick list
            for neighbor in neighbors_copy:
                if (neighbor[0] == -1 or neighbor[1] == -1 or
                        neighbor[0] == size or neighbor[1] == size or
                        grid[neighbor[0], neighbor[1]] == OPEN):
                    neighbors.remove(neighbor)
            ends_to_open.append(random.choice(neighbors))

    # Opening neighbors only after picking neighbors for all dead ends
    for i, j in ends_to_open:
        grid[i, j] = OPEN

    return grid

def place_bot_button_fire(grid):
    size = grid.shape[0]
    open_cells = []

    for i in range(size):
        for j in range(size):
            if grid[i, j] == OPEN:
                open_cells.append((i, j))
    bot_pos, button_pos, fire_pos = random.sample(open_cells, 3)
    grid[bot_pos[0], bot_pos[1]] = BOT
    grid[button_pos[0], button_pos[1]] = BUTTON
    grid[fire_pos[0], fire_pos[1]] = FIRE

    return grid, bot_pos, button_pos, fire_pos

def spread_fire(grid, q, fires_list, random_no):
    new_grid = grid.copy()
    size = grid.shape[0]
    new_fires_list = fires_list.copy()
    for i in range(size):
        for j in range(size):
            if grid[i, j] != BLOCKED:
                K = 0
                if i != 0 and grid[i - 1, j] == FIRE:
                    K += 1
                if j != 0 and grid[i, j - 1] == FIRE:
                    K += 1
                if i != size - 1 and grid[i + 1, j] == FIRE:
                    K += 1
                if j != size - 1 and grid[i, j + 1] == FIRE:
                    K += 1
                if random_no < (1 - (1 - q)**K) :
                    new_grid[i, j] = FIRE
                    new_fires_list.append((i, j))
    return new_grid, new_fires_list

def run_grids(grid, q, bot_pos, button_pos, fire_pos):
    fires_list = [fire_pos]

    # For creating movie of fire spread
    fire_t = 1
    fire_movie = np.zeros((2000, grid.shape[0], grid.shape[1]))  # assuming max 2000 steps
    fire_movie[0] = grid.copy()
    fire_movie[1] = grid.copy()  # Bot moves before fire, so it gets 1 extra timestep in the beginning to move before the 1st fire spread
    fire_movie[1][bot_pos[0]][bot_pos[1]] = OPEN


    while True:
        random_no = random.random()
        fire_t +=1
        fire_movie[fire_t], fires_list = spread_fire(fire_movie[fire_t - 1], q, fires_list, random_no)
        print(fire_movie[fire_t][button_pos[0]][button_pos[1]])
        if fire_movie[fire_t][button_pos[0]][button_pos[1]] == FIRE:
            print(f"Movie ends because button is on fire after this!")
            break
    return fire_movie[:fire_t+1], fire_t

def animate_fire_spread(fire_movie):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(fire_movie[0], cmap=cmap, norm=norm)
    def update(frame):
        im.set_array(fire_movie[frame])
        ax.set_title(f"Time Step: {frame}")
        return im,
    ani = animation.FuncAnimation(fig, update, frames=len(fire_movie), interval=200, repeat=True)
    ani.save("ship_plots/fire_spread.gif", writer="pillow", fps=5)

# I was unable to store fire spread for all ships as it was taking up too much space
# We won't get the exact same fire spread, but still useful to visualise I suppose
# Perhaps if I saved all the random_no values in a list...

# Enter ALL details from results.csv and ship_arrays
ship = np.load('ship_arrays/1790 - t0.npy')
bot_pos, button_pos, fire_pos = (24, 0), (17, 38), (27, 0)

q = 0.9

print("Bot position: ", bot_pos, ", Button position: ", button_pos, ", Fire position: ", fire_pos)

fire_movie, fire_t = run_grids(ship, q, bot_pos, button_pos, fire_pos)
print(fire_t)
# Animate fire spread, saves as fire_spread.gif
animate_fire_spread(fire_movie)

