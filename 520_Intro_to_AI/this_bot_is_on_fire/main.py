import heapq
import numpy as np
import pandas as pd
import random

BLOCKED, OPEN, BOT, FIRE, BUTTON = 0, 1, 2, 3, 4

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

# Bot movement on grid is only performed via this function to ensure safety
def bot_move(grid, bot_pos, decided_pos):
    if decided_pos[0] - bot_pos[0] not in {-1, 0, 1} or decided_pos[1] - bot_pos[1] not in {-1, 0, 1}:
        raise Exception("Error: bot is moving illegally") # Making sure there are no illegal moves
    grid[bot_pos[0], bot_pos[1]] = OPEN
    bot_pos = decided_pos
    grid[bot_pos[0], bot_pos[1]] = BOT
    return grid, bot_pos

def dist(a, b): # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# All bot think functions use A* to ensure quick runtime. Implementation involves a min heap as fringe

# For first 3 bots, cost per cell is 1 with manhattan distance as heuristic,
# ensuring the shortest path is found given the constraints for each bot
def bot1_think(grid, bot_pos, button_pos):
    size = grid.shape[0]

    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == button_pos:
            return prev, 1
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            if (0 <= child[0] < size and 0 <= child[1] < size and
                grid[child[0], child[1]] != BLOCKED and grid[child[0], child[1]] != FIRE):
                cost = totalCosts[current] + 1
                estTotalCost = cost + dist(child, button_pos)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    return prev, 0

def bot2_think(grid, bot_pos, button_pos, fire_pos, fires_list):
    size = grid.shape[0]

    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == button_pos:
            return prev, 1
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            if (0 <= child[0] < size and 0 <= child[1] < size and
                grid[child[0], child[1]] != BLOCKED and grid[child[0], child[1]] != FIRE):
                cost = totalCosts[current] + 1
                estTotalCost = cost + dist(child, button_pos)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    return prev, 0

def bot3_think(grid, bot_pos, button_pos, fire_pos, fires_list):
    size = grid.shape[0]

    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == button_pos:
            return prev, 1
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            adjacent_cells = []
            if child[0] != 0:
                adjacent_cells.append((child[0] - 1, child[1]))
            if child[1] != 0:
                adjacent_cells.append((child[0], child[1] - 1))
            if child[0] != size - 1:
                adjacent_cells.append((child[0] + 1, child[1]))
            if child[1] != size - 1:
                adjacent_cells.append((child[0], child[1] + 1))
            if (0 <= child[0] < size and 0 <= child[1] < size and
                grid[child[0], child[1]] != BLOCKED and grid[child[0], child[1]] != FIRE and
                not any(grid[i][j] == FIRE for i, j in adjacent_cells)):
                cost = totalCosts[current] + 1
                estTotalCost = cost + dist(child, button_pos)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    return bot2_think(grid, bot_pos, button_pos, fire_pos, fires_list)

# Bot 4 takes inverse of distance to original fire as cost, with average cost to goal via manhattan distance as heuristic
# More details and reasons for choice discussed in the writeup
def bot4_think(grid, bot_pos, button_pos, fire_pos,  fires_list):
    size = grid.shape[0]

    # Calculating cost of each cell via distance from fire
    Cost = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            Cost[i, j] = 1 / max(0.5, dist((i, j), fire_pos))

    avg_cost = Cost.mean()

    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == button_pos:
            return prev, 1
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            if (0 <= child[0] < size and 0 <= child[1] < size and
                    grid[child[0], child[1]] != BLOCKED and grid[child[0], child[1]] != FIRE):
                cost = totalCosts[current] + Cost[child[0], child[1]]
                estTotalCost = cost + avg_cost * dist(child, button_pos)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    return prev, 0

# Another attempt for bot 4. It was much slower yet still performed worse than even bot 2.
def bot5_think(grid, bot_pos, button_pos, fire_pos, fires_list):
    size = grid.shape[0]

    # Calculating cost of each cell via distance from fire
    Cost = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            nearest_fire_dist = min([dist((i, j), fire) for fire in fires_list])
            Cost[i, j] = 1 / max(0.5, nearest_fire_dist)

    avg_cost = Cost.mean()

    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == button_pos:
            return prev, 1
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            if (0 <= child[0] < size and 0 <= child[1] < size and
                grid[child[0], child[1]] != BLOCKED and grid[child[0], child[1]] != FIRE):
                cost = totalCosts[current] + avg_cost * Cost[child[0], child[1]]
                estTotalCost = cost + dist(child, button_pos)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    return prev, 0

def spread_fire(grid, q, fires_list, random_no):
    new_grid = grid.copy() # Making sure fire spreads all at once
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


def run_grids(grid, q, bot_pos, button_pos, fire_pos, bot_think_functions):
    fires_list = [fire_pos]
    grids = [grid.copy() for _ in bot_think_functions]  # Separate grids for each bot
    bot_positions = [bot_pos] * len(bot_think_functions)
    final_paths = [[] for _ in bot_think_functions]
    grid_status = [True] * (len(bot_think_functions))  # Track if bots are still running
    results = [0] * len(bot_think_functions)  # Track if bot succeeded or failed
    t = [0] * len(bot_think_functions) # Track steps for each bot

    # For creating movie of fire spread
    fire_t = 1
    fire_movie = np.zeros((2000, grid.shape[0], grid.shape[1])) # assuming max 2000 steps
    fire_movie[0] = grid.copy()
    fire_movie[1] = grid.copy() # Bot moves before fire, so it gets 1 extra timestep in the beginning to move before the 1st fire spread
    fire_movie[1][bot_pos[0]][bot_pos[1]] = OPEN

    # Bot 1 only thinks on 1st step
    bot1_think = bot_think_functions[0]
    prev_bot1, path_found_bot1 = bot1_think(grids[0], bot_positions[0], button_pos)

    while True:
        random_no = random.random() # passing the same random number to have the same fire spread for all bots
        fire_t +=1
        for i, bot_think in enumerate(bot_think_functions):
            if not grid_status[i]:
                continue  # Skip failed or succeeded bots
            t[i] += 1

            if i == 0:
                prev, path_found = prev_bot1, path_found_bot1 # Bot 1 moves according to its first found path
            else:
                prev, path_found = bot_think(grids[i], bot_positions[i], button_pos, fire_pos, fires_list) # Re-planing path at every step

            if not path_found:
                print(f"Bot {i + 1} could not find path. Either the button or surrounding paths are on fire! Task failed.")
                grid_status[i] = False
                results[i] = 0
                continue

            # Backtrack and unroll the path using prev
            path = []
            loc = button_pos
            while loc != bot_positions[i]:
                path.append(loc)
                loc = prev[loc[0]][loc[1]]

            decided_pos = path.pop()
            final_paths[i].append(decided_pos)

            # Bot moves first
            if decided_pos == button_pos:
                print(f"Bot {i + 1} pressed the button! Task completed.")
                grid_status[i] = False  # Mark bot as finished
                results[i] = 1
                continue

            # Fire spreads after all bots have moved
            grids[i], _ = spread_fire(grids[i], q, fires_list, random_no)

            if grids[i][decided_pos[0], decided_pos[1]] == FIRE:
                print(f"Bot {i + 1} is burning! Task failed.")
                grid_status[i] = False
                results[i] = 0
                continue

            # Movement is reflected in grid at the end because we want to save the grid one step
            # before success or failure to see what direction the bot approached the button from.
            grids[i], bot_positions[i] = bot_move(grids[i], bot_positions[i], decided_pos)

        # Add fire spread to movie
        fire_movie[fire_t], fires_list = spread_fire(fire_movie[fire_t - 1], q, fires_list, random_no)

        if fire_movie[fire_t][button_pos[0]][button_pos[1]] == FIRE:
            print(f"Movie ends because button is on fire after this!")
            grid_status[-1] = False
            break

    return grids, t, final_paths, results, fire_movie[:fire_t+1], fire_t # Cut off movie when button catches on fire

# Winnable uses the 3d fire_movie to find any legal path to a button at any step
def winnable(fire_movie, bot_pos, button_pos):
    total_time = fire_movie.shape[0]
    size = fire_movie.shape[1]
    bot_pos = (0, bot_pos[0], bot_pos[1])
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[[(None, None, None) for i in range(size)] for j in range(size)] for t in range(total_time)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current[1:3] == button_pos:
            return prev, 1
        for i, j in directions:
            child = (current[0] + 1, current[1] + i, current[2] + j) # Moves are possible exactly one step in the future
            if (0 <= child[1] < size and 0 <= child[2] < size and child[0] < total_time and
                    fire_movie[child[0], child[1], child[2]] != BLOCKED and fire_movie[child[0], child[1], child[2]] != FIRE):
                # Cost through time is irrelevant as all steps through time must be exactly of size 1,
                # and we don't care about the time it takes to reach the button
                cost = totalCosts[current] + 1
                estTotalCost = cost + dist(child[1:3], button_pos)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]][child[2]] = current
    return prev, 0

results = pd.DataFrame()
bot_think_functions = [bot1_think, bot2_think, bot3_think, bot4_think]

bot1_successes = 0
bot2_successes = 0
bot3_successes = 0
bot4_successes = 0
one_bot_successes = 0
total_winnable = 0


for test in range(2000):

    print(test)
    ship = build_ship(40)
    ship, bot_pos, button_pos, fire_pos = place_bot_button_fire(ship)
    q = round(random.choice(np.linspace(0.05, 1, 20)), 2)

    print("Bot position: ", bot_pos, ", Button position: ", button_pos, ", Fire position: ", fire_pos)

    ships, ts, final_paths, results, fire_movie, fire_t = run_grids(ship, q, bot_pos, button_pos, fire_pos, bot_think_functions)
    one_bot_won = results[0] or results[1] or results[2] or results[3]

    winning_path, winnability_result = winnable(fire_movie, bot_pos, button_pos)

    print(results, winnability_result, ts, fire_t)

    bot1_successes += results[0]
    bot2_successes += results[1]
    bot3_successes += results[2]
    bot4_successes += results[3]
    one_bot_successes += one_bot_won
    total_winnable += winnability_result

    # Recording results

    np.save("ship_arrays/" + str(test) + " - t0", ship)
    np.save("ship_arrays/" + str(test) + " - t1 - " + str(ts[0]), ships[0])
    np.save("ship_arrays/" + str(test) + " - t2 - " + str(ts[1]), ships[1])
    np.save("ship_arrays/" + str(test) + " - t3 - " + str(ts[2]), ships[2])
    np.save("ship_arrays/" + str(test) + " - t4 - " + str(ts[3]), ships[3])
    # np.save("ship_arrays/" + str(test) + " - t5 - " + str(ts[4]), ships[4])

    new_row = (test, q, bot_pos, button_pos, fire_pos, winnability_result,
               results[0], results[1], results[2], results[3],
               ts[0], ts[1], ts[2], ts[3],
               final_paths[0], final_paths[1], final_paths[2], final_paths[3])
    new_row_df = pd.DataFrame([new_row])

    new_row_df.to_csv("results.csv", mode='a', header=False, index=False)

    # Sanity check on winnability
    if winnability_result < one_bot_won:
        raise Exception("Error in winnability code")



print(bot1_successes, bot2_successes, bot3_successes, bot4_successes)
print(one_bot_successes, total_winnable)
