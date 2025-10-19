import heapq
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import ast

BLOCKED, OPEN, BOT, RAT, BOT_RAT = 0, 1, 2, 3, 4

# For coloring grid plot
cmap = ListedColormap(["black", "grey", "blue", "green", "cyan"])
bounds = [BLOCKED - 0.5, OPEN - 0.5, BOT - 0.5, RAT - 0.5, BOT_RAT - 0.5, BOT_RAT + 0.5]
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

def place_bot_rat(grid):
    size = grid.shape[0]
    open_cells = []
    for i in range(size):
        for j in range(size):
            if grid[i, j] == OPEN:
                open_cells.append((i, j))
    bot_pos, rat_pos = random.sample(open_cells, 2)
    grid[bot_pos[0], bot_pos[1]] = BOT
    grid[rat_pos[0], rat_pos[1]] = RAT

    return bot_pos, rat_pos, grid

def sense_neighbors(grid, location):
    size = grid.shape[0]
    neighbors = [(location[0] - 1, location[1] - 1),
                 (location[0] - 1, location[1]),
                 (location[0] - 1, location[1] + 1),
                 (location[0], location[1] - 1),
                 (location[0], location[1] + 1),
                 (location[0] + 1, location[1] - 1),
                 (location[0] + 1, location[1]),
                 (location[0] + 1, location[1] + 1)
                 ]
    blocked_count = 0
    for i, j in neighbors:
        if i < 0 or i >= size or j < 0 or j >= size:
            blocked_count += 1 # Out of bound cells are treated as blocked
        elif grid[i, j] == BLOCKED:
            blocked_count += 1
    return blocked_count

def detect_rat_ping(bot_pos, rat_pos, beep_Ps):
    if bot_pos == rat_pos:
        return 2
    elif random.random() < beep_Ps[dist(bot_pos, rat_pos)]:
        return 1
    else:
        return 0

# Bot movement on grid is only performed via this function to ensure safety
def bot_attempt_move(grid, bot_pos, decided_pos):
    size = grid.shape[0]
    if dist(bot_pos, decided_pos) != 1:
        raise Exception("Error: bot is moving illegally") # Making sure there are no illegal moves
    if  decided_pos[0] < 0 or decided_pos[0] >= size or decided_pos[1] < 0 or decided_pos[1] >= size:
        return bot_pos, 0
    elif grid[decided_pos] == BLOCKED:
        return bot_pos, 0
    else:
        # Update cell from which bot is leaving:
        # If the cell has both BOT and RAT, leave the RAT behind.
        if grid[bot_pos] == BOT_RAT:
            grid[bot_pos] = RAT
        else:
            grid[bot_pos] = OPEN
        bot_pos = decided_pos
        # To Update destination cell - if it already contains RAT,
        # then mark it as BOT_RAT.
        if grid[bot_pos] == RAT:
            grid[bot_pos] = BOT_RAT
        else:
            grid[bot_pos] = BOT
        return bot_pos, 1

def dist(a, b): # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_open(grid, i, j):
    size = grid.shape[0]
    if i < 0 or i >= size or j < 0 or j >= size:
        return False  # Out of bound cells are treated as blocked
    return grid[i, j] == OPEN or grid[i, j] == RAT

def beep_probabilities(grid, alpha):
    size = grid.shape[0]
    beep_Ps = np.zeros((2*size))
    for d in range(1, 2*size): # at dist = 0, probability of beep=1 is 0
        beep_Ps[d] = np.exp(-alpha * (d - 1))
    return beep_Ps

def update_Ps(p_rat, beep, beep_p, bot_pos, rat_pos):
    if beep == 1:
        p_beep_given_rat_times_p_rat = {}
        for loc in p_rat:   p_beep_given_rat_times_p_rat[loc] = beep_p[dist(loc, bot_pos)] * p_rat[loc]
        p_beep_given_rat_times_p_rat[bot_pos] = 0
        total = sum(p_beep_given_rat_times_p_rat.values())
        for loc in p_rat:   p_rat[loc] = p_beep_given_rat_times_p_rat[loc] / total
    else:
        p_no_beep_given_rat_times_p_rat = {}
        for loc in p_rat:   p_no_beep_given_rat_times_p_rat[loc] = (1 - beep_p[dist(loc, bot_pos)]) * p_rat[loc]
        p_no_beep_given_rat_times_p_rat[bot_pos] = 0
        total = sum(p_no_beep_given_rat_times_p_rat.values())
        for loc in p_rat:   p_rat[loc] = p_no_beep_given_rat_times_p_rat[loc] / total
    # Can add code to remove locations with 0 probability
    if p_rat[rat_pos] == 0:
        raise Exception("Rat pos was somehow incorrectly eliminated.")
    return p_rat

def astar_path_1(grid, bot_pos, max_loc):
    size = grid.shape[0]
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == max_loc:
            path = []
            loc = max_loc
            while loc != bot_pos:
                path.append(loc)
                loc = prev[loc[0]][loc[1]]
            return path, 1
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            if (0 <= child[0] < size and 0 <= child[1] < size and
                grid[child[0], child[1]] != BLOCKED):
                cost = totalCosts[current] + 1
                estTotalCost = cost + dist(child, max_loc)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    raise Exception("Path not found")

def astar_path_2(grid, bot_pos, max_loc, p_rat):
    size = grid.shape[0]
    # open_cells = len(p_rat)
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == max_loc:
            path = []
            loc = max_loc
            while loc != bot_pos:
                path.append(loc)
                loc = prev[loc[0]][loc[1]]
            return path, 1
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            if (0 <= child[0] < size and 0 <= child[1] < size and
                grid[child[0], child[1]] != BLOCKED):
                cost = totalCosts[current] + 1 - p_rat[child]
                estTotalCost = cost + dist(child, max_loc)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    raise Exception("Path not found")

def bot1(grid, bot_pos, rat_pos, alpha):
    neighbor_sensings, ping_tries, move_tries = 0, 0, 0
    size = grid.shape[0]

    # Phase 1
    possible_locations = {}
    for i in range(size):
        for j in range(size):
            if grid[i, j] == OPEN or grid[i, j] == BOT:
                open_directions = np.array([
                    is_open(grid, i - 1, j),  # UP
                    is_open(grid, i, j - 1),  # LEFT
                    is_open(grid, i + 1, j),  # DOWN
                    is_open(grid, i, j + 1)   # RIGHT
                ])
                possible_locations[(i, j)] = open_directions
    while len(possible_locations) > 1:
        if bot_pos not in possible_locations:
            raise Exception("Error in location filtering.")
        blocked_count = sense_neighbors(grid, bot_pos)
        neighbor_sensings += 1
        possible_locations = {location: open_directions
                              for location, open_directions in possible_locations.items()
                              if sense_neighbors(grid, location) == blocked_count}
        direction_index = random.choice([0, 1, 2, 3])
        match direction_index:
            case 0: decided_pos = (bot_pos[0] - 1, bot_pos[1])
            case 1: decided_pos = (bot_pos[0], bot_pos[1] - 1)
            case 2: decided_pos = (bot_pos[0] + 1, bot_pos[1])
            case 3: decided_pos = (bot_pos[0], bot_pos[1] + 1)
            case _: raise Exception("This index should not exist in direction_index")
        bot_pos, move_success = bot_attempt_move(grid, bot_pos, decided_pos)
        move_tries += 1
        if move_success:
            possible_locations = {location: open_directions
                                  for location, open_directions in possible_locations.items()
                                  if open_directions[direction_index]}
            match direction_index:
                case 0: possible_locations = {(i-1, j): [is_open(grid, i-1-1, j), is_open(grid, i-1, j-1), is_open(grid, i-1+1, j), is_open(grid, i-1, j+1)] for i, j in possible_locations.keys()}
                case 1: possible_locations = {(i, j-1): [is_open(grid, i-1, j-1), is_open(grid, i, j-1-1), is_open(grid, i+1, j-1), is_open(grid, i, j-1+1)] for i, j in possible_locations.keys()}
                case 2: possible_locations = {(i+1, j): [is_open(grid, i+1-1, j), is_open(grid, i+1, j-1), is_open(grid, i+1+1, j), is_open(grid, i+1, j+1)] for i, j in possible_locations.keys()}
                case 3: possible_locations = {(i, j+1): [is_open(grid, i-1, j+1), is_open(grid, i, j+1-1), is_open(grid, i+1, j+1), is_open(grid, i, j+1+1)] for i, j in possible_locations.keys()}
        else:
            possible_locations = {location: open_directions
                                  for location, open_directions in possible_locations.items()
                                  if not open_directions[direction_index]}

    if bot_pos not in possible_locations:   raise Exception("Location not found by bot.")
    else:   print("Bot location found!")

    # Correction of putting rat back on grid if it was erased via bo_attempt_move() during phase 1
    if bot_pos != rat_pos:
        grid[rat_pos] = RAT
    # Phase 2
    p_rat = {}
    for i in range(size):
        for j in range(size):
            if grid[i, j] == OPEN or grid[i, j] == BOT or grid[i, j] == RAT:
                p_rat[(i, j)] = 1
    p_rat_len = len(p_rat)
    p_rat = {location: P / p_rat_len for location, P in p_rat.items()}
    beep_Ps = beep_probabilities(grid, alpha)
    beep = detect_rat_ping(bot_pos, rat_pos, beep_Ps)
    ping_tries += 1
    if beep == 2:
        print("Rat Found!")
        return bot_pos, neighbor_sensings, ping_tries, move_tries
    else:
        p_rat = update_Ps(p_rat, beep, beep_Ps, bot_pos, rat_pos)
    while True:
        max_loc = max(p_rat, key=p_rat.get)
        path, path_found = astar_path_1(grid, bot_pos, max_loc)
        while bot_pos != max_loc:
            decided_pos = path.pop()
            bot_pos, move_success = bot_attempt_move(grid, bot_pos, decided_pos)
            move_tries += 1
            beep = detect_rat_ping(bot_pos, rat_pos, beep_Ps)
            ping_tries += 1
            if beep == 2:
                print("Rat Found!")
                return bot_pos, neighbor_sensings, ping_tries, move_tries
            else:
                p_rat = update_Ps(p_rat, beep, beep_Ps, bot_pos, rat_pos)

def bot2(grid, bot_pos, rat_pos, alpha):
    neighbor_sensings, ping_tries, move_tries = 0, 0, 0
    size = grid.shape[0]

    # Phase 1
    possible_locations = {}
    for i in range(size):
        for j in range(size):
            if grid[i, j] == OPEN or grid[i, j] == BOT:
                open_directions = np.array([
                    is_open(grid, i - 1, j),  # UP
                    is_open(grid, i, j - 1),  # LEFT
                    is_open(grid, i + 1, j),  # DOWN
                    is_open(grid, i, j + 1)   # RIGHT
                ])
                possible_locations[(i, j)] = open_directions
    while len(possible_locations) > 1:
        if bot_pos not in possible_locations:
            raise Exception("Error in location filtering.")
        blocked_count = sense_neighbors(grid, bot_pos)
        neighbor_sensings += 1
        possible_locations = {location: open_directions
                              for location, open_directions in possible_locations.items()
                              if sense_neighbors(grid, location) == blocked_count}
        direction_index = random.choice([0, 1, 2, 3])
        match direction_index:
            case 0: decided_pos = (bot_pos[0] - 1, bot_pos[1])
            case 1: decided_pos = (bot_pos[0], bot_pos[1] - 1)
            case 2: decided_pos = (bot_pos[0] + 1, bot_pos[1])
            case 3: decided_pos = (bot_pos[0], bot_pos[1] + 1)
            case _: raise Exception("This index should not exist in direction_index")
        bot_pos, move_success = bot_attempt_move(grid, bot_pos, decided_pos)
        move_tries += 1
        if move_success:
            possible_locations = {location: open_directions
                                  for location, open_directions in possible_locations.items()
                                  if open_directions[direction_index]}
            match direction_index:
                case 0: possible_locations = {(i-1, j): [is_open(grid, i-1-1, j), is_open(grid, i-1, j-1), is_open(grid, i-1+1, j), is_open(grid, i-1, j+1)] for i, j in possible_locations.keys()}
                case 1: possible_locations = {(i, j-1): [is_open(grid, i-1, j-1), is_open(grid, i, j-1-1), is_open(grid, i+1, j-1), is_open(grid, i, j-1+1)] for i, j in possible_locations.keys()}
                case 2: possible_locations = {(i+1, j): [is_open(grid, i+1-1, j), is_open(grid, i+1, j-1), is_open(grid, i+1+1, j), is_open(grid, i+1, j+1)] for i, j in possible_locations.keys()}
                case 3: possible_locations = {(i, j+1): [is_open(grid, i-1, j+1), is_open(grid, i, j+1-1), is_open(grid, i+1, j+1), is_open(grid, i, j+1+1)] for i, j in possible_locations.keys()}
        else:
            possible_locations = {location: open_directions
                                  for location, open_directions in possible_locations.items()
                                  if not open_directions[direction_index]}

    if bot_pos not in possible_locations:   raise Exception("Location not found by bot.")
    else:   print("Bot location found!")
    # Phase 2
    p_rat = {}
    for i in range(size):
        for j in range(size):
            if grid[i, j] == OPEN or grid[i, j] == BOT or grid[i, j] == RAT:
                p_rat[(i, j)] = 1
    p_rat_len = len(p_rat)
    p_rat = {location: P / p_rat_len for location, P in p_rat.items()}
    beep_Ps = beep_probabilities(grid, alpha)
    for pings in range(10):
        beep = detect_rat_ping(bot_pos, rat_pos, beep_Ps)
        ping_tries += 1
        if beep == 2:
            print("Rat Found!")
            return bot_pos, neighbor_sensings, ping_tries, move_tries
        else:
            p_rat = update_Ps(p_rat, beep, beep_Ps, bot_pos, rat_pos)
    while True:
        max_loc = max(p_rat, key=p_rat.get)
        path, path_found = astar_path_1(grid, bot_pos, max_loc)
        while bot_pos != max_loc:
            decided_pos = path.pop()
            bot_pos, move_success = bot_attempt_move(grid, bot_pos, decided_pos)
            move_tries += 1
        for pings in range(10):
            beep = detect_rat_ping(bot_pos, rat_pos, beep_Ps)
            ping_tries += 1
            if beep == 2:
                print("Rat Found!")
                return bot_pos, neighbor_sensings, ping_tries, move_tries
            else:
                p_rat = update_Ps(p_rat, beep, beep_Ps, bot_pos, rat_pos)

for test in range(5):
    ship1 = build_ship(30)
    bot_pos, rat_pos, ship1 = place_bot_rat(ship1)
    ship2 = ship1.copy()
    alpha = round(random.choice(np.linspace(0.02, 0.2, 10)), 2)
    print("rat_pos: ", rat_pos)
    print("bot_pos before: ", bot_pos)
    # plt.imshow(ship1, cmap=cmap, norm=norm)
    # plt.title("Ship config")
    # plt.show()
    bot_pos_1, neighbor_sensings_1, ping_tries_1, move_tries_1 = bot1(ship1, bot_pos, rat_pos, alpha)
    print("bot_pos_1 after: ", bot_pos_1)
    print("neighbor_sensings_1, ping_tries_1, move_tries_1: ", neighbor_sensings_1, ping_tries_1, move_tries_1)
    total_actions_1 = neighbor_sensings_1 + ping_tries_1 + move_tries_1
    print("total_actions_1 ", total_actions_1)

    bot_pos_2, neighbor_sensings_2, ping_tries_2, move_tries_2 = bot2(ship2, bot_pos, rat_pos, alpha)
    print("bot_pos_2 after: ", bot_pos_2)
    print("neighbor_sensings_2, ping_tries_2, move_tries_2: ", neighbor_sensings_2, ping_tries_2, move_tries_2)
    total_actions_2 = neighbor_sensings_2 + ping_tries_2 + move_tries_2
    print("total_actions_2 ", total_actions_2)

    new_row = (test, alpha, bot_pos, rat_pos,
               neighbor_sensings_1, ping_tries_1, move_tries_1, total_actions_1,
               neighbor_sensings_2, ping_tries_2, move_tries_2, total_actions_2)
    new_row_df = pd.DataFrame([new_row])

    # new_row_df.to_csv("results.csv", mode='a', header=False, index=False)
    plt.imshow(ship1, cmap=cmap, norm=norm)
    plt.title("Ship config")
    plt.show()


