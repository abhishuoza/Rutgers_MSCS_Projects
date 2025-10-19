import heapq
import numpy as np
import random
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm

# Constants
BLOCKED, OPEN, BOT, L_CELL = 0, 1, 2, 3
cmap = ListedColormap(["black", "grey", "blue", "yellow"])
bounds = [BLOCKED - 0.5, OPEN - 0.5, BOT - 0.5, L_CELL - 0.5, L_CELL + 0.5]
norm = BoundaryNorm(bounds, cmap.N)



def build_ship(size):
    # Create square grid with all blocked cells
    grid = np.zeros((size, size))
    local_rng = random.Random(42)
    # Generate array of random numbers

    # Open random cell
    randx = local_rng.randint(0, size)
    randy = local_rng.randint(0, size)
    grid[randx, randy] = OPEN

    # Iteratively open blocked cells
    while True:
        candidates = []
        for i in range(size):
            for j in range(size):
                counter = 0
                if grid[i, j] == BLOCKED:
                    if i != 0 and grid[i - 1, j] == OPEN:
                        counter += 1
                    if j != 0 and grid[i, j - 1] == OPEN:
                        counter += 1
                    if i != size - 1 and grid[i + 1, j] == OPEN:
                        counter += 1
                    if j != size - 1 and grid[i, j + 1] == OPEN:
                        counter += 1
                if counter == 1:
                    candidates.append((i, j))
        if len(candidates) == 0:
            break  # End while loop if no candidates left
        cell_to_open = local_rng.choice(candidates)
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
        if local_rng.random() < 0.5:  # For approximately half dead ends
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
            ends_to_open.append(local_rng.choice(neighbors))

    # Opening neighbors only after picking neighbors for all dead ends
    for i, j in ends_to_open:
        grid[i, j] = OPEN

    return grid

def place_bot(grid):
    local_rng = random.Random(42)
    size = grid.shape[0]
    open_cells = []
    for i in range(size):
        for j in range(size):
            if grid[i, j] == OPEN:
                open_cells.append((i, j))
    bot_pos = local_rng.choice(open_cells)
    print(bot_pos)
    grid[bot_pos[0], bot_pos[1]] = BOT
    return bot_pos, grid

def dist(a, b):  # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Bot movement on grid is only performed via this function to ensure safety
def bot_attempt_move(grid, bot_pos, direction):
    size = grid.shape[0]
    decided_pos = (bot_pos[0] + direction[0], bot_pos[1] + direction[1])
    if dist(bot_pos, decided_pos) != 1:
        raise Exception("Error: bot is moving illegally")
    if decided_pos[0] < 0 or decided_pos[0] >= size or decided_pos[1] < 0 or decided_pos[1] >= size:
        return bot_pos, 0
    elif grid[decided_pos] == BLOCKED:
        return bot_pos, 0
    else:
        grid[bot_pos] = OPEN
        bot_pos = decided_pos
        grid[bot_pos] = BOT
        return bot_pos, 1

def L_next(grid, L, direction):
    size = grid.shape[0]
    L_new = set()
    for loc in L:
        next_loc = (loc[0] + direction[0], loc[1] + direction[1])
        if not (0 <= next_loc[0] < size and 0 <= next_loc[1] < size) or grid[next_loc] == BLOCKED:
            L_new.add(loc)
        else:
            L_new.add(next_loc)
    return L_new

def astar_path(grid, bot_pos, target):
    size = grid.shape[0]
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # [Up, Left, Down, Right]
    fringe = []
    heapq.heappush(fringe, (0, bot_pos))
    totalCosts = {bot_pos: 0}
    prev = [[(None, None) for i in range(size)] for j in range(size)]
    while fringe:
        _, current = heapq.heappop(fringe)
        if current == target:
            path = []
            loc = target
            while loc != bot_pos:
                path.append(loc)
                loc = prev[loc[0]][loc[1]]
            return reversed(path)
        for i, j in directions:
            child = (current[0] + i, current[1] + j)
            if (0 <= child[0] < size and 0 <= child[1] < size and
                    grid[child[0], child[1]] != BLOCKED):
                cost = totalCosts[current] + 1
                estTotalCost = cost + dist(child, target)
                if child not in totalCosts or totalCosts[child] > cost:
                    heapq.heappush(fringe, (estTotalCost, child))
                    totalCosts[child] = cost
                    prev[child[0]][child[1]] = current
    raise Exception("Path not found")

def random_L0_generator(grid, bot_pos):
    open_cells = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == OPEN]
    L0_size = random.randint(0, len(open_cells))
    L0 = random.sample(open_cells, L0_size)
    L0.append(bot_pos)
    return L0, len(L0)

def all_unblocked_cells_L0_generator(grid, bot_pos):
    size = grid.shape[0]
    open_cells = [(i, j) for i in range(size) for j in range(size) if grid[i, j] != BLOCKED]
    return open_cells, len(open_cells)

def encode_input(grid, L, device):
    size = grid.shape[0]
    input_tensor = torch.zeros((2, size, size), dtype=torch.float32, device=device)
    np_open_map = (grid != BLOCKED).astype(np.float32)
    input_tensor[0] = torch.from_numpy(np_open_map).to(device)
    for loc in L:
        input_tensor[1, loc[0], loc[1]] = 1.0
    return input_tensor.unsqueeze(0)

def pi_0(grid, bot_pos, L, L_data_log=None):
    size = grid.shape[0]
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Up, Left, Down, Right
    move_sequence = []
    candidates = []
    recent_L_states = []
    MAX_REPEAT = 5
    for i in range(size):
        for j in range(size):
            loc = (i, j)
            if grid[loc] != BLOCKED:
                open_dirs = []
                for d in directions:
                    nb = (loc[0] + d[0], loc[1] + d[1])
                    if 0 <= nb[0] < size and 0 <= nb[1] < size and grid[nb] != BLOCKED:
                        open_dirs.append(d)
                if len(open_dirs) == 1 or (len(open_dirs) == 2 and open_dirs[0][0] != open_dirs[1][0] and open_dirs[0][1] != open_dirs[1][1]):
                    candidates.append(loc)

    target = random.choice(candidates)
    print("Target: ", target)
    while len(L) > 1:
        # This is for detecting excessive looping
        recent_L_states.append(frozenset(L))
        if len(recent_L_states) > MAX_REPEAT:
            recent_L_states.pop(0)

        start = random.choice(list(L))
        # print()
        # print("Start cell: ", start)
        # print("Target: ", target)
        path = astar_path(grid, start, target)
        # Move along path and update L
        for step in path:
            direction = (step[0] - start[0], step[1] - start[1])
            bot_pos, _ = bot_attempt_move(grid, bot_pos, direction)
            L = L_next(grid, L, direction)
            if L_data_log is not None:
                L_data_log.append(set(L))
            move_sequence.append(direction)

            # print("step ", len(move_sequence))
            # print("bot_pos: ", bot_pos)
            # print("Direction: ", direction)
            # print("len(L): ",len(L))
            # print("L: ", L)

            start = step
            if len(L) == 1:
                if bot_pos not in L:
                    raise Exception("error in localisation process, bot_pos is not in L.")
                else:
                    print("Localisation succeeded!")
                    break
        # Here we select a new target to break excessive looping
        if recent_L_states.count(frozenset(L)) >= MAX_REPEAT: # checks how many times the current L appears in the past 5 states encountered
            print("Looping detected — choosing new random target to break symmetry")
            possible_new_targets = [loc for loc in candidates if loc != target]
            if possible_new_targets:
                target = random.choice(possible_new_targets)
            recent_L_states.clear()
    return bot_pos, move_sequence

def pi_1(grid, bot_pos, L, model, device='cpu', L_data_log=None):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    best_cost = float('inf')
    best_direction = None
    best_L_new = None

    for d in directions:
        L_new = L_next(grid, L, d)
        if len(L_new) == 0:
            continue
        input_tensor = encode_input(grid, L_new, device)
        with torch.no_grad():
            predicted_cost = model(input_tensor).item() # Predicting cost here using pi_0 predictor
        if predicted_cost < best_cost: # This will find the lowest cost direction as best
            best_cost = predicted_cost
            best_direction = d
            best_L_new = L_new

    bot_pos, _ = bot_attempt_move(grid, bot_pos, best_direction)
    if L_data_log is not None:
        L_data_log.append(set(best_L_new))

    # For the rest of the moves, use pi_0 strategy
    return pi_0(grid, bot_pos, best_L_new, L_data_log)


def pi_2(grid, bot_pos, L, model, device='cpu', L_data_log=None):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    best_cost = float('inf')
    best_direction = None
    best_L_new = None

    for d in directions:
        L_new = L_next(grid, L, d)
        if len(L_new) == 0:
            continue
        input_tensor = encode_input(grid, L_new, device)
        with torch.no_grad():
            predicted_cost = model(input_tensor).item()  # Predicting cost here using pi_1 predictor
        if predicted_cost < best_cost: # This will find the lowest cost direction as best
            best_cost = predicted_cost
            best_direction = d
            best_L_new = L_new

    bot_pos, _ = bot_attempt_move(grid, bot_pos, best_direction)
    if L_data_log is not None:
        L_data_log.append(set(best_L_new))

    # For the rest of the moves, use pi_1 strategy
    return pi_1(grid, bot_pos, best_L_new, model, device, L_data_log)

