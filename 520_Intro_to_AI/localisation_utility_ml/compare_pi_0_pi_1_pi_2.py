import torch
from models import Pi0Predictor, Pi1Predictor
from localization_functions import (
    build_ship, place_bot, random_L0_generator, all_unblocked_cells_L0_generator,
    pi_0, pi_1, pi_2
)

def evaluate_policy(policy_fn, model=None, device='cpu', num_tests=1000):
    total_moves = 0
    ship = build_ship(30)
    bot_pos, _ = place_bot(ship)

    for test in range(num_tests):
        print(f"\nTest {test}")
        ship1 = ship.copy()
        bot_pos1 = bot_pos
        L0, _ = random_L0_generator(ship1, bot_pos1)


        if policy_fn == pi_2:
            model.load_state_dict(torch.load("models/pi1_predictor.pt", map_location=device))
            model.to(device)
            model.eval()
            bot_pos1, move_sequence = pi_2(ship1, bot_pos1, L0, model, device=device)
        elif policy_fn == pi_1:
            model.load_state_dict(torch.load("models/pi0_predictor.pt", map_location=device))
            model.to(device)
            model.eval()
            bot_pos1, move_sequence = pi_1(ship1, bot_pos1, L0, model, device=device)
        else:
            bot_pos1, move_sequence = pi_0(ship1, bot_pos1, L0)

        total_moves += len(move_sequence)

    return total_moves / num_tests


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = Pi0Predictor(30)
    model1 = Pi1Predictor(30)

    avg_pi_0 = evaluate_policy(pi_0, num_tests=1000)
    avg_pi_1 = evaluate_policy(pi_1, model=model0, device=device, num_tests=1000)
    avg_pi_2 = evaluate_policy(pi_2, model=model1, device=device, num_tests=1000)

    print(f"\nAverage moves for pi_0 over 1000 tests: {avg_pi_0:.2f}")
    print(f"Average moves for pi_1 over 1000 tests: {avg_pi_1:.2f}")
    print(f"Average moves for pi_2 over 1000 tests: {avg_pi_2:.2f}")
