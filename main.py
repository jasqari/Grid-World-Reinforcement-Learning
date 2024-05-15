import argparse
import numpy as np
from grid import Grid
from Algorithms import (
    policy_iteration,
    value_iteration,
    monte_carlo,
    sarsa,
    q_learning,
    expected_sarsa,
    double_q_learning,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int, help="The edge size of the square grid")
    parser.add_argument(
        "algorithm",
        type=str,
        choices=[
            "pol_iter",
            "val_iter",
            "mc",
            "sarsa",
            "ql",
            "exp_sarsa",
            "dql",
        ],
        help="Choice of model-based or model-free algorithm",
    )
    parser.add_argument(
        "--init_state",
        type=str,
        default="random",
        help="""Starting point of the agent in the 2D grid world.
                Provide states in the form of 'row,col' and
                type 'random' for a random initial state.""",
    )
    parser.add_argument(
        "--goal_states",
        type=str,
        default="0,0-n,n",
        help="""Goal states of the agent in the 2D grid world.
                Provide a set of states in the form of 'row1,col1-row2,col2-...'.""",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Discount factor to control how the agent should value immediate versus long-term rewards",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1e-3,
        help="Tolerance value as the stopping criterion of the iterative algorithms",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=500,
        help="Number of iterations as the stopping criterion of the iterative algorithms",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.9,
        help="Epsilon-greedy policy parameter to control the importance of exploring as opposed to exploiting",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Step size to control the process of searching for optimal values",
    )
    parser.add_argument(
        "--simulate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Simulate an episode after computing the optimal policies or not",
    )
    args = parser.parse_args()

    if args.init_state == "random":
        init_state = (np.random.randint(args.size), np.random.randint(args.size))
    else:
        x, y = args.init_state.split(",")
        init_state = (int(x), int(y))
    goal_states = args.goal_states.replace("n", str(args.size - 1)).split("-")
    goal_states = [gs.split(",") for gs in goal_states]
    goal_states = [(int(gs[0]), int(gs[1])) for gs in goal_states]
    grid = Grid(args.size, init_state, goal_states)

    possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    alg_map = {
        "pol_iter": policy_iteration.iterate_policy,
        "val_iter": value_iteration.iterate_value,
        "mc": monte_carlo.monte_carlo,
        "sarsa": sarsa.sarsa,
        "ql": q_learning.q_learn,
        "exp_sarsa": expected_sarsa.expected_sarsa,
        "dql": double_q_learning.double_q_learn,
    }
    vars_map = {
        "pol_iter": {
            "board": grid,
            "gamma": args.gamma,
            "theta": args.theta,
            "N": args.N,
            "actions": possible_actions,
        },
        "val_iter": {
            "board": grid,
            "gamma": args.gamma,
            "theta": args.theta,
            "N": args.N,
            "actions": possible_actions,
        },
        "mc": {
            "board": grid,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "N": args.N,
            "actions": possible_actions,
            "alpha": args.alpha,
        },
        "sarsa": {
            "board": grid,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "N": args.N,
            "actions": possible_actions,
        },
        "ql": {
            "board": grid,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "N": args.N,
            "actions": possible_actions,
        },
        "exp_sarsa": {
            "board": grid,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "N": args.N,
            "actions": possible_actions,
        },
        "dql": {
            "board": grid,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "N": args.N,
            "actions": possible_actions,
        },
    }

    output = alg_map[args.algorithm](**vars_map[args.algorithm])
    for o in output:
        file = open(str(o) + ".txt", "w")
        for row in output[o]:
            file.write(str(row) + ":" + str(output[o][row]) + "\n")
        file.close()

    if args.simulate:
        while grid.current_state not in grid.goal_states:
            grid.simulate(
                possible_actions[
                    np.random.choice(
                        len(possible_actions), p=[v for v in output["policy"][grid.current_state]]
                    )
                ]
            )
        grid.display()
