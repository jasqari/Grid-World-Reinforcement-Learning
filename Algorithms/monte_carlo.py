import numpy as np


def monte_carlo(board, epsilon, gamma, N, actions, alpha=None):
    Q = {
        key: 0
        for key in [
            (state, action)
            for state in board.nonterminal_states + board.goal_states
            for action in actions
        ]
    }
    Counter = dict(Q)
    Pi = {key: 0.25 * np.ones(len(actions)) for key in board.nonterminal_states}
    for episode in range(N):
        initial_state = board.nonterminal_states[
            np.random.choice([i for i in range(len(board.nonterminal_states))])
        ]
        initial_action = actions[np.random.choice(len(actions))]
        path_states, selected_actions, given_rewards = [initial_state], [initial_action], [0]
        t = 0
        # Generate and save an episode using an epsilon-greedy policy
        while True:
            At = actions[np.random.choice(len(actions), p=[v for v in Pi[path_states[t]]])]
            if t >= 1:
                selected_actions.append(At)
            next_state = (
                path_states[t][0] + selected_actions[t][0],
                path_states[t][1] + selected_actions[t][1],
            )
            if next_state not in board.nonterminal_states + board.goal_states:
                next_state = path_states[t]
            path_states.append(next_state)
            given_rewards.append(-1)
            if next_state in board.goal_states:
                selected_actions.append(None)
                break
            t += 1
        # Update the action-value estimation according to first-visits
        path_state_actions = [elem for elem in zip(path_states, selected_actions)]
        for state in board.nonterminal_states:
            for action in actions:
                if (state, action) in path_state_actions:
                    first_visit = path_state_actions.index((state, action))
                    Counter[(state, action)] += 1
                    G = sum(
                        [
                            gamma**k * given_rewards[first_visit + 1 + k]
                            for k in range(len(given_rewards) - (first_visit + 1))
                        ]
                    )
                    if alpha:
                        Q[(state, action)] = Q[(state, action)] + alpha * (G - Q[(state, action)])
                    else:
                        Q[(state, action)] = Q[(state, action)] + (1 / Counter[(state, action)]) * (
                            G - Q[(state, action)]
                        )
        # Optimize the same epsilon-greedy policy
        for state in board.nonterminal_states:
            potential_action_values = [Q[(state, action)] for action in actions]
            max_value_actions = [
                i[0]
                for i in (
                    np.argwhere(
                        np.around(potential_action_values, decimals=6)
                        == np.max(np.around(potential_action_values, decimals=6))
                    )
                ).tolist()
            ]
            A_star = np.random.choice(max_value_actions)
            for a in range(len(actions)):
                if a == A_star:
                    Pi[state][a] = 1 - epsilon + (epsilon / len(actions))
                else:
                    Pi[state][a] = epsilon / len(actions)
        # We value exploration more through the first learning steps
        epsilon = epsilon * 0.97
    return {"state_action_values": Q, "policy": Pi}
