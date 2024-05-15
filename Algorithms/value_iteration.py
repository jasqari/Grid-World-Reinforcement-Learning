import numpy as np


def iterate_value(board, gamma, theta, N, actions):
    V_0 = {key: 0 for key in board.nonterminal_states + board.goal_states}
    V = dict(V_0)
    # Solve the bellman optimality equation
    for k in range(N):
        V_prev = dict(V)
        for state in board.nonterminal_states:
            potential_action_values = []
            for action in actions:
                new_state = (state[0] + action[0], state[1] + action[1])
                if new_state not in (board.nonterminal_states + board.goal_states):
                    new_state = state
                v_action = -1 + gamma * V[new_state]
                potential_action_values.append(v_action)
            max_value = np.max(potential_action_values)
            V[state] = max_value
        # Check for convergence
        if (
            np.linalg.norm(
                np.array([val for val in V.values()]) - np.array([val for val in V_prev.values()])
            )
            < theta
        ):
            break

    # Find a greedy policy corresponding to the optimal state values
    Pi = {key: np.zeros(len(actions)) for key in board.nonterminal_states}
    for state in board.nonterminal_states:
        potential_action_values = []
        for action in actions:
            new_state = (state[0] + action[0], state[1] + action[1])
            if new_state not in (board.nonterminal_states + board.goal_states):
                new_state = state
            v_action = -1 + gamma * V[new_state]
            potential_action_values.append(v_action)
        max_value_actions = [
            i[0]
            for i in (
                np.argwhere(
                    np.around(potential_action_values, decimals=5)
                    == np.max(np.around(potential_action_values, decimals=5))
                )
            ).tolist()
        ]
        selected_action = np.random.choice(max_value_actions)
        for a in range(len(actions)):
            if a == selected_action:
                Pi[state][a] = 1
            else:
                Pi[state][a] = 0
    return {"state_values": V, "policy": Pi}
