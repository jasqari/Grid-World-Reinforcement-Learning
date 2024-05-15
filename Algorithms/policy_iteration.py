import numpy as np


def iterate_policy(board, gamma, theta, N, actions):
    V_0 = {key: 0 for key in board.nonterminal_states + board.goal_states}
    Pi_0 = {key: 0.25 * np.ones(len(actions)) for key in board.nonterminal_states}
    convergence = False
    while not convergence:
        V_new = evaluate_policy(V_0, Pi_0, board, gamma, theta, N, actions)
        Pi_new, convergence = improve_policy(Pi_0, V_new, board, gamma, actions)
        V_0 = V_new
        Pi_0 = Pi_new
    return {"state_values": V_new, "policy": Pi_new}


def evaluate_policy(V_0, Pi, board, gamma, theta, N, actions):
    V = dict(V_0)
    for k in range(N):
        V_prev = dict(V)
        for state in board.nonterminal_states:
            v_new = 0
            # Compute the new estimated value and update V(S) using Gauss–Seidel method.
            for action in actions:
                new_state = (state[0] + action[0], state[1] + action[1])
                if new_state not in (board.nonterminal_states + board.goal_states):
                    new_state = state
                v_new += Pi[state][actions.index(action)] * (-1 + gamma * V[new_state])
            V[state] = v_new
        # Check for convergence with respect to theta
        if (
            np.linalg.norm(
                np.array([val for val in V.values()]) - np.array([val for val in V_prev.values()])
            )
            < theta
        ):
            break
    return V


def improve_policy(Pi, V, board, gamma, actions):
    Pi_new = {key: np.zeros(len(actions)) for key in board.nonterminal_states}
    policy_is_stable = True
    for state in board.nonterminal_states:
        old_action = actions[np.random.choice(len(actions), p=[v for v in Pi[state]])]
        # Find the actions with maximum returns
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
        # Optimize the policy
        for a in range(len(actions)):
            if a == selected_action:
                Pi_new[state][a] = 1
            else:
                Pi_new[state][a] = 0
        # Check if the policy has converged
        if old_action not in [actions[a] for a in max_value_actions]:
            policy_is_stable = False
    return Pi_new, policy_is_stable
