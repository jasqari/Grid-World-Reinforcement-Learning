import numpy as np


def double_q_learn(board, epsilon, gamma, alpha, N, actions):
    # Initialize two action-value functions since Double Q_Learning aims to omit maximization bias
    Q1 = {
        key: 0
        for key in [
            (state, action)
            for state in board.nonterminal_states + board.goal_states
            for action in actions
        ]
    }
    Q2 = {
        key: 0
        for key in [
            (state, action)
            for state in board.nonterminal_states + board.goal_states
            for action in actions
        ]
    }
    # Initialize two policies since Q_Learning is an off-policy method
    behavior_policy = {key: 0.25 * np.ones(len(actions)) for key in board.nonterminal_states}
    target_policy = {key: np.zeros(len(actions)) for key in board.nonterminal_states}
    for episode in range(N):
        S = board.nonterminal_states[
            np.random.choice([i for i in range(len(board.nonterminal_states))])
        ]
        while True:
            # Behave under an epsilon-greedy policy
            A = actions[np.random.choice(len(actions), p=[v for v in behavior_policy[S]])]
            S_prime = (S[0] + A[0], S[1] + A[1])
            if S_prime not in board.nonterminal_states + board.goal_states:
                S_prime = S
            R = -1
            # Choose either Q to update the other with respect to a new greedy policy
            if np.random.rand() > 0.5:
                potential_action_values = [Q1[(S_prime, action)] for action in actions]
                max_value_actions = [
                    i[0]
                    for i in (
                        np.argwhere(
                            np.around(potential_action_values, decimals=6)
                            == np.max(np.around(potential_action_values, decimals=6))
                        )
                    ).tolist()
                ]
                A_star = actions[np.random.choice(max_value_actions)]
                Q1[(S, A)] = Q1[(S, A)] + alpha * (R + gamma * Q2[(S_prime, A_star)] - Q1[(S, A)])
            else:
                potential_action_values = [Q2[(S_prime, action)] for action in actions]
                max_value_actions = [
                    i[0]
                    for i in (
                        np.argwhere(
                            np.around(potential_action_values, decimals=6)
                            == np.max(np.around(potential_action_values, decimals=6))
                        )
                    ).tolist()
                ]
                A_star = actions[np.random.choice(max_value_actions)]
                Q2[(S, A)] = Q2[(S, A)] + alpha * (R + gamma * Q1[(S_prime, A_star)] - Q2[(S, A)])
            S = S_prime
            # Update the epsilon-greedy behavior policy with respect to the new action-value function
            for state in board.nonterminal_states:
                potential_action_values = [
                    (Q1[(state, action)] + Q2[(state, action)]) / len(Q1) for action in actions
                ]
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
                        behavior_policy[state][a] = 1 - epsilon + (epsilon / len(actions))
                    else:
                        behavior_policy[state][a] = epsilon / len(actions)

            if S in board.goal_states:
                break
        # We value exploration more through the first learning steps
        epsilon = epsilon * 0.97
    # Find the greedy target policy using the sum or average of both converged action-value functions
    for state in board.nonterminal_states:
        potential_action_values = [
            (Q1[(state, action)] + Q2[(state, action)]) / len(Q1) for action in actions
        ]
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
                target_policy[state][a] = 1
            else:
                target_policy[state][a] = 0
    return {"state_action_values_1": Q1, "state_action_values_2": Q2, "policy": behavior_policy}
