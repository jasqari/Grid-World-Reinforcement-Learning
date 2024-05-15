import numpy as np


def sarsa(board, epsilon, gamma, alpha, N, actions):
    Q = {
        key: 0
        for key in [
            (state, action)
            for state in board.nonterminal_states + board.goal_states
            for action in actions
        ]
    }
    Pi = {key: 0.25 * np.ones(len(actions)) for key in board.nonterminal_states}
    for episode in range(N):
        S = board.nonterminal_states[
            np.random.choice([i for i in range(len(board.nonterminal_states))])
        ]
        A = actions[np.random.choice(len(actions), p=[v for v in Pi[S]])]
        while True:
            S_prime = (S[0] + A[0], S[1] + A[1])
            if S_prime not in board.nonterminal_states + board.goal_states:
                S_prime = S
            # This will not matter since Q(goal_state) is zero. We just make sure that no key errors happen.
            if S_prime in board.goal_states:
                A_prime = actions[np.random.choice(len(actions))]
            # Behave under an epsilon-greedy policy
            else:
                A_prime = actions[np.random.choice(len(actions), p=[v for v in Pi[S_prime]])]
            R = -1
            # Update action-values using the same policy
            Q[(S, A)] = Q[(S, A)] + alpha * (R + gamma * Q[(S_prime, A_prime)] - Q[(S, A)])
            S = S_prime
            A = A_prime
            # Update the epsilon-greedy policy with respect to the new action-value function
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
            if S in board.goal_states:
                break
        # We value exploration more through the first learning steps
        epsilon = epsilon * 0.97
    return {"state_action_values": Q, "policy": Pi}
