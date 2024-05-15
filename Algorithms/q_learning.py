import numpy as np


def q_learn(board, epsilon, gamma, alpha, N, actions):
    Q = {
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
            # Update action-values using a new greedy policy
            Q[(S, A)] = Q[(S, A)] + alpha * (
                R + gamma * np.max([Q[(S_prime, a)] for a in actions]) - Q[(S, A)]
            )
            S = S_prime
            # Update the epsilon-greedy behavior policy with respect to the new action-value function
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
                        behavior_policy[state][a] = 1 - epsilon + (epsilon / len(actions))
                    else:
                        behavior_policy[state][a] = epsilon / len(actions)
            if S in board.goal_states:
                break
        # We value exploration more through the first learning steps
        epsilon = epsilon * 0.97
    # Find the greedy target policy using the converged action-value function
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
                target_policy[state][a] = 1
            else:
                target_policy[state][a] = 0
    return {"state_action_values": Q, "policy": target_policy}
