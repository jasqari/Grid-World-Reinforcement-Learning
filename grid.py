import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


class Grid:
    def __init__(self, size, initial_state, goal_states):
        self.grid_size = size
        self.current_state = initial_state
        self.goal_states = goal_states
        self.board = np.ones(size**2).reshape(size, size)
        for state in goal_states:
            self.board[state] = 3
        self.board[initial_state] = 2
        self.nonterminal_states = [
            (i, j) for i in range(size) for j in range(size) if (i, j) not in goal_states
        ]
        if initial_state in goal_states:
            self.end_of_episode = True
        else:
            self.end_of_episode = False
        self.path = [self.current_state]

    def simulate(self, action):
        if self.end_of_episode:
            return
        next_state = (self.current_state[0] + action[0], self.current_state[1] + action[1])
        if next_state in self.goal_states:
            self.board[self.current_state] = 1
            self.board[next_state] = 2
            self.current_state = next_state
            self.end_of_episode = True
            self.path.append(self.current_state)
            return -1
        if next_state in self.nonterminal_states:
            self.board[self.current_state] = 1
            self.board[next_state] = 2
            self.current_state = next_state
            self.path.append(self.current_state)
            return -1
        self.path.append(self.current_state)
        return -1

    def display(self):
        path = []
        for state in self.path:
            board = np.ones(self.board.shape)
            for fin in self.goal_states:
                board[fin] = 3
            board[state] = 2
            path.append(board)

        def generate(boards):
            while len(boards):
                yield (next_move(boards))

        def next_move(boards):
            board = boards.pop(0)
            return board

        def update(data):
            grid.set_data(data)
            return grid

        fig, ax = plt.subplots()
        plt.title("Agent Path")
        grid = ax.matshow(path[0], cmap="Greens")
        anim = animation.FuncAnimation(
            fig, update, partial(generate, path), interval=500, save_count=100
        )
        plt.show()

    def reset(self):
        self.end_of_episode = False
