import random

import numpy as np

MODES = ['mc_control', 'q_learning', 'test_mode']


class Agent:
    def __init__(self, Q, mode="test_mode", alpha=0.1, gamma=0.9, eps=0.1, eps_decay=1 - 1e-4):
        self.Q = Q
        assert mode in MODES
        self.mode = mode
        self.n_actions = 6
        # params
        self.alpha = alpha
        self.gamma = gamma
        self.init_eps = eps
        self.eps = 0 if mode == 'test_mode' else eps
        self.eps_decay = eps_decay
        self.counter = 1

    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() < self.eps:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.mode == 'mc_control':
            pass
        elif self.mode == 'q_learning':
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
            )
        if done:
            self._epsilon_decay()

    def _epsilon_decay(self, method='exponential'):
        if method == 'exponential':
            self.eps *= self.eps_decay
        elif method == 'harmonic':
            self.eps *= 1 / self.counter
            self.counter += 1
