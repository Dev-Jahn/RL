import time
import random
from collections import defaultdict

import numpy as np

MODES = ['mc_control', 'q_learning', 'test_mode']


class Agent:
    def __init__(self, Q, mode="test_mode", alpha=0.01, gamma=0.9, eps=0.1, decay_method='exponential', eps_decay=1 - 1e-4):
        self.Q = Q
        assert mode in MODES
        self.mode = mode
        self.n_actions = 6
        # params
        self.alpha = alpha
        self.gamma = gamma
        self.init_eps = eps
        self.eps = 0 if mode == 'test_mode' else eps
        self.decay_method = decay_method
        self.eps_decay = eps_decay
        self.counter = 1
        # Mode specific variables
        # Hyperparameters are optimized with Bayesian optimization
        if mode == 'mc_control':
            self.episode = []  # tuple of S, A, R
            self.alpha = 0.001221
            self.gamma = 0.7777
            self.init_eps = 0.08547
            self.eps = self.init_eps
            self.decay_method = 'harmonic'
        elif mode == 'q_learning':
            self.alpha = 0.06946
            self.gamma = 0.8898
            self.init_eps = 0.06279
            self.eps = self.init_eps
            self.decay_method = 'exponential'
            self.eps_decay = 0.9614

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
            self.episode.append((state, action, reward))
            if done:

                '''
                Impl 1
                '''
                states, actions, rewards = zip(*self.episode)
                discount_factors = np.array([self.gamma ** j for j in range(len(rewards)+1)])
                # For each state, get discounted cumulative reward
                Gs = [
                    np.sum(rewards[i:] * discount_factors[:-(i+1)])
                    for i in range(len(states))
                ]
                # Update
                for i, G in enumerate(Gs):
                    self.Q[states[i]][actions[i]] += self.alpha * (G - self.Q[states[i]][actions[i]])
                '''
                Impl 2
                '''
                # rewards = defaultdict(lambda: np.zeros(6))
                # for history in reversed(self.episode):
                #     state, action, reward = history
                #     rewards[state][action] = reward + self.gamma * rewards[state][action]
                #     self.Q[state][action] += self.alpha * (rewards[state][action] - self.Q[state][action])

                self.episode.clear()

        elif self.mode == 'q_learning':
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
            )
        if done:
            self._epsilon_decay(self.decay_method)

    def _epsilon_decay(self, method='exponential'):
        if method == 'exponential':
            self.eps *= self.eps_decay
        elif method == 'harmonic':
            self.eps *= 1 / self.counter
            self.counter += 1
