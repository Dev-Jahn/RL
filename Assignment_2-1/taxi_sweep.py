from collections import deque
from collections import defaultdict

import numpy as np
import gym
import wandb

from agent import Agent

sweep_config = {
    'name': 'MC-Control-sweep3',
    'method': 'bayes',
    'parameters': {
        'alpha': {
            'distribution': 'log_uniform_values',
            'min': 1e-3,
            'max': 1e-2,
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.99
        },
        'eps': {
            'distribution': 'log_uniform_values',
            'min': 0.05,
            'max': 0.5,
        },
        'decay_method': {
            'values': ['exponential', 'harmonic']
        },
        'eps_decay': {
            'distribution': 'log_uniform_values',
            'min': 1 - 1e-1,
            'max': 1 - 1e-5,
        },
    },
    'metric': {
        'name': 'avg_100',
        'goal': 'maximize',
    },
    'early_terminate': {
        'type': 'hyperband',
        'max_iter': 1000,
        's': 4,
        'eta': 2
    }
}

sweep_id = wandb.sweep(sweep_config, project='RL')

env = gym.make('Taxi-v3')

action_size = env.action_space.n
print("Action Space", env.action_space.n)
print("State Space", env.observation_space.n)


def testing_without_learning():
    state = env.reset()
    total_rewards = 0

    def decode(i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        return reversed(out)

    while True:
        env.render()
        print(list(decode(state)))
        print("0:down, 1:up, 2:right, 3:left, 4:pick, 5:dropoff")
        action = int(input("select action: "))
        while action not in [0, 1, 2, 3, 4, 5]:
            action = int(input("select action: "))
        next_state, reward, done, _ = env.step(action)
        print("reward:", reward)
        total_rewards = total_rewards + reward
        if done:
            print("total reward:", total_rewards)
            break
        state = next_state


def model_free_RL(Q, mode, *args, **kwargs):
    agent = Agent(Q, mode, *args, **kwargs)
    num_episodes = 50000
    last_100_episode_rewards = deque(maxlen=100)
    for i_episode in range(1, num_episodes + 1):

        state = env.reset()
        episode_rewards = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            episode_rewards += reward
            if done:
                last_100_episode_rewards.append(episode_rewards)
                break
            state = next_state

        # if (100 <= i_episode < 1000 and i_episode % 100 == 0) \
        #         or (i_episode >= 1000 and i_episode % 1000 == 0):
        if 100 <= i_episode and i_episode % 100 == 0:
            last_100_episode_rewards.append(episode_rewards)
            avg_reward = sum(last_100_episode_rewards) / len(last_100_episode_rewards)
            print(f'Episode {i_episode}/{num_episodes} || Best avg reward {avg_reward}')
            wandb.log({'avg_100': avg_reward})

    print()


def testing_after_learning(Q, mode):
    agent = Agent(Q, mode)
    n_tests = 1000
    total_test_rewards = []
    for episode in range(n_tests):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state)
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                total_test_rewards.append(episode_reward)
                break

            state = new_state

    print("avg: " + str(sum(total_test_rewards) / n_tests))


def _wrapper():
    with wandb.init() as run:
        config = wandb.config
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(
            Q, "mc_control",
            alpha=config.alpha, gamma=config.gamma, eps=config.eps,
            decay_method=config.decay_method, eps_decay=config.eps_decay
        )


Q = defaultdict(lambda: np.zeros(action_size))
while True:
    print()
    print("1. testing without learning")
    print("2. MC-control")
    print("3. q-learning")
    print("4. testing after learning")
    print("5. exit")
    try:
        menu = int(input("select: "))
    except ValueError:
        continue
    if menu == 1:
        testing_without_learning()
    elif menu == 2:
        wandb.agent(sweep_id, function=_wrapper, project='RL', count=30)
    elif menu == 3:
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(Q, "q_learning")
    elif menu == 4:
        testing_after_learning(Q, "test_mode")
    elif menu == 5:
        break
    else:
        print("wrong input!")
