import numpy as np


def policy_evaluation(env, policy, V, max_iter=10000, gamma=0.99, theta=1e-8):
    for i in range(max_iter):
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = value(env, V, s, np.argmax(policy[s]), gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def value(env, V, s, a, gamma):
    return sum([prob * (r + gamma * V[sp]) for prob, sp, r in env.MDP[s][a]])


def policy_improvement(env, policy, V, gamma=0.99):
    stable = True
    for s in range(env.nS):
        old = policy[s].copy()
        policy[s] = np.eye(env.nA)[np.argmax([value(env, V, s, a, gamma) for a in range(env.nA)])]
        if (old != policy[s]).any():
            stable = False
    return policy, stable


def policy_iteration(env, max_iter=10000, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    for i in range(max_iter):
        V = policy_evaluation(env, policy, V, max_iter, gamma, theta)
        policy, stable = policy_improvement(env, policy, V, gamma)
        if stable:
            break
    return policy, V


def value_iteration(env, max_iter=10000, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    for i in range(max_iter):
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max([value(env, V, s, a, gamma) for a in range(env.nA)])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    for s in range(env.nS):
        policy[s] = np.eye(env.nA)[np.argmax([value(env, V, s, a, gamma) for a in range(env.nA)])]
    return policy, V
