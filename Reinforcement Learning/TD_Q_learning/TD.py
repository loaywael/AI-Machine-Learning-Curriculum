import sys
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def greedy_policy(q, epsilon):
    policy_s = np.ones(len(q)) * epsilon/len(q)
    best_action = np.argmax(q)
    policy_s[best_action] = 1 - epsilon + (epsilon/len(q))
    return policy_s


def sarsa_prediction(env, policy, alpha=0.1, episodes=10000, gamma=1.0):
    V = defaultdict(float)

    for e in range(1, episodes + 1):     # iterate for episodes
        state = env.reset()
        if e % 100:
            print(f"\rEpisode {e}/{episodes}", end="")
            sys.stdout.flush()
        while True:
            action = policy[state]
            nxt_state, reward, done, _ = env.step(action)
            V[state] += alpha * (reward + (gamma*(V[nxt_state]) - V[state]))
            state = nxt_state
            if done:
                break
    return V


def sarsa_control(env, episodes=10000, alpha=0.01, gamma=1.0):
    n_actions = env.nA
    Q = defaultdict(lambda: np.zeros(n_actions))

    for e in range(1, episodes + 1):  # iterate for episodes
        if e % 100:
            print(f"\rEpisode {e}/{episodes}", end="")
            sys.stdout.flush()
        epsilon = 1/e
        state = env.reset()
        policy = greedy_policy(Q[state], epsilon)
        action = np.random.choice(n_actions, p=policy)
        while True:
            nxt_state, reward, done, _ = env.step(action)
            if not done:
                policy = greedy_policy(Q[nxt_state], epsilon)
                nxt_action = np.random.choice(n_actions, p=policy)
                Q[state][action] += alpha*(reward + (gamma*Q[nxt_state][nxt_action]) - Q[state][action])
                state, action = nxt_state, nxt_action
            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
                break
    return Q


def sarsamax_control(env, episodes=10000, alpha=0.01, gamma=1.0):
    n_actions = env.nA
    Q = defaultdict(lambda: np.zeros(n_actions))

    for e in range(1, episodes + 1):  # iterate for episodes
        if e % 100:
            print(f"\rEpisode {e}/{episodes}", end="")
            sys.stdout.flush()
        epsilon = 1/e
        state = env.reset()
        while True:
            policy = greedy_policy(Q[state], epsilon)
            action = np.random.choice(n_actions, p=policy)
            nxt_state, reward, done, _ = env.step(action)

            if not done:
                Q[state][action] += alpha*(reward + (gamma*np.max(Q[nxt_state]) - Q[state][action]))
                state = nxt_state
            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
                break
    return Q


def plot_state_value_map(V, map_size):
    map_values = np.reshape(V, map_size)
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    ax.imshow(map_values, cmap="cool")

    for (y, x), v in np.ndenumerate(map_values):
        ax.text(x, y, np.round(v, 3), fontsize=14, ha="center", va="center", color="w")
    plt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    plt.title("Map State-Values")
    plt.show()


def main():
    env = gym.make("CliffWalking-v0")
    directions = (np.ones(11), 2, 0, np.zeros(10), 2, 0, np.zeros(10), 2, 0, -1*np.ones(11))
    test_policy = np.hstack(directions)
    # V = td_state_value_prediction(env, test_policy)
    # Q = sarsa_control(env, 5000)
    Q = sarsamax_control(env, 5000)
    # V = [V[k] if k in Q else 0 for k in np.arange(48)]
    Q_plot = [np.max(Q[k]) if k in Q else 0 for k in np.arange(48)]

    def play_cliff(env):
        state = env.reset()
        rewards = 0
        while True:
            epsilon = 0.001
            policy = greedy_policy(Q[state], epsilon)
            action = np.random.choice(np.arange(env.nA), p=policy)
            nxt_state, reward, done, _ = env.step(action)
            rewards += reward
            state = nxt_state
            time.sleep(1)
            env.render()
            if done:
                break

    play_cliff(env)

    plot_state_value_map(Q_plot, (4, 12))


if __name__ == "__main__":
    main()
