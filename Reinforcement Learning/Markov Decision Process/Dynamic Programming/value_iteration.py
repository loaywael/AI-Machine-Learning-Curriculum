import gym
import time
import numpy as np
import matplotlib.pyplot as plt


def improve_policy(env, V, gamma):
    """
    Selects the optimal probable action with the highest action value for each state

    :param env:
    :param V:
    :param gamma:
    :return: opt_policy: improved policy for each state

    """
    n_states, n_actions = env.nS, env.nA
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        # selects the highest action value and give it probability of 1
        q = action_value(env, V, s, gamma)
        policy[s, np.argmax(q)] = 1
    return policy


def action_value(env, V, s, gamma):
    """

    :param env:
    :param V:
    :param s:
    :param gamma:
    :return:
    """
    n_actions = env.nA
    q = np.zeros(n_actions)
    for a_i in range(n_actions):
        for trans_prop, nxt_state, reward, done in env.P[s][a_i]:
            q[a_i] += trans_prop * (reward + gamma * V[nxt_state])
    return q


def value_iteration(env, gamma=0.9, theta=1e-8):

    n_states, n_actions = env.nS, env.nA
    V = np.zeros(n_states)
    policy = np.zeros((n_states, n_actions))

    while True:
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            V[s] = max(action_value(env, V, s, gamma))
            delta = max(delta, abs(V[s] - v_old))
        if delta < theta:
            break

        policy = improve_policy(env, V, gamma)
    return V, policy


def plot_values(V):
    """
    Plots 4 x 4 grid of all states value
    :param V: states value vector

    """
    V_sq = V.reshape((4, 4))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)  # creates grid
    im = ax.imshow(V_sq, cmap="cool")
    for (j, i), label in np.ndenumerate(V_sq):
        # writing text centered in each state-square of the grid
        ax.text(i, j, np.round(label, 5), ha="center", va="center", fontsize=12)
    # disable all axes ticks
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title("State-Value")
    plt.show()


def main():
    env_name = "FrozenLake-v0"
    env = gym.make(env_name)
    V, policy = value_iteration(env, gamma=1.0)
    # print(policy)
    plot_values(V)


if __name__ == "__main__":
    main()

