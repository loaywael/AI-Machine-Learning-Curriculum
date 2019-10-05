import gym
import time
import numpy as np
import matplotlib.pyplot as plt


# Policy Evaluation Strategy
########################################
# Decide >>> Act >>> Evaluate >> Improve
########################################


def sample_policy(env):
    """
    Returns policy of uniform random actions probability
    :param env: Environment instance
    :return: policy A(s): P(a|s)
    """
    return np.ones((env.nS, env.nA)) / env.nA


def iterative_policy_evaluation(env, policy, gamma=0.9, theta=1e-8):
    """
    Evaluates state quality value of all states following a given policy

    :param env: Environment Instance
    :param policy: actions probability for each state P[s][a]
    :param gamma: future rewards discounting factor
    :param theta: maximum change of all state value
    :return: predicted state value of all states (V) of a given policy

    """
    n_states = env.nS
    V = np.zeros(shape=n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            V_old = V[s]
            for a_i, a_prop in enumerate(policy[s]):
                for trans_prop, nxt_state, reward, done in env.P[s][a_i]:
                    v += a_prop * trans_prop * (reward + gamma*V[nxt_state])
            V[s] = v
            delta = max(delta, abs(V[s] - V_old))
        if delta < theta:
            break
    return V


def truncated_iterative_policy_eval(env, policy, gamma, theta, max_iter=100):
    """

    :param env:
    :param policy:
    :param gamma:
    :param theta:
    :param max_iter:
    :return:
    """
    counter = 0
    n_states, n_actions = env.nS, env.nA
    V = np.zeros(n_states)

    while counter < max_iter:
        delta = 0
        counter += 1
        for s in range(n_states):
            v = 0
            v_old = V[s]
            for a_i, a_prop in enumerate(policy[s]):
                for trans_prop, nxt_state, reward, done in env.P[s][a_i]:
                    v += a_prop * trans_prop * (reward + gamma*V[nxt_state])
            V[s] = v
            delta = max(delta, abs(V[s] - v_old))
        if delta < theta:
            break
    return V


def action_value(env, V, gamma=0.9):
    """
    Evaluates how good each action selected by the policy by using the
    previous state value (max-rewards obtained from previous state) as a reference.

    :param env:
    :param V:
    :param gamma:
    :return:

    """
    n_states, n_actions = env.nS, env.nA
    Q = np.zeros((n_states, n_actions))

    for s in range(n_states):
        v = 0
        for a in range(n_actions):
            for trans_prob, nxt_state, reward, done in env.P[s][a]:
                Q[s, a] += trans_prob * (reward + gamma*V[nxt_state])
    return Q


def improve_policy(env, Q):
    """
    Selects the optimal probable action with the highest action value for each state
    :param env:
    :param Q:
    :return: opt_policy: improved policy for each state

    """
    n_states, n_actions = env.nS, env.nA
    opt_policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        # selects the highest action value and give it probability of 1
        opt_policy[s, np.argmax(Q[s])] = 1
    return opt_policy


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


def policy_iteration(env, gamma=1.0, theta=1e-8, max_iter=100):
    """
    Iteratively improve the policy and state values by selecting optimum action of last iteration
    Waits until the value function yields semi perfect state values before improving the policy,
    which may take longer time and inefficient for large states usually depends on theta
    small theta: means wait longer until perfect convergence
    big theta: means wait shorter perfect values not required

    :param env:
    :param gamma:
    :param theta:
    :param max_iter:

    :return:

    """
    policy = sample_policy(env)

    while True:
        V = truncated_iterative_policy_eval(env, policy, gamma, theta, max_iter)
        Q = action_value(env, V, gamma)
        opt_policy = improve_policy(env, Q)
        if (opt_policy >= policy).all():
            break
        policy = opt_policy
    return policy, V, Q


def main():
    env_name = "FrozenLake-v0"
    env = gym.make(env_name)
    init = env.reset()
    policy, V, Q = policy_iteration(env, gamma=1)
    observation, reward, done, info = env.step()
    env.
    print(policy)
    plot_values(V)


if __name__ == "__main__":
    main()
