import gym
import numpy as np
import matplotlib.pyplot as plt


def make_env(env_name="Taxi-v2"):
    return gym.make(env_name)


def plot_env_value_map(V, map_size):
    map_values = np.reshape(V, map_size)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.imshow(map_values, cmap="cool")
    for (y, x), val in np.ndenumerate(map_values):
        ax.text(x, y, np.round(val, 3), ha="center", va="center", fontsize=14, color="w")
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.title("Map State-Values")
    plt.show()


def plot_reward_scores(scores, episodes, plot_every, axes_step=1.0, ):
    plt.plot(np.linspace(0, episodes, len(scores), endpoint=False), np.asarray(scores))
    plt.title(f"Best AVG Reward over {plot_every} Episodes: {np.max(scores)}")
    plt.xlabel("Episode")
    plt.ylabel("AVG Reward")
    plt.show()
