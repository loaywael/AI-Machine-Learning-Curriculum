import time
import pickle
import numpy as np
from Agent import Trainer
from Environment import make_env
from Environment import plot_env_value_map
from Environment import plot_reward_scores


def play_game(env, agent, Q):
    state = env.reset()
    score = 0
    while True:
        opt_policy = agent.greedy_policy(Q[state])
        action = agent.select_action(opt_policy)
        nxt_state, reward, done, _ = env.step(action)
        state = nxt_state
        score += reward
        time.sleep(1)
        env.render()
        if done:
            break
    print(f"total reward: {score}")


episodes = 500000
plot_every = 100
taxi_env = make_env("Taxi-v3")
driver = Trainer(taxi_env)
# Q, scores = driver.Q_learning(episodes)

with open("Taxi_Q-Values.txt", "rb") as Qf:
    Q, scores = pickle.load(Qf)

V = [np.max(Q[k]) if k in Q else 0 for k in np.arange(25)]
play_game(taxi_env, driver, Q)
plot_env_value_map(V, (5, 5))
plot_reward_scores(scores, episodes, plot_every)
