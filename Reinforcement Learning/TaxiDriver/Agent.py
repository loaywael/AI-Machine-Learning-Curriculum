import pickle
import numpy as np
from collections import defaultdict, deque


class Trainer:

    def __init__(self, env):
        self.env = env
        self.nS = self.env.nS
        self.epsilon = 0.0
        self.nA = self.env.nA

    def greedy_policy(self, q):
        state_policy = np.ones_like(q) * (self.epsilon/self.nA)
        state_policy[np.argmax(q)] = 1 - self.epsilon + (self.epsilon/self.nA)
        return state_policy

    def select_action(self, policy):
        action = np.random.choice(np.arange(self.nA), p=policy)
        return action

    def Q_learning(self, episodes=10000, alpha=0.01, gamma=0.9, plot_every=100):
        Q = defaultdict(lambda: np.zeros(self.nA))
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=episodes)
        for e in range(1, episodes + 1):
            score = 0
            self.epsilon = 1 / e
            if e % 1000 == 0:
                print(f"\rEpisode {e}/{episodes}", end="")
            state = self.env.reset()
            while True:
                policy = self.greedy_policy(Q[state])
                action = self.select_action(policy)
                nxt_state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    Q[state][action] += alpha*(reward - Q[state][action])
                    tmp_scores.append(score)
                    break
                Q[state][action] += alpha * (reward + gamma * (np.max(Q[nxt_state])) - Q[state][action])
                state = nxt_state
            if e % plot_every == 0:
                with open("Taxi_Q-Values.txt", "wb") as Qf:
                    pickle.dump([dict(Q), scores], Qf)
                scores.append(np.mean(tmp_scores))
        return Q, scores

