import gym
import numpy as np
import matplotlib.pyplot as plt

# LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3
# SFFF
# FHFH
# FFFH
# HFFG

policy = {0:2, 1: 2, 2: 1, 3: 0, 4:1, 6:1, 8: 2, 9: 2, 10:1, 13: 2, 14: 2}

env = gym.make('FrozenLake-v0')

n_games = 1000
win_pct = []
scores = []


for i_episode in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = policy[observation]
        observation, reward, done, info = env.step(action)
        score += reward
    scores.append(score)

    if (i_episode % 10) == 0:
        win_pct.append(np.mean(scores[-10:]))
    
env.close()

plt.plot(win_pct)
plt.show()