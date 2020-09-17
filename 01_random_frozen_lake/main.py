import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

n_games = 1000
win_pct = []
scores = []


for i_episode in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward
    scores.append(score)

    if (i_episode % 10) == 0:
        win_pct.append(np.mean(scores[-10:]))
    
env.close()

plt.plot(win_pct)
plt.show()