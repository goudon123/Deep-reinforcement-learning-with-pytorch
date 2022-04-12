import matplotlib.pyplot as plt
import pandas as pd

def smooth(scalars, weight=0.8):  
    # Weight between 0 and 1
    last = scalars[0] 
    smoothed = list()
    for i, point in enumerate(scalars):
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)                        
        last = smoothed_val                                  

    return smoothed


# Load data
ppo = pd.read_csv('run-.-tag-score_score_ppo.csv')
ddpg = pd.read_csv('run-.-tag-score_score_ddpg.csv')
sac = pd.read_csv('run-.-tag-score_score_sac.csv')

# Plot
#plt.figure(figsize=(6.8, 4.2))
x = range(len(ppo['step']))
y_ppo = ppo['value']
y_ddpg = ddpg['value']
y_sac = sac['value']


smooth_ppo = smooth(y_ppo)
smooth_ddpg = smooth(y_ddpg)
smooth_sac = smooth(y_sac)

plt.plot(x, smooth_ppo,color='b')
plt.plot(x, smooth_ddpg,color='g')
plt.plot(x, smooth_sac,color='r')

plt.plot(x, y_ppo,color='b',alpha=0.4)
plt.plot(x, y_ddpg,color='g',alpha=0.4)
plt.plot(x, y_sac,color='r',alpha=0.4)
plt.legend(['PPO','DDPG','SAC'])

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
