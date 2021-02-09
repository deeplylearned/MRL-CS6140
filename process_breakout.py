import re
import numpy as np
import matplotlib.pyplot as plt


def process_file(fname):
    rewards = []
    episodes = []

    with open(fname, 'r') as fh:
        all_lines = fh.readlines()

    for line in all_lines:
        if ("running reward") in line:
            match = re.search(r"reward: (\d+\.\d+)", line)
            reward = float(match.group(1))
            rewards.append(reward)

            match2 = re.search(r"episode (\d+)", line)
            episode = float(match2.group(1))
            episodes.append(episode)

    return np.array(episodes), np.array(rewards)

episodes_dqn, rewards_dqn = process_file("keras_baseline_breakoutv4.log")
episodes_ddqn, rewards_ddqn = process_file("keras_baseline_breakoutv4_ddqn.log")
episodes_mdqn, rewards_mdqn = process_file("keras_baseline_breakoutv4_mdqn.log")
episodes_sdqn, rewards_sdqn = process_file("keras_baseline_breakoutv4_sdqn.log")
episodes_sdqn2, rewards_sdqn2 = process_file("keras_baseline_breakoutv4_sdqn2.log")

episodes_mdqn = episodes_mdqn + 8000

fig1, ax1 = plt.subplots()

ax1.plot(episodes_dqn, rewards_dqn, label="DQN", color="C0")
ax1.plot(episodes_ddqn, rewards_ddqn, label="DDQN", color="C1")
ax1.plot(episodes_mdqn, rewards_mdqn, label="MDQN*", color="C2")

ax1.set_xlabel("Episode")
ax1.set_ylabel("Running Average Reward (100 steps)")
ax1.axhline(40, marker="_", color='r', label="target")
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.plot(episodes_dqn, rewards_dqn, label="DQN", color="C0")
ax2.plot(episodes_sdqn, rewards_sdqn, label=r"SDQN($\tau$)", color="C4")
ax2.plot(episodes_sdqn2, rewards_sdqn2, label=r"SDQN$(1-\alpha)\tau$", color="C5")
ax2.plot(episodes_mdqn, rewards_mdqn, label="MDQN*", color="C2")

ax2.set_xlabel("Episode")
ax2.set_ylabel("Running Average Reward (100 steps)")
ax2.axhline(40, marker="_", color='r', label="target")
ax2.legend()