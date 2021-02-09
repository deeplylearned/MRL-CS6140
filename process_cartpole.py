import re
import numpy as np
import matplotlib.pyplot as plt
import glob

def process_file(fname):
    rewards = []

    with open(fname, 'r') as fh:
        all_lines = fh.readlines()

    for line in all_lines:
        if ("Episode reward") in line:
            match = re.search(r"reward: (\d+\.\d+)", line)
            reward = float( match.group(1))
            rewards.append(reward)
    rewards = np.array(rewards)

    running_avg = []
    for idx in range(100, len(rewards)):
        running_avg.append(rewards[idx - 100: idx].mean())
    running_avg = np.array(running_avg)

    return running_avg

def process_trials(trials):
    min_length = len(trials[0])
    for trial in trials[1:]:
        if len(trial) < min_length:
            min_length = len(trial)

    trunc_trials = []
    for trial in trials:
        trunc_trials.append(trial[:min_length])
    trunc_trials = np.array(trunc_trials)

    return np.average(trunc_trials, axis=0), np.std(trunc_trials, axis=0)


MDQN_files = glob.glob("logs/MDQN*.txt")
DQN_files = glob.glob("logs/DQN*.txt")
DDQN_files = glob.glob("logs/DDQN*.txt")
SDQN_files = glob.glob("logs/SDQN_*.txt")
SDQN2_files = glob.glob("logs/SDQN2_*.txt")

MDQN_trials = []
DQN_trials = []
DDQN_trials = []
SDQN_trials = []
SDQN2_trials = []

for f in MDQN_files:
    MDQN_trials.append(process_file(f))
for f in DQN_files:
    DQN_trials.append(process_file(f))
for f in DDQN_files:
    DDQN_trials.append(process_file(f))
for f in SDQN_files:
    SDQN_trials.append(process_file(f))
for f in SDQN2_files:
    SDQN2_trials.append(process_file(f))

rewards_mdqn_av, rewards_mdqn_std = process_trials(MDQN_trials)
rewards_ddqn_av, rewards_ddqn_std = process_trials(DDQN_trials)
rewards_dqn_av, rewards_dqn_std = process_trials(DQN_trials)
rewards_sdqn_av, rewards_sdqn_std = process_trials(SDQN_trials)
rewards_sdqn2_av, rewards_sdqn2_std = process_trials(SDQN2_trials)
#rewards_norm_done = np.array(process_file("checkpoints/CartPole-v0/100k_openai/DDQN/log.txt"))
#rewards_ddqn = np.array(process_file("checkpoints/CartPole-v0/100k_openai/DDQN/log.txt"))
# rewards_not_norm_done = np.array(process_file("not_norm_done.txt"))

# print("Norm:", np.array(rewards_norm_done)[700:].mean())
# print("Not norm:", np.array(rewards_not_norm_done)[700:].mean())

#plt.plot(np.arange(1, len(rewards_mdqn) + 1), rewards_mdqn, label="mdqn")
#plt.plot(np.arange(1, len(rewards_ddqn) + 1), rewards_ddqn, label="ddqn")

fig1, ax1 = plt.subplots()

ax1.plot(np.arange(100, len(rewards_mdqn_av)+100), rewards_mdqn_av, label="MDQN", color='C2')
ax1.plot(np.arange(100, len(rewards_ddqn_av)+100), rewards_ddqn_av, label="DDQN", color='C1')
ax1.plot(np.arange(100, len(rewards_dqn_av)+100), rewards_dqn_av, label="DQN", color='C0')

ax1.fill_between(np.arange(100, len(rewards_mdqn_av)+100), rewards_mdqn_av - 1.96*rewards_mdqn_std/np.sqrt(5),
                 rewards_mdqn_av + 1.96*rewards_mdqn_std/np.sqrt(5), alpha=0.07, color="C2")
ax1.fill_between(np.arange(100, len(rewards_ddqn_av)+100), rewards_ddqn_av - 1.96*rewards_ddqn_std/np.sqrt(5),
                 rewards_ddqn_av + 1.96*rewards_ddqn_std/np.sqrt(5), alpha=0.07, color="C1")
ax1.fill_between(np.arange(100, len(rewards_dqn_av)+100), rewards_dqn_av - 1.96*rewards_dqn_std/np.sqrt(5),
                 rewards_dqn_av + 1.96*rewards_dqn_std/np.sqrt(5), alpha=0.07, color="C0")


ax1.axhline(rewards_mdqn_av.mean(), label="MDQN avg", color='C2', ls = '--')
ax1.axhline(rewards_ddqn_av.mean(), label="DDQN avg", color='C1', ls = '--')
ax1.axhline(rewards_dqn_av.mean(), label="DQN avg", color='C0', ls = '--')

ax1.set_xlabel("Episode")
ax1.set_ylabel("Running Average Reward (100 steps)")
ax1.axhline(195, label="target", color='r')
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(100, len(rewards_mdqn_av)+100), rewards_mdqn_av, label="MDQN", color='C2')
ax2.plot(np.arange(100, len(rewards_dqn_av)+100), rewards_dqn_av, label="DQN", color='C0')
ax2.plot(np.arange(100, len(rewards_sdqn_av)+100), rewards_sdqn_av, label=r"SDQN($\tau$)", color='C4')
ax2.plot(np.arange(100, len(rewards_sdqn2_av)+100), rewards_sdqn2_av, label=r"SDQN($(1-\alpha)\tau$)", color='C5')

ax2.fill_between(np.arange(100, len(rewards_mdqn_av)+100), rewards_mdqn_av - 1.96*rewards_mdqn_std/np.sqrt(5),
                 rewards_mdqn_av + 1.96*rewards_mdqn_std/np.sqrt(5), alpha=0.07, color="C2")
ax2.fill_between(np.arange(100, len(rewards_dqn_av)+100), rewards_dqn_av - 1.96*rewards_dqn_std/np.sqrt(5),
                 rewards_dqn_av + 1.96*rewards_dqn_std/np.sqrt(5), alpha=0.07, color="C0")
ax2.fill_between(np.arange(100, len(rewards_sdqn_av)+100), rewards_sdqn_av - 1.96*rewards_sdqn_std/np.sqrt(5),
                 rewards_sdqn_av + 1.96*rewards_sdqn_std/np.sqrt(5), alpha=0.07, color="C4")
ax2.fill_between(np.arange(100, len(rewards_sdqn2_av)+100), rewards_sdqn2_av - 1.96*rewards_sdqn2_std/np.sqrt(5),
                 rewards_sdqn2_av + 1.96*rewards_sdqn2_std/np.sqrt(5), alpha=0.07, color="C5")

ax2.axhline(rewards_mdqn_av.mean(), label="MDQN avg", color='C2', ls = '--')
ax2.axhline(rewards_dqn_av.mean(), label="DQN avg", color='C0', ls = '--')
ax2.axhline(rewards_sdqn_av.mean(), label=r"SDQN($\tau$) avg", color='C4', ls = '--')
ax2.axhline(rewards_sdqn2_av.mean(), label=r"SDQN($(1-\alpha)\tau$) avg", color='C5', ls = '--')

ax2.set_xlabel("Episode")
ax2.set_ylabel("Running Average Reward (100 steps)")
ax2.axhline(195, label="target", color='r')
ax2.legend()