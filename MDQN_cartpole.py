"""
Title: Deep Q-Learning for Atari Breakout
Author: [Jacob Chapman](https://twitter.com/jacoblchapman) and [Mathias Lechner](https://twitter.com/MLech20)
Date created: 2020/05/23
Last modified: 2020/06/17
Description: Play Atari Breakout with a Deep Q-Network.
"""
"""
## Introduction
This script shows an implementation of Deep Q-Learning on the
`BreakoutNoFrameskip-v4` environment.
### Deep Q-Learning
As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to an action. An agent will choose an action
in a given state based on a "Q-value", which is a weighted reward based on the
expected highest long-term reward. A Q-Learning Agent learns to perform its
task such that the recommended action maximizes the potential future rewards.
This method is considered an "Off-Policy" method,
meaning its Q values are updated assuming that the best action was chosen, even
if the best action was not chosen.
### Atari Breakout
In this environment, a board moves along the bottom of the screen returning a ball that
will destroy blocks at the top of the screen.
The aim of the game is to remove all blocks and breakout of the
level. The agent must learn to control the board by moving left and right, returning the
ball and removing all the blocks without the ball passing the board.
### Note
The Deepmind paper trained for "a total of 50 million frames (that is, around 38 days of
game experience in total)". However this script will give good results at around 10
million frames which are processed in less than 24 hours on a modern machine.
### References
- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)
- [Deep Q-Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)
"""
"""
## Setup
"""
import os
import sys
import time

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import sys

# Configuration paramaters for the whole setup
seed = int(sys.argv[1])
gamma = 1.  # Discount factor for past rewards
epsilon = tf.Variable(1.0, trainable=False)  # Epsilon greedy parameter
epsilon.assign(1.0)
epsilon_min = 0.02  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer

max_steps_per_episode = 10000
epsilon_greedy_decay_prop = 0.1 # proprtion of total timesteps over which epsilon is decayed
exploration_proportion = 0.001 # proportion of total timesteps in which agent engages in pure exploration
total_timesteps = 200000
num_actions = 2
timeout_steps = 200
train = True
neg_fall_reward = 5
env_string = "CartPole-v0"
exp_name = "100k_openai"
DDQN = False
MDQN = False
SDQN = False
SDQN_2 = True
tau = 0.03
alpha = 0.9
l_0 = -1.

# Use the Baseline Atari environment because of Deepmind helper functions
env = gym.make(env_string)
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
#env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

"""
## Implement the Deep Q-Network
This network learns an approximation of the Q-table, which is a mapping between
the states and actions that an agent will take. For every state we'll have four
actions, that can be taken. The environment provides the state, and the action
is chosen by selecting the larger of the four Q-values predicted in the output layer.
"""


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def create_q_mlp(input_shape=2, num_layers=2, num_hidden=64, activation=tf.tanh, hidden_units=256, num_actions=2):
    x_input = tf.keras.Input(shape=input_shape)
    # h = tf.keras.layers.Flatten(x_input)
    h = x_input
    for i in range(num_layers):
        h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)),
                                  name='mlp_fc{}'.format(i), activation=activation)(h)
    latent = tf.keras.layers.Flatten()(h)
    action_out = tf.keras.layers.Dense(units=hidden_units, activation="relu")(latent)
    q_out = tf.keras.layers.Dense(units=num_actions, activation=None)(action_out)

    network = tf.keras.Model(inputs=[x_input], outputs=[q_out])
    return network

def simulate(env_string="CartPole-v1", num_episodes=1000, exp_name=exp_name):
    model = create_q_mlp(input_shape=4, num_layers=2, num_hidden=64, num_actions=2)
    model_target = create_q_mlp(input_shape=4, num_layers=2, num_hidden=64, num_actions=2)
    env = gym.make(env_string)

    checkpoints_dir = os.path.join(".", "checkpoints", env_string, exp_name)
    checkpoint = tf.train.Checkpoint(model=model, model_target=model_target)
    manager = tf.train.CheckpointManager(checkpoint, checkpoints_dir, max_to_keep=40)

    if not manager.latest_checkpoint:
        print("No saved model")
        return

    for episode in range(num_episodes):
        state = np.array(env.reset())
        done = False
        ep_reward = 0

        while not done:
            env.render()

            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            q_tensor = model_target.predict(state_tensor)

            # Take best action
            action = tf.argmax(q_tensor[0]).numpy()
            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)
            ep_reward += reward

            state = state_next

if train:
    # The first model makes the predictions for Q-values which are used to
    # make a action.
    model = create_q_mlp(input_shape=4, num_layers=2, num_hidden=64, num_actions=2)
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = create_q_mlp(input_shape=4, num_layers=2, num_hidden=64, num_actions=2)


    """
    ## Train
    """
    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = keras.optimizers.Adam(learning_rate=1.e-3)

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    pole_fall_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    total_timestep_count = tf.Variable(0, trainable=False)
    total_timestep_count.assign(0) # sugar for chekcpointing
    # Number of frames to take random action and observe output
    exploration_timesteps = int(exploration_proportion * total_timesteps)
    print("Number of pure exploration timesteps: {}".format(exploration_timesteps))
    # Number of frames for exploration
    epsilon_greedy_timesteps = int(epsilon_greedy_decay_prop * total_timesteps)
    print("Number of timesteps over which epsilon will be decayed: {}".format(epsilon_greedy_timesteps))
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 50000
    # Train the model after 1 actions
    update_after_actions = 1
    # How often to update the target network
    update_target_network = 500
    # From OpenAI Baseline
    grad_norm_clipping = 10
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    if DDQN:
        arch = "DDQN"
    elif MDQN:
        arch = "MDQN"
    elif SDQN:
        arch = "SDQN"
    elif SDQN_2:
        arch = "SDQN_2"
    else:
        arch = "DQN"

    checkpoints_dir = os.path.join(".", "checkpoints", env_string, str(seed), exp_name, arch)
    checkpoint = tf.train.Checkpoint(model=model, model_target=model_target,
                                     epsilon=epsilon, total_timestep_count=total_timestep_count)
    manager = tf.train.CheckpointManager(checkpoint, checkpoints_dir, max_to_keep=40)
    checkpoint_timesteps = 1000

    checkpoint.restore(manager.latest_checkpoint)
    epsilon = epsilon.assign_add(0.).numpy()
    total_timestep_count = total_timestep_count.assign_add(0).numpy()

    start_time = time.time()
    while total_timestep_count < total_timesteps:
        state = np.array(env.reset())
        episode_reward = 0

        done = False
        for timestep in range(1, max_steps_per_episode):
            #env.render() # Adding this line would show the attempts
            # of the agent in a pop up window.
            total_timestep_count += 1

            if total_timestep_count % checkpoint_timesteps == 0:
                # print("Time elapsed: {:.1f}. Total number of frames: {}, epsilon:{:.1f}".format(time.time() - start_time,
                #                                                                                 total_timestep_count, epsilon))
                manager.save()

            # Use epsilon-greedy for exploration
            if total_timestep_count < exploration_timesteps or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                q_values_rep_buf = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(q_values_rep_buf[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_timesteps
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            if done and timestep < timeout_steps:
                pole_fall_history.append(True)
                # print(done, True)
            else:
                pole_fall_history.append(False)
                # print(done, False)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if total_timestep_count % update_after_actions == 0 and len(done_history) > batch_size:
                #print("here")

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices])
                pole_fall_sample = tf.convert_to_tensor(
                    [float(pole_fall_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards_targ_m = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward

                if not DDQN:
                    if SDQN:
                        pi_theta_ns = tf.keras.layers.Softmax(axis=1)(future_rewards_targ_m)

                        updated_q_values = rewards_sample + \
                                           gamma * tf.reduce_sum(
                            pi_theta_ns * (future_rewards_targ_m - tau * tf.math.log(pi_theta_ns)), axis=1)

                    elif SDQN_2:
                        tau = (1. - alpha) * tau
                        pi_theta_ns = tf.keras.layers.Softmax(axis=1)(future_rewards_targ_m)

                        updated_q_values = rewards_sample + \
                                           gamma * tf.reduce_sum(pi_theta_ns * (future_rewards_targ_m - tau * tf.math.log(pi_theta_ns)), axis=1)

                    elif MDQN:
                        pi_theta_ns = tf.keras.layers.Softmax(axis=1)(future_rewards_targ_m)
                        pi_theta = tf.keras.layers.Softmax(axis=1)(model_target.predict(state_sample))
                        unclipped_scaled_log_policy = tau * tf.math.log(tf.reduce_sum(tf.one_hot(action_sample, num_actions) * pi_theta, axis=1))

                        updated_q_values = rewards_sample + \
                                           alpha * tf.clip_by_value(unclipped_scaled_log_policy, l_0, 0) + \
                                           gamma * tf.reduce_sum(pi_theta_ns * (future_rewards_targ_m - tau * tf.math.log(pi_theta_ns)), axis=1)

                    else:
                        updated_q_values = rewards_sample + gamma * tf.reduce_max(
                            future_rewards_targ_m, axis=1)

                else:
                    future_rewards = model.predict(state_next_sample)
                    future_rewards_argmax = tf.argmax(future_rewards, axis=1)
                    future_rewards_argmax_onehot = tf.one_hot(future_rewards_argmax, num_actions, dtype=tf.float32)
                    updated_q_values = rewards_sample + gamma * tf.reduce_sum(future_rewards_argmax_onehot * future_rewards_targ_m, axis=1)

                # If done set target to zero
                updated_q_values = updated_q_values * (1. - done_sample)

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                clipped_grads = []
                for grad in grads:
                    clipped_grads.append(tf.clip_by_norm(grad, grad_norm_clipping))
                clipped_grads = grads
                grads_and_vars = zip(grads, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if total_timestep_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, cumulative timesteps {}"
                print(template.format(running_reward, episode_count, total_timestep_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                print("Time elapsed: {:.1f}. Cumulative timestep: {}. Epsilon: {:.4f}. Episode reward: {}"
                      .format(time.time() - start_time, total_timestep_count, epsilon, episode_reward))
                print("model_layer_weights", model.get_weights()[0][0][0])
                if len(done_history) > batch_size:
                    print("Average q_value: {:.1f}".format(tf.reduce_mean(q_values).numpy()))
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

else:
    simulate()