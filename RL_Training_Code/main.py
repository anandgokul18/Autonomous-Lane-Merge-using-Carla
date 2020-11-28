import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from Environment import *
from Model import *

from tqdm import tqdm

if __name__ == '__main__':

    FPS = 60  # 60

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.compat.v1.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent(loadExistingModel=None)
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))  # grayscale

    # For stats
    # Iterate over episodes
    scores = []  # ep_rewards = [-200]
    avg_scores = []

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        env.collision_hist = []

        #env.first_lane_change_on_freeway = True
        env.lane_crossings = []

        # Restarting episode - reset episode reward and step number
        score = 0  # sentdex: episode_reward
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action every epsilon to try out new moves... we have 4 actions allowed... CHANGE IF MORE ACTIONS
                action = np.random.randint(0, 4)
                # action = random.choices(population=[0,1,2,3,4],weights=[0.2,0.3,0.2,0.2,0.1],k=1)[0] # Logic: Going straight is more important
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            score += reward

            # Every step we update replay memory
            agent.update_replay_memory(
                (current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        scores.append(score)
        avg_scores.append(np.mean(scores[-10:]))

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            avg_scores.append(np.mean(scores[-AGGREGATE_STATS_EVERY:]))
            average_reward = sum(
                scores[-AGGREGATE_STATS_EVERY:])/len(scores[-AGGREGATE_STATS_EVERY:])
            min_reward = min(scores[-AGGREGATE_STATS_EVERY:])
            max_reward = max(scores[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(
                reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value #sentdex: min_reward...anand: average_reward
            if average_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            else:
                print(
                    f"[LOG] Average reward was {average_reward}, min reward was {min_reward} AND max_reward was {max_reward} ... Not saving")

        print('episode: ', episode, 'score %.2f' % score)
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(
        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
