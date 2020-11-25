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
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from threading import Thread
from Environment import *


REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16 # How many steps to use for training
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.75
MIN_REWARD = -200

EPISODES = 1000

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95  # 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10
#backend._SYMBOLIC_SCOPE.value = True

IM_WIDTH = 300 #640
IM_HEIGHT = 240 #480
INPUT_SHAPE = (IM_HEIGHT, IM_WIDTH, 3) #1 is single channel for grayscale, use 3 for rgb

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self, loadExistingModel=None):
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
    
        # Adding it to instance, becaause sess is used in train_in_loop() as well
        self.loadExistingModel = loadExistingModel
        if self.loadExistingModel:

            # adding session Ashwini
            self.sess = tf.Session().__enter__()
            self.graph = tf.compat.v1.get_default_graph()
            set_session(self.sess)

            self.model = self.load_model(loadExistingModel)
            self.target_model = self.load_model(loadExistingModel)

        else:

            self.graph = tf.compat.v1.get_default_graph()

            self.model = self.create_model()
            self.target_model = self.create_model()
        

        
        self.target_model.set_weights(self.model.get_weights())
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        return model  

    def create_model(self):

        # Anand: default weights was None. Using "imagenet" weights
        base_model = Xception(weights="imagenet", include_top=False,
                              input_shape=INPUT_SHAPE) 

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=["accuracy"])
        return model

    """
    def create_model(self):

        input = Input(shape=INPUT_SHAPE)

        cnn_1 = Conv2D(64, (7, 7), padding='same')(input)
        cnn_1a = Activation('relu')(cnn_1)
        cnn_1c = Concatenate()([cnn_1a, input])
        cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1c)

        cnn_2 = Conv2D(64, (5, 5), padding='same')(cnn_1ap)
        cnn_2a = Activation('relu')(cnn_2)
        cnn_2c = Concatenate()([cnn_2a, cnn_1ap])
        cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2c)

        cnn_3 = Conv2D(128, (5, 5), padding='same')(cnn_2ap)
        cnn_3a = Activation('relu')(cnn_3)
        cnn_3c = Concatenate()([cnn_3a, cnn_2ap])
        cnn_3ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_3c)

        cnn_4 = Conv2D(256, (5, 5), padding='same')(cnn_3ap)
        cnn_4a = Activation('relu')(cnn_4)
        cnn_4c = Concatenate()([cnn_4a, cnn_3ap])
        cnn_4ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_4c)

        cnn_5 = Conv2D(512, (3, 3), padding='same')(cnn_4ap)
        cnn_5a = Activation('relu')(cnn_5)
        #cnn_5c = Concatenate()([cnn_5a, cnn_4ap])
        cnn_5ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_5a)

        flatten = Flatten()(cnn_5ap)

        #return input, flatten

        predictions = Dense(3, activation="linear")(flatten) #grayscale. Dimensionality=1 or 3 for rgb
        model = Model(inputs=input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=["accuracy"])
        return model
    """

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0]
                                   for transition in minibatch])/255
        with self.graph.as_default():
            set_sess(self.sess)
            current_qs_list = self.model.predict(
                current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array(
            [transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            set_sess(self.sess)
            future_qs_list = self.target_model.predict(
                new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        #X = rgb2gray(X)
        
        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE,
                           verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32) #grayscale index 3 is 1 for grayscale and 3 for rgb
        y = np.random.uniform(size=(1, 3)).astype(np.float32) #grayscale (1,1), rgb (1,3)
        with self.graph.as_default():
            
            if self.loadExistingModel:
                set_session(self.sess)
            
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
