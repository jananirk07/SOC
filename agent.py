import numpy as np
import random
from collections import deque
from helper import KungFu
import tensorflow as tf

class DQNAgent:
    def __init__(self, action_size, learn_rate=0.00025, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.999995, batch_size=32, target_update_freq=10000):
        
        self.action_size = action_size
        self.learn_rate = learn_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.possible_actions = list(range(self.action_size))

        self.model = KungFu(self)
        self.target_model = KungFu(self)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.possible_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def train(self, memory):
        if len(memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(q_next[i])
            q_values[i][actions[i]] = target

        self.model.fit(states, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
