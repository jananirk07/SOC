import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, frame_height=84, frame_width=84, stack_size=4):
        self.capacity = capacity
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.stack_size = stack_size

        self.states = np.zeros((capacity, frame_height, frame_width, stack_size), dtype=np.uint8)
        self.next_states = np.zeros((capacity, frame_height, frame_width, stack_size), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.index = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices])

    def __len__(self):
        return self.size
