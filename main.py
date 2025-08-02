import gymnasium as gym
import numpy as np
import cv2
import os
from collections import deque
from memory import ReplayMemory
from agent import DQNAgent
from gymnasium.wrappers import RecordVideo

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def stack_frames(stacked_frames, frame, is_new_episode):
    processed = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([processed] * 4, maxlen=4)
    else:
        stacked_frames.append(processed)
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

ENV_NAME = "ALE/KungFuMaster-v5"
EPISODES = 2000
REPLAY_SIZE = 100000
BATCH_SIZE = 32
SAVE_EVERY = 500
VIDEO_DIR = "videos"
MODEL_DIR = "models"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env = gym.make(ENV_NAME, render_mode="rgb_array")
env = RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda x: x % 100 == 0)
action_size = env.action_space.n

agent = DQNAgent(action_size=action_size)
memory = ReplayMemory(REPLAY_SIZE)

stacked_frames = deque(maxlen=4)

for episode in range(1, EPISODES + 1):
    obs, _ = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, obs, True)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_obs, reward, done, truncated, _ = env.step(action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_obs, False)
        memory.push(state, action, reward, next_state, done)
        agent.train(memory)
        state = next_state
        total_reward += reward

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    if episode % SAVE_EVERY == 0:
        agent.model.save(os.path.join(MODEL_DIR, f"dqn_episode_{episode}.h5"))

env.close()
