import logging
from collections import deque
from typing import Deque

import gymnasium as gym
import numpy as np
import torch
from torch.nn import Module


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def worker_loop(
    proc_id: int,
    exp_queue,
    global_model: Module,
    rollout_steps: int = 2048
) -> None:
    torch.manual_seed(proc_id)
    env = gym.make("BipedalWalker-v3", render_mode=None)
    obs, _ = env.reset()

    while True:
        obs_buffer: Deque = deque()
        action_buffer: Deque = deque()
        reward_buffer: Deque = deque()
        done_buffer: Deque = deque()

        episode_reward, episode_steps = 0.0, 0

        for _ in range(rollout_steps):
            state = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist, value = global_model(state)
                action = dist.sample()[0].clamp(-1.0, 1.0)
            next_obs, reward, terminated, truncated, _ = env.step(action.numpy())

            episode_reward += reward
            episode_steps  += 1
            done = terminated or truncated

            obs_buffer.append(obs)
            action_buffer.append(action.numpy())
            reward_buffer.append(reward)
            done_buffer.append(done)

            if done:
                obs, _ = env.reset()
                episode_reward, episode_steps = 0.0, 0
            else:
                obs = next_obs

        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            _, last_value = global_model(state)

        exp_queue.put((
            list(obs_buffer),
            list(action_buffer),
            list(reward_buffer),
            list(done_buffer),
            float(last_value.item())
        ))