import numpy as np, gymnasium as gym
from collections import deque
from gymnasium.spaces import Box

class FrameStack(gym.Wrapper):
    def __init__(self,env,k):
        super().__init__(env)
        self.k=k
        h,w=env.observation_space.shape
        self.frames=deque(maxlen=k)
        self.observation_space=Box(0,255,shape=(k,h,w),dtype=np.uint8)
    def reset(self,**kw):
        obs=self.env.reset(**kw)[0]
        for _ in range(self.k):self.frames.append(obs)
        return self._get_obs(),{}
    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(),reward,terminated,truncated,info
    def _get_obs(self):
        return np.stack(self.frames,axis=0)