import gymnasium as gym
import numpy as np
import torch

def worker(proc_id,exp_queue,global_model,num_steps):
    torch.manual_seed(proc_id)
    env=gym.make('LunarLander-v3',continuous=False)
    state=env.reset()[0] if isinstance(env.reset(),tuple) else env.reset()
    while True:
        states=[]
        actions=[]
        rewards=[]
        dones=[]
        for _ in range(num_steps):
            arr=np.asarray(state,dtype=np.float32)
            states.append(arr)
            dist,_=global_model(torch.from_numpy(arr).unsqueeze(0))
            action=dist.sample().item()
            step=env.step(action)
            if len(step)==5:
                nxt,r,term,trunc,_=step
                done=term or trunc
            else:
                nxt,r,done,_=step
            rewards.append(r)
            dones.append(done)
            actions.append(action)
            if done:
                state=env.reset()[0] if isinstance(env.reset(),tuple) else env.reset()
            else:
                state=nxt
        last_arr=np.asarray(state,dtype=np.float32)
        with torch.no_grad():
            _,last_value=global_model(torch.from_numpy(last_arr).unsqueeze(0))
        exp_queue.put((states,actions,rewards,dones,float(last_value.item())))