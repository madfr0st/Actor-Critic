import gymnasium,numpy as np,torch
from gymnasium.wrappers import AtariPreprocessing
import ale_py

from final.part_3__Pong_v5_modules.FrameStack import FrameStack


def worker_loop(pid,exp_q,global_model,rollout_steps,device):
    torch.manual_seed(pid)
    env=gymnasium.make('ALE/Pong-v5')
    env=AtariPreprocessing(env,screen_size=84,grayscale_obs=True,frame_skip=1)
    env=FrameStack(env,4)
    while True:
        obs_buf,act_buf,rew_buf,done_buf=[],[],[],[]
        obs,_=env.reset()
        for _ in range(rollout_steps):
            inp=torch.from_numpy(obs.astype(np.float32)/255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                dist,_=global_model(inp)
            a=dist.sample().item()
            nxt,r,term,trunc,_=env.step(a)
            done=term or trunc
            obs_buf.append(obs);act_buf.append(a);rew_buf.append(r);done_buf.append(done)
            obs=nxt if not done else env.reset()[0]
        inp=torch.from_numpy(obs.astype(np.float32)/255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            _,lv=global_model(inp)
        exp_q.put((obs_buf,act_buf,rew_buf,done_buf,float(lv.item())))