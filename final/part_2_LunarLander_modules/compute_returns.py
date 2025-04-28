import torch
import numpy as np

def compute_returns(rewards,dones,last_state,model,gamma):
    if isinstance(last_state,(float,int)):
        R=float(last_state)
    else:
        arr=np.asarray(last_state,dtype=np.float32)
        with torch.no_grad():
            _,v=model(torch.from_numpy(arr).unsqueeze(0))
        R=float(v.item())
    returns=[]
    for r,d in zip(reversed(rewards),reversed(dones)):
        R=r+gamma*R*(1.0-float(d))
        returns.insert(0,R)
    return returns