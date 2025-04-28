def calculate_returns(rewards,dones,last_value,gamma=0.99):
    R=last_value
    out=[]
    for r,d in zip(reversed(rewards),reversed(dones)):
        R=r+gamma*R*(1.0-float(d))
        out.insert(0,R)
    return out