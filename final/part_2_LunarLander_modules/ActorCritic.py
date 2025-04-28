import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4, hidden_sizes=(256,256)):
        super().__init__()
        layers=[]
        last=obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last,h))
            layers.append(nn.Tanh())
            last=h
        self.trunk=nn.Sequential(*layers)
        self.policy_head=nn.Linear(last,n_actions)
        self.value_head=nn.Linear(last,1)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight,gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight,gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
    def forward(self,x):
        if x.dim()==1:
            x=x.unsqueeze(0)
        f=self.trunk(x)
        logits=self.policy_head(f)
        dist=Categorical(logits=logits)
        value=self.value_head(f).squeeze(-1)
        return dist,value