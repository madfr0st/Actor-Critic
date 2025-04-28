import torch, torch.nn as nn
from torch.distributions import Categorical

class A2CActorCritic(nn.Module):
    def __init__(self,num_actions=6,input_channels=4):
        super().__init__()
        self.feature_extractor=nn.Sequential(
            nn.Conv2d(input_channels,32,8,4),nn.ReLU(inplace=False),
            nn.Conv2d(32,64,4,2),nn.ReLU(inplace=False),
            nn.Conv2d(64,64,3,1),nn.ReLU(inplace=False),
            nn.Conv2d(64,64,3,1),nn.ReLU(inplace=False)
        )
        with torch.no_grad():
            flat=self.feature_extractor(torch.zeros(1,input_channels,84,84)).flatten(1).size(1)
        self.shared=nn.Sequential(
            nn.Linear(flat,512),nn.ReLU(inplace=False),
            nn.Linear(512,512),nn.ReLU(inplace=False)
        )
        self.ln=nn.LayerNorm(512)
        self.policy_head=nn.Linear(512,num_actions)
        self.value_head=nn.Linear(512,1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.orthogonal_(m.weight,gain=nn.init.calculate_gain('relu'));nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,state,temp=1.0):
        if state.dim()==3:state=state.unsqueeze(0)
        x=self.feature_extractor(state).flatten(1)
        x=self.ln(self.shared(x))
        logits=self.policy_head(x)/temp
        dist=Categorical(logits=logits)
        value=self.value_head(x).squeeze(-1)
        return dist,value