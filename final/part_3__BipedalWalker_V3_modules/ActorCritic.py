import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

torch_logger = logging.getLogger(__name__)
torch_logger.setLevel(logging.INFO)
torch_logger.addHandler(logging.StreamHandler())


class A2CActorCritic(nn.Module):
    def __init__(
        self,
        num_actions: int = 4,
        input_dim: int = 24,
        hidden_sizes: Tuple[int, ...] = (256, 256)
    ):
        super().__init__()

        layers = []
        in_f = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_f, h), nn.ReLU(inplace=False)]
            in_f = h

        self.feature_extractor = nn.Sequential(*layers)
        self.shared = nn.Identity()
        self.ln     = nn.LayerNorm(in_f)


        self.policy_head = nn.Linear(in_f, num_actions)
        self.log_std     = nn.Parameter(torch.full((num_actions,), -0.5))


        self.value_head  = nn.Linear(in_f, 1)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if m is self.policy_head else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

        torch_logger.info(
            f"A2CActorCritic: obs={input_dim}, actions={num_actions}, "
            f"params={sum(p.numel() for p in self.parameters())/1e3:.1f}k"
        )

    def forward(self, state: torch.Tensor, temp: float = 1.0):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        assert state.dim() == 2, "state must be B×24 vector"

        x = self.feature_extractor(state)
        x = self.ln(x)

        mean = self.policy_head(x) / temp


        log_std_clamped = self.log_std.clamp(-3.0, 0.5)
        std = log_std_clamped.exp().expand_as(mean)


        dist = Normal(mean, std)
        value = self.value_head(x).squeeze(-1)
        return dist, value