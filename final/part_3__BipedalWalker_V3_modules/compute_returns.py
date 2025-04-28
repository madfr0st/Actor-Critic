from typing import List, Union

import numpy as np
import torch


def calculate_returns(
    rewards: List[float],
    dones:   List[bool],
    last_value_or_state: Union[float, np.ndarray],
    model: torch.nn.Module = None,
    gamma: float = 0.99
) -> List[float]:

    if isinstance(last_value_or_state, (float, int)):
        R = float(last_value_or_state)
    else:
        assert model is not None, "Model required for bootstrapping"
        obs = torch.as_tensor(last_value_or_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value = model(obs)
        R = float(value.item())

    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - float(d))
        returns.insert(0, R)
    return returns