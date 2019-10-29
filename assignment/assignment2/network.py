"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

from torch.distributions import Categorical
from torch import nn
import torch

def network_factory(in_size, num_actions, env):
    """

    :param in_size:
    :param num_actions:
    :param env: The gym environment. You shouldn't need this, but it's included regardless.
    :return: A network derived from nn.Module
    """
    return nn.Sequential(
        nn.Linear(in_size, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, num_actions, bias=True),
        nn.Softmax(dim=-1)
    )


class PolicyNetwork(nn.Module):
    def __init__(self, network):
        super(PolicyNetwork, self).__init__()
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        probs = self.network(inputs)
        return Categorical(probs)


    def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
        dist = self.forward(inputs)
        return dist.sample().item()

class ValueNetwork(nn.Module):
    def __init__(self, in_size):
        super(ValueNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = nn.Sequential(
            nn.Linear(in_size, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True),
        )

    def forward(self, inputs):
        inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        return self.network(inputs)


    def get_value(self, inputs):
        """
        return value of given states
        """
        return self.forward(inputs).item()







