#!/usr/bin/env python3

"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

"""
You are free to additional imports as needed... except please do not add any additional packages or dependencies to
your virtualenv other than those specified in requirements.txt. If I can't run it using the virtualenv I specified,
without any additional installs, there will be a penalty.

I've included a number of imports that I think you'll need.
"""
import torch
import matplotlib
import matplotlib.pyplot as plt
import gym
from network import *
import argparse
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter
import os

# prevents type-3 fonts, which some conferences disallow.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def make_env():
    env = gym.make('CartPole-v0')
    return env


def sliding_window(data, N):
    """
    For each index, k, in data we average over the window from k-N-1 to k. The beginning handles incomplete buffers,
    that is it only takes the average over what has actually been seen.
    :param data: A numpy array, length M
    :param N: The length of the sliding window.
    :return: A numpy array, length M, containing smoothed averaging.
    """

    idx = 0
    window = np.zeros(N)
    smoothed = np.zeros(len(data))

    for i in range(len(data)):
        window[idx] = data[i]
        idx += 1

        smoothed[i] = window[0:idx].mean()

        if idx == N:
            window[0:-1] = window[1:]
            idx = N - 1

    return smoothed


if __name__ == '__main__':

    """
    You are free to add additional command line arguments, but please ensure that the script will still run with:
    python main.py --episodes 10000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-e", default=10000, type=int, help="Number of episodes to train for")
    parser.add_argument("--total_run", default=1, type=int, help="Number of runs with different random initialization")
    parser.add_argument("--save", "-s", action='store_true', help="Save model or not")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot result or not")
    parser.add_argument("--write", "-w", action="store_true", help="Whether write the .csv file")
    parser.add_argument("--tensorboard", "-t", action="store_true", help="Whether write to tensorboard")
    parser.add_argument("--gamma_correct", action="store_true", help="Use gamma corrected policy gradient or not")
    parser.add_argument("--load_model", "-l", action="store_true", help="Load trained model")
    args = parser.parse_args()

    episodes = args.episodes
    save_or_not = args.save
    plot_or_not = args.plot
    load_or_not = args.load_model
    total_runs = args.total_run
    write_csv = args.write
    write_tensorboard = args.tensorboard
    gamma_correct = args.gamma_correct


    """
    It is unlikely that the GPU will help in this instance (since the size of individual operations is small) - in fact 
    there's a good chance it could slow things down because we have to move data back and forth between CPU and GPU.
    Regardless I'm leaving this in here. For those of you with GPUs this will mean that you will need to move your 
    tensors to GPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()

    in_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    gamma = 0.99

    gamma_list = [gamma**t for t in range(1000)]
    gamma_list = torch.tensor(gamma_list)
    for _ in range(total_runs):

        network = network_factory(in_size=in_size, num_actions=num_actions, env=env)
        network.to(device)

        # TODO: after each episode add the episode return (the total sum of the undiscounted rewards received in the
        #  episode) to this list.
        ep_returns = []

        # initialize policy network, value network and tensorboard writer
        policy = PolicyNetwork(network)
        policy_optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        if load_or_not:
            network.load_state_dict(torch.load('saved.pkl'))
        value = ValueNetwork(in_size)
        value_optimizer = torch.optim.Adam(value.parameters(), lr=3e-3)
        summary_writer = SummaryWriter()

        # main loop for generating episodes and training
        for ep in range(episodes):
            done = False
            state = env.reset()
            states, actions, rewards,  = [], [], []
            ep_returns.append(0)
            while not done:
                action = policy.get_action(state)
                next_state, reward, done, info = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                ep_returns[-1] += reward
                state = next_state
            returns = []
            # compute G_t
            last_value = 0
            for r in reversed(rewards):
                last_value = gamma * last_value + r
                returns.append(last_value)
            returns.reverse()
            # value update
            returns = torch.tensor(returns, dtype=torch.float32)
            v = value.forward(states)
            value_loss = torch.mean((returns - v)**2)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            # policy update
            dist = policy.forward(torch.tensor(states, dtype=torch.float32))
            adv = returns - value.forward(states)
            if gamma_correct:
                policy_loss = -torch.mean(gamma_list[:len(rewards)]*dist.log_prob(torch.tensor(actions, dtype=torch.int32))\
                       * adv)
            else:
                policy_loss = -torch.mean(dist.log_prob(torch.tensor(actions, dtype=torch.int32))\
                       * adv)
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # print episode information, write to tensorboard and local .csv file
            print("Episode {} Return: {}".format(ep, np.mean(ep_returns[:-10])))
            if write_tensorboard:
                summary_writer.add_scalar('Return', ep_returns[-1], ep)
                summary_writer.add_scalar('Loss/policy_loss', policy_loss, ep)
                summary_writer.add_scalar('Loss/value_loss', value_loss, ep)
                summary_writer.add_scalar('Loss/total_loss', policy_loss + 0.01 * value_loss, ep)
                for name, params in zip(network.state_dict().keys(), network.parameters()):
                    avg_grad = torch.mean(params.grad**2)
                    summary_writer.add_scalar('Gradient/'+str(name), avg_grad, ep)
        if write_csv:
            with open('returns.csv', 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(ep_returns)

        # don't plot when we need multiple runs
        if plot_or_not:
            plt.plot(sliding_window(ep_returns, 100))
            plt.title("Episode Return")
            plt.xlabel("Episode")
            plt.ylabel("Average Return (Sliding Window 100)")
            plt.show()

        # TODO: save your network
        if save_or_not:
            torch.save(network.state_dict(), './saved.pkl')
