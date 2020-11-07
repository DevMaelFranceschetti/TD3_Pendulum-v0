
from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import gym
import gym.spaces
import numpy as np
from tqdm import tqdm

from models import RLNN, Actor
from random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from memory import Memory
from util import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


if __name__ == "__main__":

    ###########################################
    ##           Code Parameters:            ##
    ###########################################

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='Pendulum-v0', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)
    parser.add_argument('--file', default="actor_2", type=str)#the policy to evaluate, without extension .pkl
    parser.add_argument('--policies_dir', default="actors", type=str)#the directory where the policies ar stored


    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')


    args = parser.parse_args()
    
    # parameters :
    output_video_dir = './video_actor' # /!\ Directory for video output if used
    actor_directory = args.policies_dir # /!\ Directory of the actor to load
    actor_filename = args.file # /!\ actor's filename (without extension)
    reward = 0  
    rewards_list = []
    nb_runs = 900

    for run in range(nb_runs):
        # creating gym env :
        print("Creating environment")
        env = gym.make(args.env)

        #creating a monitor for video output if used :
        #env = gym.wrappers.Monitor(env, output_video_dir,force=True)

        # initialisation :
        state = env.reset()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = int(env.action_space.high[0])
        tot_reward = 0

        done = 0  
        #actor load:
        print("Loading actor")
        actor = Actor(state_dim, action_dim, max_action, args)
        actor.load_model(actor_directory,actor_filename)

        # executing the actor on an episode:
        print("Running...")
        while not done:
            # converting state in tensor for the policy:
            stateTorch = FloatTensor(np.array(state).reshape(-1))
            # select action and convert it to a flatten numpy :
            action = actor.forward(stateTorch).cpu().data.numpy().flatten()
            # running the action:
            state, reward, done, _ = env.step(action)
            tot_reward+=reward
        rewards_list.append(tot_reward)
        print("total reward : "+str(tot_reward))
        #print("Video saved in : "+str(output_video_dir))

        # closing the env:
        env.close()
    print("mean reward : "+str(np.mean(rewards_list)))
