import torch
from meas_get_data import *
import os
from meas_dataset import *
from torch.utils.data import DataLoader
import numpy as np

# Data Loader for training (Ventil + robot arm)
def get_dataloader(input_data, params, robot_mode=False):
    

    #Split data into train and test sets
    np.random.seed(1234)
    num_of_inits_train = int(len(input_data)*params["percentage_of_data"])
    train_inits = np.random.choice(np.arange(len(input_data)),num_of_inits_train,replace=False)
    test_inits = np.array([x for x in range(len(input_data)) if x not in train_inits])
    np.random.shuffle(train_inits)
    np.random.shuffle(test_inits)
    #train_data = input_data[train_inits,:input_data.size(dim=1)-params["cut_off_timesteps"],:]
    #cut off timesteps at the start
    train_data = input_data[train_inits,params["cut_off_timesteps"]:,:]
    test_data = input_data[test_inits,:,:]

    # dataloader for batching during training
    if "OR" in params["model_flag"]:
        train_set = custom_simple_dataset(train_data, window_size=params["window_size"])
        train_loader = DataLoader(train_set, batch_size=int(params["batch_size"]), pin_memory=True)
    if  params["model_flag"] in ["LSTM", "RNN", "GRU"]:
        train_set = CustomDataset(train_data, window_size=params["window_size"])
        train_loader = DataLoader(train_set, batch_size=int(params["batch_size"]), pin_memory=True)
    if "TCN" == params["model_flag"]:
        train_set = CustomDataset(train_data, window_size=params["window_size"])
        train_loader = DataLoader(train_set, batch_size=int(params["batch_size"]), pin_memory=True)
    if "MLP" == params["model_flag"]:
        train_set = CustomDataset_mlp(train_data, window_size=params["window_size"])
        train_loader = DataLoader(train_set, batch_size=int(params["batch_size"]), pin_memory=True)


    return train_loader, test_data
