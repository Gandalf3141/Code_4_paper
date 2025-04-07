import torch
from meas_get_data import *
from meas_NN_classes import *
from meas_dataset import *
from tqdm import tqdm
from meas_test_func_fs import *
from meas_dataloader import *

# train functions

def train_lstm_no_or_derivative(traindataloader, model, optimizer, lr_scheduler, use_lr_scheduler=False):
   
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = []

    for k, (inp, label) in enumerate(traindataloader):  # inp = (u, x) label = x

        inp=inp.to(device)
        label=label.to(device)

        # Predict one timestep :
        output, _ = model.simple_forward(inp)
        out = inp[:,-1:, 2:] + output[:, -1:, :]

        # reset the gradient
        
        optimizer.zero_grad(set_to_none=True)
        # calculate the error

        loss = loss_fn(out[:,-1,:], label[:, 2:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss.append(loss.detach().cpu().numpy())
    
    if use_lr_scheduler:
        lr_scheduler.step()
   # return the average error of the next step prediction
    return np.mean(total_loss)

def train_mlp_no_or_derivative(traindataloader, model, optimizer, lr_scheduler, use_lr_scheduler=False):
 
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = []
  
    for k, (x, x_last, y) in enumerate(traindataloader):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)
        x_last = x_last.to(device)
        x_last = x_last.squeeze()
        
        output = model.simple_forward(x)
        pred = x_last[:,2:] + output[:, :]

        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(pred, y[:,2:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
 
        total_loss.append(loss.detach().cpu().numpy())
    
    if use_lr_scheduler:
        lr_scheduler.step()
        
   # return the average error of the next step prediction
    return np.mean(total_loss)

def train_tcn_no_or_derivative(traindataloader, model, optimizer, lr_scheduler, use_lr_scheduler=False):
 
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = []
  
    for k, (x,y) in enumerate(traindataloader):  # inp = (u, x) label = x
        
        x = x.to(device)
        y = y.to(device)

        x = x.transpose(1,2)

        output = model.simple_forward(x)
        out = x[:, 2:, -1] + output

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(out, y[:, 2:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss.append(loss.detach().cpu().numpy())
    
    if use_lr_scheduler:
        lr_scheduler.step()
    return np.mean(total_loss)

def train(traindataloader, model, optimizer, lr_scheduler, use_lr_scheduler=False):
    
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = []

    if model.get_flag() == "LSTM":
        return train_lstm_no_or_derivative(traindataloader, model, optimizer, lr_scheduler)
    if model.get_flag() == "MLP":
        return train_mlp_no_or_derivative(traindataloader, model, optimizer, lr_scheduler)
    if model.get_flag() == "TCN":
        return train_tcn_no_or_derivative(traindataloader, model, optimizer, lr_scheduler)

    for k, (x,y) in enumerate(traindataloader):  
        
        x = x.to(device)
        y = y.to(device)

        if model.get_flag() == "OR_LSTM":
            output, _ = model(x)
        if model.get_flag() == "OR_MLP":
            output = model(x)
        if model.get_flag() == "OR_TCN":
            x = x.transpose(1,2)
            y = y.transpose(1,2)
            output = model(x)

        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss.append(loss.item())
    
    if use_lr_scheduler:
        lr_scheduler.step()
    
    return np.mean(total_loss)



# deprecated:

# def train_lstm(input_data, model,  optimizer, lr_scheduler, use_OR=True):
 
#     loss_fn = nn.MSELoss()
#     model.train()
#     total_loss = []


#     for k, (x,y) in enumerate(input_data):  # inp = (u, x) label = x
        
#         x = x.to(device)
#         y = y.to(device)
        
#         output, _ = model(x)
  
#         # reset the gradient
#         optimizer.zero_grad(set_to_none=True)
        
#         # calculate the error
#         loss = loss_fn(output, y)

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         total_loss.append(loss.item())
#     lr_scheduler.step()
#     #print(lr_scheduler.get_last_lr())

#     return np.mean(total_loss)

# def train_mlp(input_data, model,  optimizer, lr_scheduler):
#  return
# def train_tcn(input_data, model,  optimizer, lr_scheduler):
#  return