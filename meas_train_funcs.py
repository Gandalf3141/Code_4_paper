import torch
from meas_get_data import *
from meas_NN_classes import *
from meas_dataset import *
from tqdm import tqdm
from meas_test_func_fs import *
from meas_dataloader import *
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# train functions

# also for RNN and GRU
def train_lstm_no_or_derivative(traindataloader, model, optimizer, lr_scheduler, use_lr_scheduler=False):

    
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = []
    #device = next(model.parameters()).device

    for k, (inp, label) in enumerate(traindataloader):  # inp = (u, x) label = x
        
        inp=inp.to(device)
        label=label.to(device)
        num_outs = int(inp.size(dim=2) / 2) # number of outputs = number of states which is half of the input size

        # Predict one timestep :
        output, _ = model.simple_forward(inp)
        out = inp[:,-1:, num_outs:] + output[:, -1:, :]

        # reset the gradient
        
        optimizer.zero_grad(set_to_none=True)
        # calculate the error

        loss = loss_fn(out[:,-1,:], label[:, num_outs:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss.append(loss.detach().cpu().numpy())
    
    if use_lr_scheduler:
        lr_scheduler.step()
   # return the average error of the next step prediction
    return np.mean(total_loss)def train_mlp_no_or_derivative(traindataloader, model, optimizer, lr_scheduler, use_lr_scheduler=False):
 
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = []
    #device = next(model.parameters()).device
    
  
    for k, (x, x_last, y) in enumerate(traindataloader):  # inp = (u, x) label = x
        
        

        x = x.to(device)
        y = y.to(device)
        x_last = x_last.to(device)
        x_last = x_last.squeeze()
        
        
        output = model.simple_forward(x)
        num_outs = int(y.size(dim=1) / 2) # number of outputs = number of states which is half of the input size
        #print(num_outs , x_last.shape, x.shape, y.shape, output.shape)
        pred = x_last[:,num_outs:] + output[:, :]

        # reset the gradient
        optimizer.zero_grad(set_to_none=True)
        
        # calculate the error
        loss = loss_fn(pred, y[:,num_outs:])
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
    #device = next(model.parameters()).device
  
    for k, (x,y) in enumerate(traindataloader):  # inp = (u, x) label = x
        
       
        x = x.to(device)
        y = y.to(device)
        num_outs = int(y.size(dim=1) / 2) # number of outputs = number of states which is half of the input size

        x = x.transpose(1,2)

        output = model.simple_forward(x)
        out = x[:, num_outs:, -1] + output

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(out, y[:, num_outs:])
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
    #device = next(model.parameters()).device

    if model.get_flag() in ["LSTM", "RNN", "GRU"]:
        return train_lstm_no_or_derivative(traindataloader, model, optimizer, lr_scheduler)
    if model.get_flag() == "MLP":
        return train_mlp_no_or_derivative(traindataloader, model, optimizer, lr_scheduler)
    if model.get_flag() == "TCN":
        return train_tcn_no_or_derivative(traindataloader, model, optimizer, lr_scheduler)

    for k, (x,y) in enumerate(traindataloader):  
        
        x = x.to(device)
        y = y.to(device)

        if model.get_flag() in ["OR_LSTM", "OR_RNN", "OR_GRU"]:
            output, _ = model(x)
        if model.get_flag() == "OR_MLP":
            output = model(x)
        if model.get_flag() == "OR_TCN":
            x = x.transpose(1,2)
            y = y.transpose(1,2)
            output = model(x)

        optimizer.zero_grad(set_to_none=True)
        
        # physics loss: 
        # rescale data first?
        # how to calc s'' and p' fast? 

        # s, p rescale to original units
        # s' = s.autodiff()...

        # s'' = 1/m (A (p - p_0)- c (s - s_0) + f_r(v) + f_k(s,v))
        # p' = k/V(s) (-A v p + R_s T_0 (\rho_0 (Q_NC - Q_NO)))
        # simple physics loss:
        # 0 = 1/m (A (p - p_0)- c (s - s_0)

        # calculate the error
        loss = loss_fn(output, y) # + lambda * loss_physics

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss.append(loss.item())
    
    if use_lr_scheduler:
        lr_scheduler.step()
    
    return np.mean(total_loss)
