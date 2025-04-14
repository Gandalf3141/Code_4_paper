import torch
from meas_get_data import *
#from meas_NN_classes import *
from meas_dataset import *
import os 
from tqdm import tqdm
import logging
from meas_test_func_fs import *
from meas_dataloader import *
from meas_train_funcs import *
from model_params import get_model_params
import timeit

class OR_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, layers, window_size=4, flag="OR_LSTM"):

        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ws = window_size
        self.flag = flag
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)

        # Define linear layer
        self.linear = nn.Linear(hidden_size, out_size)
    def get_flag(self):
        return self.flag

    def forward(self, one_full_traj):

        #init out
        out = torch.zeros(one_full_traj.size(0), one_full_traj.size(1), 2, device=one_full_traj.device)

        seq = one_full_traj[:, 0:self.ws, :]
        lstm_out, hidden = self.lstm(seq)           
        pred = self.linear(lstm_out)
        #only update next step
        out = one_full_traj[:, self.ws-1:self.ws, 2:] + pred[:, -1: , :]

        for t in range(1, self.ws): # für RK : range(1, self.ws + 2):

            #tmp = torch.cat((one_full_traj[:,self.ws+(t-1):self.ws+(t-1)+(out.size(dim=1)), 0:2] , out[:, :, :]), dim=2)
            #seq = torch.cat((one_full_traj[:, t:self.ws, :], torch.cat((one_full_traj[:,self.ws+(t-1):self.ws+(t-1)+(out.size(dim=1)), 0:2] , out[:, :, :]), dim=2)), dim=1)

            lstm_out, hidden = self.lstm(torch.cat((one_full_traj[:, t:self.ws, :], torch.cat((one_full_traj[:,self.ws+(t-1):self.ws+(t-1)+(out.size(dim=1)), 0:2] , out[:, :, :]), dim=2)), dim=1))           
            #pred = self.linear(lstm_out)
            out = torch.cat((out, out[:, -1:, :] + self.linear(lstm_out)[:, -1: , :]), dim=1)
            
        for t in range(self.ws, one_full_traj.size(dim=1) - self.ws):

            #seq = torch.cat((one_full_traj[:, t : t + self.ws, 0:2], out[:, t - self.ws : t , :]), dim=2)
            
            lstm_out, hidden = self.lstm(torch.cat((one_full_traj[:, t : t + self.ws, 0:2], out[:, t - self.ws : t , :]), dim=2))           
            #pred = self.linear(lstm_out)

            out = torch.cat((out, out[:, t-1:t, :] + self.linear(lstm_out)[:, -1: , :]), dim=1)

        return out, hidden 

    def simple_forward(self, seq):

        lstm_out, hidden = self.lstm(seq)           
        pred = self.linear(lstm_out)

        return pred, hidden  
    
    def time_forward(self):
        data = torch.rand(40, 300, 4)
        self.forward(data)
 

torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device="cpu"
print("this device is available : ", device)

def save_model_with_versioning(model, base_path):
    version = 1
    save_path = base_path
    while os.path.exists(save_path):
        version += 1
        save_path = f"{base_path.rsplit('.', 1)[0]}_v{version}.pth"

    print(f"Model saved at: {save_path}")
    return save_path

def main(parameters):

    if "LSTM" in parameters["model_flag"]:
        model = OR_LSTM(input_size=4, hidden_size=parameters["h_size"], out_size=2, 
                        layers=parameters["l_num"], window_size=parameters["window_size"], flag=parameters["model_flag"]).to(device)


    # Generate input data (the data is normalized and some timesteps are cut off)
    if os.name == "nt":
        path_train_data=r"C:\Users\StrasserP\Documents\NN_Paper\Code_4_paper\messdaten\messdaten_900traj_500steps.csv"
    else:

        path_train_data=r"/home/rdpusr/Documents/Code_4_paper/messdaten/messdaten_900traj_500steps.csv"

    train_data = get_data(path_train_data,num_inits=parameters["part_of_data"])
    train_loader, test_data = get_dataloader(train_data, parameters)

    elapsed_time = timeit.timeit(model.time_forward, number=20) #39
    print(f"Elapsed time for one forward pass: {elapsed_time} seconds")

if __name__ == '__main__':

    # toggle to test everything with a small amount of data
    testing_mode = False

    parameters =    {
                        "model_flag" : "OR_LSTM",
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "percentage_of_data" : 0.2,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000),
                        }
 
    main(parameters)
