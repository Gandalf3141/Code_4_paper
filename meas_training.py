import torch
from meas_get_data import *
from meas_NN_classes import *
from meas_dataset import *
import os 
from tqdm import tqdm
import logging
from meas_test_func_fs import *
from meas_dataloader import *
from meas_train_funcs import *
from model_params import get_model_params

torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("this device is available : ", device)


def main(parameters):

    # Configure logging
    log_file = f"training_model_{parameters['model_flag']}_{parameters['experiment_number']}.log"
    filemode = 'a' if os.path.exists(log_file) else 'w'
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize the LSTM model
    # Use the flag to confirm the right model is used and to save it
    if "LSTM" in parameters["model_flag"]:
        model = OR_LSTM(input_size=4, hidden_size=parameters["h_size"], out_size=2, 
                        layers=parameters["l_num"], window_size=parameters["window_size"], flag=parameters["model_flag"]).to(device)

    if "MLP" in parameters["model_flag"]:

        model = OR_MLP(input_size=4*parameters["window_size"], hidden_size=parameters["h_size"],
                        output_size=2, l_num=parameters["l_num"], window_size=parameters["window_size"], flag=parameters["model_flag"]).to(device)

    if "TCN" in parameters["model_flag"]:
        num_channels = [parameters["n_hidden"]] * parameters["levels"]
        model = OR_TCN(input_size=4, output_size=2, num_channels=num_channels,
                        kernel_size=parameters["kernel_size"], dropout=parameters["dropout"], windowsize=parameters["window_size"], flag=parameters["model_flag"]).to(device)
    
    # Generate input data (the data is normalized and some timesteps are cut off)
    if os.name == "nt":
        path_train_data=r"C:\Users\StrasserP\Documents\NN_Paper\Code_4_paper\messdaten\messdaten_900traj_500steps.csv"
    else:

        path_train_data=r"/home/rdpusr/Documents/Code_4_paper/messdaten/messdaten_900traj_500steps.csv"

    #train_data = get_data(path_train_data,num_inits=parameters["part_of_data"])
    train_data = get_data(path_train_data,num_inits=parameters["part_of_data"])
    train_loader, test_data = get_dataloader(train_data, parameters)

    #average_traj_err_train_lstm = []

    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = parameters["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = parameters["T_max"], eta_min=0, last_epoch=-1, verbose=False)

    if parameters["model_flag"] != model.get_flag():
        print("Parameter list model flag does not match the model flag used!")

    #Training loop
    print(f"Starting Training with {model.get_flag()}")
    for e in tqdm(range(parameters["epochs"])):

        train_error = train(train_loader, model, optimizer=optimizer, lr_scheduler=lr_scheduler, use_lr_scheduler=False)

        if (e+1) % 50 == 0:
            print(f"({model.get_flag()}) - Training error : ", train_error)
        
        if (e+1) % parameters["test_every_epochs"] == 0:
            test_error = test(data=test_data, model=model, window_size=parameters["window_size"])
            print(f"({model.get_flag()}) - Testing error : ", test_error)
        
    # Save trained model
    path = f'Trained_networks/modeltype_{model.get_flag()}_expnumb_{parameters["experiment_number"]}.pth'

    torch.save(model.state_dict(), path)

    print(f"Run finished!")
    print(path)

    # Log parameters

    logging.info(f"hyperparams (modeltype_{model.get_flag()}_expnumb_{parameters['experiment_number']}) : {parameters}")
    #logging.info(f"LSTM - Experiment number {parameters['experiment_number']}_{average_traj_err_train_lstm}")   
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")

if __name__ == '__main__':

    # toggle to test everything with a small amount of data
    testing_mode = True

    parameter_list = get_model_params(testing_mode)
    
    seems_to_work = []#["OR_LSTM", "OR_MLP", "OR_TCN", "LSTM", "MLP", "TCN"]

    for parameters in parameter_list:
        if parameters["model_flag"] in seems_to_work:
            continue
        main(parameters)
        