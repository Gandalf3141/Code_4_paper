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
from model_params_robot import get_model_params_robot
import pandas as pd

torch.set_default_dtype(torch.float32)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device="cpu"
print("this device is available : ", device)

def save_model_with_versioning(base_path):
    version = 1
    save_path = base_path
    while os.path.exists(save_path):
        version += 1
        save_path = f"{base_path.rsplit('.', 1)[0]}_v{version}.pth"

    print(f"Model saved at: {save_path}")
    return save_path

def main(parameters):

    if robot_mode == True:
        inp_size = 12
        output_size = 6
        nn_folder = "Trained_networks_robot"
    else:
        inp_size = 4
        output_size = 2
        nn_folder = "Trained_networks" 
        

    # Configure logging
    log_file = f"training_model_{parameters['model_flag']}.log" if robot_mode == False \
    else f"robot_training_model_{parameters['model_flag']}.log"  
    filemode = 'a' if os.path.exists(log_file) else 'w'
    logging.basicConfig(filename=log_file, filemode=filemode, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Initialize the LSTM model
    # Use the flag to confirm the right model is used and to save it
    if "LSTM" in parameters["model_flag"]:
        model = OR_LSTM(input_size=inp_size, hidden_size=parameters["h_size"], out_size=output_size, 
                        layers=parameters["l_num"], window_size=parameters["window_size"], flag=parameters["model_flag"]).to(device)

    if "MLP" in parameters["model_flag"]:

        model = OR_MLP(input_size=inp_size*parameters["window_size"], hidden_size=parameters["h_size"],
                        output_size=output_size, l_num=parameters["l_num"], window_size=parameters["window_size"], flag=parameters["model_flag"]).to(device)

    if "TCN" in parameters["model_flag"]:
        num_channels = [parameters["n_hidden"]] * parameters["levels"]
        model = OR_TCN(input_size=inp_size, output_size=output_size , num_channels=num_channels,
                        kernel_size=parameters["kernel_size"], dropout=parameters["dropout"], windowsize=parameters["window_size"], flag=parameters["model_flag"]).to(device)

    if "RNN" in parameters["model_flag"]:
        model = OR_RNN(input_size=inp_size, hidden_size=parameters["h_size"], out_size=output_size, 
                        layers=parameters["l_num"], window_size=parameters["window_size"], flag=parameters["model_flag"]).to(device)
        
    if "GRU" in parameters["model_flag"]:
        model = OR_GRU(input_size=inp_size, hidden_size=parameters["h_size"], out_size=output_size, 
                        layers=parameters["l_num"], window_size=parameters["window_size"], flag=parameters["model_flag"]).to(device)

    # Generate input data (the data is normalized and some timesteps are cut off)
    if os.name == "nt":
        path_train_data=r"C:\Users\StrasserP\Documents\NN_Paper\Code_4_paper\messdaten\messdaten_900traj_500steps.csv"
    else:
        path_train_data=r"/home/rdpusr/Documents/Code_4_paper/messdaten/messdaten_900traj_500steps.csv"

    if robot_mode == True:
        train_data, test_data = get_data_robot() #train data split still works! but test data is just the full robot test data.
        train_loader, _ = get_dataloader(train_data, parameters)
    else:
        train_data = get_data(path_train_data,num_inits=parameters["part_of_data"])
        train_loader, test_data = get_dataloader(train_data, parameters)

    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = parameters["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = parameters["T_max"], eta_min=0, last_epoch=-1, verbose=False)

    if parameters["model_flag"] != model.get_flag():
        print("Parameter list model flag does not match the model flag used!")

    #Training loop
    if robot_mode:
     print(f"Starting Training with {model.get_flag()} -- robot")
    else:
     print(f"Starting Training with {model.get_flag()} -- ventil")
     
    test_error=0 # test error 0 until first test with testdata is done
    for e in tqdm(range(parameters["epochs"])):

        train_error = train(train_loader, model, optimizer=optimizer, lr_scheduler=lr_scheduler, use_lr_scheduler=False)

        if (e+1) % 50 == 0:
            print(f"({model.get_flag()}) - Training error : ", train_error)
        
        if (e+1) % parameters["test_every_epochs"] == 0:
            test_error = test(data=test_data, model=model, window_size=parameters["window_size"])
            print(f"({model.get_flag()}) - Testing error : ", test_error)
            if test_error < 0.001:
                print("Early stopping: Test error is below the threshold -> Training stopped.")
                break
        
        error_dic[model.get_flag() + "_test_err"].append(test_error)
        error_dic[model.get_flag() + "_train_err"].append(train_error)
        
    # Save trained model
    if robot_mode == True:
        path = f'{nn_folder}/modeltype_{model.get_flag()}_robot.pth'
    else:
        path = f'{nn_folder}/modeltype_{model.get_flag()}_ventil.pth'
    torch.save(model.state_dict(), save_model_with_versioning(path))


    print(f"Run finished!")

    print(path)

    # Log parameters

    logging.info(f"hyperparams: final_test_err {error_dic[model.get_flag() + '_test_err'][-1]} (modeltype_{model.get_flag()}_expnumb_{parameters['experiment_number']}) : {parameters}")
    #logging.info(f"LSTM - Experiment number {parameters['experiment_number']}_{average_traj_err_train_lstm}")   
    logging.info("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("\n")
    return error_dic

if __name__ == '__main__':

    # toggle to test everything with a small amount of data
    testing_mode = False
    robot_mode = True
    param_search_mode = True

    if robot_mode:
        parameter_list = get_model_params_robot(testing_mode, robot_mode, params_search=True, params_specific="OR_LSTM")
    else:
        parameter_list = get_model_params(testing_mode, robot_mode, params_search=True)

    list_of_NNs_to_train = ["OR_LSTM", "LSTM", "OR_TCN", "TCN", "OR_MLP", "MLP"] #["OR_LSTM", "OR_MLP", "OR_TCN", "OR_RNN", "OR_GRU", "LSTM", "MLP", "TCN", "RNN", "GRU"]
    error_dic = {x : [] for x in [x + "_train_err" for x in list_of_NNs_to_train] + [x + "_test_err" for x in list_of_NNs_to_train]}

    if param_search_mode:
        for param_search_nets in ["OR_LSTM", "OR_MLP", "OR_TCN", "OR_RNN", "OR_GRU", "LSTM", "MLP", "TCN", "RNN", "GRU"]:
            parameter_list = get_model_params_robot(testing_mode, robot_mode, params_search=True, params_specific=param_search_nets)
            for parameters in parameter_list:
                main(parameters)
    else:
        for parameters in parameter_list:
            if parameters["model_flag"] not in list_of_NNs_to_train:
                continue

            main(parameters)

    #save errors as csv
    max_length = max(len(v) for v in error_dic.values())
    for key, value in error_dic.items():
        if len(value) < max_length:
            error_dic[key] = value + [0] * (max_length - len(value))

    if robot_mode:
        base_path = "robot_train_test_errors.csv"
    else:
        base_path = "train_test_errors.csv"
    save_path = base_path
    version = 0
    while os.path.exists(save_path):
        version += 1
        save_path = f"{base_path.rsplit('.', 1)[0]}_v{version}.csv"

    df = pd.DataFrame(error_dic)
    df.to_csv(save_path, index=False)
    print("errors saved")