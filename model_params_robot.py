import numpy as np

# Comments on parameters:
# 
# We take parameters that worked for other training runs..
# the big difference between OR and TF models is the batch size!

# ONLY ROBOT ARM PARAMS HERE!

# Param Optimization
# 1 run for each parameter (hsize, batchsize, windowsize)
# After that
# -> 1 big training run on full data

def search_params_robot(params_specific):

    if params_specific == "OR_LSTM":
        p1 =    {
                        "model_flag" : "OR_LSTM",
                        "window_size" : 8, # 8 best
                        "h_size" : 16, # 16 best
                        "l_num" : 1,
                        "batch_size" : 20,
                        "experiment_number" : np.random.randint(0,1000),
        
                }
        key = "l_num"  
        p2 = p1.copy()
        p2[key] = 2
        p3 = p1.copy()
        p3[key] = 3
        p4 = p1.copy()
        p4[key] = 4
        p5 = p1.copy()
        p5[key] = 5

    if params_specific == "LSTM":
        p1 =    {
                        "model_flag" : "OR_LSTM",
                        "window_size" : 8, #best
                        "h_size" : 16, # best
                        "l_num" : 1,
                        "batch_size" : 20,
                        "experiment_number" : np.random.randint(0,1000),
        
                }
        key = "l_num"  
        p2 = p1.copy()
        p2[key] = 2
        p3 = p1.copy()
        p3[key] = 3
        p4 = p1.copy()
        p4[key] = 4
        p5 = p1.copy()
        p5[key] = 5

    if params_specific == "OR_MLP":
               
        p1 =    {
                    "model_flag" : "OR_MLP",
                    "window_size" : 20,
                    "h_size" : 16, # 16 best
                    "l_num" : 1,
                    "learning_rate" : 0.001,
                    "batch_size" : 80,
                    "experiment_number" : np.random.randint(0,1000)
                    }
        key = "l_num"  
        p2 = p1.copy()
        p2[key] = 2
        p3 = p1.copy()
        p3[key] = 3
        p4 = p1.copy()
        p4[key] = 4
        p5 = p1.copy()
        p5[key] = 5
    
    if params_specific == "MLP":
               
        p1 =    {
                    "model_flag" : "OR_MLP",
                    "window_size" : 20,
                    "h_size" : 8,
                    "l_num" : 1,
                    "learning_rate" : 0.001,
                    "batch_size" : 8,
                    "experiment_number" : np.random.randint(0,1000)
                    }
        key = "l_num"  
        p2 = p1.copy()
        p2[key] = 2
        p3 = p1.copy()
        p3[key] = 3
        p4 = p1.copy()
        p4[key] = 4
        p5 = p1.copy()
        p5[key] = 5

    if params_specific == "OR_TCN":
            
            p1 =    {
                "model_flag" : "OR_TCN",
                "window_size" : 20,
                "n_hidden" : 5, #best
                "levels" : 2,
                "kernel_size" : 3,
                "dropout" : 0,
                "batch_size" : 30,
                "experiment_number" : np.random.randint(0,1000)                        
                        }
            key = "levels"  
            p2 = p1.copy()
            p2[key] = 3
            p3 = p1.copy()
            p3[key] = 4
            p4 = p1.copy()
            p4[key] = 5
            p5 = p1.copy()
            p5[key] = 6 

    if params_specific == "TCN":
            
            p1 =    {
                "model_flag" : "TCN",
                "window_size" : 20,
                "n_hidden" : 5,
                "levels" : 4,
                "kernel_size" : 3,
                "dropout" : 0,
                "batch_size" : 30,
                "experiment_number" : np.random.randint(0,1000)                        
                        }
            key = "levels"  
            p2 = p1.copy()
            p2[key] = 3
            p3 = p1.copy()
            p3[key] = 4
            p4 = p1.copy()
            p4[key] = 5
            p5 = p1.copy()
            p5[key] = 6 

    if params_specific == "OR_GRU":
            
            p1 =    {
                "model_flag" : "OR_GRU",
                "window_size" : 16,
                "h_size" : 4,
                "l_num" : 1,
                "learning_rate" : 0.001,
                "batch_size" : 20,
                "experiment_number" : np.random.randint(0,1000)                        
                        }
            key = "batch_size"  
            p2 = p1.copy()
            p2[key] = 16
            p3 = p1.copy()
            p3[key] = 24
            p4 = p1.copy()
            p4[key] = 32
            p5 = p1.copy()
            p5[key] = 48

    if params_specific == "GRU":
            
        p1 =    {
                "model_flag" : "GRU",
                "window_size" : 16,
                "h_size" : 4,
                "l_num" : 1,
                "learning_rate" : 0.001,
                "batch_size" : 20,
                "experiment_number" : np.random.randint(0,1000)                        
                        }
        key = "l_num"  
        p2 = p1.copy()
        p2[key] = 2
        p3 = p1.copy()
        p3[key] = 3
        p4 = p1.copy()
        p4[key] = 4
        p5 = p1.copy()
        p5[key] = 5

    if params_specific == "OR_RNN":
            
        p1 =    {
                "model_flag" : "OR_RNN",
                "window_size" : 16,
                "h_size" : 4,
                "l_num" : 1,
                "learning_rate" : 0.001,
                "batch_size" : 20,
                "experiment_number" : np.random.randint(0,1000)                        
                        }
        key = "l_num"  
        p2 = p1.copy()
        p2[key] = 2
        p3 = p1.copy()
        p3[key] = 3
        p4 = p1.copy()
        p4[key] = 4
        p5 = p1.copy()
        p5[key] = 5

    if params_specific == "RNN":
            
        p1 =    {
                "model_flag" : "RNN",
                "window_size" : 16,
                "h_size" : 4,
                "l_num" : 1,
                "learning_rate" : 0.001,
                "batch_size" : 20,
                "experiment_number" : np.random.randint(0,1000)                        
                        }
        key = "l_num"  
        p2 = p1.copy()
        p2[key] = 2
        p3 = p1.copy()
        p3[key] = 3
        p4 = p1.copy()
        p4[key] = 4
        p5 = p1.copy()
        p5[key] = 5
    
    params_list = [p1, p2, p3, p4, p5]
    return params_list

def get_model_params_robot(testing_mode=True, robot_mode = False, params_specific="", params_search=True):
    
    parameters_or_lstm =    {
                        "model_flag" : "OR_LSTM",
                        "window_size" : 48,
                        "h_size" : 46,
                        "l_num" : 2,
                        "learning_rate" : 0.0012515,
                        "batch_size" : 8,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000),
                        }
    
    parameters_or_rnn =    {
                        "model_flag" : "OR_RNN",
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000),
                        }

    parameters_or_gru =    {
                        "model_flag" : "OR_GRU",
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 20,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000),
                        }
       
    parameters_or_mlp =    {
                        "model_flag" : "OR_MLP",
                        "window_size" : 20,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 80,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,
                        "experiment_number" : np.random.randint(0,1000)
                        
                        }
    
    parameters_or_tcn =    {
                        "model_flag" : "OR_TCN",
                        "window_size" : 30,
                        "n_hidden" : 5,
                        "levels" : 4,
                        "kernel_size" : 7,
                        "l_num" : 3,
                        "dropout" : 0,
                        "learning_rate" : 0.001,
                        "batch_size" : 30,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 3000,
                        "test_every_epochs" : 2,
                        "T_max" : 4000,
                        "experiment_number" : np.random.randint(0,1000)
                        
                        }

    parameters_lstm =    {
                        "model_flag" : "LSTM",
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 1500,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000),
                        }

    parameters_rnn =    {
                        "model_flag" : "RNN",
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 1500,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000),
                        }

    parameters_gru =    {
                        "model_flag" : "GRU",
                        "window_size" : 16,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 1500,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,

                        "experiment_number" : np.random.randint(0,1000),
                        }

    parameters_mlp =    {
                        "model_flag" : "MLP",
                        "window_size" : 20,
                        "h_size" : 8,
                        "l_num" : 3,
                        "learning_rate" : 0.001,
                        "batch_size" : 1500,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 2000,
                        "test_every_epochs" : 2,
                        "T_max" : 1000,
                        "experiment_number" : np.random.randint(0,1000)
                        
                        }
    
    parameters_tcn =    {
                        "model_flag" : "TCN",
                        "window_size" : 30,
                        "n_hidden" : 5,
                        "levels" : 4,
                        "kernel_size" : 7,
                        "l_num" : 3,
                        "dropout" : 0,
                        "learning_rate" : 0.001,
                        "batch_size" : 1500,
                        "percentage_of_data" : 0.8,
                        "cut_off_timesteps" : 0,
                        "part_of_data" : 0,
                        "epochs" : 3000,
                        "T_max" : 4000,
                        "experiment_number" : np.random.randint(0,1000)
                        
                        }
    
    param_list = [ 
                   parameters_or_mlp,parameters_mlp,
                   parameters_or_rnn,parameters_rnn,
                   parameters_or_gru,parameters_gru,
                   parameters_or_lstm, parameters_lstm,
                   parameters_or_tcn, parameters_tcn
                   ]

    # amount of data used should be constant across all networks
    for parameters in param_list:

        parameters["percentage_of_data"]  = 0.8
        parameters["cut_off_timesteps"]  = 0
        parameters["part_of_data"]  = 0
        parameters["epochs"]  = 1000
        parameters["T_max"] =  int(parameters["epochs"] / 2)
        parameters["test_every_epochs"]  = int(parameters["epochs"]/10)
        parameters["experiment_number"]  = np.random.randint(0,1000)

        if params_specific == parameters["model_flag"] and params_search == False:
            return [parameters]
        
    if testing_mode:
   
       
        for parameters in param_list:

            parameters["percentage_of_data"]  = 0.2
            parameters["cut_off_timesteps"]  = 0
            parameters["part_of_data"]  = 50
            parameters["epochs"]  = 5
            parameters["test_every_epochs"]  = 2
            parameters["experiment_number"]  = np.random.randint(0,1000)
        
    if params_search: 
        
        param_list = search_params_robot(params_specific)
        for parameters in param_list:
            parameters["learning_rate"] = 0.001 #leave learning rate fixed because of scheduler and Adamoptimizer
            parameters["percentage_of_data"]  = 0.2
            parameters["cut_off_timesteps"]  = 0
            parameters["part_of_data"]  = 0
            parameters["epochs"]  = 200
            parameters["T_max"] =  int(parameters["epochs"] / 2)
            parameters["test_every_epochs"]  = int(parameters["epochs"]/2)
            parameters["experiment_number"]  = np.random.randint(0,1000)

    return param_list

            

































































        # parameters_or_lstm =    {
        #         "model_flag" : "OR_LSTM",
        #         "window_size" : 16,
        #         "h_size" : 8,
        #         "l_num" : 3,
        #         "learning_rate" : 0.001,
        #         "batch_size" : 5,
        #         "percentage_of_data" : 0.1,
        #         "cut_off_timesteps" : 0,
        #         "part_of_data" : 200,
        #         "epochs" : 10,
        #         "test_every_epochs" : 2,
        #         "T_max" :  5,
        #         "experiment_number" : np.random.randint(0,1000)
        #         }
        # parameters_or_mlp =    {
        #         "model_flag" : "OR_MLP",
        #         "window_size" : 20,
        #         "h_size" : 24,
        #         "l_num" : 3,
        #         "learning_rate" : 0.001,
        #         "batch_size" : 20,
        #         "percentage_of_data" : 0.1,
        #         "cut_off_timesteps" : 0,
        #         "part_of_data" : 200,
        #         "epochs" : 10,
        #         "test_every_epochs" : 2,
        #         "T_max" : 5,
        #         "experiment_number" : np.random.randint(0,1000)
        #         }
        # parameters_or_tcn =    {
        #         "model_flag" : "OR_TCN",
        #         "window_size" : 30,
        #         "h_size" : 8,
        #         "n_hidden" : 5,
        #         "levels" : 4,
        #         "kernel_size" : 7,
        #         "l_num" : 3,
        #         "dropout" : 0,
        #         "learning_rate" : 0.001,
        #         "batch_size" : 20,
        #         "percentage_of_data" : 0.1,
        #         "cut_off_timesteps" : 0,
        #         "part_of_data" : 200,
        #         "epochs" : 10,
        #         "test_every_epochs" : 2,
        #         "T_max" : 5,
        #         "experiment_number" : np.random.randint(0,1000)
        #         }
            
        # parameters_lstm =    {
        #         "model_flag" : "LSTM",
        #         "window_size" : 16,
        #         "h_size" : 8,
        #         "l_num" : 3,
        #         "learning_rate" : 0.001,
        #         "batch_size" : 100,
        #         "percentage_of_data" : 0.1,
        #         "cut_off_timesteps" : 0,
        #         "part_of_data" : 200,
        #         "epochs" : 10,
        #         "test_every_epochs" : 2,
        #         "T_max" :  5,
        #         "experiment_number" : np.random.randint(0,1000)
        #         }
        # parameters_mlp =    {
        #         "model_flag" : "MLP",
        #         "window_size" : 20,
        #         "h_size" : 24,
        #         "l_num" : 3,
        #         "learning_rate" : 0.001,
        #         "batch_size" : 100,
        #         "percentage_of_data" : 0.1,
        #         "cut_off_timesteps" : 0,
        #         "part_of_data" : 200,
        #         "epochs" : 10,
        #         "test_every_epochs" : 2,
        #         "T_max" : 5,
        #         "experiment_number" : np.random.randint(0,1000)
        #         }
        # parameters_tcn =    {
        #         "model_flag" : "TCN",
        #         "window_size" : 30,
        #         "h_size" : 8,
        #         "n_hidden" : 5,
        #         "levels" : 4,
        #         "kernel_size" : 7,
        #         "l_num" : 3,
        #         "dropout" : 0,
        #         "learning_rate" : 0.001,
        #         "batch_size" : 100,
        #         "percentage_of_data" : 0.1,
        #         "cut_off_timesteps" : 0,
        #         "part_of_data" : 200,
        #         "epochs" : 10,
        #         "test_every_epochs" : 2,
        #         "T_max" : 5,
        #         "experiment_number" : np.random.randint(0,1000)
        #         }

