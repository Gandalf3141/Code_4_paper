

import torch
from meas_get_data import *
from meas_NN_classes import *
from meas_dataset import *
from meas_test_func_fs import *
import warnings

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


def plot_results(x, pred, rescale=False, window_size=1, sim_data_index=0, features=4):
    

    if rescale:
        x = normalize_invert(x)
        pred = normalize_invert(pred)

    if x.dim() == 3:
        x = x.view(x.size(dim=1), x.size(dim=2))
    if pred.dim() == 3:
        pred = pred.view(pred.size(dim=1), pred.size(dim=2))

    if features == 4:
        figure , axs = plt.subplots(4,1, figsize=(9,9))
        figure.tight_layout(pad=2.0)

        stepsize = 2 * 2e-5 
        time = np.linspace(0,x.size(dim=0)* stepsize, x.size(dim=0))

        axs[0].plot(time, x.detach().cpu().numpy()[:, 0], color="darkgreen", label="data")
        axs[0].set_title("NC : Input Voltage 1")
        axs[0].set_ylabel("[V]")

        axs[1].plot(time, x.detach().cpu().numpy()[:, 1], color="darkgreen", label="data")
        axs[1].set_title("NO : Input Voltage 2")
        axs[1].set_ylabel("[V]")

        axs[2].plot(time, pred.detach().cpu().numpy()[:, 2], color="red", label="pred")
        axs[2].plot(time, x.detach().cpu().numpy()[:, 2], color="blue", label="data", linestyle="dashed")
        axs[2].set_title("pressure")
        axs[2].set_ylabel("[Pa]")

        axs[3].plot(time, pred.detach().cpu().numpy()[:, 3], color="red", label="pred")
        axs[3].plot(time, x.detach().cpu().numpy()[:, 3], color="blue", label="data", linestyle="dashed")
        axs[3].set_title("position")
        axs[3].set_ylabel("[m]")
        axs[3].set_xlabel(f"time [s]")

        axs[2].axvline(x=time[window_size], color='black', linestyle='--', label='start of prediction')
        axs[3].axvline(x=time[window_size], color='black', linestyle='--', label='start of prediction')


        if rescale and features==4:
            u1_max = 200  #Spannung in [V]              ... [0, 200]
            u1_min = 0
            u2_max = 200
            u2_min = 0
            p_max = 3.5*1e5 #Druck in [bar]             ... [1, 3.5]
            p_min = 1.0*1e5 #Umgebungsdruck in [bar]
            s_max = 0.605*1e-3     #Position [m]          ... [0, 0.0006]
            s_min = 0.0

            axs[0].set_ylim(u1_min-10, u1_max+10)
            axs[1].set_ylim(u2_min-10, u2_max+10)
            axs[2].set_ylim(p_min-0.1*p_max, p_max+0.1*p_max)
            axs[3].set_ylim(s_min-+0.1*s_max, s_max+0.1*s_max)

    else:
        figure , axs = plt.subplots(12,1, figsize=(9,9))
        #figure.tight_layout(pad=2.0)

        stepsize = 0.1 # 100 ms ???
        time = np.linspace(0,x.size(dim=0)* stepsize, x.size(dim=0))

        for i in range(6):
            axs[i].plot(time, x.detach().cpu().numpy()[:, i], color="darkgreen", label="data")

            axs[i].set_title(f"tau {i}")
            axs[i].set_ylabel("[Nm]")
        for i in range(6,12):
            axs[i].plot(time, pred.detach().cpu().numpy()[:, i], color="red", label="pred")
            axs[i].plot(time, x.detach().cpu().numpy()[:, i], color="blue", label="data")

            axs[i].set_title(f"q {i}")
            axs[i].set_ylabel("[deg°]")

        #axs[2].axvline(x=time[window_size], color='black', linestyle='--', label='start of prediction')
        #axs[3].axvline(x=time[window_size], color='black', linestyle='--', label='start of prediction')

    for i in range(2*features):
        axs[i].grid(True)
        axs[i].legend()



    plt.grid(True)
    plt.legend()
    plt.show()

#old version
def plot_histogramm(error_position : list,
                    error_pressure : list,
                    error_position_simulink : list,
                    error_pressure_simulink : list)->None:
    
    SMALL_SIZE = 15
    MEDIUM_SIZE = 25
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsiz

    params = {'axes.titlesize': 20}
    plt.rcParams.update(params)

    fig, (ax1, ax2) = plt.subplots(2, 1)#, figsize=(9,9))  # Create a figure with 2 subplots

    bins = 5
    ax1.hist(error_position, alpha=0.5, label='position neural net', color='red', edgecolor = "black", hatch = "//", linewidth=2)
    ax1.hist(error_position_simulink, alpha=0.3, label='position simulink', color='green', edgecolor = "black", hatch = "||", linewidth=2)
    
    ax1.axvline(np.array(error_position).mean(), color='orange', linewidth=2, label=f"Mean {np.round(np.array(error_position).mean(),5)}: MSE position")
    ax1.axvline(np.array(error_position_simulink).mean(), color='blue', linewidth=2, label=f"Mean {np.round(np.array(error_position_simulink).mean(),5)}: MSE position simulink")
    
    ax1.set_xlabel('MSE position')
    ax1.legend()
    ax1.grid(True)

    # Plot C and D in the second subplot (ax2)
    ax2.hist(error_pressure, bins=bins, alpha=0.5, label='pressure neural net', color='red', edgecolor = "black", hatch = "//", linewidth=2)
    ax2.hist(error_pressure_simulink, bins=bins, alpha=0.3, label='pressure simulink', color='green', edgecolor = "black", hatch = "||", linewidth=2)

    ax2.axvline(np.array(error_pressure).mean(), color='orange', linewidth=2, label=f"Mean {np.round(np.array(error_pressure).mean(),5)}: MSE pressure")
    ax2.axvline(np.array(error_pressure_simulink).mean(), color='blue', linewidth=2, label=f"Mean {np.round(np.array(error_pressure_simulink).mean(),5)}: MSE pressure simulink")
    
    ax2.set_xlabel('MSE pressure')
    ax2.legend()
    ax2.grid(True)

    ax1.set_ylabel(f"frequency", fontsize=15)
    ax2.set_ylabel(f"frequency", fontsize=15)

    # Display the figure
    plt.tight_layout()
    plt.show()
    
    return


# test function for all networks - works on both datasets!
def test(
    data, 
    model, 
    window_size: int = 1, 
    display_plots: bool = False, 
    numb_of_inits: int = 1, 
    fix_random: bool = True, 
    rescale: bool = False, 
    specific_index: int = -1, 
    error_histogramm: bool = False
) -> float:

    if window_size==1:
        warnings.warn("Window size 1 is not supported!", UserWarning)

    if fix_random:
     np.random.seed(1234)
    else:
     np.random.seed(seed=None)
      
    test_inits = data.size(dim=0)
    ids = np.random.choice(test_inits, min([numb_of_inits, test_inits]), replace=False)
    ids = np.unique(ids)

    

    if specific_index >= 0:
        data = data[specific_index:specific_index+1,:, :]
    else :
        if not plot_histogramm: #use all data for histogramm
         data = data[ids,:, :]   

    loss_fn = nn.MSELoss()
    timesteps = data.size(dim=1)
    features = int(data.size(dim=2) / 2)

    total_loss = 0

    error_position = []
    error_pressure = []

    error_position_simulink = []
    error_pressure_simulink = []

    for i, x in enumerate(data):

        with torch.inference_mode():
            


            x=x.to(device)        
            x = x.view(1,x.size(dim=0), x.size(dim=1))                
            pred = torch.zeros((timesteps, 2*features), device=device)

            pred[0:window_size, :] = x[0, 0:window_size, :]
            pred[:, 0] = x[0, :, 0]

            x_test = x.clone()
            x_test[:,window_size:,features:] = 0
            x_test = x_test.to(device)

            # note that both OR and regular NNs use the same function during inferece
            # which is just the forward pass of the OR model
            if model.get_flag() in ["OR_LSTM", "LSTM", "OR_RNN", "RNN", "OR_GRU", "GRU"]:
                out, _ = model(x_test)
                pred[window_size:, features:] = out
 
            if model.get_flag() in ["OR_MLP", "MLP"]:
                out = model(x_test)
                pred[window_size:, features:] = out

            if model.get_flag() in ["OR_TCN", "TCN"]:
                out = model(x_test.transpose(1,2))
                pred[window_size:, features:] = out.squeeze(0).transpose(0,1)
            
            total_loss += loss_fn(pred[window_size:, features:], x[0, window_size:, features:]).detach().cpu().numpy()

            #error_position.append(loss_fn(pred[window_size:, 2:3], x[0, window_size:, 2:3]).detach().cpu().numpy())
            #error_pressure.append(loss_fn(pred[window_size:, 3:4], x[0, window_size:, 3:4]).detach().cpu().numpy())

            if display_plots:
                if specific_index>=0:
                    plot_results(x, pred, rescale=rescale, window_size=window_size, sim_data_index=specific_index, features=features)
                else:
                    plot_results(x, pred, rescale=rescale, window_size=window_size, sim_data_index=ids[i], features=features)
    
    #if error_histogramm:
        
        #plot_histogramm(error_position, error_pressure, error_position_simulink, error_pressure_simulink)

    return total_loss/data.size(0)