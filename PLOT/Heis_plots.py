import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#### This is required on my system, advised to be removed on other systems
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from MPS_TimeOp import MPS
from MPS_initializations import create_maxmixed_normstate, calculate_thetas_singlesite
sys.path.pop(0)

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker



def load_state(folder, name, new_ID):
    """ loads a pickled state from folder 'folder' with name 'name' - note: name must include .pkl """
    filename = folder + name
    with open(filename, 'rb') as file:  
        loaded_state = pickle.load(file)
    globals()[loaded_state.name] = loaded_state
    
    loaded_state.ID = new_ID
    if loaded_state.is_density:
        loaded_state.name = "DENS"+str(new_ID)
    else: 
        loaded_state.name = "MPS"+str(new_ID)
    return loaded_state


Sz = np.array([[1,0],[0,-1]])


current_loglog = False
magnetization_profile = False
current_difference_dt = True


##### Plot the current
if current_loglog:
    N = np.linspace(5, 55)
    s_coup = 1
    
    asymptotic_values_Delta05 = s_coup * (np.sqrt(81 + 74*s_coup**2 + 9*s_coup**4) - 7 - 3*s_coup**2) / (4*(1+s_coup**2))
    asymptotic_values_Delta1 = np.pi**2 / N**2
    
    measure_xdata = np.array([15, 20, 25, 30, 35, 40, 45, 50])
    measured_values_Delta05 = np.array([0.350553, 0.350649, 0.350617, 0.350627, 0.350628, 0.350628, 0.350628, 0.350628])
    #measured_values_Delta1 = np.array([0.040193, 0.028404, 0.013697, 0.009153, 0.006460, 0.007796, 0.003575, 0.002761])
    corrected_values_Delta1 = np.array([0.044174, 0.025271, 0.016207, 0.011268, 0.008267, 0.0063287, 0.004992, 0.004037])
    
    plt.figure(dpi=200)
    plt.hlines(asymptotic_values_Delta05, 5,55, color="limegreen", linewidth=0.5)
    plt.loglog(N, asymptotic_values_Delta1, color="dodgerblue", linewidth=0.5)
    plt.ylim(0.001, 1)
    plt.xlim(10,55)
    plt.loglog(measure_xdata[:], measured_values_Delta05, linestyle="", marker=".", label="$\\Delta$=0.5", color="limegreen")
    plt.loglog(measure_xdata[:], corrected_values_Delta1, linestyle="", marker=".", label="$\\Delta$=1", color="dodgerblue")
    
    ax=plt.gca()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    
    plt.xlabel("Chain length")
    plt.ylabel("Current")
    plt.legend(bbox_to_anchor=(0.1, 0.2), loc="center left", borderaxespad=0)
    #plt.grid(True, which="both")
    plt.show()



#### Magnetization profile
if magnetization_profile:
    folder = "Heis_data\\"
    filename_Delta05 = "Delta05_HPC_DENS1_N50_chi40.pkl"
    filename_Delta1 = "Delta1_HPC_DENS1_N50_chi75.pkl"
    
    DENS05 = load_state(folder, filename_Delta05, 5)
    NORM_state_05 = create_maxmixed_normstate(DENS05.N, 2, DENS05.chi)
    NORM_state_05.singlesite_thetas = calculate_thetas_singlesite(NORM_state_05)
    
    DENS1 = load_state(folder, filename_Delta1, 1)
    NORM_state_1 = create_maxmixed_normstate(DENS1.N, 2, DENS1.chi)
    NORM_state_1.singlesite_thetas = calculate_thetas_singlesite(NORM_state_1)
    
    
    plt.figure(dpi=200)
    #plt.plot(DENS05.Lambda_mat[25], linestyle="", marker=".")
    plt.plot(DENS1.Lambda_mat[25], linestyle="", marker=".")
    plt.ylabel("$\\lambda_{\\alpha}$")
    plt.xlabel("$\\alpha$")
    plt.show()
    
    
    mag_profile_05, trace_05 = DENS05.expval_chain(np.kron(Sz, np.eye(2)), NORM_state_05)
    mag_profile_05 *= 1/trace_05
    
    mag_profile_1, trace_1 = DENS1.expval_chain(np.kron(Sz, np.eye(2)), NORM_state_1)
    mag_profile_1 *= 1/trace_1
    
    plt.figure(dpi=200)
    plt.plot(mag_profile_05, linestyle="", marker=".", label="$\Delta$=0.5")
    plt.plot(mag_profile_1, linestyle="", marker=".", label="$\Delta$=1")
    plt.ylabel("<$S_z$>")
    plt.xlabel("j")
    plt.legend()
    plt.show()


#### Current convergence with smaller timesteps, Delta=1
if current_difference_dt:
    folder = "Heis_data\\dt_investigation\\"
    
    filename_dt04 = "HPC_dt04_0520_1129_DENS1_N15_chi45.pkl"
    filename_dt02 = "HPC_dt02_0517_2236_DENS1_N15_chi80.pkl"
    filename_dt01 = "HPC_dt01_0520_1457_DENS1_N15_chi45.pkl"
    filename_dt005 = "HPC_dt005_0520_1439_DENS1_N15_chi45.pkl"
    
    filename_list = [filename_dt04, filename_dt02, filename_dt01, filename_dt005]
    dt_list = [0.04, 0.02, 0.01, 0.005]
    num_steps = np.divide(-200, dt_list)
    num_steps[-1] *= 1/2
    num_steps = np.ndarray.astype(num_steps, int)
    
    fig, axs = plt.subplots(2,2, dpi=200)
    
    for i in range(len(filename_list)):
        j = i//2
        DENS1 = load_state(folder, filename_list[i], 1)
        in_data, = axs[j,i-2*j].plot(DENS1.spin_current_values[num_steps[i]:], label="In")
        out_data, =axs[j,i-2*j].plot(DENS1.spin_current_out[num_steps[i]:], label="Out")
        avg_data, = axs[j,i-2*j].plot((DENS1.spin_current_values[num_steps[i]:] + DENS1.spin_current_out[num_steps[i]:])/2, label="$\\frac{In+Out}{2}$")
        
        axs[j,i-2*j].set_title(f"dt={dt_list[i]}")
        axs[j,i-2*j].set_ylim(0.033,0.055)
        axs[j,i-2*j].set_xlim(0)
    
    x_ticks = [0, 10000, 20000]
    x_ticklabels = ["$T_f -200$", "$T_f - 100$", "$T_f$"]
    
    axs[0,0].set_xticks([0,2500,5000])
    axs[0,1].set_xticks([0,5000,10000])
    
    for i in[0,1]:
        for j in [0,1]:
            axs[i,j].set_yticks([0.04,0.05])
    
    for ax in axs[0,:]:
        ax.set_xticklabels([])
    for ax in axs[1, :]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)
    for ax in axs[:,1]:
        ax.set_yticklabels([])
    #axs.set_ylim[0.035,0.055]
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Current', va='center', rotation='vertical')
    
    fig.legend(handles=[out_data, avg_data, in_data], bbox_to_anchor=(0.95, 0.5), loc="center left", ncol=1)
    
        
    plt.tight_layout(pad=2.5)
    plt.show()























