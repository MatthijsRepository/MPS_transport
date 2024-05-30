import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

### These imports are done in the 'main()' function to avoid multiprocessing-related errors

#from MPS_TimeOp_no_locsize import MPS, Time_Operator
#from MPS_TimeOp import MPS, Time_Operator

#from MPS_initializations import *


##############################################################################################

def load_state(folder, name, new_ID):
    from MPS_TimeOp import MPS
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
    
def create_superket(State, newchi):
    from MPS_TimeOp import MPS
    """ create MPS of the density matrix of a given MPS """
    gammas, lambdas, locsize = State.construct_vidal_supermatrices(newchi)
    
    name = "DENS" + str(State.ID)
    newDENS = MPS(State.ID, State.N, State.d**2, newchi, True)
    newDENS.Gamma_mat = gammas
    newDENS.Lambda_mat = lambdas
    newDENS.locsize = locsize
    globals()[name] = newDENS
    return newDENS

##############################################################################################

def global_apply_twosite(TimeOp, normalize, Lambda_mat, Gamma_mat, locsize, d, chi):
        """ Applies a two-site operator to sites i and i+1 """
        #theta = self.contract(i,i+1) #(chi, chi, d, d)
        theta = np.tensordot(np.diag(Lambda_mat[0,:locsize[0]]), Gamma_mat[0,:,:locsize[0],:locsize[1]], axes=(1,1)) #(chi, d, chi)
        theta = np.tensordot(theta,np.diag(Lambda_mat[1,:locsize[1]]),axes=(2,0)) #(chi, d, chi) 
        theta = np.tensordot(theta, Gamma_mat[1,:,:locsize[1],:locsize[2]],axes=(2,1)) #(chi,d,d,chi)
        theta = np.tensordot(theta,np.diag(Lambda_mat[2,:locsize[2]]), axes=(3,0)) #(chi, d, d, chi)   
        #operator is applied, tensor is reshaped
        TimeOp = np.reshape(TimeOp, (d,d,d,d))
        theta_prime = np.tensordot(theta, TimeOp,axes=([1,2],[2,3])) #(chi,chi,d,d)     
        theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(d*locsize[0], d*locsize[2])) #first to (d, chi, d, chi), then (d*chi, d*chi)
        X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T

        if normalize:
            Lambda_mat[1,:locsize[1]] = Y[:locsize[1]] * 1/np.linalg.norm(Y[:locsize[1]])
        else:
            Lambda_mat[1,:locsize[1]] = Y[:locsize[1]]
        
        #truncation, and multiplication with the inverse lambda matrix of site i, where care is taken to avoid divides by 0
        X = np.reshape(X[:d*locsize[0], :locsize[1]], (d, locsize[0], locsize[1])) 
        inv_lambdas  = Lambda_mat[0, :locsize[0]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:locsize[0],:locsize[1]],axes=(1,1)) #(chi, d, chi)
        Gamma_mat[0,:,:locsize[0],:locsize[1]] = np.transpose(tmp_gamma,(1,0,2))
        
        #truncation, and multiplication with the inverse lambda matrix of site i+2, where care is taken to avoid divides by 0
        Z = np.reshape(Z[:d*locsize[2], :locsize[1]], (d, locsize[2], locsize[1]))
        Z = np.transpose(Z,(0,2,1))
        inv_lambdas = Lambda_mat[2, :locsize[2]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        Gamma_mat[1,:,:locsize[1],:locsize[2]] = np.tensordot(Z[:,:locsize[1],:locsize[2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
        return (Lambda_mat[1], Gamma_mat)
 

def global_apply_hopping(TimeOp_hopping, normalize, Lambda_mat, Gamma_mat, locsize, d, chi):
    #Swap
    Lambda_mat[2], Gamma_mat[1:3] = global_apply_twosite(swap_op, normalize, Lambda_mat[1:4], Gamma_mat[1:3], locsize[1:4], d, chi)
    
    #Apply gates
    Lambda_mat[1], Gamma_mat[0:2] = global_apply_twosite(TimeOp_hopping, normalize, Lambda_mat[:3], Gamma_mat[:2], locsize[:3], d, chi)
    Lambda_mat[3], Gamma_mat[2:4] = global_apply_twosite(TimeOp_hopping, normalize, Lambda_mat[2:5], Gamma_mat[2:4], locsize[2:5], d, chi)
    
    #Swap
    Lambda_mat[2], Gamma_mat[1:3] = global_apply_twosite(swap_op, normalize, Lambda_mat[1:4], Gamma_mat[1:3], locsize[1:4], d, chi)
    return (Lambda_mat[1:4], Gamma_mat)
       
def TEBD_Hub_multi(State, incl_SOC, TimeOp_Coul, TimeOp_hopping, TimeOp_SOC, diss_index, diss_TimeOp, normalize, diss_bool):
    """ Performing the TEBD steps in parallel using python's 'pool' method """
    
    #Coulomb interactions
    for j in [0,1]:
        new_matrices = p.starmap(global_apply_twosite, [(TimeOp_Coul, normalize, State.Lambda_mat[i:i+3], State.Gamma_mat[i:i+2], State.locsize[i:i+3], State.d, State.chi) for i in range(j, State.N-1, 2)])
        for i in range(j, State.N-1, 2):
            State.Lambda_mat[i+1] = new_matrices[int(i//2)][0]
            State.Gamma_mat[i:i+2] = new_matrices[int(i//2)][1]
      
    #Hopping interactions
    for j in [0,2]:
        new_matrices = p.starmap(global_apply_hopping, [(TimeOp_hopping, normalize, State.Lambda_mat[i:i+5], State.Gamma_mat[i:i+4], State.locsize[i:i+5], State.d, State.chi) for i in range(j, State.N-3, 4)])
        for i in range(j, State.N-3, 4):
            State.Lambda_mat[i+1:i+4] = new_matrices[int((i-j)//4)][0]
            State.Gamma_mat[i:i+4] = new_matrices[int((i-j)//4)][1]
            
    if incl_SOC:
        State.TEBD_SOC(TimeOp_SOC, normalize)
            
    
    if diss_bool:
        for i in range(len(diss_index)):
            State.apply_singlesite(diss_TimeOp[i], diss_index[i])
    pass 

##############################################################################################

def init_TimeOp():
    from MPS_TimeOp import Time_Operator
    """ Initialize time operator for the XXZ chain """
    TimeEvol_obj = Time_Operator(N, d, diss_bool, True)
    
    TimeEvol_obj.Ham_Coul = U_coulomb/4 * TimeEvol_obj.calc_dens_Ham_term(num_op, num_op, True)
    
    TimeEvol_obj.Ham_hopping = np.zeros((d**4, d**4), dtype=complex)
    TimeEvol_obj.Ham_hopping += -t_hopping/2 * TimeEvol_obj.calc_dens_Ham_term(Sx, Sx, True)
    TimeEvol_obj.Ham_hopping += -t_hopping/2 * TimeEvol_obj.calc_dens_Ham_term(Sy, Sy, True)
    
    TimeEvol_obj.TimeOp_Coul = TimeEvol_obj.Create_TimeOp(TimeEvol_obj.Ham_Coul, dt, use_CN)
    TimeEvol_obj.TimeOp_hopping = TimeEvol_obj.Create_TimeOp(TimeEvol_obj.Ham_hopping, dt, use_CN)
    
    
    if incl_SOC:
        from helix import create_helix_coordinates, create_v_vectors
        helix = create_helix_coordinates(N, R, pitch, dPhi, N_per_cycle, is_righthanded)
        v_list = create_v_vectors(helix)
        
        TimeEvol_obj.Ham_SOC = np.ones((int(N/2)-2,4, d**4, d**4), dtype=complex)
        TimeEvol_obj.Ham_SOC *= 1j * s_SOC * (TimeEvol_obj.calc_dens_Ham_term(Sp, Sm, True) - TimeEvol_obj.calc_dens_Ham_term(Sm, Sp, True))
        for i in range(int(N/2)-2):
            TimeEvol_obj.Ham_SOC[i,0] *= v_list[i,2]
            TimeEvol_obj.Ham_SOC[i,1] *= (v_list[i,0] - 1j*v_list[i,1])
            TimeEvol_obj.Ham_SOC[i,2] *= (v_list[i,0] + 1j*v_list[i,1])
            TimeEvol_obj.Ham_SOC[i,3] *= -1*v_list[i,2]
    
        TimeEvol_obj.TimeOp_SOC = np.ones((int(N/2)-2,4, d**4, d**4), dtype=complex)
        for i in range(int(N/2)-2):
            for j in range(4):
                TimeEvol_obj.TimeOp_SOC[i,j] = TimeEvol_obj.Create_TimeOp(TimeEvol_obj.Ham_SOC[i,j], dt, use_CN)
    else: TimeEvol_obj.TimeOp_SOC = None
    
    #Note, we must pass an imaginary time here, because the create_TimeOp function multiplies by -1j
    TimeEvol_obj.add_dissipative_term(0, np.array([np.sqrt(up_factor)*mu_min*Sp, np.sqrt(up_factor)*mu_plus*Sm]), 1j*dt, use_CN)
    TimeEvol_obj.add_dissipative_term(1, np.array([np.sqrt(down_factor)*mu_min*Sp, np.sqrt(down_factor)*mu_plus*Sm]), 1j*dt, use_CN)
    TimeEvol_obj.add_dissipative_term(N-2, np.array([mu_plus*Sp, mu_min*Sm]), 1j*dt, use_CN)
    TimeEvol_obj.add_dissipative_term(N-1, np.array([mu_plus*Sp, mu_min*Sm]), 1j*dt, use_CN)
    return TimeEvol_obj


def time_evolution(TimeEvol_obj, State, steps, track_Sz):
    """ Perform the time evolution steps and calculate the observables """
    if TimeEvol_obj.is_density != State.is_density:
        print("Error: time evolution operator type does not match state type (MPS/DENS)")
        return
    print(f"Starting time evolution of {State}")
    
    if track_n:
        State.n_expvals = np.zeros((State.N, steps))
    
    t1 = time.time() #Time evolution start time
    for t in range(steps):
        #if (t%2000==0 and t>0 and save_state_bool):
        #    State.store()
        if (t%20==0 and t>0):
            #print(str(t) + " / " + str(steps) + " (" + str(np.round(t/steps*100, decimals=0)) + "% completed)")
            print(str(t) + " / " + str(steps) + " (" + str(np.round(t/steps*100, decimals=0)) + "% completed), approx. " + str(np.round((steps/t - 1)*(time.time()-t1), decimals=0)) + "s left" )
            
        State.normalization = np.append(State.normalization, State.calculate_vidal_inner(State))
        if State.is_density:
            State.trace = np.append(State.trace, State.calculate_vidal_inner(NORM_state))
            
        if track_n:
            State.n_expvals[:,t], temp_trace = State.expval_chain(np.kron(num_op, np.eye(2)), NORM_state)
            State.n_expvals[:,t] *= 1/temp_trace
        
        #In currents for the up and down channels, middle site
        State.swap(current_site_index-1, normalize)
        State.spin_current_in = np.append(State.spin_current_in, np.real( State.expval_twosite(spin_current_op, current_site_index-2, NORM_state, normalize) ))
        State.spin_current_in_down = np.append(State.spin_current_in_down, np.real( State.expval_twosite(spin_current_op, current_site_index, NORM_state, normalize) ))
        State.swap(current_site_index-1, normalize)
        
        #Out currents for the up and down channels
        State.swap(current_site_index+1, normalize)
        State.spin_current_out = np.append(State.spin_current_out, np.real( State.expval_twosite(spin_current_op, current_site_index, NORM_state, normalize) ))
        State.spin_current_out_down = np.append(State.spin_current_out_down, np.real( State.expval_twosite(spin_current_op, current_site_index+2, NORM_state, normalize) ))       
        State.swap(current_site_index+1, normalize)
        
        State.cross_current = np.append(State.cross_current, np.real( State.expval_twosite(spin_current_op, current_site_index, NORM_state, normalize) ))
        
        
        if State.is_density:
            State.spin_current_in[-1] *= 1/State.trace[t]
            State.spin_current_out[-1] *= 1/State.trace[t]
            State.spin_current_in_down[-1] *= 1/State.trace[t]
            State.spin_current_out_down[-1] *= 1/State.trace[t]
            State.cross_current[-1] *= 1/State.trace[t]
        
        #State.TEBD_Hub(incl_SOC, TimeEvol_obj.TimeOp_Coul, TimeEvol_obj.TimeOp_hopping, TimeEvol_obj.TimeOp_SOC, TimeEvol_obj.diss_index, TimeEvol_obj.diss_TimeOp, normalize, diss_bool)
        TEBD_Hub_multi(State, incl_SOC, TimeEvol_obj.TimeOp_Coul, TimeEvol_obj.TimeOp_hopping, TimeEvol_obj.TimeOp_SOC, TimeEvol_obj.diss_index, TimeEvol_obj.diss_TimeOp, normalize, diss_bool)
    
    plt.plot(State.cross_current)
    plt.title("cross_current")
    plt.grid()
    plt.show()
    
    if steps>300:
        print("Cross current")
        print(np.average(State.cross_current[-100:]))
        
        print("Avg current up")
        print(np.average(State.spin_current_in[-100:] + State.spin_current_out[-100:])/2)
        print("Avg current down")
        print(np.average(State.spin_current_in_down[-100:] + State.spin_current_out_down[-100:])/2)
    pass


def plot_results(State):
    """ Plot the time evolution results """
    plt.plot(State.Lambda_mat[int(State.N/2)], linestyle="", marker=".")
    plt.title(f"Singular values of site {int(State.N/2)}")
    plt.grid()
    plt.show()
    
    plt.plot(State.normalization)
    plt.xlabel("Timesteps")
    plt.ylabel("Normalization")
    plt.show()
    
    if State.is_density:
        plt.plot(State.trace)
        plt.xlabel("Timesteps")
        plt.ylabel("Trace")
        plt.show()
        
    if hasattr(State, 'n_expvals'):
        for i in range(State.N):
            plt.plot(State.n_expvals[i], label=f"Site {i}")
        plt.xlabel("Timesteps")
        plt.ylabel("<n>")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.show()
        
    if plot_n_final:
        if hasattr(State, 'n_expvals'):
            occup = State.n_expvals[:,-1]
        else:
            occup, temp_trace = State.expval_chain(np.kron(num_op, np.eye(d)), NORM_state)
            occup *= 1/temp_trace
        even = np.arange(0, State.N, 2)
        odd = np.arange(1, State.N, 2)
        plt.plot(occup[even], linestyle="", marker=".", label="Even sites")
        plt.plot(occup[odd], linestyle="", marker=".", label="Odd sites")
        plt.xlabel("Physical sites")
        plt.ylabel("<n>")
        plt.legend()
        plt.grid()
        plt.show()
        print("Approx. Linear coefficients of bulk density profile:")
        print("Up: " + str((occup[State.N-4] - occup[2]) / (State.N-4)/2))
        print("Down: " + str((occup[State.N-3] - occup[3]) / (State.N-4)/2))
    
    plt.plot(State.spin_current_in, label="In")
    plt.plot(State.spin_current_out, label="Out")
    plt.plot( (State.spin_current_in + State.spin_current_out)/2, label="Average current")
    plt.title("Current through 'up' channel")
    plt.xlabel("Timesteps")
    plt.ylabel("Current")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.plot(State.spin_current_in[-100:], label="In")
    plt.plot(State.spin_current_out[-100:], label="Out")
    plt.plot( (State.spin_current_in[-100:] + State.spin_current_out[-100:])/2, label="Average current")
    plt.title("Current through 'up' channel")
    plt.xlabel("Timesteps")
    plt.ylabel("Current")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.plot(State.spin_current_in_down, label="In")
    plt.plot(State.spin_current_out_down, label="Out")
    plt.plot( (State.spin_current_in_down + State.spin_current_out_down)/2, label="Average current")
    plt.title("Current through 'down' channel")
    plt.xlabel("Timesteps")
    plt.ylabel("Current")
    plt.legend()
    plt.grid()
    plt.show()
    pass
    



##############################################################################################

max_cores = 4

t0 = time.time()
#### Simulation variables
N=8
d=2
chi=10      #MPS truncation parameter
newchi=60   #DENS truncation parameter

im_steps = 0
im_dt = -0.03j
steps=700
dt = 0.02


current_site_index = int(np.round(N/2))-2 #Site of which we will track the currents in and out of


normalize = True
use_CN = False #choose if you want to use Crank-Nicolson approximation
diss_bool = True
track_n = False
plot_n_final = True

incl_SOC = False

#### Hamiltonian and coupling constants
t_hopping = 1   #NOTE: t must be incorporated for spin current operator
U_coulomb = 1
s_SOC = 1


s_coup = 1
mu = 1
polarization_is_up = True

if incl_SOC:
    polarization_factor = 2
else:
    polarization_factor = 1

mu_plus = np.sqrt(s_coup*(1+mu))
mu_min = np.sqrt(s_coup*(1-mu))

if polarization_is_up:
    up_factor = s_coup
    down_factor = s_coup / polarization_factor
else:
    up_factor = s_coup / polarization_factor
    down_factor = s_coup
   
    
#### Helix constants
R = 1
pitch = 2
dPhi = None
N_per_cycle = N-1
is_righthanded = True
    

#### Spin matrices
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

#### JW-transformed number operator
num_op = np.array([[1,0],[0,0]])

#### Swap operator, used such that swaps can be performed using the global_apply_twosite function
swap_op = np.zeros((d**4,d**4))
for i in range(d**2):
    for j in range(d**2):
        swap_op[i*d**2 + j, j*d**2 +i] = 1


#### Spin current operators for DENS objects
#spin_current_op = -t_hopping*  1/2 * ( np.kron( np.kron(Sx, np.eye(d)) , np.kron(Sy, np.eye(d))) - np.kron( np.kron(Sy, np.eye(d)) , np.kron(Sx, np.eye(d))) )
spin_current_op = -t_hopping * 1j* ( np.kron( np.kron(Sp, np.eye(d)) , np.kron(Sm, np.eye(d))) - np.kron( np.kron(Sm, np.eye(d)) , np.kron(Sp, np.eye(d))) )

#### NORM_state definition, is initialized in main() function due to multiprocessing reasons
NORM_state = None

#### Loading and saving states
loadstate_folder = "data\\"
loadstate_filename = ""

savestate_folder = "data\\"
savestring = "mu" + str(int(mu*10)) + "_uf" + str(int(up_factor*10)) + "_s" + str(int(s_SOC*10)) + "_"

save_state_bool = False
load_state_bool = False

##############################################################################################

def main():
    #Import is done here instead of at the beginning of the code to avoid multiprocessing-related errors
    from MPS_TimeOp import MPS
    from MPS_initializations import initialize_halfstate, initialize_LU_RD
    from MPS_initializations import create_maxmixed_normstate, calculate_thetas_singlesite, calculate_thetas_twosite
    
    global NORM_state
    NORM_state = create_maxmixed_normstate(N, d, newchi)
    NORM_state.singlesite_thetas = calculate_thetas_singlesite(NORM_state)
    NORM_state.twosite_thetas = calculate_thetas_twosite(NORM_state)
        
    #load state or create a new one
    if load_state_bool:
        DENS1 = load_state(loadstate_folder, loadstate_filename, 1)
    else:
        MPS1 = MPS(1, N,d,chi, False)
        #MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_halfstate(N,d,chi)
        MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_LU_RD(N,d,chi, scale_factor = -0.8 )
        #temp = np.zeros((d,chi,chi))
        #temp[0,0,0] = np.sqrt(4/5)
        #temp[1,0,0] = 1/np.sqrt(5)
        #MPS1.Gamma_mat[0] = temp
        
        DENS1 = create_superket(MPS1, newchi)
    
    #creating time evolution object
    TimeEvol_obj1 = init_TimeOp()
    
    time_evolution(TimeEvol_obj1, DENS1, steps, track_n)
    
    plot_results(DENS1)
    
    if save_state_bool:
        DENS1.store(savestate_folder, savestring, True)
    pass
    

t0 = time.time()

if __name__=="__main__":
    p = Pool(processes=max_cores)
    main()
    p.close()

elapsed_time = time.time()-t0
print()
print(f"Elapsed simulation time: {elapsed_time}")







