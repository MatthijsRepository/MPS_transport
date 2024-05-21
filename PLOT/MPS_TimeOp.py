import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from scipy.linalg import expm

import pickle
from datetime import datetime

########################################################################################################

class MPS:
    def __init__(self, ID, N, d, chi, is_density):
        self.ID = ID
        self.N = N
        self.d = d
        self.chi = chi
        self.is_density = is_density
        if is_density:
            self.name = "DENS"+str(ID)
            self.trace = np.array([])
        else: 
            self.name = "MPS"+str(ID)
        
        self.Lambda_mat = np.zeros((N+1,chi))
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)

        self.normalization = np.array([])
        self.locsize = np.zeros(N+1, dtype=int)     #locsize tells us which slice of the matrices at each site holds relevant information
        
        self.spin_current_in = np.array([])
        self.spin_current_out = np.array([])
        
        self.spin_current_in_2 = np.array([])       #Used for the spin-down channel in the Hubbard model
        self.spin_current_out_2 = np.array([])      #Used for the spin-down channel in the Hubbard model
        return
        
    def __str__(self):
        if self.is_density:
            return f"DENS{self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
        else:
            return f"MPS{self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
            
    def store(self, folder, custom_string, include_timestr):
        """ Stores the object to memory using pickle """
        if include_timestr:
            time = str(datetime.now())
            timestr = time[5:7] + time[8:10] + "_" + time[11:13] + time[14:16] + "_"  #get month, day, hour, minute
        else:
            timestr = ""
        
        filename = custom_string + timestr+self.name+"_N"+str(self.N)+"_chi"+str(self.chi)+".pkl"
        
        file = open(folder + filename, 'wb')
        pickle.dump(self, file)

        print(f"Stored as {folder+filename}")
        pass        
          
    def construct_vidal_supermatrices(self, newchi):
        """ Constructs the matrices for the superket MPDO in Vidal decomposition for this MPS -- see the function 'create superket' """
        sup_Gamma_mat = np.zeros((self.N, self.d**2, newchi, newchi), dtype=complex)
        sup_Lambda_mat = np.zeros((self.N+1, newchi))
        for i in range(self.N):
            sup_Gamma_mat[i,:,:,:] = np.kron(self.Gamma_mat[i], np.conj(self.Gamma_mat[i]))[:,:newchi,:newchi]
            sup_Lambda_mat[i,:] = np.kron(self.Lambda_mat[i], self.Lambda_mat[i])[:newchi]
        sup_Lambda_mat[self.N,:] = np.kron(self.Lambda_mat[self.N], self.Lambda_mat[self.N])[:newchi]
        sup_locsize = np.minimum(self.locsize**2, newchi)
        return sup_Gamma_mat, sup_Lambda_mat, sup_locsize
    
    def contract(self, begin, end):
        """ Contracts the gammas and lambdas between sites 'begin' and 'end' """
        theta = np.diag(self.Lambda_mat[begin,:self.locsize[begin]]).copy()
        theta = theta.astype(complex)
        for i in range(end-begin+1):
            theta = np.tensordot(theta, self.Gamma_mat[begin+i,:,:self.locsize[begin+i],:self.locsize[begin+i+1]], axes=(-1,1)) #(chi,...,d,chi)
            theta = np.tensordot(theta, np.diag(self.Lambda_mat[begin+i+1, :self.locsize[begin+i+1]]), axes=(-1,1)) #(chi,...,d,chi)
        theta = np.rollaxis(theta, -1, 1) #(chi, chi, d, ..., d)
        return theta
    
    def decompose_contraction(self, theta, i, normalize):
        """ decomposes a given theta back into Vidal decomposition. i denotes the leftmost site contracted into theta """
        """ NOTE: theta must be in format (chi, chi, d, d, d, ...), i.e. the bond indices must come first """
        num_sites = np.ndim(theta)-2 # The number of sites contained in theta
        temp = num_sites-1           # Total number of loops required
        for j in range(temp):
            theta = theta[:self.locsize[i+j], :self.locsize[i+num_sites]]
            theta = theta.reshape((self.locsize[i+j], self.locsize[i+num_sites], self.d, self.d**(temp-j)))
            theta = theta.transpose(2,0,3,1) #(d, chi, d**(temp-j), chi)
            theta = theta.reshape((self.d*self.locsize[i+j], self.d**(temp-j)*self.locsize[i+num_sites]))
            X, Y, Z = np.linalg.svd(theta); Z = Z.T
            #This can be done more efficiently by leaving out the Z=Z.T and only doing so in case of j==2
            
            if normalize==True:
                self.Lambda_mat[i+j+1,:self.locsize[i+j+1]] = Y[:self.locsize[i+j+1]] *1/np.linalg.norm(Y[:self.locsize[i+j+1]])
            else:
                self.Lambda_mat[i+j+1,:self.locsize[i+j+1]] = Y[:self.locsize[i+j+1]]
                
            self.Lambda_mat[i+j+1] = np.round(self.Lambda_mat[i+j+1], decimals=18)
            
            X = np.reshape(X[:self.d*self.locsize[i+j],:self.locsize[i+j+1]], (self.d, self.locsize[i+j], self.locsize[i+j+1]))
            inv_lambdas = self.Lambda_mat[i+j, :self.locsize[i+j]].copy()
            inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
            X = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i+j],:self.locsize[i+j+1]],axes=(1,1)) #(chi, d, chi)
            X = X.transpose(1,0,2)
            self.Gamma_mat[i+j, :, :self.locsize[i+j],:self.locsize[i+j+1]] = X

            theta_prime = np.reshape(Z[:self.locsize[i+num_sites]*self.d**(temp-j),:self.locsize[i+j+1]], (self.d**(temp-j), self.locsize[i+num_sites], self.locsize[i+j+1]))
            theta_prime = theta_prime.transpose(0,2,1)
            if j==(temp-1):
                inv_lambdas  = self.Lambda_mat[i+j+2, :self.locsize[i+j+2]].copy()
                inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
                tmp_gamma = np.tensordot(theta_prime[:,:self.locsize[i+j+1],:self.locsize[i+j+2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
                self.Gamma_mat[i+j+1, :, :self.locsize[i+j+1],:self.locsize[i+j+2]] = tmp_gamma 
            else:
                #Here we must contract Lambda with V for the next SVD. The contraction runs over the correct index (the chi resulting from the previous SVD, not the one incorporated with d**(temp-j))
                theta_prime = np.tensordot(np.diag(Y[:self.locsize[i+j+1]]), theta_prime, axes=(1,1))
        return
    
    def apply_singlesite(self, TimeOp, i):
        """ Applies a single-site operator to site i """
        theta = self.contract(i,i)
        theta_prime = np.tensordot(theta, TimeOp, axes=(2,1)) #(chi, chi, d)

        inv_lambdas  = self.Lambda_mat[i,:self.locsize[i]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(np.diag(inv_lambdas), theta_prime, axes=(1,0)) #(chi, chi, d) 
        
        inv_lambdas = self.Lambda_mat[i+1,:self.locsize[i+1]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(theta_prime, np.diag(inv_lambdas), axes=(1,0)) #(chi, d, chi)
        self.Gamma_mat[i,:,:self.locsize[i],:self.locsize[i+1]] = np.transpose(theta_prime, (1,0,2))
        return
    
    def apply_twosite(self, TimeOp, i, normalize):
        """ Applies a two-site operator to sites i and i+1 """
        theta = self.contract(i,i+1) #(chi, chi, d, d)
        theta = theta.reshape(self.locsize[i], self.locsize[i+2], self.d**2)

        theta_prime = np.tensordot(theta, TimeOp, axes=(2,1))
        theta_prime = theta_prime.reshape((self.locsize[i], self.locsize[i+2], self.d, self.d))

        self.decompose_contraction(theta_prime, i, normalize)
        return 
    
    def swap(self, i, normalize):
        theta = self.contract(i,i+1)
        theta = theta.transpose(0,1,3,2)
        self.decompose_contraction(theta, i, normalize)
        pass
    
    def TEBD_Heis(self, TimeOp_Heis, diss_index, diss_TimeOp, normalize, diss_bool):
        """ TEBD algorithm for the Heisenberg model """
        #Regular TEBD steps
        for i in range(0, self.N-1, 2):
            self.apply_twosite(TimeOp_Heis, i, normalize)
        for i in range(1, self.N-1, 2):
            self.apply_twosite(TimeOp_Heis, i, normalize)
        
        #Dissipative steps
        if diss_bool:
            for i in range(len(diss_index)):
                self.apply_singlesite(diss_TimeOp[i], diss_index[i])        
        return
    
    def TEBD_SOC(self, TimeOp_SOC, normalize):
        """ TEBD algorithm for the spin-orbit coupling terms """
        for i in range(0, 1):
            #Swap sites i+2 and i+3 with sites i+4 and i+5
            self.swap(i+3, normalize)
            self.swap(i+4, normalize)
            self.swap(i+2, normalize)
            self.swap(i+3, normalize)
            
            #"""
            for j in (i, i+4):
                if j>=self.N-3:
                    break #if the physical system has odd length this prevents the code from exceeding N
                self.swap(j, normalize)
                self.apply_twosite(TimeOp_SOC[int(j/4),0], j+1, normalize) #j up & j+2 up
                self.swap(j+2, normalize)
                self.apply_twosite(TimeOp_SOC[int(j/4),1], j+1, normalize) #j up & j+2 down
                self.swap(j, normalize)
                self.apply_twosite(TimeOp_SOC[int(j/4),3], j+1, normalize) #j down & j+2 down
                self.swap(j+2, normalize)
                self.apply_twosite(TimeOp_SOC[int(j/4),2], j+1, normalize) #j down & j+2 up
            #"""
            #Swap sites i+2 and i+3 with sites i+4 and i+5 back to their original positions
            self.swap(i+3, normalize)
            self.swap(i+4, normalize)
            self.swap(i+2, normalize)
            self.swap(i+3, normalize)
        pass
            
    def TEBD_Hub(self, include_SOC, TimeOp_Coul, TimeOp_hop, TimeOp_SOC, diss_index, diss_TimeOp, normalize, diss_bool):
        """ TEBD algorithm for the Hubbard model """
        #Coulomb terms        
        for i in range(0,self.N-1,2):
            self.apply_twosite(TimeOp_Coul, i, normalize)
         #Hopping terms       
        for i in (0,2):
            for j in range(i, self.N-3, 4):
                # Apply swap (2,3) -> (3,2)
                self.swap(j+1, normalize)
                
                self.apply_twosite(TimeOp_hop, j, normalize)
                self.apply_twosite(TimeOp_hop, j+2, normalize)
                
                # Apply swap (3,2) -> (2,3)
                self.swap(j+1, normalize)
        
        #Spin-Orbit Coupling terms
        if include_SOC:
            self.TEBD_SOC(TimeOp_SOC, normalize)
            
        #Dissipative terms
        if diss_bool:
            for i in range(len(diss_index)):
                self.apply_singlesite(diss_TimeOp[i], diss_index[i])
        return
    
    def expval(self, Op, site, NORM_state):
        """ Calculates the expectation value of an operator Op for a single site """
        if self.is_density:     #In case of density matrices we must take the trace  
            Gamma_temp = self.Gamma_mat[site].copy()
            self.apply_singlesite(Op, site)
            result = self.calculate_vidal_inner(NORM_state)
            self.Gamma_mat[site] = Gamma_temp
            return result
        else:
            theta = self.contract(site,site) #(chi, chi, d)
            theta_prime = np.tensordot(theta, Op, axes=(2,1)) #(chi, chi, d)
            return np.real(np.tensordot(theta_prime, np.conj(theta), axes=([0,1,2],[0,1,2])))
    
    def expval_chain(self, Op, NORM_state):
        """ Calculates expectation values from the left side, by reusing the already
            contracted part left of the site we want to know our expectation value of
            after completing calculation, also immediately returns the normalization of the state """
        expvals = np.zeros(self.N)
        Left_overlap = np.eye(self.chi)
        
        for i in range(self.N):
            temp_Gamma = self.Gamma_mat[i].copy()
            #Calculating expectation value for site i, using Left_overlap on the left, and constructing the rest of the overlap in the regular way
            self.apply_singlesite(Op, i)
            st1 = np.tensordot(self.Gamma_mat[i,:,:,:],np.diag(self.Lambda_mat[i+1,:]), axes=(2,0)) #(d, chi, chi)
            sub_expval = np.tensordot(Left_overlap, np.conj(st1), axes=(0,1)) #(chi, d, chi)
            sub_expval = np.tensordot(sub_expval, NORM_state.singlesite_thetas, axes=([1,0],[0,1])) #(chi, chi)
            for j in range(i+1, self.N):
                temp = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
                sub_expval = np.tensordot(sub_expval, np.conj(temp), axes=(0,1)) #(chi, d, chi)
                sub_expval = np.tensordot(sub_expval, NORM_state.singlesite_thetas, axes=([1,0],[0,1])) #(chi, chi)
            expvals[i] = np.real(sub_expval[0,0])
            
            #Placing the old Gamma matrix back at the correct place
            self.Gamma_mat[i] = temp_Gamma
            
            #Updating Left_overlap
            st1 = np.tensordot(self.Gamma_mat[i,:,:,:],np.diag(self.Lambda_mat[i+1,:]), axes=(2,0)) #(d, chi, chi)
            Left_overlap = np.tensordot(Left_overlap, np.conj(st1), axes=(0,1)) #(chi, d, chi)
            Left_overlap = np.tensordot(Left_overlap, NORM_state.singlesite_thetas, axes=([1,0],[0,1])) #(chi, chi)
        norm = np.real(Left_overlap[0,0])
        return expvals, norm
    
    def expval_twosite(self, Op, site, NORM_state, normalize):
        """ Calculates the expectation value of an operator Op that acts on two sites """
        if self.is_density:
            temp_gamma = self.Gamma_mat[site:site+2].copy()
            temp_lambda = self.Lambda_mat[site+1].copy()
            self.apply_twosite(Op, site, normalize)
            result = self.calculate_vidal_inner(NORM_state)
            self.Gamma_mat[site:site+2] = temp_gamma
            self.Lambda_mat[site+1] = temp_lambda
            return result
        else:
            Op = np.reshape(Op, (self.d,self.d,self.d,self.d))
            theta = self.contract(site,site+1) #(chi, chi, d, d)
            theta_prime = np.tensordot(theta, Op, axes=(2,3,2,3)) #(chi, chi, d, d)
            return np.real(np.tensordot(theta_prime, np.conj(theta), axes=([0,1,2,3],[0,1,2,3])))
        
    def calculate_vidal_inner(self, MPS2):
        """ Calculates the inner product of the MPS with another MPS """
        m_total = np.eye(self.chi)
        temp_gammas, temp_lambdas = MPS2.Gamma_mat, MPS2.Lambda_mat  #retrieve gammas and lambdas of MPS2
        for j in range(0, self.N):
            st1 = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            st2 = np.tensordot(temp_gammas[j,:,:,:],np.diag(temp_lambdas[j+1,:]), axes=(2,0)) #(d, chi, chi)
            m_total = np.tensordot(m_total, np.conj(st1), axes=(0,1)) #(chi, d, chi)
            m_total = np.tensordot(m_total, st2, axes=([1,0],[0,1])) #(chi, chi)
        return np.real(m_total[0,0])

########################################################################################    
class Time_Operator:
    def __init__(self,N, d, diss_bool, is_density):
        self.N = N
        self.d = d
        self.is_density = is_density
        self.diss_bool = diss_bool
        
        self.diss_index = np.array([],dtype=int)
        self.diss_TimeOp = []
        return
    
    def calc_dens_Ham_term(self, site1, site2, is_twosite):
        "Given operators, creates the singlesite or twosite Hamiltonian term for the MPDO"
        if is_twosite == False:
            return np.kron(site1,np.eye(self.d)) - np.transpose(np.kron(np.eye(self.d),site1))
        else:
            term1 = np.kron(np.kron(site1, np.eye(self.d)), np.kron(site2, np.eye(self.d)))
            term2 = np.kron(np.kron(np.eye(self.d), site1), np.kron(np.eye(self.d), site2))
            return term1 - np.transpose(term2)

    def Create_TimeOp(self, Ham, dt, use_CN):
        """ Computes the matrix exponential or Crank_Nicolson approximation of a given square matrix, including a timestep """
        if use_CN:
            U = self.create_crank_nicolson(Ham, dt)
        else:
            U = expm(-1j*dt*Ham)
        U = np.around(U, decimals=15)        #Rounding out very low decimals 
        return U

    def create_crank_nicolson(self, H, dt):
        """ Creates the Crank-Nicolson operator from a given Hamiltonian """
        H_top=np.eye(H.shape[0])-1j*dt*H/2
        H_bot=np.eye(H.shape[0])+1j*dt*H/2
        return np.linalg.inv(H_bot).dot(H_top)

    def calc_diss_site(self, Lind_Op):
        """ Creates the dissipative term for a single site """
        """ Lind_Op is shape (k,d,d) or (d,d) -- the k-index is in case multiple different lindblad operators act on a single site """
        Diss = np.zeros((self.d**2, self.d**2), dtype=complex)
        if Lind_Op.ndim==2:     #If only a single operator is given, this matrix is used
            Diss += 2*np.kron(Lind_Op, np.conj(Lind_Op))
            Diss -= np.kron(np.matmul(np.conj(np.transpose(Lind_Op)), Lind_Op), np.eye(self.d))
            Diss -= np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op), np.conj(Lind_Op)))
        else:                   #If multiple matrices are given, the sum of Lindblad operators is used
            for i in range(np.shape(Lind_Op)[0]):
                Diss += 2*np.kron(Lind_Op[i], np.conj(Lind_Op[i]))
                Diss -= np.kron(np.matmul(np.conj(np.transpose(Lind_Op[i])), Lind_Op[i]), np.eye(self.d))
                Diss -= np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op[i]), np.conj(Lind_Op[i])))
        return Diss
    
    def add_dissipative_term(self, site, Op, dt, use_CN):
        """ Construct dissipative term for a (or multiple) operator(s) 'Op', which act on site 'index' """
        self.diss_index = np.append(self.diss_index, site)
        temp_TimeOp = self.Create_TimeOp(self.calc_diss_site(Op), dt, use_CN)
        self.diss_TimeOp.append(temp_TimeOp)
        return
