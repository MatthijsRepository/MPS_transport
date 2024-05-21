import numpy as np
import matplotlib.pyplot as plt

def create_helix_coordinates(N, R, pitch, dPhi, N_per_cycle, is_righthanded):
    if dPhi==None:
        dPhi = 2*np.pi/N_per_cycle
    else:
        N_per_cycle = 2*np.pi/dPhi
    
    coordinates = np.zeros((3,N))
    
    angles = np.arange(N)*dPhi
    dZ = pitch/N_per_cycle
    
    if is_righthanded==False:
        angles *= -1
    
    coordinates[0] = R*np.cos(angles)
    coordinates[1] = R*np.sin(angles)
    coordinates[2] = dZ*np.arange(N)
    coordinates = np.around(coordinates, decimals=10)
    return coordinates.transpose()

def create_v_vectors(lattice):
    #Creating d_m+1 vectors
    d_m1 = lattice[:-1] - lattice[1:]
    d_m1 = d_m1 / np.linalg.norm(d_m1, axis=1)[:, np.newaxis]
   
    #Creating d_m+2 vectors
    d_m2 = lattice[:-2] - lattice[2:]
    d_m2 = d_m2 / np.linalg.norm(d_m2, axis=1)[:, np.newaxis]
    
    #Calculating v_m vectors
    return np.cross(d_m1[:-1], d_m2)




N=8
R=5
pitch=10
dPhi=None
N_per_cycle = N-1

helix = create_helix_coordinates(N, R, pitch, dPhi, N_per_cycle, True)
print(helix)

vm = create_v_vectors(helix)
print(vm)



#"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(helix[:,0], helix[:,1], helix[:,2], marker='o')
plt.show()    
#"""  
    















