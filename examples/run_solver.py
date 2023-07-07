from py2d.Py2D_solver import Py2D_solver

# Script to call the function with the given parameters
Py2D_solver(Re=20e3, # Reynolds number
               fkx=4, # Forcing wavenumber in x
               fky=4, # Forcing wavenumber in y
               alpha=0.1, # Rayleigh drag coefficient
               beta=20, # Coriolis parameter
               NX=128, # Number of grid points in x and y '32', '64', '128', '256', '512'
               SGSModel_string='DLEITH', # SGS model to use 'NoSGS', 'SMAG', 'DSMAG', 'LEITH', 'DLEITH', 'PiOmegaGM2', 'PiOmegaGM4', 'PiOmegaGM6'
               eddyViscosityCoeff=0.17, # Coefficient for eddy viscosity models: SMAG and LEITH
               dt=5e-4, # Time step
               saveData=True, # Save data
               tSAVE=10, # Time interval to save data
               tTotal=50, # Total time of simulation
               readTrue=False, 
               ICnum=1, # Initial condition number: Choose between 1 to 20
               resumeSim=False, # tart new simulation (False) or resume simulation (True) 
               jobName='')
#%%
import scipy.io as sio
# directory = '/home/rm99/Mount/py2d/examples/results/Re20k_fkx4fky0_r0.1/DLEITH_/NX128/dt0.0005_IC1/data/'
directory = '/home/rm99/Mount/py2d/examples/results/Re20k_fkx4fky4_r0.1/DLEITH_/NX128/dt0.0005_IC1/data/'

filename = '5.mat'
mat_contents = sio.loadmat(directory+filename)
#%%
mat_contents.keys()
Omega = mat_contents['Omega']
#%%
plt.contourf(Omega,cmap='bwr',levels=110)
#%%
plt.pcolor(Omega)

#%%
plt.contourf(X,Y,Omega,cmap='bwr',levels=110)

