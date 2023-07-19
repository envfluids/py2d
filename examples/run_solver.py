from py2d.Py2D_solver import Py2D_solver

# Script to call the function with the given parameters
Py2D_solver(Re=20e3, # Reynolds number
               fkx=4, # Forcing wavenumber in x
               fky=0, # Forcing wavenumber in y
               alpha=0.1, # Rayleigh drag coefficient
               beta=0, # Coriolis parameter
               NX=128, # Number of grid points in x and y '32', '64', '128', '256', '512'
               SGSModel_string='DLEITH', # SGS model to use 'NoSGS', 'SMAG', 'DSMAG', 'LEITH', 'DLEITH', 'PiOmegaGM2', 'PiOmegaGM4', 'PiOmegaGM6'
               eddyViscosityCoeff=0.17, # Coefficient for eddy viscosity models: SMAG and LEITH
               dt=5e-4, # Time step
               saveData=True, # Save data
               tSAVE=0.1, # Time interval to save data
               tTotal=1, # Total time of simulation
               readTrue=False, 
               ICnum=1, # Initial condition number: Choose between 1 to 20
               resumeSim=False, # tart new simulation (False) or resume simulation (True) 
               )
