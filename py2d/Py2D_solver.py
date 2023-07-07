# ----------------------------------------------------------------------
# Created : Yifei Guan, Rambod Mojgani  2023
# Revised : Moein Darman | Karan Jakhar May 2023
# ----------------------------------------------------------------------

# 2D Turbulence Solver using Fourier-Fourier Pseudo-spectral Method
# Navier-Stokes equation is in vorticity-stream function form

# Import os module
import os
os.chdir('../../py2d/')
from pathlib import Path

# Import Python Libraries
from prettytable import PrettyTable
import jax
from jax import jit
import numpy as nnp
import jax.numpy as np
from scipy.io import loadmat, savemat
import time as runtime
from timeit import default_timer as timer
print(jax.default_backend())
print(jax.devices())

# Import Custom Module
from py2d.convection_conserved import *
from py2d.convert import *
from py2d.aposteriori_analysis import eddyTurnoverTime_2DFHIT
from py2d.SGSModel import *
# from py2d.uv2tau_CNN import *
from py2d.initialize import *

# Enable x64 Precision for Jax
jax.config.update('jax_enable_x64', True)

## -------------- Initialize the kernels in JIT --------------
Omega2Psi_2DFHIT_jit = jit(Omega2Psi_2DFHIT)
Psi2Omega_2DFHIT_jit = jit(Psi2Omega_2DFHIT)
Psi2UV_2DFHIT_jit = jit(Psi2UV_2DFHIT)
Tau2PiOmega_2DFHIT_jit = jit(Tau2PiOmega_2DFHIT)
initialize_wavenumbers_2DFHIT_jit = jit(initialize_wavenumbers_2DFHIT)
prepare_data_cnn_jit = jit(prepare_data_cnn)
postproccess_data_cnn_jit = jit(postproccess_data_cnn)
eddyTurnoverTime_2DFHIT_jit = jit(eddyTurnoverTime_2DFHIT)

# Start timer
startTime = timer()

def Py2D_solver(Re, fkx, fky, alpha, beta, NX, SGSModel_string, eddyViscosityCoeff, dt, saveData, tSAVE, tTotal, readTrue, ICnum, resumeSim, jobName=''):
    
    # -------------- RUN Configuration --------------
    # Use random initial condition or read initialization from a file or use
    #     readTrue = False
    # False: IC from file
    # True: Random IC (not validated)

    # Read the following initial condition from a file. ICnum can be from 1 to 20.
    #     ICnumx = 5 # 1 to 20

    # Resume simulation from last run
    #     resumeSim = False
    # True: You are ontinuing the simulation
    # False: You are starting the simulation from a given IC (First simulation)

    # Save data at the end of the simulation
    #     saveData = True

    # Save snapshot at every time step 
    #     tSAVE = 0.1

    # Length of simulation (in time)
    #     tTotal = 1

    # -------------- Geometry and mesh Parameters --------------
    # Number of grid points in each direction
    #     NX = 128

    # -------------- Flow specifications --------------
    # Reynolds number
    #     Re = 20e3

    # Time step
    #     dt = 5e-4

    # Density
    rho = 1
    
    # Kinematic Viscosity
    nu = 1.0 / Re
    
    # Linear drag coefficient
    #     alpha = 0.1

    # SGS Model
    #     SGSModel_string = 'NoSGS' # SMAG DSMAG LEITH DLEITH CNN GAN
    
    # Eddy Viscosity Coefficient
    # eddyViscosityCoeff = 0.1 needed for SMAG and LEITH

    # SGS Model - PiOmega numerical scheme (Crank nicholson scheme used for time integration and eddy viscosity)
    PiOmega_numerical_scheme = 'E1' # E1: Euler 1st order, AB2: Adam Bashforth 2nd order

    # -------------- Deterministic forcing Parameters--------------
    # Wavenumber in x direction
    #     fkx = 4

    # Wavenumber in y direction
    #     fky = 0

    # -------------- Geometry and mesh Calculation --------------

    # Domain length
    Lx = 2 * np.pi
    
    # Filter Width
    Delta = 2 * Lx / NX
    
    Lx, _, X, Y = gridgen(Lx, NX)
    # -------------- Create the meshgrid both in physical and spectral space --------------
    Kx, Ky, Ksq = initialize_wavenumbers_2DFHIT(NX, NX, Lx, Lx)

    # Numpy to jax
    X = np.array(X)
    Y = np.array(Y)    
    Kx = np.array(Kx)
    Ky = np.array(Ky)
    Ksq = np.array(Ksq)

    # -------------- Deterministic forcing Calculation --------------

    # Deterministic forcing in Physical space
    Fk = fky * np.cos(fky * Y) + fkx * np.cos(fkx * X)
    
    # Deterministic forcing in Fourier space
    Fk_hat = np.fft.fft2(Fk)
    
    # -------------- RUN Configuration --------------

    # Save data at every Nth iteration
    NSAVE = int(tSAVE / dt)
    
    # Total number of iterations
    maxit = int(tTotal / dt)
    
    # -------------- Directory to store data ------------------
    # Snapshots of data save at the following directory
    dataType_DIR = 'Re' + str(int(Re / 1000)) + 'k_fkx' + str(fkx) + 'fky' + str(fky) + '_r' + str(alpha) + '/'
    SAVE_DIR = 'results/' + dataType_DIR + SGSModel_string + '_' + jobName + '/NX' + str(NX) + '/dt' + str(dt) + '_IC' + str(ICnum) + '/'
    SAVE_DIR_DATA = SAVE_DIR + 'data/'
    SAVE_DIR_IC = SAVE_DIR + 'IC/'

    # Create directories if they aren't present
    try:
        os.makedirs(SAVE_DIR_DATA)
        os.makedirs(SAVE_DIR_IC)
    except OSError as error:
        print(error)

    # -------------- Print the run configuration --------------

    # Create pretty table instances for each category
    table_run_config = PrettyTable()
    table_geometry_mesh = PrettyTable()
    table_flow_spec = PrettyTable()

    # Add the columns and their corresponding values for each table
    table_run_config.field_names = ["Run Configuration", "Value"]
    table_geometry_mesh.field_names = ["Geometry and Mesh", "Value"]
    table_flow_spec.field_names = ["Flow Specification", "Value"]

    # Add variables and their values to the Run Configuration table
    table_run_config.add_row(["Time Step (dt)", dt])
    table_run_config.add_row(["Resume Simulation", resumeSim])    
    table_run_config.add_row(["Read Initialization (readTrue), If False: Will read IC from a file", readTrue])
    table_run_config.add_row(["Saving Data  (saveData)", saveData])
    table_run_config.add_row(["Save data every t th timestep (tSAVE)", tSAVE])
    table_run_config.add_row(["Save data every Nth iteration (NSAVE)", NSAVE])
    table_run_config.add_row(["Length of simulation (tTotal)", tTotal])
    table_run_config.add_row(["Maximum Number of Iterations (maxit)", maxit])

    # Add variables and their values to the Geometry and Mesh table
    table_geometry_mesh.add_row(["Number of Grid Points (NX)", NX])
    table_geometry_mesh.add_row(["Domain Length (L)", Lx])
    table_geometry_mesh.add_row(["Mesh size (dx)", dx])

    # Add variables and their values to the Flow Specification table
    table_flow_spec.add_row(["Reynolds Number (Re)", Re])
    table_flow_spec.add_row(["Linear Drag Coefficient (alpha)", alpha])
    table_flow_spec.add_row(["Deterministic forcing (f_kx, f_ky)", [fkx,fky]])
    table_flow_spec.add_row(["SGS Model", SGSModel_string])
    table_flow_spec.add_row(["Eddy Viscosity (Only for SMAG and LEITH", SGSModel_string])

    # directory to save data
    table_flow_spec.add_row(["Saving Directory (SAVE_DIR)", SAVE_DIR])

    # Print the tables
    print(table_run_config)
    print(table_geometry_mesh)
    print(table_flow_spec)

    # -------------- Initialization Section--------------
    print("-------------- Initialization Section--------------")

    # -------------- Initialize PiOmega Model --------------

    PiOmega_eddyViscosity_model = SGSModel()  # Initialize SGS Model
    PiOmega_eddyViscosity_model.set_method(SGSModel_string) # Set SGS model to calculate PiOmega and Eddy Viscosity
    
    if SGSModel_string == 'CNN':
        model_path = "best_model_mcwiliams_exact.pt"
        cnn_model = init_model(model_type='mcwiliams', model_path=model_path) 

    if readTrue:

        # -------------- Initialization using pertubration --------------
        w1_hat, psi_hat, psiPrevious_hat, psiCurrent_hat = initialize_perturbation(NX, Kx, Ky)
        time = 0.0

    else:
        
        if resumeSim:
            # Get the last file name (filenames are integers)
            last_file_number_data = get_last_file(SAVE_DIR_DATA)
            last_file_number_IC = get_last_file(SAVE_DIR_IC)

            # Print last file names (filenames are integers)
            if last_file_number_data is not None:
                print(f"Last data file number: {last_file_number_data}")
            else:
                print("No .mat files found")

            if last_file_number_IC is not None:
                print(f"Last IC file number: {last_file_number_IC}")
            else:
                raise ValueError("No .mat initialization files found to resume the simulation")

            # Load initial condition to resume simulation
            # Resume from the second last saved file - 
            # the last saved file is often corrupted since the jobs stop (reach wall clocktime limit) while the file is being saved.
            last_file_number_data = last_file_number_data - 1
            last_file_number_IC = last_file_number_IC - 1

            data_Poi = loadmat(SAVE_DIR_IC + str(last_file_number_IC) + '.mat')
            Omega0_hat_cpu = data_Poi["Omega0_hat"]
            Omega1_hat_cpu = data_Poi["Omega1_hat"]
            time = data_Poi["time"]

            # Convert numpy initialization arrays to jax array
            Omega0_hat = np.array(Omega0_hat_cpu)
            Omega1_hat = np.array(Omega1_hat_cpu)
            time = time[0][0]

            Psi0_hat = Omega2Psi_2DFHIT_jit(Omega0_hat, Kx, Ky, Ksq)
            Psi1_hat = Omega2Psi_2DFHIT_jit(Omega1_hat, Kx, Ky, Ksq)
            
        else:
            # Path of Initial Conditions

            # Get the absolute path to the directory
            base_path = Path(__file__).parent.absolute()

            # Construct the full path to the .mat file
            # Go up one directory before going into ICs
            IC_DIR = 'data/ICs/NX' + str(NX) + '/' 
            IC_filename = str(ICnum) + '.mat'
            file_path = os.path.join(base_path, "..", IC_DIR, IC_filename)

            # Resolve the '..' to compute the actual directory
            file_path = Path(file_path).resolve()

            # -------------- Loading Initial condition (***) --------------

            data_Poi = loadmat(file_path)
            Omega1 = data_Poi["Omega"]
            Omega1_hat = np.fft.fft2(Omega1)
            Omega0_hat = Omega1_hat
            Psi1_hat = Omega2Psi_2DFHIT_jit(Omega1_hat, Kx, Ky, Ksq)
            Psi0_hat = Psi1_hat
            time = 0.0

            # Get the last file name (filenames are integers)
            last_file_number_data = get_last_file(SAVE_DIR_DATA)
            last_file_number_IC = get_last_file(SAVE_DIR_IC)

            # Set last File numbers 
            if last_file_number_data is None:
                print(f"Last data file number: {last_file_number_data}")
                last_file_number_data = 0
                print("Updated last data file number to " + str(last_file_number_data))
            else:
                raise ValueError("Data already exists in the results folder for this case, either resume the simulation (resumeSim = True) or delete data to start a new simulation")

            if last_file_number_IC is None:
                print(f"Last data file number: {last_file_number_IC}")
                last_file_number_IC = 0
                print("Updated last data file number to " + str(last_file_number_IC))
            else:
                raise ValueError("Data already exists in the results folder for this case, either resume the simulation (resumeSim = True) or delete data to start a new simulation")

            # Save variables of the solver 
            variables = {
                'readTrue': readTrue,
                'ICnum': ICnum,
                'resumeSim': resumeSim,
                'saveData': saveData,
                'NSAVE': NSAVE,
                'tSAVE': tSAVE,
                'tTotal': tTotal,
                'maxit': maxit,
                'NX': NX,
                'Lx': Lx,
                'Re': Re,
                'dt': dt,
                'nu': nu,
                'rho': rho,
                'alpha': alpha,
                'SGSModel': SGSModel_string,
                'fkx': fkx,
                'fky': fky,
                'SAVE_DIR': SAVE_DIR,
            }

            # Open file in write mode and save parameters
            with open(SAVE_DIR + 'parameters.txt', 'w') as f:
                # Write each variable to a new line in the file
                for key, value in variables.items():
                    f.write(f'{key}: {value}\n')

            print("Parameters of the flow saved to saved to " + SAVE_DIR + 'parameter.txt')

    # -------------- Main iteration loop --------------
    print("-------------- Main iteration loop --------------")
    ## 0 meanns previous time step, 1 means current time step
    start_time = runtime.time()
    
    for it in range(maxit):

        if it == 0:
            U0_hat, V0_hat = Psi2UV_2DFHIT_jit(Psi0_hat, Kx, Ky, Ksq)
            U1_hat, V1_hat = U0_hat, V0_hat
            convec0_hat = convection_conserved(Omega0_hat, U0_hat, V0_hat, Kx, Ky)

        convec1_hat = convection_conserved(Omega1_hat, U1_hat, V1_hat, Kx, Ky)

        # 2 Adam bash forth 
        convec_hat = 1.5*convec1_hat - 0.5*convec0_hat

        diffu_hat = -Ksq*Omega1_hat
        
        if SGSModel_string == 'NoSGS':
            PiOmega1_hat, eddyViscosity = PiOmega_eddyViscosity_model.calculate()
        elif SGSModel_string == 'SMAG':
            PiOmega1_hat, eddyViscosity, eddyViscosityCoeff = PiOmega_eddyViscosity_model.calculate(
                Psi_hat=Psi1_hat, Kx=Kx, Ky=Ky, Ksq=Ksq, Cs=eddyViscosityCoeff, Delta=Delta)
        elif SGSModel_string == 'LEITH':
            PiOmega1_hat, eddyViscosity, eddyViscosityCoeff = PiOmega_eddyViscosity_model.calculate(
                Omega_hat=Omega1_hat, Kx=Kx, Ky=Ky, Cl=eddyViscosityCoeff, Delta=Delta)
        elif SGSModel_string == 'DSMAG' or SGSModel_string == 'DLEITH':
            PiOmega1_hat, eddyViscosity, eddyViscosityCoeff = PiOmega_eddyViscosity_model.calculate(
                Psi_hat=Psi1_hat, Omega_hat=Omega1_hat, Kx=Kx, Ky=Ky, Ksq=Ksq, Delta=Delta)
        elif SGSModel_string == 'PiOmegaGM2' or SGSModel_string == 'PiOmegaGM4' or SGSModel_string == 'PiOmegaGM6':
            PiOmega1_hat, eddyViscosity = PiOmega_eddyViscosity_model.calculate(
                Omega_hat=Omega1_hat, U_hat=U1_hat, V_hat=V1_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        elif SGSModel_string == 'CNN':
            eddyViscosity = 0.0
            input_data = prepare_data_cnn_jit(Psi1_hat, Kx, Ky, Ksq)    
            # input_data_normalized = normalize_data(input_data) 
            output_normalized = PiOmega_eddyViscosity_model.calculate(cnn_model, input_data=input_data, Kx=Kx, Ky=Ky, Ksq=Ksq)
            # # Diagnosis 
            # print("The stats of the output are: ")
            # print("Mean: " + str(output_normalized.mean(axis=(1,2))))
            # print("Std: " + str(output_normalized.std(axis=(1,2))))


            # output_mean = np.array([0.0088, 5.1263e-05, 0.0108]).reshape((3,1,1))
            # output_std = np.array([0.0130, 0.0080, 0.0145]).reshape((3,1,1))
            
            # output_denomralized = denormalize_data(output_normalized, mean= output_mean, std= output_std) 
            # output_denomralized = np.zeros((3, output_normalized.shape[1], output_normalized.shape[2]))
            # for i in range(3):
            #     updated_values = denormalize_data(output_normalized[i], mean= output_mean[i], std= output_std[i])
            #     output_denomralized = output_denomralized.at[i, :, :].set(updated_values)
            PiOmega1_hat = postproccess_data_cnn_jit(output_normalized[0], output_normalized[1], output_normalized[2], Kx, Ky, Ksq)
            # print(np.abs(PiOmega_hat[0]).mean())
            # PiOmega_hat = PiOmegaModel.calculate()


        # Numerical scheme for PiOmega_hat
        if PiOmega_numerical_scheme == 'E1':
            PiOmega_hat = PiOmega1_hat

        elif PiOmega_numerical_scheme == 'AB2':

            if it ==0:
                PiOmega0_hat = PiOmega1_hat

            PiOmega_hat = 1.5*PiOmega1_hat-0.5*PiOmega0_hat
        
        # 2 Adam bash forth Crank Nicolson
        RHS = Omega1_hat - dt*(convec_hat) + dt*0.5*(nu+eddyViscosity)*diffu_hat - dt*(Fk_hat+PiOmega_hat) + dt*beta*V1_hat

        # Older version of RHS: Moein delete later
        # RHS = Omega1_hat + dt*(-1.5*convec1_hat + 0.5*convec0_hat) + dt*0.5*(nu+ve)*diffu_hat + dt*(Fk_hat-PiOmega_hat)

        Omega_hat_temp = RHS/(1+dt*alpha + 0.5*dt*(nu+eddyViscosity)*Ksq)

        # Replacing the previous time step with the current one
        Omega0_hat = Omega1_hat
        Omega1_hat = Omega_hat_temp
        convec0_hat = convec1_hat
        PiOmega0_hat = PiOmega1_hat

        # Poisson equation for Psi
        Psi0_hat = Psi1_hat
        Psi1_hat = Omega2Psi_2DFHIT_jit(Omega1_hat, Kx, Ky, Ksq)
        U1_hat, V1_hat = Psi2UV_2DFHIT_jit(Psi1_hat, Kx, Ky, Ksq)

        time = time + dt

        if saveData and np.mod(it+1, (NSAVE)) == 0:

            Omega = np.real(np.fft.ifft2(Omega1_hat))
            # Psi = np.real(np.fft.ifft2(Psi1_hat))

            # Converting to numpy array
            Omega_cpu = nnp.array(Omega)
            # Psi = nnp.array(Psi)
            Omega0_hat_cpu = nnp.array(Omega0_hat)
            Omega1_hat_cpu = nnp.array(Omega1_hat)
            eddyTurnoverTime = eddyTurnoverTime_2DFHIT_jit(Omega)

            last_file_number_data = last_file_number_data + 1
            last_file_number_IC = last_file_number_IC + 1

            filename_data = SAVE_DIR_DATA + str(last_file_number_data)
            filename_IC = SAVE_DIR_IC + str(last_file_number_IC)

            try:
                if np.isnan(eddyTurnoverTime).any():
                    filename = SAVE_DIR + 'unstable.txt'
                    error_message = "eddyTurnoverTime is NaN. Stopping execution at time = " + str(time) 

                    with open(filename, 'w') as f:
                        f.write(error_message)
                    raise ValueError(error_message)
                else:
                    savemat(filename_data + '.mat', {"Omega":Omega_cpu, "time":time})
                    savemat(filename_IC + '.mat', {"Omega0_hat":Omega0_hat_cpu, "Omega1_hat":Omega1_hat_cpu, "time":time, "eddyViscosity":eddyViscosity, "eddyViscosityCoeff":eddyViscosityCoeff})

            except ValueError as e:
                print(str(e))
                quit()

            print('Time = {:.6f} -- Eddy Turnover Time = {:.6f} -- C = {:.4f} -- Eddy viscosity = {:.6f} ** Saved {}'.format(time, eddyTurnoverTime, eddyViscosityCoeff, eddyViscosity, filename_data))
            
    # Print elapsed time
    print('Total Iteration: ', it+1)
    endTime = timer()
    print('Total Time Taken: ', endTime-startTime)

if __name__ == '__main__':
    import sys
    sys.path.append('examples')
    sys.path.append('py2d')
    sys.path.append('.')

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
                   tSAVE=1, # Time interval to save data
                   tTotal=10, # Total time of simulation
                   readTrue=False, 
                   ICnum=1, # Initial condition number: Choose between 1 to 20
                   resumeSim=False, # tart new simulation (False) or resume simulation (True) 
                   jobName='')

