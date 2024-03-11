#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 18:10:24 2023

@author: rm99
"""
import os
import glob
# --------------------------------------------------------------------------
#def load_IC():

# --------------------------------------------------------------------------
def gen_path(NX, dt, ICnum, Re, 
             fkx, fky, alpha, beta, SGSModel_string):
    # Snapshots of data save at the following directory
    dataType_DIR = 'Re' + str(int(Re)) + '_fkx' + str(fkx) + 'fky' + str(fky) + '_r' + str(alpha) + '_b' + str(beta) + '/'
    SAVE_DIR = 'results/' + dataType_DIR + SGSModel_string + '/NX' + str(NX) + '/dt' + str(dt) + '_IC' + str(ICnum) + '/'
    SAVE_DIR_DATA = SAVE_DIR + 'data/'
    SAVE_DIR_IC = SAVE_DIR + 'IC/'

    # Create directories if they aren't present
    try:
        os.makedirs(SAVE_DIR_DATA, exist_ok=True)
        os.makedirs(SAVE_DIR_IC, exist_ok=True)
    except OSError as error:
        print(error)
        
    return SAVE_DIR, SAVE_DIR_DATA, SAVE_DIR_IC
# --------------------------------------------------------------------------
def get_last_file(file_path):
    # Get all .mat files in the specified directory
    mat_files = glob.glob(os.path.join(file_path, '*.mat'))
    
    # Extract the integer values from the filenames
    file_numbers = [int(os.path.splitext(os.path.basename(file))[0]) for file in mat_files]
    
    # Find the highest integer value
    if file_numbers:
        last_file_number = max(file_numbers)
        return last_file_number
    else:
        return None
    return file_numbers
# --------------------------------------------------------------------------
def set_last_file(last_file_number_data, last_file_number_IC):
    
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
    
    return last_file_number_data, last_file_number_IC
# --------------------------------------------------------------------------
def save_settings(readTrue,ICnum,resumeSim,saveData,
         NSAVE,tSAVE,tTotal,maxit,NX, Lx, Re,dt,nu,rho,alpha, SGSModel_string,
         fkx, fky, SAVE_DIR):
    
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

def pretty_print_table(title, table_content):
    longest_key = max([len(str(row[0])) for row in table_content])
    longest_value = max([len(str(row[1])) for row in table_content])
    
    # Title
    print('+' + '-' * (longest_key+2) + '+' + '-' * (longest_value+2) + '+')
    print('| ' + title.center(longest_key) + ' | ' + 'Value'.center(longest_value) + ' |')
    print('+' + '-' * (longest_key+2) + '+' + '-' * (longest_value+2) + '+')
    
    # Content
    for row in table_content:
        print('| ' + str(row[0]).center(longest_key) + ' | ' + str(row[1]).center(longest_value) + ' |')
    print('+' + '-' * (longest_key+2) + '+' + '-' * (longest_value+2) + '+')
    print()