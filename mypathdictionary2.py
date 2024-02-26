#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:29:03 2023

@author: rmojgani
"""

def mypathdictionaryRL(CASENO, NX, METHOD):

    if CASENO==1:
        if NX==32:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 1.0

            elif 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 2#_000
                BANDWIDTH_R = 10.0

        elif NX==64:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 1_000
                BANDWIDTH_R = 1.0

            elif 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 1_000
                BANDWIDTH_R = 1.0

        elif NX==128:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 2#_000
                BANDWIDTH_R = 10.0

            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 1#_000
                BANDWIDTH_R = 5.0

    elif CASENO==2:
        if NX==32:
            # Redo
            # if 'SMAG' in METHOD :
            #     path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs_redo/experiments/flowControl_turb_code/'
            #     path_RL += '_result_vracer_C2_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
            #     # ongoing
            # elif 'LEITH' in METHOD :
            #     path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs_redo/experiments/flowControl_turb_code/'
            #     path_RL += '_result_vracer_C2_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
                # ongoing
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 1_000
                BANDWIDTH_R = 5.0
                # ongoing
            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 1_000
                BANDWIDTH_R = 5.0
                # ongoing
        if NX==64:
            if 'SMAG' in METHOD :
                #path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                #path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CSpost'
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 5_00
                BANDWIDTH_R = 5.0
            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 300
                BANDWIDTH_R = 5.0
        if NX == 128:
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor50000.0_NumRLSteps5000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 350
                BANDWIDTH_R = 5.0

            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor50000.0_NumRLSteps5000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 300
                BANDWIDTH_R = 5.0

    elif CASENO==4:
        print('CASE 4: RL load')
        if NX == 128:
            if 'SMAG' in METHOD or 'Leith' in METHOD:
                path_RL = '/mnt/Mount/jetstream_volume3/docker/RLonKorali2/experiments/flowControl_turb_code_dsmag/'
                path_RL += '_result_vracer_C4_N128_R_ke_State_energy_Action_CS/dsmag'
        if NX == 256:
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume3/docker/RLonKoraliGPU/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N256_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost/'
            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume3/docker/RLonKoraliGPU/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N256_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost/'

        NUM_DATA_RL = 1#00
        BANDWIDTH_R = 5.0

    if CASENO==10:
        if NX==32:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code_100k/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'


    # delter
    NUM_DATA_RL = 2#_000
    BANDWIDTH_R = 1.0
    return path_RL, NUM_DATA_RL, BANDWIDTH_R


def mypathdictionaryclassic(CASENO, NX, METHOD):

    if CASENO==1 or CASENO==10:
        dataType = 'results/Re20000_fkx4fky4_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'

    elif CASENO==2:
        # dataType = 'results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
        dataType = 'results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
        # dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------

    elif CASENO==4:
        # dataType = 'results/Re20000_fkx25fky25_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
        dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx25fky25_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------

    return dataType
