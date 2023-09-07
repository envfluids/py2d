#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:29:03 2023

@author: rmojgani
"""

def mypathdictionary(CASENO, NX, METHOD):

    if CASENO==1:
        if NX==32:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'
            elif 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
        elif NX==64:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
            elif 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
        elif NX==128:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'

    elif CASENO==2:
        if NX==64:
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CSpost'
            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
        if NX == 128:
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost/'
            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0_56/CLpost/'


    if CASENO==10:
        if NX==32:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code_100k/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'

    return path_RL