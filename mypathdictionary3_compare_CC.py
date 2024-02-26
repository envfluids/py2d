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
                #path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_interperiodic/experiments/flowControl_turb_code/'
                #path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'


                # path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_interperiodicfft/experiments/flowControl_turb_code/'
                # path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'

                # Periodic - degree2
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_interperiodic_deg2_range2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'

                NUM_DATA_RL = int(2_000/1)
                BANDWIDTH_R = 50.0

            elif 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_interperiodic/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'

                # path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_interperiodicfft/experiments/flowControl_turb_code/'
                # path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'

                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 50.0

        elif NX==64:
            if 'LEITH' in METHOD :
                # Periodic - degree2
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_interperiodic_deg2_range2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'

                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 1.0

            elif 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 1.0

        elif NX==128:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 10.0

            if 'SMAG' in METHOD :
                # Periodic - degree2
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_interperiodic_deg2_range2cs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C1_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 10.0

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
                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 10.0
                # ongoing
            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 2_000
                BANDWIDTH_R = 5.0
                # ongoing

        if NX==64:
            if 'SMAG' in METHOD :
                #path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                #path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CSpost'
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnucs/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 3_00
                BANDWIDTH_R = 20.0

            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 3_00
                BANDWIDTH_R = 10.0

        if NX == 128:
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor50000.0_NumRLSteps5000.0_EPERU1.0/CSpost'
                NUM_DATA_RL = 300
                BANDWIDTH_R = 50.0

            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/RLonKorali_beta_rewardxy_localnucs_redo2/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C2_N128_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor50000.0_NumRLSteps5000.0_EPERU1.0/CLpost'
                NUM_DATA_RL = 300
                BANDWIDTH_R = 5.0

    elif CASENO==4:
        print('CASE 4: RL load')
        if NX == 128:
            if 'SMAG' in METHOD :
                #path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU_CS/experiments/flowControl_turb_code/'
                #path_RL += '_result_vracer_C4_N128_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost/'

                # path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU_CS/experiments/flowControl_turb_code/'
                # path_RL += 'pass_result_vracer_C4_N128_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost'

                path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU_CS/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N128_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost'


                # path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKorali_FDNS_FF/experiments/flowControl_turb_code/_result_vracer_C4_N128_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost/ICs/'

                path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKorali_CC/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N128_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost/IC1'

                NUM_DATA_RL = 1000
                BANDWIDTH_R = 5.0

            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N128_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost/'

                # path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU_positiveclipping/experiments/flowControl_turb_code/'
                # path_RL += '_result_vracer_C4_N128_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost'

                NUM_DATA_RL = 128#1000
                BANDWIDTH_R = 5.0

        elif NX == 192:
            if 'SMAG' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKorali_FDNS_FF/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N192_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost'

                NUM_DATA_RL = 128#1000
                BANDWIDTH_R = 5.0

            elif 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N192_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost/'

                NUM_DATA_RL = 128#1000
                BANDWIDTH_R = 5.0

        elif NX == 256:
            if 'SMAG' in METHOD :
                # path_RL = '/mnt/Mount/jetstream_volume3/docker/RLonKoraliGPU/experiments/flowControl_turb_code/'
                # path_RL += '_result_vracer_C4_N256_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost/'
                path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKorali_FDNS_FF/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N256_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost'
                NUM_DATA_RL = 2000
                BANDWIDTH_R = 5.0

            elif 'LEITH' in METHOD :
                #path_RL = '/mnt/Mount/jetstream_volume3/docker/RLonKoraliGPU/experiments/flowControl_turb_code/'
                #path_RL += '_result_vracer_C4_N256_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost/'
                #path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU/experiments/flowControl_turb_code/'
                # path_RL += '_result_vracer_C4_N256_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost/'


                # path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU/experiments/flowControl_turb_code/'
                path_RL = '/mnt/Mount/jetstream_volumephy/redo/RLonKoraliGPU_positiveclipping/experiments/flowControl_turb_code/'
                path_RL += '_result_vracer_C4_N256_R_z1_State_enstrophy_Action_CL_nAgents16/CLpost'

                NUM_DATA_RL = 2000
                BANDWIDTH_R = 5.0

        # NUM_DATA_RL = 1000
        # BANDWIDTH_R = 5.0

    if CASENO==10:
        if NX==32:
            if 'LEITH' in METHOD :
                path_RL = '/mnt/Mount/jetstream_volume2/RLonKorali_beta_rewardxy_localnu/experiments/flowControl_turb_code_100k/'
                path_RL += '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost'


    # delter
    # NUM_DATA_RL = 1_000
    #BANDWIDTH_R = 1.0
    return path_RL, NUM_DATA_RL, BANDWIDTH_R


def mypathdictionaryclassic(CASENO, NX, METHOD):

    if CASENO==1 or CASENO==10:
        if not (METHOD=='SMAG'):
            dataType = 'results/Re20000_fkx4fky4_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
        else:
            #dataType =  '/mnt/Mount/jetstream_volumephy/py2d/py2d/' + dataType
            dataType = '/mnt/Mount/jetstream_volumephy/py2d/py2d/results/Re20000_fkx4fky4_r0.1_b0/SMAG/NX'+str(NX)+'/dt0.0005_IC1/data'

    elif CASENO==2:
        if not (METHOD=='SMAG'):
            dataType = 'results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
            #dataType = '/mnt/Mount/jetstream_volumephy/ENVFLUIDS/results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
            #dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
        else:
            # dataType = 'results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
            #dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx4fky4_r0.1_b20/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
            dataType = '/mnt/Mount/jetstream_volumephy/py2d/py2d/results/Re20000_fkx4fky4_r0.1_b20/SMAG/NX'+str(NX)+'/dt0.0005_IC1/data'

    elif CASENO==4:
        # dataType = 'results/Re20000_fkx25fky25_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'
        # dataType = '/mnt/Mount/envfluids/PostDoc/Projects/py2d/results/Re20000_fkx25fky25_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
        if not (METHOD=='SMAG'):
            dataType = '/mnt/Mount/jetstream_volumephy/ENVFLUIDS/results/Re20000_fkx25fky25_r0.1_b0/'+METHOD+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
        #dataType = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/py2d/results/Re20000_fkx25fky25_r0.1_b0/SMAG0d17/'+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
        # dataType = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/py2d/results/Re20000_fkx25fky25_r0.1_b0/SMAGn0d17/'+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
        # dataType = '/media/rmojgani/hdd/PostDoc/Projects/py2d_local/py2d/results/Re20000_fkx25fky25_r0.1_b0/SMAG/'+'/NX'+str(NX)+'/dt0.0005_IC1/data'#----------
        else:
            dataType = '/mnt/Mount/jetstream_volumephy/py2d/py2d/results/Re20000_fkx25fky25_r0.1_b0/SMAG/NX'+str(NX)+'/dt0.0005_IC1/data'


    print('METHOD:', METHOD)
    print('path:', dataType)

    return dataType
