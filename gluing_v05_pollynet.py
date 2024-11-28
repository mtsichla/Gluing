# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:42:34 2024

@author: TSICHLA
"""


import numpy as np
import matplotlib.pyplot as plt
import necessary_functions as nes_fun
import os
import netCDF4
from netCDF4 import Dataset
import scipy.signal
import scipy
import math
from skimage.filters import gaussian
#import cv2
import scipy.stats as st
from scipy.special import factorial
import numpy.ma as ma
import datetime
import matplotlib.colors as colors
import matplotlib.dates as mdates

#    ##### =============================================================================
#    ##### GLUE functions
#    ##### =============================================================================



def calculate_norm_factors_klett(repeats, averaged_signals_array, channel_gluing_indexes, overlap, corr_coef_threshold, snr_nf_threshold):
    bin_low_list = []
    bin_high_list = []
    mean_norm_factor_list = []
    corrcoef_list = []
    snr_nf_value_list = []
    
    for i in range(repeats):     
        nf_signal = averaged_signals_array[i, :, channel_gluing_indexes[1]]
        ff_signal = averaged_signals_array[i, :, channel_gluing_indexes[0]]

        next_max_corr_coef_position = nes_fun.optimum_norm_region(
            nf_signal, ff_signal, overlap, corr_coef_threshold, snr_nf_threshold)
        
        bin_low, bin_high, mean_norm_factor = nes_fun.gluing_window_parameters(
            nf_signal, ff_signal, overlap, corr_coef_threshold, snr_nf_threshold)
        
        bin_low_list.append(bin_low)
        bin_high_list.append(bin_high)
        mean_norm_factor_list.append(mean_norm_factor)
        corrcoef_list.append(next_max_corr_coef)
        snr_nf_value_list.append(snr_nf_value)
        
    
    print('End loop for %s for bin_low, bin_high, mean_norm_factor' % if_center[channel_gluing_indexes[0]])
    
    return bin_low_list, bin_high_list, mean_norm_factor_list, corrcoef_list, snr_nf_value_list, corr_coef, snr_nf, snr_ff

def process_gluing_2nd_step_klett(repeats, mean_norm_factor_list, bin_low_list, bin_high_list, average_interval, time, depol_cal_angle, depol_cal_angle_value, range_corr_arr, channel_gluing_indexes, glued_signal):
    for k in range(repeats):  
        mean_norm_factor = mean_norm_factor_list[k]
        bin_low = bin_low_list[k]
        bin_high = bin_high_list[k]
        limit = (k + 1) * average_interval
        if limit > time:
            limit = time
        
        for i in range(k * average_interval, limit):
            if int(depol_cal_angle[i]) != depol_cal_angle_value:
                continue
            
            nf_signal = range_corr_arr[i, :, channel_gluing_indexes[1]]
            ff_signal = range_corr_arr[i, :, channel_gluing_indexes[0]]
            
            glued_signal[i, :, channel_gluing_indexes[0]], ff_signal, nf_adjusted_signal, average_signal, averaging_weights = nes_fun._signal_gluing_2nd(
                nf_signal, ff_signal, mean_norm_factor, bin_low, bin_high)
    
    print('End loop for %s in gluing 2nd step' % if_center[channel_gluing_indexes[0]])
    



    
def calculate_norm_factors_raman(repeats, averaged_signals_array, channel_gluing_indexes, overlap, corr_coef_threshold, snr_nf_threshold):
    bin_low_list = []
    bin_high_list = []
    mean_norm_factor_list = []
    corrcoef_list = []
    snr_nf_value_list = []
    
    for i in range(repeats):
        if np.nanmean(averaged_signals_array[i, 0:10, channel_gluing_indexes[1]]) == 0:  ### mean betwen 0:10 first heights to ensure that the raman channels was closed and not one specific height because statistical that could be zero
            continue
        else:
            nf_signal = averaged_signals_array[i, :, channel_gluing_indexes[1]]
            ff_signal = averaged_signals_array[i, :, channel_gluing_indexes[0]]
            
            corr_coef, snr_nf, snr_ff, next_max_corr_coef, next_max_corr_coef_position, snr_nf_value = nes_fun._optimum_norm_region(
                nf_signal, ff_signal, overlap, corr_coef_threshold, snr_nf_threshold)
            
            bin_low, bin_high, mean_norm_factor = nes_fun._signal_gluing(
                nf_signal, ff_signal, overlap, corr_coef_threshold, snr_nf_threshold)
            
            bin_low_list.append(bin_low)
            bin_high_list.append(bin_high)
            mean_norm_factor_list.append(mean_norm_factor)
            corrcoef_list.append(next_max_corr_coef)
            snr_nf_value_list.append(snr_nf_value)
            

    print('End loop for %s for bin_low, bin_high, mean_norm_factor' % if_center[channel_gluing_indexes[0]])
    
    return bin_low_list, bin_high_list, mean_norm_factor_list, corrcoef_list, snr_nf_value_list, corr_coef, snr_nf, snr_ff

def process_gluing_2nd_step_raman(time, mean_norm_factor_list, bin_low_list, bin_high_list, average_interval, depol_cal_angle, depol_cal_angle_value, range_corr_arr, channel_gluing_indexes, glued_signal):
    t = 0
    k = 0
    while t < time:
        if np.nanmean(range_corr_arr[t, 0:3, channel_gluing_indexes[1]]) == 0: ### mean betwen 0:3 first heights to ensure that the raman channels was closed and not one specific height because statistical that could be zero
            t += 1      
            continue
        else:
            mean_norm_factor = mean_norm_factor_list[k]
            bin_low = bin_low_list[k]
            bin_high = bin_high_list[k]
            limit = t + average_interval - (t % average_interval)
            
            if limit > time:
                limit = time
            
            print(limit)
            for i in range(t, limit):
                if np.nanmean(range_corr_arr[i, 0:3, channel_gluing_indexes[1]]) == 0:
                    continue
                if int(depol_cal_angle[i]) != depol_cal_angle_value:
                    continue
                
                nf_signal = range_corr_arr[i, :, channel_gluing_indexes[1]]
                ff_signal = range_corr_arr[i, :, channel_gluing_indexes[0]]
                
                glued_signal[i, :, channel_gluing_indexes[0]], ff_signal, nf_adjusted_signal, average_signal, averaging_weights = nes_fun._signal_gluing_2nd(
                    nf_signal, ff_signal, mean_norm_factor, bin_low, bin_high)
            
            t = limit
            k += 1
        t += 1

    print('End loop for %s in gluing 2nd step' % if_center[channel_gluing_indexes[0]])

def find_indexes_of_channels_to_glue(array1, array2, array3):
    matching_indexes = []
    for i in range(len(array1)):
        for j in range(i + 1, len(array1)):
            if (
                array1[i] == array1[j] and             # Same value in array1 at indexes i and j
                array2[i] == array2[j] and             # Same value in array2 at indexes i and j
                array3[i] != array3[j]             # Different value in array3 at indexes i and j
            ):
                matching_indexes.append((i, j))
    return matching_indexes   


##### =============================================================================
##### USER INPUT
##### =============================================================================

folder_path = r"C:\Users\TSICHLA\Documents\papers\gluing"

#overlap = overlap_bin - first_bin

overlap_355 = 80 ####80 corresponds to 600m and 133 to 1000m and 66 to 500m and 107 to 800m
overlap_387 = 80
overlap_532 = 80 ## for the systems with 532s near field channel it takes the same overlap as for 532
overlap_607 = 80

##### SENSITIVITY PARAMETERS

corr_coef_threshold_355 = 0.75
snr_nf_theshold_355 = 5
corr_coef_threshold_387 = 0.75
snr_nf_theshold_387 = 5
corr_coef_threshold_532 = 0.75
snr_nf_theshold_532 = 5
corr_coef_threshold_607 = 0.75
snr_nf_theshold_607 = 5

average_interval = 45

depol_cal_angle_value = 0




##### =============================================================================
##### RUN FOR ALL THE FILES IN THE FOLDER PATH
##### =============================================================================

all_file_list = os.listdir(folder_path)
for file in all_file_list: 
    print(file)
    file_path = os.path.join (folder_path, file)

    
    
    #################extract variables, dimensions and global attributes from netcdf############
    d = Dataset(file_path)
    
    time = len(d.variables["measurement_time"]) #time index
    if time == 0:
        continue
    channels = len(d.dimensions['channel'])
    height = len(d.dimensions['height'])
    
    depol_cal_angle = d.variables["depol_cal_angle"][:]
    if_center = d.variables["if_center"][:]
    polstate = d.variables["polstate"][:]
    telescope = d.variables["telescope"][:]
    
    channel_list = []
    for i in range (len(if_center)):
        if polstate[i]==0 and telescope[i]==0:
            channel_list.append('%s' % int(if_center[i]))
        elif polstate[i]==1 and telescope[i]==0:
            channel_list.append('%sco' % int(if_center[i]))
        elif polstate[i]==2 and telescope[i]==0:
            channel_list.append('%ss' % int(if_center[i]))            
        elif polstate[i]==0 and telescope[i]==1:
            channel_list.append('%sNF' % int(if_center[i]))              
        elif polstate[i]==1 and telescope[i]==1:
            channel_list.append('%scoNF' % int(if_center[i]))  
        elif polstate[i]==2 and telescope[i]==1:
            channel_list.append('%ssNF' % int(if_center[i]))
    print('channel_list = ', channel_list)
            
    channels_dictionary = {'355': 0, '355s': 1, "387": 2, "407": 3,
                "532": 4, "532s": 5, "607": 6, "1064": 7,
                "532NF": 8, "607NF": 9, "355NF": 10, "387NF": 11, "532sNF": 12, "1058":13, "1064s":14}
    
    channels_dictionary_inverse = {0:"355", 1:"355s", 2:"387", 3:"407",
                4:"532", 5:"532s", 6:"607", 7:"1064",
                8:"532NF", 9:"607NF", 10:"355NF", 11:"387NF", 12:"532sNF", 13:"1058", 14:"1064s"}
    

    
        
    
    ################## AVERAGING  ####################
    

    repeats = (time + average_interval - 1) // average_interval  # Ceiling division to include partial intervals
    averaged_signals_array = np.zeros((repeats, len(range_corr_arr[1]), channels))
    
    t = 0
    z_list_list = []
    z_list_for_raman_list = []
    
    # Function to process an interval
    def process_interval(start, end):
        for ch in channel_list:
            z_list = []
            for z in range(start, end):
                if int(depol_cal_angle[z]) != depol_cal_angle_value:
                    continue
                if '%s' % ch in {"387", "407", "607", "607NF", "387NF", "1058"}:
                    if np.nanmean(range_corr_arr[z, 0:3, channels_dictionary['%s' % ch]]) == 0:
                        continue
                z_list.append(z)
            if z_list:  # Ensure z_list is not empty before averaging
                averaged_signals_array[t, :, channels_dictionary['%s' % ch]] = np.nanmean(
                    range_corr_arr[z_list, :, channels_dictionary['%s' % ch]], axis=0
                )
    
    # Main loop for full intervals
    for i in range(0, time, average_interval):
        j = min(i + average_interval, time)  # Ensure the interval doesn't exceed `time`
        process_interval(i, j)
        t += 1

    ##### =============================================================================
    ##### FIND CHANNELS TO GLUE
    ##### =============================================================================
       
    channel_gluing_indexes_pairs = find_indexes_of_channels_to_glue(if_center, polstate, telescope)


    ##### =============================================================================
    ##### GLUE_CORRECTION
    ##### =============================================================================

    
    glued_signal = range_corr_arr
    #glued_signal[glued_signal.mask] = np.nan
    

    for ch in channel_gluing_indexes_pairs:
        print(ch)
        if '%s'%int(if_center[ch[0]])=="387" or '%s'%int(if_center[ch[0]])=="607" or '%s'%int(if_center[ch[0]])=="1058":
            
            # Access the value of the dynamically named variable
            overlap = globals().get(f"overlap_{int(if_center[ch[0]])}")
            corr_coef_threshold = globals().get(f"corr_coef_threshold_{int(if_center[ch[0]])}")
            snr_nf_theshold = globals().get(f"snr_nf_theshold_{int(if_center[ch[0]])}")
            
            bin_low_list, bin_high_list, mean_norm_factor_list, corrcoef_list, snr_nf_value_list, corr_coef, snr_nf, snr_ff = calculate_norm_factors_raman(
            repeats, averaged_signals_array, ch, overlap, corr_coef_threshold, snr_nf_theshold)


            var_name_bin_low_list = f"bin_low_list_{if_center[ch[0]]}"

            globals()[var_name_bin_low_list] = globals().get("bin_low_list")

            var_name_bin_high_list = f"bin_high_list_{if_center[ch[0]]}"

            globals()[var_name_bin_high_list] = globals().get("bin_high_list")

            var_name_mean_norm_factor_list = f"mean_norm_factor_list_{if_center[ch[0]]}"

            globals()[var_name_mean_norm_factor_list] = globals().get("mean_norm_factor_list")
            
            var_name_corrcoef_list = f"corrcoef_list_{if_center[ch[0]]}"

            globals()[var_name_corrcoef_list] = globals().get("corrcoef_list")

            var_name_snr_nf_value_list = f"snr_nf_value_list_{if_center[ch[0]]}"

            globals()[var_name_snr_nf_value_list] = globals().get("snr_nf_value_list")

#            corr_coef = f"corr_coef_{if_center[ch[0]]}"
#            snr_nf = f"snr_nf_{if_center[ch[0]]}"
#            snr_ff = f"snr_ff_{if_center[ch[0]]}"
            
            process_gluing_2nd_step_raman(
            time, mean_norm_factor_list, bin_low_list, bin_high_list, average_interval, depol_cal_angle, depol_cal_angle_value, range_corr_arr, ch, glued_signal)

        else:
            
            overlap = globals().get(f"overlap_{int(if_center[ch[0]])}")
            corr_coef_threshold = globals().get(f"corr_coef_threshold_{int(if_center[ch[0]])}")
            snr_nf_theshold = globals().get(f"snr_nf_theshold_{int(if_center[ch[0]])}")
             
            bin_low_list, bin_high_list, mean_norm_factor_list, corrcoef_list, snr_nf_value_list, corr_coef, snr_nf, snr_ff = calculate_norm_factors_klett(
                repeats, averaged_signals_array, ch, overlap, corr_coef_threshold, snr_nf_theshold)
            
            process_gluing_2nd_step_klett(
                repeats, mean_norm_factor_list, bin_low_list, bin_high_list, average_interval, time, depol_cal_angle, depol_cal_angle_value, range_corr_arr, ch, glued_signal)

            var_name_bin_low_list = f"bin_low_list_{if_center[ch[0]]}"

            globals()[var_name_bin_low_list] = globals().get("bin_low_list")

            var_name_bin_high_list = f"bin_high_list_{if_center[ch[0]]}"

            globals()[var_name_bin_high_list] = globals().get("bin_high_list")

            var_name_mean_norm_factor_list = f"mean_norm_factor_list_{if_center[ch[0]]}"

            globals()[var_name_mean_norm_factor_list] = globals().get("mean_norm_factor_list")
            
            var_name_corrcoef_list = f"corrcoef_list_{if_center[ch[0]]}"

            globals()[var_name_corrcoef_list] = globals().get("corrcoef_list")

            var_name_snr_nf_value_list = f"snr_nf_value_list_{if_center[ch[0]]}"

            globals()[var_name_snr_nf_value_list] = globals().get("snr_nf_value_list")    
    

