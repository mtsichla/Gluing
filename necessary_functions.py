# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:33:13 2024

@author: TSICHLA
"""

import numpy as np
import pandas as pd
import scipy.signal #import savgol_filter
import matplotlib.pyplot as plt
import math
import os
import netCDF4
from netCDF4 import Dataset
import scipy.signal
#import cv2
import scipy
from skimage.filters import gaussian
import scipy.stats as st
from scipy.special import factorial
import numpy.ma as ma
import datetime
from datetime import datetime, timedelta

def _deadtime_correction(raw_signal, dead_time, measurement_shots):
    """
    Apply the non-paralyzable dead time correction to the signal.
        
    Parameters
    ----------
    measured_counts: integer vector
       The measured number on photons.
    measurement_interval: float
       The total time interval. [s]
    dead_time: float
       The dead time of the detector. [s]
    
    Returns
    -------
    corrected_counts: float vector
       The dead-time corrected number of photons.
    """
    # Apply the dead-time correction.
    corrected_counts = raw_signal / (1 - raw_signal * dead_time)
    #    print(f'{corrected_counts}')
    #Guiseppe deadtime correction
#    corrected_counts = measured_counts / (1 - measured_counts * dead_time)
    # Return the corrected counts.
    return corrected_counts

def _deadtime_correction_from_verlauf(raw_signal, dead_time, measurement_shots):
    """
    Apply the non-paralyzable dead time correction to the signal.
        
    Parameters
    ----------
    measured_counts: integer vector
       The measured number on photons.
    measurement_interval: float
       The total time interval. [s]
    dead_time: float
       The dead time of the detector. [s]
    
    Returns
    -------
    corrected_counts: float vector
       The dead-time corrected number of photons.
    """
    # Apply the dead-time correction.
    P_meas = 0.5*raw_signal/(measurement_shots*0.05)
    P_corr = P_meas / (1 - dead_time / 1000 * P_meas)
    corrected_counts = P_corr * measurement_shots *0.05 / 0.5
    # Return the corrected counts.
    return corrected_counts

def _background_correction(signal, idx_min, idx_max):
    """
    Subtracts the background level from the signal.
            
    Parameters
    ----------
    signal: vector
       The signal vector.
    idx_min: integer
       The minimum index to calculate the background level
    idx_max: integer
       The maximum index to calculate the background level
       
    Returns
    -------
    corrected_signal: float array
       The signal without the background level
    """
    # Calculate the mean background signal.
#    background_mean = int(round(np.mean(signal[idx_min:idx_max], axis = 0)))
    background_mean = np.mean(signal[idx_min:idx_max], axis = 0)    
    # Apply the background correction.
#    corrected_signal = np.round(np.array(signal - background_mean))
    corrected_signal = signal - background_mean   
    # Return the corrected signal.
    return  corrected_signal, background_mean


def _range_correction(signal, distance):
    """
    Apply range correction to the signal.

    Parameters
    ----------
    signal: vector
       The signal vector.
    distance: vector
       The distance vector that corresponds to each spatial bin. [m]

    Returns
    -------
    corrected_signal: vector
       The range corrected signal.
    """
    # Apply range correction.
    corrected_signal = np.round(np.array(signal * distance ** 2))
    # Return the range corrected signal
    return corrected_signal   


def vec_gaussian(img: np.ndarray, variance: float) -> np.ndarray:
    # For applying gaussian function for each element in matrix.
    img = (img - img.mean())/img.std()
    sigma = math.sqrt(variance)
    cons = 1 / (sigma * math.sqrt(2 * math.pi))
    return cons * np.ma.exp(-img**2/(2*sigma**2))

def vec_poisson(img: np.ndarray, lambda_value) -> np.ndarray:
    # For applying poisson function for each element in matrix.
    # return np.ma.power(lambda_value, img)*np.ma.exp(-lambda_value)/factorial(img)
    return np.ma.power(img, lambda_value)*np.ma.exp(-img)/factorial(lambda_value)

# def vec_PDF_for_gamma(img: np.ndarray, lambda_value) -> np.ndarray:
#     # For applying poisson function for each element in matrix.
#     img_final = np.zeros(img.shape)
#     for i in range(len(img)):
#         for j in range(len(img)):
#             img_final[i][j] = scipy.stats.gamma.pdf(img[i][j], lambda_value, loc=0, scale=1)
#             # np.ma.power(img[i][j], (lambda_value-1))*np.ma.exp(-img[i][j])/math.gamma(lambda_value)
#     return img_final#scipy.stats.gamma.pdf(img, lambda_value, loc=0, scale=1)#scipy.stats.gamma(img)#img_final
 
def get_slice(img: np.ndarray, x: int, y: int, kernel_size: int) -> np.ndarray:
    half = kernel_size // 2
    return img[x - half : x + half + 1, y - half : y + half + 1]


# def get_gauss_kernel(kernel_size: int, spatial_variance: float) -> np.ndarray:
#     # Creates a gaussian kernel of given dimension.
#     arr = np.zeros((kernel_size, kernel_size))
#     for i in range(0, kernel_size):
#         for j in range(0, kernel_size):
#             arr[i, j] = math.sqrt(
#                 abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2
#             )
#     return vec_gaussian(arr, spatial_variance), arr

# a, b = get_gauss_kernel(3,10)

# c = np.sum(a)

def get_gauss_kernel(kernel_size: int, spatial_variance: float) -> np.ndarray:
    arr = np.zeros((kernel_size, kernel_size))
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            arr[i, j] = math.sqrt(
                abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2
            )
    sigma = math.sqrt(spatial_variance)
    cons = 1 / (sigma * math.sqrt(2 * math.pi))
    gaussKer = cons * np.ma.exp(-(arr/sigma) ** 2 * 0.5)
    return gaussKer, arr
#
#a, b = get_gauss_kernel(3,10)
#
#c = np.sum(a)

#def poisson_reducing_bilateral_filter(
#    img: np.ndarray,
#    spatial_variance: float,
#    kernel_size: int,
#    division_value: int
#) -> np.ndarray:
#    padded_img = np.pad(img, kernel_size//2, mode='reflect')
#    max_value = np.nanmax(padded_img)
#    img2 = np.zeros(img.shape)
#    gaussKer, arr = get_gauss_kernel(kernel_size, spatial_variance)
#    sizeX, sizeY = padded_img.shape
#    if max_value == 0:
#        img2 = img
#    else:
#        for i in range(kernel_size // 2, sizeX - kernel_size // 2):
#            for j in range(kernel_size // 2, sizeY - kernel_size // 2):
#                imgS = get_slice(padded_img, i, j, kernel_size)
#
#                if imgS[kernel_size // 2, kernel_size // 2] ==0:
#                    continue
#                else:
#                    reducted_array = imgS/division_value ####we divide with a large value because some values are so large that the poisson equation with the factor (1000!) can't be computed
#                    imgIG = vec_poisson(reducted_array, reducted_array[kernel_size // 2, kernel_size // 2]) ###here is the poisson weight
#                    weights = np.multiply(gaussKer, imgIG)
#                    vals = np.multiply(imgS, weights)
#                    val = np.sum(vals) / np.sum(weights) ## weighted average
#                    img2[i-kernel_size // 2, j-kernel_size // 2] = val ##here is where we replace the center bin with the weighted average of all the bins in the kernel
#    return img2, imgS, imgIG, weights, vals, val, padded_img, gaussKer


def magnitude_order(num):
    if num == 0:
        return 0

    absnum = abs(num)
    order = math.log10(absnum)
    res = math.floor(order)

    return res

def poisson_reducing_bilateral_filter(
    img: np.ndarray,
    spatial_variance: float,
    kernel_size: int,
    division_value: int
) -> np.ndarray:
    padded_img = np.pad(img, kernel_size//2, mode='reflect')
    img2 = np.zeros(img.shape)
    gaussKer, arr = get_gauss_kernel(kernel_size, spatial_variance)
    sizeX, sizeY = padded_img.shape
    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):
            imgS = get_slice(padded_img, i, j, kernel_size)
            scale = magnitude_order(np.max(imgS))   ###this command changes the division value according to the maximum of each window
            if scale==0 :#or scale==1 or scale==2 or scale==3:
                division_value = 1
            else:
                division_value=np.power(10,scale)
            reducted_array = imgS/division_value ####we divide with a large value because some values are so large that the poisson equation with the factor (1000!) can't be computed
            # lambda_value = np.nanmean(reducted_array)
            lambda_value = reducted_array[kernel_size // 2, kernel_size // 2]
            imgIG = vec_poisson(reducted_array, lambda_value)#reducted_array[kernel_size // 2, kernel_size // 2]) ###here is the poisson weight
            # imgIG = vec_PDF_for_gamma(reducted_array, lambda_value)
            weights = np.multiply(gaussKer, imgIG)
            vals = np.multiply(imgS, weights)
            val = np.sum(vals) / np.sum(weights) ## weighted average
            img2[i-kernel_size // 2, j-kernel_size // 2] = val ##here is where we replace the center bin with the weighted average of all the bins in the kernel
    return img2, imgS, imgIG, weights, vals, val, padded_img, gaussKer, lambda_value


def bilateral_filter(
    img,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int
):
    vals_list = []
    padded_img = np.pad(img, kernel_size//2, mode='reflect')
    img2 = np.zeros(img.shape)
    gaussKer, arr = get_gauss_kernel(kernel_size, spatial_variance)
    sizeX, sizeY = padded_img.shape
    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):
 
            imgS = get_slice(padded_img, i, j, kernel_size)
            imgI = imgS - imgS[kernel_size // 2, kernel_size // 2] ## we do this to take the second Gaussian filter which is a function of pixel difference. The Gaussian function of intensity difference makes sure that only those bins with similar intensities to the central bin are considered for blurring.
            imgIG = vec_gaussian(imgS, intensity_variance)
            weights = np.multiply(gaussKer, imgIG)
            vals = np.multiply(imgS, weights)
            vals_list.append(vals)
            val = np.sum(vals) / np.sum(weights)
            img2[i-kernel_size // 2, j-kernel_size // 2] = val
    return img2, imgS, imgI, imgIG, weights, vals_list, val, padded_img, gaussKer, arr

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def _optimum_norm_region(nf_signal_smoothed, ff_signal, overlap, corr_coef_threshold, snr_nf_theshold):
    """
    
    Parameters
    ----------
    nf_signal: vector
       The Near Field signal.
    ff_signal: vector
       The Far Field signal.
       
    Returns
    -------
    norm_region: integer list
       The vertical region in bins, where the signal normalization is performed. [start, end]
    corr_coef: float list
       The list with the correlation coefficients for each bin.
    """
    iterations = 26 #  94bins  705m  #54bins  400m
    minimum_window = 20
    gluing_window = 21 #150m ayto shmainei oti to gluing window pou eksetazoume auksanei apo 150-300m 
#    overlap = 386
    corr_coef = np.zeros((iterations, gluing_window))
    snr_nf = np.zeros((iterations, gluing_window))
    snr_ff = np.zeros((iterations, gluing_window))
    for i in range (0,iterations):
        for j in range (0,gluing_window):            #331 #384 #357=800m
            corr_coef[i][j] = np.corrcoef(nf_signal_smoothed[overlap+i:overlap+i+minimum_window+j], ff_signal[overlap+i:overlap+i+minimum_window+j])[(0,1)]       #14 window antistoixei se 100 m
            snr_nf[i][j] = signaltonoise(nf_signal_smoothed[overlap+i:overlap+i+minimum_window+j], axis=0, ddof=0)
            snr_ff[i][j] = signaltonoise(ff_signal[overlap+i:overlap+i+minimum_window+j], axis=0, ddof=0)
    max_corr_coef = np.amax(corr_coef)
    
    
    if max_corr_coef!=max_corr_coef:
        next_max_corr_coef = np.nan
        next_max_corr_coef_position = ([0],[0])
        snr_nf_value = np.nan
    else:
        max_corr_coef_position = np.where(corr_coef==max_corr_coef)
        snr_nf_initial = snr_nf[max_corr_coef_position[0][0]][max_corr_coef_position[1][0]]
    
    ##############loupa gia to epomeno megalytero
        next_max_corr_coef = max_corr_coef
        next_max_corr_coef_position = max_corr_coef_position
        snr_nf_value = snr_nf_initial
    
        while snr_nf_value<snr_nf_theshold:
            indexes = np.where (corr_coef<next_max_corr_coef)
            if not (indexes[0].any() or indexes[1].any()):# or indexes[2].any()):
                if np.max(corr_coef[indexes])<corr_coef_threshold: #not (indexes[0].any() or indexes[1].any() or indexes[2].any()):
                    snr_nf_theshold -= 0.05
                    snr_nf_value = snr_nf_initial
                    next_max_corr_coef = max_corr_coef
                    next_max_corr_coef_position = max_corr_coef_position
                else:
                    next_max_corr_coef = max_corr_coef - np.min(max_corr_coef - corr_coef[indexes])
                    next_max_corr_coef_position = np.where(corr_coef==next_max_corr_coef)
                    snr_nf_value = snr_nf[next_max_corr_coef_position[0][0]][next_max_corr_coef_position[1][0]]
            else:
                break
        
        
    return  corr_coef, snr_nf, snr_ff, next_max_corr_coef, next_max_corr_coef_position, snr_nf_value


#def _optimum_norm_region(nf_signal_smoothed, ff_signal, overlap, corr_coef_threshold, snr_nf_theshold):
#    """
#    
#    Parameters
#    ----------
#    nf_signal: vector
#       The Near Field signal.
#    ff_signal: vector
#       The Far Field signal.
#       
#    Returns
#    -------
#    norm_region: integer list
#       The vertical region in bins, where the signal normalization is performed. [start, end]
#    corr_coef: float list
#       The list with the correlation coefficients for each bin.
#    """
#    iterations = 26 #  94bins  705m  #54bins  400m
#    minimum_window = 20
#    gluing_window = 21 #150m ayto shmainei oti to gluing window pou eksetazoume auksanei apo 150-300m 
##    overlap = 386
#    corr_coef = np.zeros((iterations, gluing_window))
#    snr_nf = np.zeros((iterations, gluing_window))
#    snr_ff = np.zeros((iterations, gluing_window))
#    for i in range (0,iterations):
#        for j in range (0,gluing_window):            #331 #384 #357=800m
#            corr_coef[i][j] = np.corrcoef(nf_signal_smoothed[overlap+i:overlap+i+minimum_window+j], ff_signal[overlap+i:overlap+i+minimum_window+j])[(0,1)]       #14 window antistoixei se 100 m
#            snr_nf[i][j] = signaltonoise(nf_signal_smoothed[overlap+i:overlap+i+minimum_window+j], axis=0, ddof=0)
#            snr_ff[i][j] = signaltonoise(ff_signal[overlap+i:overlap+i+minimum_window+j], axis=0, ddof=0)
#    max_corr_coef = np.amax(corr_coef)
#    
#
#
#    max_corr_coef_position = np.where(corr_coef==max_corr_coef)
#    snr_nf_initial = snr_nf[max_corr_coef_position[0][0]][max_corr_coef_position[1][0]]
#
###############loupa gia to epomeno megalytero
#    next_max_corr_coef = max_corr_coef
#    next_max_corr_coef_position = max_corr_coef_position
#    snr_nf_value = snr_nf_initial
#
#    while snr_nf_value<snr_nf_theshold:
#        indexes = np.where (corr_coef<next_max_corr_coef)
#        if not (indexes[0].any() or indexes[1].any()):# or indexes[2].any()):
#            if np.max(corr_coef[indexes])<corr_coef_threshold: #not (indexes[0].any() or indexes[1].any() or indexes[2].any()):
#                snr_nf_theshold -= 0.05
#                snr_nf_value = snr_nf_initial
#                next_max_corr_coef = max_corr_coef
#                next_max_corr_coef_position = max_corr_coef_position
#            else:
#                next_max_corr_coef = max_corr_coef - np.min(max_corr_coef - corr_coef[indexes])
#                next_max_corr_coef_position = np.where(corr_coef==next_max_corr_coef)
#                snr_nf_value = snr_nf[next_max_corr_coef_position[0][0]][next_max_corr_coef_position[1][0]]
#        else:
#            break
#    
##    else:
##        next_max_corr_coef = np.nan
##        next_max_corr_coef_position = ([0],[0])
##        snr_nf_value = np.nan
#        
#    return  corr_coef, snr_nf, snr_ff, next_max_corr_coef, next_max_corr_coef_position, snr_nf_value


def _signal_gluing(nf_signal_smoothed, ff_signal, overlap, corr_coef_threshold, snr_nf_theshold):
    """
    Glue the adjusted Near Field signal with the Far Field signal, after 
    performing a weighted averaging for a specified vertical region.
    
    Parameters
    ----------
    nf_signal: vector
       The Near Field signal.
    ff_signal: vector
       The Far Field signal.
       
    Returns
    -------
    glued_signal: vector
       The glued signals.
    nf_adjusted_signal: vector
       The normalized Near Field signal to the FF signal.
    gluing_region: integer list
       The vertical region in bins, where the signal gluing is performed. [start, end]
    """
    # Determine the ideal gluing region.
    corr_coef, snr_nf, snr_ff, next_max_corr_coef, next_max_corr_coef_position, snr_nf_value = _optimum_norm_region(nf_signal_smoothed, ff_signal, overlap, corr_coef_threshold, snr_nf_theshold)
    minimum_window = 20
#    overlap = 386
#    if len(max_corr_coef_positions[0])==0:
#        bin_low = 384
#        bin_high = 398

    bin_low = overlap + next_max_corr_coef_position[0][0] #to np.where gyrnaei tuples apo arrays kai emeis theloume mono thn prwth timh tou array
    bin_high = overlap + next_max_corr_coef_position[0][0]+ minimum_window + next_max_corr_coef_position[1][0]
    
    mean_norm_factor = np.mean(ff_signal[bin_low:bin_high]) / np.mean(nf_signal_smoothed[bin_low:bin_high])
    
#    X = ff_signal[bin_low:bin_high]
#    Y = nf_signal_smoothed[bin_low:bin_high]
#    
#    # Building the model
#    X_mean = np.mean(X)
#    Y_mean = np.mean(Y)
#
#    num = 0
#    den = 0
#    for i in range(len(X)):
#        num += (X[i] - X_mean)*(Y[i] - Y_mean)
#        den += (X[i] - X_mean)**2
#    mean_norm_factor = num / den


#    from sklearn.linear_model import LinearRegression
#    lr = LinearRegression()
#    lr.fit(ff_signal[bin_low:bin_high].reshape(-1, 1), nf_signal_smoothed[bin_low:bin_high].reshape(-1, 1))
#    mean_norm_factor = lr.coef_[0]
    
    
#    norm_factor_list = []
#    for i in range (bin_low, bin_high):
#        norm_factor = ff_signal[i] / nf_signal_smoothed[i]
#        norm_factor_list.append(norm_factor)
#        
#    mean_norm_factor = np.mean(norm_factor_list)
    return bin_low, bin_high, mean_norm_factor  


def _signal_gluing_2nd(nf_signal, ff_signal, mean_norm_factor, bin_low, bin_high):
    """
    Glue the adjusted Near Field signal with the Far Field signal, after 
    performing a weighted averaging for a specified vertical region.
    
    Parameters
    ----------
    nf_signal: vector
       The Near Field signal.
    ff_signal: vector
       The Far Field signal.
       
    Returns
    -------
    glued_signal: vector
       The glued signals.
    nf_adjusted_signal: vector
       The normalized Near Field signal to the FF signal.
    gluing_region: integer list
       The vertical region in bins, where the signal gluing is performed. [start, end]
    """
                
    nf_adjusted_signal = nf_signal * mean_norm_factor

    # Create "weight" vectors for the averaging.
    nf_weight = np.linspace(1, 0, (bin_high-bin_low)) 
    ff_weight = 1 - nf_weight
        
    #Weights me vash thn sygmoeidh synarthsh
#    x = np.arange(0,(bin_high-bin_low), 1)
#    ff_weight = (1+np.tanh((x-(12)/6))/2)/np.nanmax(1+np.tanh((x-(12)/6))/2)
#    nf_weight = 1 - ff_weight
    
    averaging_weights = np.column_stack((nf_weight, ff_weight))
    
    # Assign variables according to the selected vertical range for the NF-FF comparison.
    nf = nf_adjusted_signal[bin_low : bin_high] #360 to bin pou antistoixei sta 1000m(ta prwta 250 bins einai to pre-trigger) kai 493 to bin pou antistoixei sta 2000m
    ff = ff_signal[bin_low : bin_high]
    
    # Calculate weighted average.
    average_signal = np.average(np.column_stack((nf, ff)), axis=1, weights=averaging_weights)    

    # Glue the signals.
    glued_signal = np.concatenate((nf_adjusted_signal[:bin_low], average_signal, ff_signal[bin_high:]), axis=0)
#    glued_signal = np.round(np.array(np.concatenate((nf_adjusted_signal[:bin_low], ff_signal[bin_low:]), axis=0)))

    return glued_signal, ff_signal, nf_adjusted_signal, average_signal, averaging_weights  

def reverse_range_correction(signal, distance):

    corrected_signal = signal / (distance ** 2)
    
    return corrected_signal   

def reverse_background_correction(signal, background_mean):

    corrected_signal = signal + background_mean

    return  corrected_signal

def reverse_deadtime_correction(corrected_counts, dead_time, measurement_shots):
    
    raw_signal = corrected_counts / (1 + corrected_counts * dead_time)
        #    print(f'{corrected_counts}')

    return raw_signal

def reverse_deadtime_correction_from_verlauf(corrected_counts, dead_time, measurement_shots):
    
    P_corr = corrected_counts * 10 / measurement_shots
    P_meas = P_corr / (1 + P_corr * dead_time / 1000)
    raw_signal = P_meas * measurement_shots / 10

    return raw_signal

def _bins_to_time(measurement_time, temporal_index):
    """
    Convert the time-domain bins to time (HHMM format).
    
    Parameters
    ----------
    measurement_time: integer
       The "measurement_time" variable of the PollyxT netCDF file.
    temporal_index: string
       The temporal index of the given bin.
    
    Returns
    -------
    date_object: date object
       The date object that corresponds to the input time string.
    """
    # Select the date of the index and convert it to datetime object.
    date_only = datetime.strptime(str(measurement_time[temporal_index, 0]),"%Y%m%d")
    
    # Select the seconds of the index.
    seconds_only = int(measurement_time[temporal_index,1])
    
    # Add the seconds to the date.
    date_object = date_only + timedelta(seconds = seconds_only)
    
    return date_object