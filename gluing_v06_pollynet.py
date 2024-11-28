# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:42:34 2024

@author: TSICHLA
"""


import necessary_functions as nes_fun


corr_coef_threshold = 0.75
snr_nf_threshold = 5
overlap = 80
        
bin_low, bin_high, mean_norm_factor = nes_fun.gluing_window_parameters(
    nf_signal, ff_signal, overlap, corr_coef_threshold, snr_nf_threshold)
        
glued_signal = ff_signal
glued_signal = nes_fun.signal_gluing(
    nf_signal, ff_signal, mean_norm_factor, bin_low, bin_high)
    

    

