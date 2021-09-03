#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Standard parameters. These should match the paper. 
'''

SOURCE_YF  = 0
SOURCE_YN  = 1
SOURCE_YR  = 2
SOURCE_ZF  = 3
SOURCE_YR2 = 4
WEIGHTS_Σz = 0
WEIGHTS_Σy = 1
WEIGHTS_R  = 0

######################################################################
# Baseline options
options = {
    'features' :'ougaussian',# Type of encoding drift
    'readout'  :'ring',      # Structure of readout tuning 
    'geometry' :'ring',      # Structure of θ
    'seed'     :0,           # Random seed to use

    # Simulated drift parameters
    'L' :60    ,# Length of track
    'K' :120   ,# Number of features
    'T' :1005  ,# Number of "days" to simulate
    'P' :1     ,# Number of independent place maps
    'σ' :0.10  ,# std of spatial low-pass filter for features 
    'τ' :100   ,# Correlation time of random drift
    'r' :0.05  ,# Per-day excess code variability
    'Δ' :5     ,# Features changes before running homeostasis
    'f' :1.0   ,# Input rate scaling

    # Simulated readout parameters
    'M' :60    ,# Number of decoder units
    'ν' :0.06  ,# Width of decoder unit receptive field
    'I' :100   ,# Iterations for self-healing plasticity loop
    'n' :0.01  ,# Per-timepoint readout synapse Gausian noise
    'ρ' :1e-4  ,# Weight decay rate (L2 regularization)
    'ι' :0.5   ,# ﬁltering weight
    
    # Options for negative feeback (method=='predictive')
    'ηz' :0.01 ,# step size
    'Iz' :100  ,# iterations
    'zk' :1.0  ,# prior weight
    'rec':WEIGHTS_Σz, # source of recurrent weights; 012→Σz Σy R
    
    # Options for predictive and recurrent methods 012→y_f y_n y_r
    'hom':SOURCE_YR2, # Source of homeostatic error signals 
    'heb':SOURCE_YR , # Source of Hebbian learning signals
    
    # Extra options
    'normalize'          :True , # Response normalization?
    'linearize'          :False, # Remove nonlinearity?
    'disable_homeostasis':False, # Disable gain/bias homeostasis
    'fail_early'         :False, # Don't stop early if NRMSE>0.5
}

################################################################################
# Tune rates for each model

homeostat_rates = {'η':.01,'ηβ':0.1,'ηγ':0.001}
map_rates       = {'η':1.0,'ηβ':0.1,'ηγ':0.001}
hebb_rates      = {'η':1.0,'ηβ':0.1,'ηγ':0.001}
feedback_rates  = {'η':1.0,'ηβ':5.0,'ηγ':0.005}
rates = {
    'homeostat' : homeostat_rates,
    'hebbhomeo' : hebb_rates,
    'predictive': feedback_rates,
    'recurrent' : map_rates
}


