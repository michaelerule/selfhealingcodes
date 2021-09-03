#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import os,sys,traceback,h5py
# Get PPC routines and neurotools;
# Add your local paths to this list
[sys.path.insert(0,os.path.abspath(os.path.expanduser(p))) for p in [
    "../",
    "../../"]]
from ppc_analysis import *
from numpy import concatenate as cat
from neurotools.tools import find, amap

import ppc_analysis

def constrained_analysis_sweep(x,y,NGRID=20):
    x = list(map(np.array,x))
    y = list(map(np.array,y))
    # Same-day fits
    ww0 = array([reglstsq(xi,yi) for (xi,yi) in zip(x,y)])
    # Concatenated fit
    wc  = reglstsq(cat(x),cat(y))
    wwc = outer(ones(len(x)),wc)
    # Compute covariances and cross covariances
    sYY = np.array([(yi.T @ yi)/yi.shape[0] for  yi     in y       ])
    sXX = np.array([(xi.T @ xi)/xi.shape[0] for  xi     in x       ])
    sXY = np.array([(xi.T @ yi)/yi.shape[0] for (xi,yi) in zip(x,y)])  
    # Baseline values for eror and jacobian of OLS penalty
    e0 = np.sum(sYY)
    j0 = -2*sXY.ravel()
    # Inter-day difference operator,
    # defines quadratic penalty on the weigth changes.
    D = len(x)
    G = (-eye(D)+eye(D,k=1))[:-1,:]
    Q = G.T@G
    # Error functions for the OLS and Δw penalties
    err1 = lambda w:e0+einsum('is,ist,it',w,sXX,w)-2*np.sum(w*sXY)
    err2 = lambda w:einsum('si,st,ti',w,Q,w)
    # Rescale both OLS and constraint contribution to objective to be similar
    # (improved numeric conditioning, able to cover reasonable range of
    # constraint values with a fixed grid search)
    emin1    = err1(ww0) # Best-case  OLS error: same-day
    emax1    = err1(wwc) # Worst-case OLS error: concatenated
    emin2    = 0         # Best-case  Δw penalty: concatenated (0 Δw)
    emax2    = err2(ww0) # Worst-case Δw penalty: same-day fits
    scale_e1 = 1/(emax1-emin1) 
    scale_e2 = 1/(emax2-emin2) 
    # Grid search over convex combinations
    ll = linspace(0,1,NGRID)
    allww = [ww0.ravel(),ww0.ravel()]
    for l in ll:
        w0 = (2*allww[-1]-allww[-2]).ravel()
        # Objective and Jacobian combining the OLS and Δw penalties
        a,b = (1-l)*scale_e1, l*scale_e2
        def objective(w):
            w  = w.reshape(ww0.shape)
            return a*(err1(w)-emin1) + b*(err2(w)-emin2)
        def jacobian(w):
            w  = w.reshape(ww0.shape) 
            j1 = 2*einsum('ist,it->is',sXX,w).ravel()+j0
            j2 = 2*einsum('si,st->ti' ,w,Q  ).ravel()
            return a*j1 + b*j2
        allww.append(minimize_retry(objective,w0,jacobian,
            tol          =1e-6,
            show_progress=False,
            printerrors  =False))
    allww = array(allww[2:]).reshape((NGRID,)+ww0.shape)
    return ww0,wwc,allww,ll

def constrained_sweep_crossvalidated(X,Y,NXVAL=10,NGRID=20,errmethod='L1',matched=True):
    efn = neurotools.stats.error_functions[errmethod]
    X   = [array(x) for x in X]
    Y   = [array(y) for y in Y]
    D   = len(X)
    N   = len(cat(X))
    M   = N/D**2/NXVAL
    def partition(x,y):
        K      = len(x)
        trials = arange(K)
        Ntest  = int(M           if matched else K/NXVAL)
        Ntrain = int(M*(NXVAL-1) if matched else K-Ntest)
        train  = int32(np.random.choice(trials,Ntrain,replace=False))
        test   = int32(np.random.choice(list(set(trials)-set(train)),Ntest,replace=False))
        return x[train],y[train],x[test],y[test]
    trnX,trnY,tstX,tstY = [amap(cat,v) for v in zip(*[partition(x,y) for (x,y) in zip(X,Y)])]
    ww0,wwc,allww,ll = constrained_analysis_sweep(trnX,trnY,NGRID)
    results = cat([[(0,ww0)],list(zip(ll,allww)),[(1,wwc)]])
    return [{'MAW'  :mean(abs(w)),
             'RMSW' :mean(abs(w)**2)**0.5,
             'MADW' :mean(abs(diff(w,axis=0))),
             'RMSDW':mean(abs(diff(w,axis=0))**2)**0.5,
             'MAE'  :mean([mean(abs(y-x@w)) for (w,x,y) in zip(w,tstX,tstY)]),
             'RMSE' :mean([mean((y-x@w)**2) for (w,x,y) in zip(w,tstX,tstY)])**0.5,
             'MERR' :mean([efn(y,x@w)       for (w,x,y) in zip(w,tstX,tstY)])} 
            for (l,w) in results],results

def get_data_constrained_analysis_2(animal,sessions,predict,
    permute=False,
    split=1):    
    '''
    Get data pre-processed for performing the constrained analyses. 

    We extract good trials, z-score the dF/F calcium signals, and zero-mean
    the kinematic variables, within each trial. 

    Parameters
    ----------
    animal: int
        Which subject to use
    sessions: list of ints
        Which sessions to use
    predict: int
        Which kinematic variable to predict

    Other Parametes
    ---------------
    permute: bool, default False
        Whether to randomly scramble the neuronal identities. 
        Used for shuffle chance level assessment. 
    split: int, default 1
        Split days into `split` pieces. 
    
    Returns
    -------
    X: list
        List of neural trial data for each session
    Y: list
        List of kinematic trial data for each session
    '''
    # Get units in common
    units,uidxs = get_units_in_common(animal,sessions)
    X,Y = [],[]
    # Store same-day fit information
    sameday = {s:{} for s in sessions}
    for s in sessions:
        # Get trials for this session
        f  = get_dFF(animal,s)[:,units]
        if permute:
            f = f[:,np.random.permutation(len(units))]
        k  = ppc_analysis.kininfo[predict]['get'](animal,s)
        x  = array([add_constant(x) for x in extract_in_trial(f,animal,s,dozscore=True)])
        y  = array(extract_in_trial(k,animal,s,dozeromean=True))
        n  = len(x)
        b  = int(n//split)
        for i in range(split):
            X += [x[i*b:] if i==split-1 else x[i*b:(i+1)*b]]
            Y += [y[i*b:] if i==split-1 else y[i*b:(i+1)*b]]
    return X,Y

def compute_constrained_sweep(animal,sessions,predict,
                              NXVAL=10,
                              REPL=1,
                              NGRID=20,
                              errtype='L1',
                              matched=False,
                              permute=False,
                              split=1):
    emth = (errtype+'_degrees') if predict==4 else errtype
    X,Y  = get_data_constrained_analysis_2(animal,sessions,predict,permute,split)
    return constrained_sweep_crossvalidated(X,Y,NXVAL,NGRID,emth,matched)



# Parallel run of all cross-validation samples
def constrained_analysis_helper(p):
    i,p = p
    return i,compute_constrained_sweep(*p)[0]

@memoize
def do_parallel_constrained_analysis(animal,sessions,predict,NXVAL=10,REPL=100,NGRID=20,errtype='L1',matched=False):
    reset_pool()
    jobs = [
        (animal,sessions,predict,NXVAL,1,NGRID,errtype,matched,False) 
        for i in range(REPL)
        ]+[
        (animal,sessions,predict,NXVAL,1,NGRID,errtype,matched,True) 
        for i in range(REPL)
        ]
    results = parmap(constrained_analysis_helper,enumerate(jobs))
    print('done')
    shuffle = results[REPL:]
    results = results[:REPL]
    return results,shuffle

