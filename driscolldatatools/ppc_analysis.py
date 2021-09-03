#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import os,sys,traceback,h5py,platform
from neurotools.nlab import *
import ppc_data_loader
from ppc_data_loader import *
from ppc_trial       import *
from ppc_plot        import *
from neurotools.hdfmat import printmatHDF5, hdf2dict, getHDF
from itertools import product
from numpy import concatenate as cat

kininfo = {
    0:{'name':'X position'    ,'units':'m'  ,'get':get_x},
    1:{'name':'Y position'    ,'units':'m'  ,'get':get_y},
    2:{'name':'X velocity'    ,'units':'m/s','get':get_dx},
    3:{'name':'Y velocity'    ,'units':'m/s','get':get_dy},
    4:{'name':'Head direction','units':'°'  ,'get':get_theta}
}

def run_LMS(X,Y,w0=None,rate=1,reg=1e-12,normalized=False,delta=None,clip=None):
    '''
    Parameters
    ----------
    X: array-like; Nsamples x Nfeatures
        Independent/input variables
    Y: array-like; Nsamples x Noutputs
        Dependent/output variables
    
    Other Parameters
    ----------------
    w0: array-like
        Initial parameter weighers
    rate: scalar, default 1
        Learning rate
    reg: scalar, default 1e-12
        L2 regularization penalty to shrink weight
    normalized: bool, default True
        Whether to run the Normalized LMS
    delta: positive scalar

    Returns
    -------
    all_w:
    all_e;
    all_yh:
    '''
    X = np.array(X)
    Y = np.array(Y)
    if len(Y.shape)==1:
        Y = array([Y]).T

    rate  = np.float32(rate)

    if not delta is None:
        delta = np.abs(np.float32(delta))
    if not clip  is None:
        clip  = np.abs(np.float32(clip))
 
    Nsamples   = Y.shape[0]
    Nfeatures  = X.shape[1]
    Nkinematic = Y.shape[1]
    # Initial weights
    w = zeros((Nfeatures,Nkinematic)) if w0 is None\
        else np.array(w0).reshape((Nfeatures,Nkinematic))
    # Tracking weights
    u = w.copy()
    # Store a history of weights, errors, and decoder estiamtes
    all_w  = zeros((Nsamples+1,Nfeatures,Nkinematic))
    all_e  = zeros((Nsamples,Nkinematic))
    all_yh = zeros((Nsamples,Nkinematic))

    all_w[0] = w
    # Shrinkage penalty
    λ = exp(-reg)
    # Run LMS algorithm
    for t in range(Nsamples):
        x    = X[t:t+1] # 1×Nfeat; intput features
        y    = Y[t:t+1] # 1×Nkine; decoded kinematics
        yh   = x.dot(w)    # 1×Nfeat∙Nfeat×Nkine; prediction
        e    = y-yh     # 1×Nkine; errors
        # 1x1; weight update strength (normalized)
        α    = (rate/x.dot(x.T)) if normalized else rate 
        dw   = (α*x.T.dot(e)) # Nfeat×1∙1×Nkine; weight update

        # L2 regularizing potential
        dw = (w+dw)*λ-w

        # Clipped weight updates
        w   += dw if clip is None else np.clip(dw,-clip,clip)

        # Clipped tracking
        u = w if delta is None else u+np.clip(w-u,-delta,delta)

        all_w [t+1] = u
        all_e [t  ] = e
        all_yh[t  ] = yh

        if not clip is None:
            assert(np.all(np.abs(all_w[t+1]-all_w[t])<=clip*(1+1e-7)))
        if not delta is None:
            assert(np.all(np.abs(all_w[t+1]-all_w[t])<=delta*(1+1e-7)))
    
    if not delta is None:
        all_yh = np.sum(X[:,:,None] * all_w[:-1],axis=1)
        all_e  = Y - all_yh
    all_w = all_w[1:]

    return all_w,all_e,all_yh

@memoize
def prepare_data(animal,sessions,topredict,units=None):
    NDAYS    = len(sessions)
    if units is None:
        units = get_units_in_common(animal,sessions)[0]
    # Train on first session
    s  = sessions[0]
    f  = get_dFF(animal,s)[:,units]
    k  = array([kfunctions[j](animal,s) for j in topredict]).T
    x  = add_constant(cat(extract_in_trial(f,animal,s)))
    y  = cat(extract_in_trial(k,animal,s))
    w0 = reglstsq(x,y)
    # Test on following sessions
    x,y = [],[]
    for i,s in enumerate(sessions[1:]):
        f  = get_dFF(animal,s)[:,units]
        k  = array([kfunctions[j](animal,s) for j in topredict]).T
        x += extract_in_trial(f,animal,s)
        y += extract_in_trial(k,animal,s)
    X = add_constant(cat(x))
    Y = cat(y)
    return X,Y,w0

def test_LMS_full(X,Y,w0,sessions,rate,polar_error=False,reg=1e-12,delta=None,clip=None):
    '''
    Test least mean-squares, returning the weights as well as the
    statistical summary information. 
    '''
    X,T = np.array(X),np.array(Y)
    if len(Y.shape)==1: Y=array([Y]).T
    w,e,yh  = run_LMS(X,Y,w0,
        rate       = rate,
        reg        = reg,
        normalized = False,
        delta      = delta,
        clip       = clip)
    NDAYS = len(sessions)
    sampd = (X.shape[0]/(NDAYS-1))
    RMSW  = np.sqrt(np.mean(w[:,:,0]**2))
    MABSW = np.mean(abs(w[:,:,0]))
    dw    = np.diff(w[:,:,0],axis=0)
    RMSD  = np.sqrt(np.mean(dw**2))
    RMSE, MABSE = polar_error_degrees(Y,yh) if polar_error else\
                  np.sqrt(np.mean(np.abs(e)**2)),np.mean(np.abs(e))
    MABSD = np.mean(np.abs(dw))
    MAXD  = np.max (np.abs(dw))

    if not clip is None:
        if np.any(np.abs(dw)>clip*(1+1e-6)):
            raise ValueError('!!!!Sanity-check failed: some Δw larger than clip value of %f!!!!'%clip)
        assert(MAXD<clip*(1+1e-6))
        assert(MABSD<clip*(1+1e-6))

    return {
        'w' :w,
        'e' :e,
        'yh':yh,
        'RMSY' :np.std(Y),
        'MABSY':np.mean(np.abs(Y)),
        'RMSE' :RMSE,
        'MABSE':MABSE,
        'RMSW' :RMSE,
        'MABSW':MABSW,
        'RMSD' :RMSD,
        'MABSD':MABSD,
        'MAXD' :MAXD
    }

def test_LMS_summary(X,Y,w0,sessions,rate,polar_error=False,reg=1e-12,delta=None,clip=None):
    '''
    Test least mean-squares. Don't return the individual weights,
    errors, or predictions, for each sample, to save space. 

    This is a wrapper for `test_LMS_full`.
    '''
    result = test_LMS_full(X,Y,w0,sessions,rate,polar_error,reg,delta,clip)
    del result['w' ]
    del result['e' ]
    del result['yh']
    return result

print('Defined LMS algorithm')

@memoize
def get_catd_error(animal,sessions,topredict):
    units = get_units_in_common(animal,sessions)[0]
    xdata,ydata = [],[]
    for i,s in enumerate(sessions):
        dff     = get_dFF(animal,s)[:,units]
        kin     = array([kfunctions[j](animal,s) for j in topredict]).T
        xdata  += extract_in_trial(dff,animal,s)
        ydata  += extract_in_trial(kin,animal,s)
    O = permutation(len(xdata))
    X = cat(array(xdata)[O])
    Y = cat(array(ydata)[O])
    X = add_constant(X)
    W,Yh,_,_ = crossvalidated_least_squares(X,Y,10)
    err  = Y-Yh
    rmse = sqrt(mean(abs(err)**2))
    mabs = mean(abs(err))
    if topredict[0]==4:
        # Wrap head direction error as a special case
        rmse, mabs = polar_error_degrees(Y,Yh)
    maxrms = std(Y)
    maxabs = mean(abs(Y))
    return rmse,maxrms,mabs,maxabs

@memoize
def LMS_cached_result(animal,sessions,units,topredict,rate):
    '''
    Disk-caching LMS routine. 

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : list of ints
        Which session IDs to use
    '''
    X,Y,w0 = prepare_data(animal,sessions,topredict,units,verbose=False)
    result = test_LMS(X,Y,w0,sessions,rate,verbose=False,polar_error=topredict[0]==4)
    return tuple([(k,result[k]) for k in sorted(result.keys())])

def counter_wrapper(f):
    '''
    Decorator for objective functions passed to scipy.optimize.minimize, 
    which prints the number of times the objective function has been called
    as well as the current error, each time the optimizer calls the objective
    function.

    Parameters
    ----------
    f : function
        Objective function to be wrapped. Function should return a single
        scalar error values

    Returns
    -------
    obj : function
        Wrapped objcetive function.
    '''
    iter = 0
    def obj(u):
        nonlocal iter
        e = f(u)
        iter += 1
        sys.stdout.write('\riter %4d: error=%0.10f'%(iter,np.mean(e))+' '*40)
        sys.stdout.flush()
        return e
    return obj

def get_data_constrained_analysis(animal,sessions,predict,permute=False):    
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
        k  = kininfo[predict]['get'](animal,s)
        x  = array([add_constant(x) for x in extract_in_trial(f,animal,s,dozscore=True)])
        y  = array(extract_in_trial(k,animal,s,dozeromean=True))
        X += [x]
        Y += [y]
    return X,Y

@memoize
def get_same_day_performances(animal,sessions,predict,NXVAL=10,errmethod=None):
    '''
    Assess decoding performance for single sessions using ordinary least
    squares (OLS), with cross-validation. 

    Parameters
    ----------
    animal: int
        Which subject to use
    sessions: list of ints
        Which sessions to use
    predict: int
        Which kinematic variable to predict

    Other Parameters
    ----------------
    NXVAL : int, default 10
        Number of crossvalidation blocks to use when fitting
    errmethod : str, default None
        For this analysis, we use 'L1' for position and velocity, and
        'L1_degrees' for head direction. 

    Returns
    -------
    sameday: dict
        Dictionary of model fitting results. Contains an entry for each
        session, which is a dictionary containing the values
        `w0`, the OLS regression weights using all available data to train,
        `e0`, the training error associated with `w0`,
        `wx`, NXVAL copies of OLS weights from crossvalidated training, and
        `e0`, NXVAL copies of the error under cross-validation
    '''
    if errmethod is None:
        # Handle head direction as a special case  
        errmethod='L1_degrees' if predict==4 else 'L1'
    errfn = neurotools.stats.error_functions[errmethod]
    # Get trial data (mean-centered; neural signals z-scored)
    X,Y = get_data_constrained_analysis(animal,sessions,predict)
    # Store same-day fit information
    sameday = {s:{} for s in sessions}
    for s,x,y in zip(sessions,X,Y):
        # Get overfit same-day performance
        xx,yy = concatenate(x), concatenate(y)
        w0 = reglstsq(xx,yy)
        e0 = errfn(yy, xx @ w0)
        sameday[s]['w0'] = w0
        sameday[s]['e0'] = e0
        # Get cross-validated same-day performance
        wx,yx,ex = trial_crossvalidated_least_squares(x,y,NXVAL,shuffle=True,errmethod=errmethod)
        sameday[s]['wx'] = wx
        sameday[s]['ex'] = ex
    return sameday

def constrained_analysis_regression(x,y,dw,tol=1e-6,maxiter=3000,wwi=None):
    '''
    Perform multiple least-squares regressions over several days, constraining
    the overal root-mean-squared (RMS) difference in weights between days to be
    less than `dw`.

    This uses the `SLSQP` solver in `scipy.optimize.minimize`.

    Parameters
    ----------
    x: list
        List of training data for each session, dependent variables.
        Each item should be a Nsamples x Nfeatures array. 
    y: list
        List of training data for each session, independent variable.
        Each item should be a length Nsamples vector. 
    dw: float
        The upper limit on the RMS change in weight across days.
    
    Other Parameters
    ----------------
    tol: float, default 1e-3
        The tolerance stopping criteron to forward to the `minimize` solver. 
    maxiter: int, default 3000
        The maximum number of iterations before giving up
    wwi: array, default None
        Initial guess for the constrained weights. This should be a
        Nsessions x Nfeatures sized array of weights for each day. 
        If `None`, the function will try to guess sensible initial conditions. 

    Returns
    -------
    weigts: array
        The Nsessions x Nfeatures sized array of linear regression coefficients
        for each day, fit such that the RMS weight change across days is less
        than `dw`.
    '''
    x = list(map(np.array,x))
    y = list(map(np.array,y))
    if wwi is None:
        # Initializer: interpolate between same-day and concatenated
        wwi   = array([reglstsq(xi,yi) for (xi,yi) in zip(x,y)])
        rmsdw = sqrt(mean(diff(wwi,axis=0)**2))
        if rmsdw<=dw: return wwi
    ww1   = outer(ones(len(x)),reglstsq(cat(x),cat(y)))
    rmsdw = sqrt(mean(diff(wwi,axis=0)**2))
    wwi   = (wwi-ww1)*(dw/rmsdw)+ww1
    # Compute covariances and cross covariances
    sXX = np.array([(xi.T.dot(xi))/xi.shape[0] for  xi     in x       ])
    sXY = np.array([(xi.T.dot(yi))/yi.shape[0] for (xi,yi) in zip(x,y)])  
    j0  = -2*sXY.ravel()
    @counter_wrapper
    def objective(wr):
        w = wr.reshape(wwi.shape) 
        # np.sum([w[i].T@(sXX[i]@w[i]-2*sXY[i]) for i in range(D)])
        return einsum('is,ist,it',w,sXX,w)-2*np.sum(w*sXY)
    def jacobian(w):
        w = w.reshape(wwi.shape) 
        # np.array([2*(sXX[i]@w[i]-sXY[i]) for i in range(D)]).ravel()
        return j0+2*einsum('ist,it->is',sXX,w).ravel()
    def constraint(w):
        w = w.reshape(wwi.shape) 
        return dw**2-mean(diff(w,axis=0)**2)
    result = scipy.optimize.minimize(
        objective, 
        wwi.ravel(), 
        method      = 'SLSQP',
        jac         = jacobian, 
        tol         = tol,
        constraints = [{'type':'ineq','fun':constraint}],
        options     = {'maxiter':maxiter,'ftol':tol})
    print(result.success)
    print(result.message)
    return result.x.reshape(wwi.shape)

def constrained_analysis_crossvalidated(X,Y,dw,NXVAL=10,errmethod='L1',matched=True):
    '''
    Performs `constrained_analysis_regression` using K-fold crossvalidation. 

    Parameters
    ----------
    X : list 
        List of Ntimes x Nfeatures independent variable arrays,
        one for each group/session/day
    Y : list 
        List of Ntimes x 1 dependent variable arrays,
        one for each group/session/day
    dw : scalar
        Maximum root-mean-squared weight change to allow
    
    Other Parameters
    ----------------
    NXVAL: int, default 10
        Number of cross-validation blocks
    errmethod: str, default 'L1'
        Which error metric to use for summarizing results
    matched: bool, default True
        Whether to randomly reduce the number of trials used so that the total number
        of trials used to train the models matches the typical number of trials on each 
        day.
        
    Returns
    -------
    ww : 
        List of weight fits from each crossvalidation block.
        Each fit is a list of weights on each day
    ee : 
        List of average errors from each crossvalidation block. 
    '''
    efn   = neurotools.stats.error_functions[errmethod]
    NBLK  = NXVAL*len(X) if matched else NXVAL
    sets  = [partition_trials_for_crossvalidation(x,NBLK,shuffle=True) for x in X]
    ww,ee = [],[]
    wwi = constrained_analysis_regression    (map(cat,X),map(cat,Y),dw)
    for k in range(NXVAL):
        trn  = [cat(s[:k]+s[k+1:NXVAL]) for s in sets]
        tst  = [cat(s[k:k+1]+s[NXVAL:]) for s in sets] # [s[k] for s in sets]
        trnX = [cat(x[t]) for (x,t) in zip(X,trn)]
        trnY = [cat(y[t]) for (y,t) in zip(Y,trn)]
        tstX = [cat(x[t]) for (x,t) in zip(X,tst)]
        tstY = [cat(y[t]) for (y,t) in zip(Y,tst)]
        w    = constrained_analysis_regression(trnX,trnY,dw)
        e    = mean([efn(y,x.dot(w)) for (w,x,y) in zip(w,tstX,tstY)])
        ww += [w]
        ee += [e]
    return np.array(ww),np.array(ee)

@memoize
def compute_constrained_analysis(animal,sessions,predict,pctdw,NXVAL=10,errtype='L1'):
    '''

    Parameters
    ----------
    animal: int
        Which subject to use
    sessions: list of ints
        Which sessions to use
    predict: int
        Which kinematic variable to predict
    pctdw: float
        Percent weight change to tolerate. Average weight magnitude is 
        defined as the mean absolute weight size, for un-regularized models
        trained on single sessions.
    
    Other Parameters
    ----------------
    NXVAL: int, default 10
        Number of cross-validation blocks
    errtype: str, default 'L1'
        Which error metric to use for summarizing results
    
    Returns
    -------
    ed:
    ec:
    ex:
    
    '''
    NDAYS    = len(sessions)
    emth    = (errtype+'_degrees') if predict==4 else errtype
    # Get data
    X,Y     = get_data_constrained_analysis(animal,sessions,predict)
    sameday = get_same_day_performances    (animal,sessions,predict,NXVAL,emth)
    # Same-day
    ed = array([sameday[s]['ex'] for s in sessions]) 
    # Concatenated: need to thin the trials to match amount of training data!
    x,y  = cat(X),cat(Y)
    nuse = len(x)//NDAYS
    iuse = np.random.choice(arange(len(x)),nuse,replace=False)
    xuse = x[iuse]
    yuse = y[iuse]
    ec = trial_crossvalidated_least_squares(x,y,NXVAL,shuffle=True,errmethod=emth)[2]
    # Constrained
    w0    = array([sameday[s]['wx'] for s in sessions])
    dw    = mean(abs(w0))*pctdw/100
    wx,ex = constrained_analysis_crossvalidated(X,Y,dw,NXVAL,emth)
    return ed,ec,ex



