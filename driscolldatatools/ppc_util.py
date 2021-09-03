#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from   neurotools.nlab import *
import os,sys,traceback,h5py
from   sklearn.decomposition import FactorAnalysis
from   scipy.linalg import *
import numpy as np

from sklearn.decomposition import FactorAnalysis
import warnings

def get_factor_analysis(X,NFACTORS):
    '''
    Wrapper to fit factor analysis model, extract the model,
    #and sort by factor importance
    '''
    fa = FactorAnalysis(n_components=NFACTORS)
    Y = fa.fit_transform(X)
    Sigma = fa.noise_variance_
    F     = fa.components_
    # Get eigenvalues/loadings
    lmbda = diag(F.dot(F.T))
    # Sort by importance
    #order = argsort(abs(lmbda))[::-1]
    #lmbda = lmbda[order]
    #F     = F[order,:]
    return Y,Sigma,F,lmbda,fa

def project_factors(X,F,S):
    '''
    Project observations X with noise variances S onto latent factors F.
    This uses the same argument/return conventions as scipy's factor analysis.
    
    Parameters
    ----------
    X : array-like
        data
    F : array-like
        factor matrix
    S : array-like
        i.i.d variances
    '''
    Nfactors, Nobserved = F.shape
    assert(S.shape==(Nobserved,))
    P = 1/S
    FP = F*P
    Px = np.eye(Nfactors) + FP.dot(F.T)
    return lstsq(Px,FP.dot(X.T))[0].T

def predict_latent(fa,predict_from,X):
    '''
    Predict mean, variance of some factors from other
    '''
    S = fa.noise_variance_
    F = fa.components_
    e = diag(F.dot(F.T))
    N = F.shape[0]
    Xf = X[:,predict_from]
    Ff = F[:,predict_from]
    Pf  = np.diag(1/S[predict_from])
    I   = np.eye(N)
    FPf = Ff.dot(Pf)
    Px  = FPf.dot(Ff.T)
    pPx = I+Px
    # Predict means
    latents = scipy.linalg.lstsq(pPx,FPf.dot(Xf.T))[0]
    return latents

def factor_predict(fa,predict_from,predict_to,X):
    '''
    Predict mean, variance of some factors from other
    '''
    S = fa.noise_variance_
    F = fa.components_
    e = diag(F.dot(F.T))
    N = F.shape[0]

    Xf = X[:,predict_from]
    Xt = X[:,predict_to  ]

    Ff = F[:,predict_from]
    Ft = F[:,predict_to  ]

    Pf  = np.diag(1/S[predict_from])
    I   = np.eye(N)

    FPf = Ff.dot(Pf)
    Px  = FPf.dot(Ff.T)
    pPx = I+Px

    # Predict means
    latents = scipy.linalg.lstsq(pPx,FPf.dot(Xf.T))[0]
    Xthat   = Ft.T.dot(latents)

    # Predict variance
    #iFf = numpy.linalg.pinv(Ff)
    #Sf  = np.diag(S[predict_from])
    St  = np.diag(S[predict_to])
    #M   = lstsq(Ff,Ft)[0]
    #Xtc = M.T.dot(Sf).dot(M)+St

    Xtc = Ft.T.dot(scipy.linalg.lstsq(pPx,Ft)[0]) + St

    return Xthat,Xtc

def deep_tuple(x):
    '''
    Convert x to tuple, deeply.
    Defaults to the identity function if x is not iterable
    '''
    if type(x)==str:
        return x
    try:
        result = tuple(deep_tuple(i) for i in x)
        if len(result)==1: result = result[0]
        return tuple(result)
    except TypeError:
        return x
    assert 0

def deep_map(f,tree):
    '''
    Maps over a tree like structure
    '''
    if hasattr(tree, '__iter__') and not type(tree) is str:
        return tuple([deep_map(f,t) for t in tree])
    else:
        return f(tree)

def to_indices(x):
    '''
    There are two ways to extract a subset from numpy arrays:
    1. providing a boolean array of the same shape
    2. providing a list of indecies
    
    This function is designed to accept either, and return a list 
    of indecies.
    '''
    x = np.array(x)
    if x.dtype==np.dtype('bool'):
        # typed as a boolean, convert this to indicies
        return deep_tuple(np.where(x))
    # Array is not boolean; 
    # Several possibilities
    # It could already be a list of indecies
    # OR it could be boolean data encoded in another numeric type
    symbols = np.unique(x.ravel())
    bool_like = np.all((symbols==1)|(symbols==0))
    if bool_like:
        if len(symbols)<2:
            warnings.warn('Indexing array looks boolean, but contains only the value %s?'%symbols[0])
        return deep_tuple(np.where(x!=0))
    if np.all((symbols==-1)|(symbols==-2)):
        warnings.warn('Numeric array for indexing contains only (-2,-1); Was ~ applied to an int32 array? Assuming -1=True')
        x = (x==-1)
        return deep_tuple(np.where(x))
    if np.all(np.int32(x)==x):
        # Seems like it is already integers?
        return deep_tuple(x)
    raise ValueError('Indexing array is numeric, but contains non-integer numbers?')
    
    
def factors_tree(F,labels=None):
    '''
    F is Nfactors x Nvariables factor matrix from factor analysis
    This code needs improvement.
    '''
    def PSDpeak(J):
        '''
        Find row and column of largest off-diagonal value in 
        a symmetric matrix.
        '''
        JT = J.copy()
        JT *= np.tri(*JT.shape)
        np.fill_diagonal(JT,0)
        a,b = np.unravel_index(np.argmax(JT),JT.shape)
        return sorted([a,b])
    def mergedrows(a,b):
        '''
        Average two vectors together and normalize result
        '''
        if np.dot(a,b)<0: a = -a
        c = (a+b)/2
        return c/np.dot(c,c)**0.5
    def collapse(F,a,b):
        ja,jb = F[:,(a,b)].T
        jab = mergedrows(ja,jb)
        F2 = np.zeros((F.shape[0],F.shape[1]-1))
        order = sorted(list(set(np.arange(F.shape[1]))-set([a,b])))
        F2[:,1:] = F[:,order]
        F2[:,0] = jab
        return F2
    N = F.shape[1]
    if labels is None:
        labels = ['%d'%i for i in arange(F.shape[1])]
    tree = labels#list(arange(N))
    while F.shape[1]>1:
        p1 = scipy.linalg.pinv(F)
        J0 = np.abs(p1.dot(p1.T))
        a,b = PSDpeak(J0)
        #print(a,b)
        F = collapse(F,a,b)
        node = [(tree[a],tree[b])]
        del tree[a]
        del tree[b-1]
        tree = node+tree
    return tree

def invert_permutation(p):
    '''
    Return inverse of a permutation
    '''
    p = np.int32(np.array(p))
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return tuple(np.int32(np.array(s)))
    
def interp_all(t,tp,fp,**kwargs):
    '''
    '''
    return np.array([np.interp(t,tp,fi,**kwargs) for fi in fp])

def pseudotime(signal,N=100):
    '''
    '''
    T = np.array(signal).shape[0]
    pseudotime = linspace(0,1,N)
    realtime = linspace(0,1,T)
    return interp_all(pseudotime,realtime,signal.T).T

def virtual_x(x,y,PADDING=5,VXMAX=4.65):
    '''
    '''
    virtual_x = y+np.abs(x)
    virtual_x[:PADDING ][virtual_x[:PADDING ]>0.1] = 0
    stop = np.where(np.diff(virtual_x)[-PADDING:]<0)[0]
    if len(stop)>0:
        stop = len(x)-PADDING+stop[0]-1
        virtual_x[stop:] = virtual_x[stop]
    return unitscale(virtual_x)*VXMAX

def derivative_direction(x):
    c = covariance(x)
    e,v = real_eig(c)
    w = v[:,0]
    return w*sign(sum(x.dot(w)))

def centermean(x):
    return 0.5*(x[1:,...]+x[:-1,...])

def rmse(X1,X2,axis=None):
    '''
    Root mean-squared error
    '''
    return np.mean((X1-X2)**2,axis=axis)**0.5

def normalize(x):
    return x/numpy.linalg.linalg.norm(x)

def unitsum(x):
    return x/np.sum(x)

amap = lambda f,x:array(list(map(f,x)))

def make_lagged(x,NLAGS=5,LAGSPACE=1):
    '''
    Parameters
    ----------
    x : vector
        Vector from which to build time-lagged features.
    
    Other Parmaeters
    ----------------
    NLAGS: int, default 5
        Number of time lags to create
    LAGSPACE: int, default 1
        Sample separation between time lags
    '''
    if not len(x.shape)==1:
        raise ValueError('Signal should be one-dimensional')
    t = arange(len(x))
    return array([np.interp(t-LAG,t,x) for LAG in arange(NLAGS)*LAGSPACE])
    
def onehot(ids):
    '''
    Generate so-called "one-hot"
    representation of class labels from 
    a vector of class identities
    
    Returns
    -------
    labels:
        labels corresponding to each index
    r:
        One-hot label format
    '''
    ids      = np.array(ids)
    labels   = sorted(list(set(ids)))
    nsamples = len(ids)
    nlabels  = len(labels)
    r = np.zeros((nlabels,nsamples))
    for i,l in enumerate(labels):
        r[i,ids==l] = 1
    return labels,r
    
def el2(w,x,y):
    return mean((y.ravel()-(x.dot(w.ravel())).ravel())**2)**0.5

def add_constant(data):
    return cat([data,ones((data.shape[0],1))],axis=1)

def polar_error_degrees(x,xh):
    # Report error in physical units
    e = abs(x-xh)
    e[e>180] = 360-e[e>180]
    return mean(abs(e)**2)**.5, mean(abs(e))

    
def block_shuffle(x,BLOCKSIZE=None):
    '''
    Shuffles array in blocks along first dimension
    '''
    T = x.shape[0]
    if BLOCKSIZE is None:
        BLOCKSIZE = max(10,T//100)
    nblocks = int(np.ceil(T/BLOCKSIZE))
    PAD = nblocks*BLOCKSIZE-T
    if PAD>0:
        x2 = np.zeros((nblocks*BLOCKSIZE,)+x.shape[1:])
        x2[:T,...] = x
        x = x2
    R = x.reshape((nblocks,BLOCKSIZE)+x.shape[1:])
    R = R[np.random.permutation(nblocks),...]
    R = R.reshape((nblocks*BLOCKSIZE,)+x.shape[1:])
    return R[:T,...]
    
def block_crossvalidated_linear_reconstruction(W,X,NXVAL=10,BLOCKSIZE=100):
    # First shuffle (coherently both features and predicted covariate) in blocks
    WX = np.concatenate([W,X[:,None]],axis=1)
    WX = block_shuffle(WX,BLOCKSIZE)
    Ws = WX[:,:W.shape[-1]]
    Xs = WX[:,W.shape[-1]:]
    # Next iterate over cross-validations
    T = Ws.shape[0]
    XVALSIZE = int(np.ceil(T/NXVAL))
    Xhat = []
    for T0 in arange(NXVAL)*XVALSIZE:
        # Build training data
        Wtrain = np.concatenate([Ws[:T0],Ws[T0+XVALSIZE:]])
        Xtrain = np.concatenate([Xs[:T0],Xs[T0+XVALSIZE:]])
        # Build testing data
        Wtest = Ws[T0:T0+XVALSIZE]
        Xtest = Xs[T0:T0+XVALSIZE]
        # Solve AX=B for X = A^{-1}B
        M  = lstsq(Wtrain,Xtrain)[0]
        # Predict X
        Xhat.extend(Wtest.dot(M))
    return array(Xhat)

def leave_one_out_crossvalidation(W,X,NTRIALS,NSUBJECTS,TIMERES):
    Q = W.reshape(NTRIALS*NSUBJECTS,TIMERES,W.shape[-1])
    X = X.reshape(NTRIALS*NSUBJECTS,TIMERES)
    NBLOCKS = X.shape[0]
    Xhat = []
    trial_set = set(range(NBLOCKS))
    for trial in range(NBLOCKS):
        train = np.array(list(trial_set-{trial}))
        Xtrain = X[train,...].reshape((NBLOCKS-1)*TIMERES,*X.shape[2:])
        Qtrain = Q[train,...].reshape((NBLOCKS-1)*TIMERES,*Q.shape[2:])
        Xtest  = X[trial,...]
        Qtest  = Q[trial,...]
        # Solve AX=B for X = A^{-1}B
        M  = scipy.linalg.lstsq(Qtrain,Xtrain)[0]
        # Predict X
        Xhat.extend(Qtest.dot(M))
    Xhat = array(Xhat)
    return Xhat
