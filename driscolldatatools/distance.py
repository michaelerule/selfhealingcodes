#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from neurotools.nlab import *

from pylab import *
from numpy import *

def mdistance(a,b):
    return norm(a-b)/sqrt(norm(a)*norm(b))

def mdistance_submatrix_sampled_helper(params):
    iproblem,(A,B,k,m,shuffle) = params
    r,c = A.shape
    d = []
    for i in arange(m):
        #
        subset_r = np.random.randint(0,r,k)
        subset_c = np.random.randint(0,c,k)
        a = A[subset_r,:][:,subset_c]
        if shuffle:
            subset_r = np.random.randint(0,r,k)
            subset_c = np.random.randint(0,c,k)
        b = B[subset_r,:][:,subset_c]
        d.append(mdistance(a,b))
    return iproblem,np.array(d)

reset_pool()

def mdistance_submatrix_sampled(A,B,k,nsample=10000,shuffled=False):
    '''
    Get the normalized L2 distance between matrix `A` and `B`
    for a size `k` subsample
    
    Parameters
    ----------
    A : matrix
    B : matrix
    k : subsample size
    nsample : number of samples to use
    
    Returns
    -------
    m : mean of sampled distance
    e : 1.96*standard errror (95% level) of sampled distance
    v : variance of sampled distance
    '''
    if not B.shape==A.shape: raise ValueError('Matrices must be same shape')
    if not len(A.shape)==2 : raise ValueError('Matrices should be two-dimensional')
    r,c = A.shape
    if r<k: raise ValueError('Matrix rows cannot be less than sample size')
    if c<k: raise ValueError('Matrix rows cannot be less than sample size')
    nthreads  = 200#neurotools.jobs.parallel.cpu_count()
    blocksize = nsample//nthreads
    problems = [(i,(A,B,k,blocksize,shuffled)) for i in range(nsample//blocksize+1)]
    d = parmap(mdistance_submatrix_sampled_helper,problems,debug=False)
    d = array(d).ravel()[:nsample]
    m = np.mean(d)
    v = np.var(d)
    s = sqrt(v/k)
    e = 2.96*s
    return m,e,v
