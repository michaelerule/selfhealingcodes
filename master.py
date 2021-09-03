#!/usr/bin/python
# -*- coding: UTF-8 -*-

######################################################################
# INITIALIZATION 
# Set to True to have worker threads report progress
# Turn off to avoid printing too much when runnign parallel jobs
PRINT_LOGGING = False
# Set to true to use Jax jit compiling; very fast
# Turn off to get usable error messages when debugging
# Turn off if you see weird behavior when running parallel jobs
USE_JAX = True
# Common configuration for self-healing codes
from config           import *
from standard_options import *
memoize = neurotools.jobs.initialize_system_cache.memoize

'''
Logging for debugging purposes, with fun colors (:
the global __log_prefix__ allows us to distinguish messages
from different worker processes.
''' 
__log_prefix__ = 'main'
bg = lambda i:'\x1b[48;5;%dm'%i
fg = lambda i:'\x1b[38;5;%dm'%i
def LOG(msg,
        newline   = True,
        bgcolor   = bg(53),
        logcolor1 = fg(174),
        logcolor2 = fg(226)):
    global __log_prefix__
    if not PRINT_LOGGING: return
    def subroutine(msg,flush=True,force_ascii=False):
        if force_ascii:
            msg = msg.encode("ascii","replace").decode('ascii')
        print('\r'+bgcolor+logcolor1 
            +(__log_prefix__+': ').rjust(10)
            +'\x1b[0m'+bgcolor+logcolor2+msg.ljust(70)
            +'\x1b[0m',end='\n' if newline else '',
            flush=flush);
    flush = True
    force_ascii = False
    for retry in range(3):
        try:
            subroutine(msg,flush,force_ascii)
            return
        except UnicodeEncodeError:
            force_ascii=True
        except BlockingIOError:
            flush = False

'''
Jax is fast, but may cause bugs in multiprocessing.
'''
if USE_JAX:
    # Shadow all pylab functions and numpy with the Jax versions
    # Keep numpy around as np0 for easier RNG, array assignments
    import jax
    import jax.numpy as np
    import numpy.random as npr
    from jax.api           import jit, grad, vmap
    from jax.config        import config
    from jax.scipy.special import logsumexp
    from jax.numpy         import *
    from jax import jacfwd, jacrev
    from jax import lax
    fori = lax.fori_loop
    def hess(f,inparam):
        return jacfwd(jacrev(f,inparam),inparam)
    import numpy as np0
else:
    # Map numpy to np and np0
    # make jit a no-op
    # mimic the fori loop function
    import numpy as np
    import numpy as np0
    jit = lambda x:x
    def fori(lower, upper, body_fun, init_val):
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val
    
'''
Define the firing-rate nonlinearity and its inverse
We're using an exponential nonlinearity at the m ment,
with some checks to avoid numerical overflow.
'''
# Version of exp and log that won't over/under flow
def slog(x,eps=9.1469e-12):
    return np.log(np.maximum(eps,x))
def sexp(x,limit=9.4192):
    return np.exp(np.clip(x,-limit,limit))
def σ(x,limit=9.4192):
    x = np.clip(x,-limit,limit)
    return sexp(-np.logaddexp(0.0,-x))
# Use exponential nonlinearity
φ         = sexp
φ_inverse = slog

'''
Prepare options for multithreading/multiprocessing.

Set **everything** within a process to single-threaded so multiprocessing doesn't
burn resources waiting for cores to sync/communicate on linear algebra calls
(I'm not sure how much of this is needed)

(Jax was formerly imported here but was removed due to bugs)
Limit ourselves to single-threaded jax/xla operations to avoid thrashing.
See https://github.com/google/jax/issues/743.
'''
import os
import multiprocess as multi
import traceback
from   threadpoolctl import threadpool_info,threadpool_limits
def limit_cores(CORES_PER_THREAD=1): 
    keys = [
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'OMP_NUM_THREADS ',
        'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS']
    for k in keys: os.environ[k] = str(CORES_PER_THREAD)
    os.environ["XLA_FLAGS"] = \
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=%d"\
    %CORES_PER_THREAD
    threadpool_limits(limits=CORES_PER_THREAD, user_api='blas')
    NCORES = multi.cpu_count()
    if PRINT_LOGGING:
        print('>>> %d cores available.'%NCORES)
        print('>>> limited each process to %d cores.'%CORES_PER_THREAD)
    
# Run this if you plan to use multiprocessing
#(It is best to limit each process to one core)
#limit_cores(1)

######################################################################
# Define different geometries

from scipy.linalg import expm

# Graph adjacency matrix for circular variable
def circle_graph(L):
    A = eye(L,k=  1)+eye(L,k= -1)+eye(L,k=L-1)+eye(L,k=1-L)
    return A

# Graph adjacency matrix for linear variable
def linear_graph(L):
    return eye(L,k=1) + eye(L,k=-1)

# Graph adjacency matrix for T-maze variable
def tmaze_graph(L):
    # Length of each arm of the T
    La = L//4#2//7
    # Lenth of the neck of the T
    Ln = L - 2*La
    # Index offset of start ot second arm of T
    L1 = Ln + La
    # Start with two linear segments: neck and arms
    neck = eye(Ln,k=1) + eye(Ln,k=-1)
    arm1 = eye(La,k=1) + eye(La,k=-1)
    arm2 = eye(La,k=1) + eye(La,k=-1)
    O = zeros((Ln,La))
    A = block([[neck,O],[O.T,arm1]])
    O = zeros((L1,La))
    A = np0.block([[A,O],[O.T,arm2]])    
    # Join up the segments
    A[Ln-1,Ln  ] = A[Ln  ,Ln-1] = 2 # Neck to arm 1
    A[Ln-1,L1  ] = A[L1  ,Ln-1] = 2 # Neck to arm 2
    A[Ln  ,L1  ] = A[L1  ,Ln  ] = 2 # Arm 1 to arm 2
    return A

# Get Laplacian-based diffusion map from graph adjacency matrix
def getmap(graph,L,σ):
    t = σ**2/2
    A = graph(L)
    D = diag(sum(A,0))
    l = D-A
    m = expm(-t*l)
    return m/sum(m,axis=0)[:,None]

geometries = {
    'ring' :lambda L,σ:getmap(circle_graph,L,σ),
    'line' :lambda L,σ:getmap(linear_graph,L,σ),
    'tee'  :lambda L,σ:getmap(tmaze_graph ,L,σ)}

######################################################################
'''
Simulate different types of encoding drift

This code is messy (mix of old hacks I havn't had time to clean up)

Define different ways that the encoding (input) population might 
drift. Presently, we simlulate different types of gradual drift
based on Ornstein Uhlenbeck random walk of tuning curves, as well
as variants that change input tunings abruptly. To add a new model
of encoding drift, define a function that returns a
Ntimepoints×Nfeatures×Nlocations array, defining how tuned
responses of each encoding cell ("feature") in the population
changes over time.

- `oneatatime`: change one of the $K$ features every timepoint
- `ougaussian`: tuning curves undergo Ornstein Uhlenbeck random walk
- `oumomentum`: tuning curves undergo Ornstein Uhlenbeck random walk, 
       filtered so that changes are correlated across days

The homeostasis model for the encoding population is assumed to be 
fixed. If you change these, you must clear the disk cache for
changes to take effect. 
'''

# Homeostasis model for drifting feature initialization
ηβ = .2 # Threshold homeostasis gain
ηγ = .1 # Variability homeostasis gain
μt = 4  # Target mean rate
σt = 5  # Target variability
γ0 = 0  # Initial value for gain
β0 = -2 # Initial value for bias

@jit
def homeostasis_loop(γβ,x,μt,σt):
    # Subroutine for the simple_homeostasis function.
    γ,β = γβ
    λ  = φ(x*γ+β)
    β += ηβ*(μt-mean(λ))
    γ += ηγ*(1-std(λ)/σt)
    return γ,β

@jit
def simple_homeostasis(x,
    N  = 5,   # Iterations
    γ  = γ0,  # Initial value for gain
    β  = β0,  # Initial value for bias
    μ0 = μt,  # Target for mean
    σ0 = σt): # Target for standard deviation
    # Homeostasis for a single cell
    γ2,β2 = fori(0, N*10,
        lambda i,γβ: homeostasis_loop(γβ,x,μ0,σ0),
        (γ,β))
    return γ2,β2,φ(x*γ2+β2)

def get_oneatatimelogGP_features(L,K,T,σ,geometry='ring'):
    '''
    Simulate a population of K features on a length L circular track,
    drifting for T timepoints. Features are spatially low-pass filterd
    with Gaussian kernel with standard deviation σ. One feature
    changes at each time point. 
    
    Parameters
    ----------
    L: Number of discrete positions/latent states
    K: Number of initial features to generate
    T: Number of random features to samples
    σ: Spatial scale of features
    '''
    if not geometry in geometries:
        raise ValueError('geometry %s is not supported'%geometry)
    kern = geometries[geometry](L,σ)
    def randfeature(L):
        return simple_homeostasis(kern@randn(L),μ0=μt,σ0=σt)[2]
    s = np0.array([randfeature(L) for k in range(K)])
    Ξ = [np0.copy(s)]
    for t in range(T-1):
        s[t%K] = randfeature(L)
        Ξ += [np0.copy(s)]
    return array(Ξ)

@jit
def hhelper(x,γ,β,μt,σt):
    '''
    Vectorized homeostasis helper routine
    
    Parameters
    ----------
    x: K×L array of synaptic activations for K cells at L locations
    γ: lenthl L array of initial gains
    β: lenthl L array of initial biases
    ηβ: Threshold homeostasis gain (default .2)
    ηγ: Variability homeostasis gain (default .1)
    μt: Target mean rate (default 5)
    σt: Target rate standard deviation (default 5)
    '''
    μx = mean(x,1)
    x  = x - μx[:,None]
    for i in range(5):
        λ  = φ(x*γ[:,None]+(β+μx)[:,None])
        β += ηβ*(μt-mean(λ,1))
        γ += ηγ*(1-std(λ,1)/σt)
    return γ,β,φ(x*γ[:,None]+(β+μx)[:,None])

# Encoding witu Ornstein Uhlenbeck drift
def get_OUlogGP_features_vectorized(L,K,T,σ,τ,N=10,geometry='ring'):
    '''
    Parameters
    ----------
    L: Number of discrete positions/latent states
    K: Number of initial features to generate
    T: Number of random features to samples
    σ: Spatial scale of features
    τ: Correlation time for drift
    N: Iterarions for initial homeostasis (default 10)
    γ: Initial value for gain (default 0.5)
    β: Initial value for bias (default -2)
    '''
    if not geometry in geometries:
        raise ValueError('geometry %s is not supported'%geometry)
    k = geometries[geometry](L,σ)
    a = randn(K,L)
    x = a@k.T
    γ = ones(K)*γ0
    β = ones(K)*β0
    for i in range(N):
        γ,β,z = hhelper(x,γ,β,μt,σt)
    Z = []
    # Discrete time decay rate and noise variance
    α = exp(-1/τ) 
    ξ = 1-α**2
    for t in range(T):
        a = a*α + randn(K,L)*sqrt(ξ)
        γ,β,z = hhelper(a@k.T,γ,β,μt,σt)
        Z += [z]
    return np.array(Z)

# Encoding that drifts smoothly
def get_momentumOUlogGP_features_vectorized(L,K,T,σ,τ,N=10,geometry='ring'):
    '''
    Parameters
    ----------
    L: Number of discrete positions/latent states
    K: Number of initial features to generate
    T: Number of random features to samples
    σ: Spatial scale of features
    τ: Correlation time for drift
    N: Iterarions for initial homeostasis (default 10)
    γ: Initial value for gain (default 0.5)
    β: Initial value for bias (default -2)
    '''
    if not geometry in geometries:
        raise ValueError('geometry %s is not supported'%geometry)
    k = geometries[geometry](L,σ)
    a = randn(K,L)
    x = a@k.T
    γ = ones(K)*γ0
    β = ones(K)*β0
    for i in range(N):
        γ,β,z = hhelper(x,γ,β,μt,σt)
    Z = []
    ω = randn(L)
    τ = τ/2
    α = exp(-1/τ) 
    ξ = 1-α**2
    for t in range(T):
        ω = ω*α + randn(K,L)*sqrt(ξ)
        a = a*α + ω/τ
        γ,β,z = hhelper(a@k.T,γ,β,μt,σt)
        Z += [z]
    return np.array(Z)

# Completely different features on each day
def get_random_features_vectorized(L,K,T,σ,N=10,geometry='ring'):
    '''
    Parameters
    ----------
    L: Number of discrete positions/latent states
    K: Number of initial features to generate
    T: Number of random features to samples
    σ: Spatial scale of features
    N: Iterarions for initial homeostasis (default 10)
    '''
    if not geometry in geometries:
        raise ValueError('geometry %s is not supported'%geometry)
    k = geometries[geometry](L,σ)
    # In this case homeostasis just sets γ,β to values that are
    # typically OK given the desired moments and nonlinearity.
    a = randn(K,L)
    x = a@k.T
    γ = ones(K)*γ0
    β = ones(K)*β0
    for i in range(N):
        γ,β,z = hhelper(x,γ,β,μt,σt)
    Z = []
    for t in range(T):
        a = randn(K,L)
        γ,β,z = hhelper(a@k.T,γ,β,μt,σt)
        Z += [z]
    return np.array(Z)

######################################################################
######################################################################
######################################################################
######################################################################
# Cache features to disk once and re-use them.
@memoize
def get_features(L,K,T,P,σ,τ,features,geometry,seed):
    '''
    Options for `features`:
    - `oneatatime`: change one of the $K$ features every timepoint
    - `ougaussian`: tuning curves undergo Ornstein Uhlenbeck random walk
    - `oumomentum`: OU, filtered so that changes are correlated across days
    - 'ramdom'    : no relationship between days
    
    Options for `geometry`
    - 'ring'  : circular environment(s)
    - 'line'  : linear environment(s)
    - 'tee'   : T-maze environment(s)
    - 'mixed' : if P≥3, cycle between `ring`, `line`, and `tee` in rotation
    
    Parameters
    ----------
    L: Number of discrete positions/latent states
    K: positive int, Number of initial features to generate
    T: positive int, Number of random features to samples
    P: positive int, Number of distinct place maps to concatenate
    σ: positive float, Spatial scale of encoding features
    features: which style of drift to sample
    seed: int, random seed to use
    geometry: geometry of latent variable θ
    
    Returns
    -------
    Q: T×K×P×L array of drifting encoding features
    '''
    np0.random.seed(seed)
    LOG('Drift: %s %s seed %d LKTPστ=%d %d %d %d %0.1f %0.1f , '\
        %(features,geometry,seed,L,K,T,P,σ,τ))
    # Get geometries for each of the P environments
    if geometry=='mixed':
        geos = [['ring','line','tee'][p%3] for p in range(P)]
    else:
        geos = (geometry,)*P
    # Change features one at a time, one per iteration
    if   features=='oneatatime':
        Q = [get_oneatatimelogGP_features(L,K,T,σ,
             geometry=geos[p]) for p in range(P)]
    # OU drift on a Gaussian process
    elif features=='ougaussian':
        Q = [get_OUlogGP_features_vectorized(L,K,T,σ,τ,
             geometry=geos[p]) for p in range(P)]
    # OU drift with momentum (changes are correlated over days)
    elif features=='oumomentum':
        Q = [get_momentumOUlogGP_features_vectorized(L,K,T,σ,τ,
             geometry=geos[p]) for p in range(P)]
    # No relationship between days
    elif features=='random':
        Q = [get_random_features_vectorized(L,K,T,σ,
             geometry=geos[p]) for p in range(P)]
    else:
        raise ValueError('Feature type %s not supported'%features)
    LOG('Finishing')
    Q = np.concatenate(Q,axis=-1)
    Q = Q.reshape(T,K,P,L)
    LOG('Returning')
    return np.array(Q)
    
def prepare_features(Q,σ,r,geometry,seed,linearize=False):
    '''
    Add per-day variation on top of an existing model of drift.
    This allows us to vary this noise level without re-simulating
    the drift (if the drifting features have been cached to disk).
    
    Parameters
    ----------
    Q: T×K×L×P ndarray
        Encoding features returned by `get_features`
    σ: positive float
        Spatial scale of encoding features
    r: float∈[0,1]
        Amount of per-timepoint noise to blend into features
    linearize: bool
        If set to true, this undoes the nonlinearity applied to the features.
        This was added to confirm that the readout stability doesn't depend 
        on the encoding being sparse. It was added here to avoid complicating
        the existing code which generates drifting features. 
    '''
    T,K,P,L = shape(Q)
    if r>0: 
        # Remove nonlinearity
        x = φ_inverse(Q.reshape(T*K*P,L))
        # Sample noise
        ξ = get_features(L,K,T,P,σ,0,'random',geometry,seed+35353)
        ξ = φ_inverse(ξ.reshape(T*K*P,L))
        # Mix in noise
        x = x*sqrt(1-r) + ξ*sqrt(r)
        # Finish with homeostasis
        γ = ones(T*K*P)*.5
        β = ones(T*K*P)*-2
        for i in range(10):
            γ,β,z = hhelper(x,γ,β,μt,σt)
        Q = z
    # Remove nonlinearity if needed
    if linearize:
        Q = φ_inverse(Q)
        Q = Q*σt/std(Q)
    # Combine and mean-center
    Q = Q.reshape(T,K,P*L)
    Q = Q - mean(Q,2)[:,:,None]
    Q = concatenate([Q,ones((T,1,L*P))],1)
    return Q

######################################################################
######################################################################
######################################################################
######################################################################
# Define readout-population tuning curves
# We focus on decoding the location on a ring. Can the readout can 
# stabilize something a bit more complicated? We also check whether 
# it is possible to stabilize a code where each readout cell reponds 
# to two location. 
# 
# It turns out this is unstable without recurrence, because the 
# error-correcting mechanism is attracted to single-bump solutions. 
# This is related to Hebbian plasticity picking up the lowest-
# frequency mode ($\sin(\theta)$, $\cos(\theta)$), which is
# responsible for the largest amount of variability in the encoding
# population. 
# 
# You can add other population readout patterns here and explore their
# stability.
# 
# - `circular1`: Decode location in a periodic environment. If there
#        are multiple $P$ place-cell maps, decode from them all 
#        consistently. 
# - `circular2`: Readout cells have two peaks on opposite sites of 
#        the circular environment. If there are multiple $P$ place-
#        cell maps, decode from them all consistently. 
# - `circles`: If there are multiple $P$ place-cell maps, treat 
#        them as separate circular environments.

def get_readout_tuning(L,M,P,ν,readouttype,height=0.05):
    '''
    Generate target readout tuning curves.
    
    Original options
    
    - `circular1`: bumps wrapping once around a circle 
    - `circular2`: two bumps around a circle
    - `circles`  : P independent rings
    
    New extended options
    
    - `ring` : This should be redundant to the `circular1` option
    - `line` : Linear geometry
    - `tee`  : T-shaped geometry (c.f. Driscoll et al.)
    
    Extra special option
    
    - `mixed`: If P>1 this cycles between ring, line, and T shapes 
    '''
    x = arange(L)
    C = linspace(0,L,M+1)[:-1]
    if readouttype=='circular1':
        r = L/2
        Y = array([unitscale(npdf(r,ν,r-abs((x-c+r)%(2*r)-r))) for c in C])
        return tile(Y,P)*height
    elif readouttype=='circular2':
        r = L/4
        Y = array([unitscale(npdf(r,ν,r-abs((x-c+r)%(2*r)-r))) for c in C])
        return tile(Y,P)*height
    elif readouttype=='circles':
        if not M%P==0:
            raise ValueError('To define multiple separate environments, ensure'
                             ' that #environments P divides #readouts M')
        m = M//P
        C = linspace(0,L,m+1)[:-1]
        r = L/2
        Y = array([unitscale(npdf(r,ν,r-abs((x-c+r)%(2*r)-r))) for c in C])
        return kron(eye(P),Y)*height # Check this
    elif readouttype in {'ring','line','tee'}:
        # One of the new geometries
        if M!=L:
            raise RuntimeError(
                'Using a different number of readout neurons (M) and spatial'
                ' locations (L) is not yet implemented for the geometries '
                'ring, line, tee, and tloop. M=%d, L=%d. Please ensure that '
                'M=L.'%(M,L))
        Y = geometries[readouttype](M,ν)
        Y = amap(unitscale,Y)
        return tile(Y,P)*height
    elif readouttype=='mixed':
        # Extra special!
        if not M%P==0:
            raise ValueError('To define multiple separate environments, ensure'
                             ' that #environments P divides #readouts M')
        Y = []
        for p in range(P):
            ttype = ['ring','line','tee'][p%3]
            Y += [amap(unitscale,geometries[ttype](L,ν))*height]
        return scipy.linalg.block_diag(*Y)
    else:
        raise ValueError('Readout type %s unsupported'%readouttype)

        
#    


######################################################################
######################################################################
######################################################################
######################################################################
# Helper functions for self-healing readout
# 
# - `summarize_stability` Summarizing accuracy of location decoding
# - `train_readout` Training the initial decoding weights 
# - `population_figure` Summary figure plotting code
@jit
def summarize_stability(Z,Y):
    '''
    Summarize code stability as normalized root-mean-squared-error; 
    
    Parameters
    ----------
    Z: time×readouts×locations ndarray; the current tuning 
    Y: time×readouts×locations ndarray; the reference tuning
    '''
    Y,Z = array(Y),array(Z)
    Y = Y - mean(Y,axis=1)[:,None]
    Y = Y / std (Y,axis=1)[:,None]
    Z = Z - mean(Z,axis=2)[:,:,None]
    Z = Z / std (Z,axis=2)[:,:,None]
    return sqrt(mean((Z-Y[None,:,:])**2,(1,2)))/sqrt(2)

def train_readout(x,y,
    reg=0.01,
    rtol=1e-7,
    maxiter=100000,
    show_progress=True,
    last_feature_is_constant=True):
    '''
    Gradient descent learning of the initial tuning. 
    
    Parameters
    ----------
    x: Inputs: Ninputcells x Nlocations array
        Rows: cells, Columns: locations
        Last feature should be constant (all 1s)
    y: Outputs: Noutputcells x Nlocations array
    
    Other Parameters
    ----------------
    reg=0.01:
        Quadratic (L2) regularization strength
    rtol=1e-7:
        Relative tolerance for convergence. 
    maxiter=100000:
        Maximum number of iterations
    show_progress=True:
        Animate a progress bar
    last_feature_is_constant=True:
        Expect the last feature to be a constant all-ones vector
    '''
    # Last element of x assumed to be a constant feature
    K,L  = x.shape
    M,L2 = y.shape
    assert L==L2
    # Don't apply weight decay to threshold parameter
    if last_feature_is_constant:
        m = cat([ones((K-1,M)),zeros((1,M))],0)
    else:
        m = ones((K,M))
    # Poisson loss
    @jit
    def loss(v):
        lnλ = einsum('xm,xl->ml',v,x)
        return -mean(y*lnλ-sexp(lnλ))+reg*mean((v*m)**2)
    # Learning with weight decay
    @jit
    def g(v):
        λ = φ(einsum('xm,xl->ml',v,x))
        return x@(λ-y).T/L + 2*reg*v*m/K
    v = concatenate([zeros((K-1,M)),ones((1,M))*-4],axis=0)
    l = loss(v)
    r = .1
    for i in range(maxiter):
        v1 = v-r*g(v)
        l1 = loss(v1)
        # Adaptive stepping
        if l1<l:
            v = v1
            r *=1.01
        else:
            r *= 0.5
        if r<rtol or abs(l/l1-1)<rtol: break
        l,v = l1,v1
        if show_progress and i%10==0:
            sys.stdout.write('\r%0.12f'%l+'   ')
            sys.stdout.flush()
    if show_progress:
        sys.stdout.write('\r%0.12f'%l+'   (done)\n')
        sys.stdout.flush()
    return v

######################################################################
######################################################################
######################################################################
######################################################################
# Self-healing neural codes 
'''
Self-healing neural codes 

- `run_experiment`: A giant function to explore various plasticity
       rules for different paramaters and types of drift
- `initialization`: Caches common initialization (encoding drift
       simulation and initial weight traiing) to save time

Note: for historic reasons the recurrent weights are called V here
(the variable R is used for something else); This differs from the
text    
'''
@jit
def add_const(x):
    '''
    Adds a constant (threshold) feature 
    '''
    K_,L = shape(x)
    return concatenate([array(x),ones((1,L))],axis=0)

@memoize
def initialization(
    features,  # How to change the encoding features
    readout,   # geometry of θ use to model the output
    geometry,  # geometry of θ used to model the input
    seed,      # Random seed to use
    L,         # Number of bins for  θ
    K,         # Number of encoding features
    T,         # Number of drift timepoints to simulate
    M,         # Number of readout features
    P,         # Number of distinct enironments for θ
    σ,         # Standard deviation of encoding feature correlations, in units of bins
    τ,         # Time constant for gradual drift, if applicable
    ν,         # Standard deviation of decoding feature bumps, in units of bins
    ρ,         # Weight decay penalthy used during initial weight trainings
    r,         # Per-timepoint excess variability fraction 
    f,         # Feature magnitude 
    linearize, # Whether to remove the sparsifying nonlinearty from the features
    ):
    '''
    Options for `features`:
    - `oneatatime`: change one of the $K$ features every timepoint
    - `ougaussian`: tuning curves undergo Ornstein Uhlenbeck random walk
    - `oumomentum`: OU, filtered so that changes are correlated across days
    
    Options for `readout`
    - `circular1`: bumps wrapping once around a circle 
    - `circular2`: two bumps around a circle
    - `circles`  : P independent rings
    - `ring`     : Revised option; redundant to`circular1`
    - `line`     : Linear geometry
    - `tee`      : T-shaped geometry (c.f. Driscoll et al.)
    - `tloop`    : T-shaped geometry with end looped back to beginning
    - `mixed`: If P>1 this cycles between ring, line, and T shapes 

    Options for `geometry`
    - 'ring'  : circular environment(s)
    - 'line'  : linear environment(s)
    - 'tee'   : T-maze environment(s)
    - 'tloop  : T-maze, looped so that the end touches the beginning
    - 'mixed' : if P≥3, cycle between `ring`, `line`, and `tee` in rotation
    
    Parameters
    ----------
    features  : How to change the encoding features
    readout   : geometry of θ use to model the output
    geometry  : geometry of θ used to model the input
    seed      : Random seed to use
    L         : Number of bins for  θ
    K         : Number of encoding features
    T         : Number of drift timepoints to simulate
    M         : Number of readout features
    P         : Number of distinct enironments for θ
    σ         : Standard deviation of encoding feature correlation, (bins)
    τ         : Time constant for gradual drift, if applicable
    ν         : Standard deviation of decoding feature bumps, in units of bins
    ρ         : Weight decay penalthy used during initial weight trainings
    r         : Per-timepoint excess variability fraction 
    f         : Feature magnitude 
    linearize : Whether to remove the sparsifying nonlinearty from the features
    '''
    # Reset seed
    np0.random.seed(seed)
    tag = 'i '
    LOG(tag+'%s %s sd %d'%(features,readout,seed))
    LOG(tag+'LKTMP %d %d %d %d %d'%(L,K,T,M,P))
    LOG(tag+'σνρrf '+' '.join(map(shortscientific,(σ,ν,ρ,r,f))))
    
    # Get drifting encoding
    LOG(tag+'Sampling encoding drift...')
    Ξ = get_features(L,K,T,P,σ,τ,features,geometry,seed)
    
    # Adjust features and add per-timepoint variability
    LOG(tag+'Preparing encoding features...')
    Ξ = prepare_features(Ξ,σ,r,geometry,seed,linearize)*f
    x = Ξ[0]
    LOG(tag,'Preparing encoding features (done)')
    assert all(isfinite(Ξ))
    LOG(tag+'<X>=%0.3f, std(X)=%0.3f'%(mean(Ξ),std(Ξ)))
    
    # Readout tuning curves
    LOG(tag+'Preparing initial readout state...')
    Y  = get_readout_tuning(L,M,P,ν,readout)
    LOG(tag+'Preparing initial readout state (done)')
    assert all(isfinite(Y))
    
    # Forward weight learning
    LOG(tag+'Training forward weights...')
    W = train_readout(x,Y,ρ,show_progress=False)
    LOG(tag+'Training forward weights (done)')
    LOG("%s %d %d"%(shape(W),K+1,M))
    assert all(isfinite(W))
    
    # Recurrent weight learning
    LOG(tag+'Training recurrent weights...')
    R = train_readout(add_const(Y),Y,ρ,
                      show_progress=False)
    LOG(tag+'Training recurrent weights (done)')
    assert all(isfinite(R))
    return Ξ,x,Y,W,R

@memoize
def run_experiment(
    method    = 'homeostat', # Readout plasticity rule
    features  = 'oneatatime',# Type of encoding drift
    readout   = 'ring',      # Style of readout tuning 
    geometry  = 'ring',      # Environmental correlation structure
    saveall   = False,       # Return full evolution of decoder tuning?
    seed      = 0,           # Random seed to use
    L         = 60,          # Length of track
    K         = 100,         # Number of features
    T         = 1005,        # Number of "days" to simulate
    M         = 60,          # Number of decoder units
    P         = 1,           # Number of distinct place maps
    τ         = 100,         # Correlation time of drift (`oneatatime` ignores this)
    σ         = 0.10,        # Standard deviation of spatial scale of features 
    ν         = 0.06,        # Width of decoder unit receptive field
    I         = 100,         # Weight-update (replay) iterations per "day"
    ρ         = 1e-4,        # Weight decay rate (L2 regularization)
    r         = 0.05,        # Per-day code variability (fraction of variance)
    n         = 0.01,        # Per-day readout synapse drift (fraction of variance)
    Δ         = 5,           # Features to change before running homeostasis
    ηβ        = 0.01,        # Mean rate homeostasis
    ηγ        = 0.0015,      # Gain homeostasis
    η         = 1.0,         # Learning rate
    f         = 1.0,         # Input rate scaling
    ηz        = 0.01,        # method 'predictive' step size
    Iz        = 100,         # method 'predictive' iterations
    zk        = 0.1,         # method 'predictive' prior confidence
    ι         = 0.3,         # Homeostatic "inertia"
    eps       = 1e-6,        # Avoid divide by zero in response normalization
    hom       = 4,           # Source of hom; 012→y_f y_n y_r, if applicable
    heb       = 2,           # Source of Heb; 012→y_f y_n y_r, if applicable
    rec       = 0,           # Source of recurrent weights; 012→Σz Σy R (method predictive)
    normalize = True,        # Divisive normalization to population responses?
    linearize = False,       # Remove sparsifying nonlinearity on encoding
    disable_homeostasis = False, # Disable gain/bias homeostasis
    fail_early          = False, # Stop simulating if readout drift NRMSE > 0.75
    ):
    '''
    Options for `method`:
    - `homeostat`  : naïve homeostasis; adjusts gain and threshold
    - `hebbhomeo`  : Hebbian homeostasis; 
    - `recurrent`  : uses a recurrent map to error-correct
    - `predictive` : uses error feedback to error correct training signal
    
    To simulate response normalization, use `hebbhomeo` with `normalize=True`.
    
    Options for `features`:
    - `oneatatime`: change one of the $K$ features every timepoint
    - `ougaussian`: tuning curves undergo Ornstein Uhlenbeck random walk
    - `oumomentum`: OU, filtered so that changes are correlated across days
    
    Options for `readout`
    - `circular1`: bumps wrapping once around a circle 
    - `circular2`: two bumps around a circle
    - `circles`  : P independent rings
    - `ring`     : Revised option; redundant to`circular1`
    - `line`     : Linear geometry
    - `tee`      : T-shaped geometry (c.f. Driscoll et al.)
    - `mixed`: If P>1 this cycles between ring, line, and T shapes 

    Options for `geometry`
    - 'ring'  : circular environment(s)
    - 'line'  : linear environment(s)
    - 'tee'   : T-maze environment(s)
    - 'mixed' : if P≥3, cycle between `ring`, `line`, and `tee` in rotation
    
    Parameters
    ----------
    method    = 'homeostat' : Readout plasticity rule
    features  = 'oneatatime': Type of encoding drift
    readout   = 'circular1' : Style of readout tuning 
    geometry  = 'ring'      : Environmental correlation structure
    saveall   = False       : Return full evolution of decoder tuning?
    seed      = 0           : Random seed to use
    L  = 60     : Length of track
    K  = 100    : Number of features
    T  = 605    : Number of "days" to simulate
    M  = 60     : Number of decoder units
    P  = 1      : Number of distinct place maps
    τ  = 100    : Correlation time of drift (`oneatatime` ignores this)
    σ  = 0.10   : Standard deviation of spatial scale of features 
    ν  = 0.05   : Width of decoder unit receptive field
    I  = 50     : Weight-update (replay) iterations per "day"
    ρ  = 0.001  : Weight decay rate (L2 regularization)
    r  = 0.1    : Per-day code variability (fraction of variance)
    n  = 0.01   : Per-day readout synapse drift (fraction of variance)
    Δ  = 5      : Features to change before running homeostasis
    ηβ = 0.5    : Mean rate homeostasis
    ηγ = 0.04   : Gain homeostasis
    η  = 0.07   : Learning rate
    f  = 1.0    : Input rate scaling
    ηz = 1e-2   : Method 'predictive': step size
    Iz = 100    : Method 'predictive': iterations
    zk = 0.1    : Method 'predictive': prior weighting
    eps = 1e-6  : Avoid divide by zero in response normalization
    hom = 0     : Source of homeo. error; {0,1,2}→{yf,yn,yr} (recur., pred.)
    heb = 2     : Source of Hebb. update; {0,1,2}→{yf,yn,yr} (recur., pred.)
    rec = 0     : Source of recurrent weights; {0,1,2}→{Σz,Σy,R} (method predictive)
    normalize = True        : Apply divisive normalization to responses?
    linearize = False       : Override sparsifying nonlinearity on coding features
    disable_homeostasis = False : Disable gain and bias homeostasis
    fail_early          = True  : Stop simulating if readout drift NRMSE > 0.75
    '''
    LOG('! %s %s %s %s s%d'%(method,features,readout,geometry,seed),
        bgcolor=bg(243))
    LOG('! LKTMPIΔ %d %d %d %d %d %d %d'%(L,K,T,M,P,I,Δ),
        bgcolor=bg(243))
    LOG('! τσνρrn '+' '.join([shortscientific(q) for q in (τ,σ,ν,ρ,r,n)]),
        bgcolor=bg(243))
    if disable_homeostasis and method in {'homeostat','hebbhomeo'}:
        raise ValueError('Can\'t disable homeostasis for method="%s"'%method+
                         'please set disable_homeostasis=False')
    
    # Convert scale params from relative to absolute
    σ*=L
    ν*=L
        
    # Initialize encoding features and readout weights
    Ξ,x,Y,W,R = initialization(features, 
                               readout, 
                               geometry, 
                               seed, 
                               L, K, T, M, P, 
                               σ, τ, ν, ρ, r, f, 
                               linearize)
    
    # Note: z, y, etc are shaped as Nreadouts × Nlocations.
    # Averaging over axis 0 gets you the population mean rate
    # Averaging over axis 1 gets you the mean rate per neuron

    # Initialize constants for predictive routine
    z  = φ_inverse(Y)
    μz = mean(z,1)
    if method == 'predictive':
        # Use activation correlation for feedback
        # (physiological?)
        Δz = z - μz[:,None]
        Σz = (Δz@Δz.T)/M
        # Use rate correlations for feedback (Hebbian?)
        μy = mean(Y,1)
        Δy = Y - μy[:,None]
        Σy = (Δy@Δy.T)/M
        Σy = Σy*np0.max(Σz)/np0.max(Σy)
        # Use the same recurrent matrix for feedback
        # (more directly comparable to the recurrent map?)
        R  = R[:-1,:]
        R  = R*np0.max(Σz)/np0.max(R)
        R  = [Σz,Σy,R][rec]
    
    # Subroutine for response normalization
    μy = mean(Y)
    nrmlz = jit((lambda y:μy*(eps+y)/(eps+mean(y,0))) if normalize else (lambda yf:yf))
    # Subroutines for different scenarios 
    @jit
    def replay_trial_homeostat(w,γ,β,x):   # Naïve homeostasis
        a  = w.T@x                         # Feed-forward activation
        z  = a*γ[:,None]+β[:,None]         # Adjusted forward activation
        yf = φ(z)                          # Forward rate response
        yn = nrmlz(yf)                     # Response normalization
        q  = [yf,yn,yn,z,yn][hom]          # Select source of homeostatic errors
        εμ = μt - mean(q,1)                # Mean-rate error
        εσ = 1  - std(q,1)/σt              # Variability error
        β += η*(ηβ*εμ-mean(a*εσ[:,None],1))# Update bias
        γ += η*εσ                          # Update gain
        return w,γ,β,yn                    # Return values passed to next iteration
    @jit
    def replay_trial_recurrent(w,γ,β,x):  # Hebbian homeostasis with linear-nonlienar map and error integration
        z  = w.T@x                         # Forward synaptic activation
        yf = φ(z)                          # Forward response
        yn = nrmlz(yf)                     # Response normalization
        yr = φ(R[:-1].T@yn+R[-1][:,None])  # Appy a recurrent map 
        yr = nrmlz(yr)                     # Normalize
        q  = [yf,yn,yr,z,yr][hom]          # Select source of homeostatic errors
        εμ = μt - mean(q,1)                # Mean-rate error
        εσ = (σt-std(q,1)+eps)/(σt+eps)    # Variability error
        if not disable_homeostasis:        # Leaky integrate homeostatic errors
            β = β*ι + εμ                   # Threshold homeostasis
            γ = γ*ι + εσ                   # Gain homeostasis 
        y  = [yf,yn,yr][heb]               # Select source of signals for Hebbian learning
        Δγ = ηγ*γ*(x[:-1]@y.T/L-w[:-1])    # Weight updates
        Δμ = ηβ*β                          # Bias updates
        Δw = concatenate([Δγ,array([Δμ])]) # All updates
        w += η*(Δw - w*ρ[:,None])          # Update weights; weight decay ρ; learning rate η.
        return w,γ,β,yr                    # Return values passed to next iteration
    @jit
    def replay_trial_predictive(w,γ,β,x): # Hebbian homeostasis with recurrent feedback and error integration
        z  = w.T@x                         # Forward synaptic activation
        yf = φ(z)                          # Forward response
        yn = nrmlz(yf)                     # Response normalization
        # Simulate recurrent feedback dynamics
        zh = fori(0, Iz,lambda i,zh:zh+ηz*(-zh+(1/zk)*R@(yn-φ(zh+μz[:,None]))),z)
        yr = φ(zh+μz[:,None])              # Result of recurrent dynamics
        yr = nrmlz(yr)                     # Normalize
        q  = [yf,yn,yr,z,yr][hom]          # Select source of homeostatic errors
        εμ = μt - mean(q,1)                # Mean-rate error
        εσ = (σt-std(q,1)+eps)/(σt+eps)    # Variability error
        if not disable_homeostasis:        # Leaky integrate homeostatic errors
            β = β*ι + εμ                   # Threshold homeostasis
            γ = γ*ι + εσ                   # Gain homeostasis 
        y  = [yf,yn,yr][heb]               # Select source of signals for Hebbian learning
        Δγ = ηγ*γ*(x[:-1]@y.T/L-w[:-1])    # Weight updates
        Δμ = ηβ*β                          # Bias updates
        Δw = concatenate([Δγ,array([Δμ])]) # All updates
        w += η*(Δw - w*ρ[:,None])          # Update weights; weight decay ρ; learning rate η.
        return w,γ,β,yr                    # Return values passed to next iteration
    @jit
    def replay_trial_hebbhomeo(w,γ,β,x):  # Hebbian homeostasis with error integration
        z  = w.T@x                         # Forward synaptic activation
        yf = φ(z)                          # Forward rate response
        yn = nrmlz(yf)                     # Response normalization
        q  = [yf,yn,yn,z,yn][hom]          # Select source of homeostatic errors
        εμ = μt - mean(q,1)                # Mean-rate error
        εσ = 1  - std(q,1)/σt              # Variability error
        if not disable_homeostasis:        # Leaky integrate homeostatic errors
            β = β*ι + εμ                   # Threshold homeostasis
            γ = γ*ι + εσ                   # Gain homeostasis 
        y  = [yf,yn,yn][heb]               # Select source of signals for Hebbian learning
        Δγ = ηγ*γ*(x[:-1]@y.T/L-w[:-1])    # Weight updates
        Δμ = ηβ*β                          # Bias updates
        Δw = concatenate([Δγ,array([Δμ])]) # All updates
        w += η*(Δw - w*ρ[:,None])          # Update weights; weight decay ρ; learning rate η.
        return w,γ,β,yn                    # Return values passed to next iteration
    
    methods = {
        'homeostat'  :replay_trial_homeostat  ,
        'hebbhomeo'  :replay_trial_hebbhomeo  ,
        'recurrent'  :replay_trial_recurrent  ,
        'predictive' :replay_trial_predictive 
    }
    replay_trial = methods[method]
    
    LOG('Initializing...')
    # Mask for weight decay (no decay on threshold)
    ρ = array([*(ones(K)*(ρ*2/K))]+[0])
    # Add noise to readout synapses, (but not threshold)
    n = n*np0.ones(shape(W))
    n[-1,:] = 0
    # Simulation initial conditions
    u,s = np0.copy(W),np.array(x)
    MP  = shape(W)[1]
    β,γ = zeros(MP),ones(MP)
    # The latest versions express γ as a perturbation from 1
    if method in ('hebbhomeo3','recurrent3','predictive3'):
        γ *= 0.0
        
    # Get initial outputs
    # Mock up a dummy σt/μy so we can re-use the simulation code
    # Ignore any updates to weights, γ, β returned at this point
    μt = mean(Y)
    σt = std (Y)   
    # Initial tuning curves
    Y0 = replay_trial(u,γ,β,x)[-1]
        
    LOG('Preparing homeostatic targets...')
    # Using the initial training signal as targets
    if hom in (0,1,2):
        μt = mean(Y)
        σt = std (Y)   
    # Targets on synaptic activation
    elif hom==3:
        μt = mean(z)
        σt = std (z)
    elif hom==4:
        # Targets based on baseline recurrent dynamics
        # Used if hom = 4
        # To get these we need to run the recurrent dynamics
        # Now, use the recurrent statistics as the homeostatic targets
        μt = mean(Y0)
        σt = std (Y0)   
    else:
        raise ValueError('The homeostatic target source should bee hom∈{0,1,2,3,4}')
    
    LOG('Simulating...')
    # Set seed
    np0.random.seed(seed)
    # Simulation loop
    Z, U = [],[]
    G, B = [],[]
    fail_condition = False
    for t in range(T):
        u = randn(K+1,MP)*sqrt(n)*std(u[:-1]) + sqrt(1-n)*u
        if t%Δ==Δ-1:
            if not fail_condition: 
                s = Ξ[t]
                for i in range(I):
                    u,γ,β,z = replay_trial(u,γ,β,s)
                    G.append(np0.copy(γ))
                    B.append(np0.copy(β))
                if fail_early and not saveall:
                    if summarize_stability(array([Y0]),Y)>0.75:
                        fail_condition = True
                        u *= nan
                        z *= nan
                        LOG('Readout failure condition encountered')
            Z.append(np0.copy(z))
            U.append(np0.copy(u))
    Z = array(Z)
    U = array(U)
    LOG('Simulating (done)')
    
    # Return
    if saveall: 
        return W,Ξ,Y0,Z,U,R,G,B
    else:
        LOG('Summarizing performance')
        result = summarize_stability(Z,Y)
        LOG('Returning')
        return result

