#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
# Prepare for parallel simulations; Define subroutines

- `options`: default values for each parameters
- `try_variations`: generate parameters for several experiments 
- `helper`: subroutine for calling and error handling with multiprocessing
- `_parmap`: parallel map function to run experiments using multiprocessing
'''

import config, master, standard_options
from config           import *
from master           import *
from standard_options import *

import numpy

methods = [
    ('homeostat'  , False ),
    ('hebbhomeo3' , False ),
    ('hebbhomeo3' , True  ),
    ('predictive3', True  ),
    ('recurrent3' , True  )]

method_names = [
    'Homeostasis',
    'Hebbian homeostasis',
    'Response normalization',
    'Recurrent feedback',
    'Linear-nonlinear map',
]

def try_variations(variations,nseeds=1,override={}):
    seeds = arange(nseeds)
    return [{'method'   :method, 
             'normalize':normalize, 
             **options,
             **override, 
             parameter:value, 
             'seed':seed}
        for method,normalize  in methods
        for parameter, values in variations.items()
        for value             in values
        for seed              in seeds]

def helper(params):
    global __log_prefix__
    try:
        i,job = params
        # There is a reason for this weird code but it would take too long to explain
        from importlib import reload  
        import config, master, standard_options
        config           = reload(config)
        master           = reload(master)
        standard_options = reload(standard_options)
        master.__log_prefix__ = __log_prefix__ = str(i)
        master.PRINT_LOGGING  = False
        initialize_cache()
        limit_cores(1)
        return i,master.run_experiment(**job)
    except Exception as e:
        LOG(str(i),'Failed')
        traceback.print_exc()
        return e
    except BaseException as e:
        LOG(str(i),'Interrupted')
        traceback.print_exc()
        raise e
    
pool = None
def start_pool(pool_limit=1):
    global pool
    '''
    As far as I can tell, repeatedly spawning a pool leaks... something,
    and over the course of the program one gets fewer and fewer cores.
    To circumvent this we will need to initialize a pool once and only
    once. 
    '''
    if pool is None:
        use_cores = multi.cpu_count()
        print('Parallel using %d cores'%use_cores)
        limit_cores(pool_limit)
        pool = multi.Pool(use_cores)
    return pool
        
def _parmap(f,jobs,debug=False):
    '''
    Wrapper to distribute work over multiple processes.
    
    `pool_limit` is the number of cores per worker. In most cases, 
    using additional parallelism within each worker will slow things 
    down, since cores will have to communicate with each-other. It is
    better to many workers on single cores, so this should be "1".
    
    `random order` shuffles the order that jobs are processed. If 
    a sequential block of jobs all require the same resource (or all
    recompute the same cache file), some of the parallel efficiency
    is lost. Sometimes evaluating the jobs in random order helps a 
    bit here.
    
    As far as I can tell, repeatedly spawning a pool leaks... something,
    and over the course of the program one gets fewer and fewer cores.
    To circumvent this we will need to initialize a pool once and only
    once. 
    '''
    print('Shuffling job sequence')
    #permutation = np0.arange(len(jobs))
    permutation = np0.random.permutation(len(jobs))
    inverse     = invert_permutation(permutation)
    jobs        = np0.array([*jobs])[permutation]
    njobs       = len(jobs)
    pool        = start_pool(pool_limit=1)
    print('Preparing to run %d jobs'%njobs)
    results      = {}
    lastprogress = -inf
    thisprogress = 0.0
    def pbar_update(i):
        nonlocal thisprogress, lastprogress
        thisprogress = ((i+.01)*100./njobs)
        if (thisprogress - lastprogress)>0:
            r = thisprogress*40/100
            k = int(r)
            bar = ' ▏▎▍▌▋▊▉'[int((r-k)*8)]
            bar = '█'*k+bar+' '*(39-k)
            bar = '\r[%s]%7.3f%% '%(bar[:40],thisprogress)
            print(bar,end='',flush=True)
            lastprogress = thisprogress
    print('Starting...')
    pbar_update(0)
    problems   = np0.array([*enumerate(jobs)])
    enumerator = map(f,problems) if debug else pool.imap(f,problems)
    for ir in enumerator:
        if isinstance(ir, Exception):
            notify('Hey something bad happened')
            print('EXC  :',ir)
            print('TRACE:',ir.__traceback__)
            traceback.print_exc()
            raise ir.with_traceback(ir.__traceback__)
        i,result = ir
        pbar_update(i)
        if isinstance(result,tuple) and len(result)==1:
            result=result[0]
        if isinstance(result, Exception):
            print('Error processing job',i)
            raise result
        results[i]=result
    pbar_update(njobs)
    results = [results[i] if i in results else None \
        for i,k in enumerate(jobs)]
    return numpy.array(results)[inverse]
        
def paramsweep_summary(parname,results,variations,nseeds,options,
                       vmin=0,
                       vmax=1,
                       cmap='roma_r'):
    figure(figsize=(6,2))
    iplot = 1
    ijob  = 0
    if vmin is None: vmin = min(0,np.nanmin(results))
    if vmax is None: vmax = np.nanmax(results)
    for method in methods:
        for parameter, values in variations.items():
            performance = []
            for value in values:
                replicas = []
                for seed in range(nseeds):
                    replicas += [results[ijob]]
                    ijob += 1
                replicas = array(replicas)
                performance += [mean(replicas,0)]
            performance = np.float64(performance)
            subplot(1,4,iplot)
            imshow(performance,cmap=cmap,vmin=vmin,vmax=vmax,
                  extent=(1,options['T']-options['Δ'],0,len(values)))
            if iplot <= 4: title(method)
            if iplot == 1:
                xlabel('Iteration')
                ylabel(parname)
                if isinstance(variations[parameter],ndarray) and values.dtype==dtype('float64'):
                    values = [*map(shortscientific,values)]
                yticks(linspace(.5,len(values)-.5,len(values)),[*map(str,values)])
            else:
                yticks([])
            simpleaxis()
            iplot += 1
    tight_layout()
    good_colorbar(vmin,vmax,cmap,'MSE')