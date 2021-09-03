#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Use multiprocess rather than multiprocessing to avoid pickling 
# errors. We MUST use spawn or forkserver to use Jax with
# multiprocess(ing). This MUST be called before other
# initialization takes place.
import multiprocess as multi
try:
    multi.set_start_method('spawn')
except RuntimeError:
    pass
    #print('>>> !! couldn\'t set context')
#print('>>> Set multiprocessing method for Jax compatability')

# Add local libraries to path. A stripped-down copy of neurotools, and 
# some code for extracting the Driscoll datasets, is included. 
# https://github.com/michaelerule/neurotools/blob/master/README.md
import os,sys
sys.path.insert(0,os.path.abspath("./"))
sys.path.insert(0,os.path.abspath("./driscolldatatools/"))
import neurotools
from   neurotools.nlab import *
memoize = neurotools.jobs.ndecorator.memoize

# Matplotlib configuration
import matplotlib as mpl

TEXTWIDTH        = 6.26894 # inch
TEXTHEIGHT       = 9.69481
PNAS_SMALL_WIDTH = 4.488189
PNAS_LARGE_WIDTH = 7.007874
matplotlib.rcParams['figure.figsize'] = (PNAS_SMALL_WIDTH, PNAS_SMALL_WIDTH/sqrt(2))

# Fonts
SMALL  = 6
MEDIUM = 7
LARGE  = 8
mpl.rcParams['font.size'           ]=SMALL  # controls default text sizes
mpl.rcParams['axes.titlesize'      ]=MEDIUM # fontsize of the axes title
mpl.rcParams['axes.labelsize'      ]=MEDIUM # fontsize of the x and y labels
mpl.rcParams['xtick.labelsize'     ]=SMALL  # fontsize of the tick labels
mpl.rcParams['ytick.labelsize'     ]=SMALL  # fontsize of the tick labels
mpl.rcParams['legend.fontsize'     ]=SMALL  # legend fontsize
mpl.rcParams['figure.titlesize'    ]=LARGE  # fontsize of the figure title
mpl.rcParams['lines.solid_capstyle']='round'
mpl.rcParams['savefig.dpi'         ]=140
mpl.rcParams['figure.dpi'          ]=140

lw = .5
for k in ['axes.linewidth',
          'xtick.major.width',
          'xtick.minor.width',
          'ytick.major.width',
          'ytick.minor.width']:
    mpl.rcParams[k] = lw
tl = 3
for k in ['xtick.major.size',
    'xtick.minor.size',
    'ytick.major.size',
    'ytick.minor.size']:
    mpl.rcParams[k] = tl
    
import numpy as np0
np0.seterr(all='ignore')
np0.set_printoptions(precision=3)

# Configuration for images we'll be plotting
mpl.rcParams['figure.dpi'] =140
plt.rcParams.update({'image.%s'%k:v for (k,v) in 
    [('aspect'       ,'auto'   ),
     ('origin'       ,'lower'  ),
     ('interpolation','nearest'),
     ('cmap'         ,'bone_r' )]})

# Customize colors further
from itertools import product
import copy as cpy
cm = cpy.copy(matplotlib.cm.get_cmap('bone_r'))
cm.set_bad(color='m')

# For jupyter notebooks: trigger browser to notify when done
def speak(text):
    from IPython.display import Javascript as js, clear_output
    # Escape single quotes
    text = text.replace("'", r"\'")
    display(js('''
    if(window.speechSynthesis) {{
        var synth = window.speechSynthesis;
        synth.speak(new window.SpeechSynthesisUtterance('{text}'));
    }}
    '''.format(text=text)))
    # Clear the JS so that the notebook doesn't speak again when reopened/refreshed
    #clear_output(False)
def notify(what='attention'):
    #os.system("echo -n '\a'")
    speak(what+'!')
    
# We're switching back to dumb process-level paralleism becase
# multiprocessing in python is broken. See 'dispatch.py'
jobfilename = 'jobs.pkl'
import neurotools.jobs.cache as cache
import pickle
def prepare_jobs(jobs):
    # First we need to check that the jobs havn't been run
    todo = []
    for j in jobs:
        fn,sig,path,filename,location = cache.locate_cached(cache.cache_root,f,run_experiment,**j)
        if not os.path.isfile(location):
            todo += [j]
    print('%d jobs remaining'%len(todo))
    with open(jobfilename,"wb") as jobfile:
        pickle.dump(todo,jobfile)
        
## Cache must be static-initialized BEFORE importing master
## (or master must be re-initiazed and its imports re-bound)
'''
The @memoize decorator can be used to save the result of function
calls to disk. These results are stored in separate files, indexed by
a base-64 encoding of the function arguments. Cached results use a
hash of the current source code to identify a function. 
Changing the source (even whitespace or comments) will therefore 
invalidate the cache. Stale cache files must be deleted manually (you
do NOT want to trust this code with the power to delete files!)
'''
# Caching: Cache folder name and where to put it
CACHE_NAME   = 'cache'
PY_CACHE_DIR = './'
import os,sys
def initialize_cache():
    from neurotools.tools import ensure_dir
    ensure_dir(PY_CACHE_DIR+os.sep+CACHE_NAME)
    # Configure disk caching
    from neurotools.jobs.initialize_system_cache import initialize_caches
    initialize_caches(PY_CACHE_DIR,CACHE_IDENTIFIER=CACHE_NAME,verbose=False)
    #print('>>> Cache initialized to %s'%PY_CACHE_DIR)
initialize_cache()

