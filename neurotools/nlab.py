#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

'''
Pulls in the pylab namespace, and many other functions. 

This has been modified from the main neurotools
https://github.com/michaelerule/neurotools
To remove all functionality not needed for the self-healing-codes scripts
'''

import os, sys, pickle, random, traceback, warnings
from   collections       import *
from   itertools         import *
from   os.path           import *
#from   multiprocessing   import Process, Pipe, cpu_count, Pool

import scipy
import scipy.optimize
try:
    from   sklearn.metrics   import roc_auc_score,roc_curve,auc
except Exception as e:
    print('could not find sklearn; ROC and AUC will be missing')
from scipy.stats       import wilcoxon
from scipy.signal      import *
from scipy.optimize    import leastsq
from scipy.interpolate import *
from scipy.io          import *
from scipy.signal      import butter,filtfilt,lfilter


#from neurotools.spikes.spikes            import *
#from neurotools.spikes.waveform          import *
from neurotools.tools                    import *
#from neurotools.text                     import *
from neurotools.functions                import *
from neurotools.linalg.operators         import *
from neurotools.graphics.color           import *
from neurotools.graphics.colormaps       import *
from neurotools.graphics.plot            import *
from neurotools.linalg.matrix            import *
#from neurotools.models.lif               import *
#from neurotools.models.izh               import *
#from neurotools.spatial.dct              import *
#from neurotools.spatial.array            import *
#from neurotools.spatial.distance         import *
#from neurotools.spatial.fftzeros         import *
#from neurotools.spatial.spatialPSD       import *
#from neurotools.spatial.phase            import *
#from neurotools.spatial.spiking          import *
#from neurotools.spatial.kernels          import *
import neurotools.stats
from neurotools.stats                    import *
from neurotools.stats.minimize           import *
#from neurotools.stats.density            import *
#from neurotools.stats.distributions      import *
#from neurotools.stats.mixtures           import *
#from neurotools.stats.entropy            import *
#from neurotools.stats.GLMFit             import *
#from neurotools.stats.glm                import *
#from neurotools.stats.hmm                import *
#from neurotools.stats.gmm                import *
#from neurotools.stats.mvg                import *
#from neurotools.stats.history_basis      import *
#from neurotools.stats.kent_reimann       import *
#from neurotools.stats.modefind           import *
#from neurotools.stats.regressions        import *
#from neurotools.stats.circular           import *
#from neurotools.signal.linenoise         import *
#from neurotools.signal.morlet_coherence  import *
#from neurotools.signal.morlet            import *
from neurotools.signal.savitskygolay     import *
from neurotools.signal                   import *
#from neurotools.signal.conv              import *
from neurotools.getfftw                  import *


# Depends on the spectrum package and will not import if this is missing
#try:
#    from neurotools.signal.multitaper        import *
#except ImportError:
#    print('Skipping the neurotools.signal.multitaper module')

# Depends on the spectrum package and will not import if this is missing
#try:
#    from neurotools.signal.ppc               import *
#except ImportError:
#    print('Skipping the neurotools.signal.ppc module')

# Depends on the nitime package and will not import if this is missing
#try:
#    from neurotools.signal.coherence         import *
#except ImportError:
#    print('Skipping the neurotools.signal.coherence module')

# Sometimes this fails?
#try:
#    from neurotools.jobs.parallel            import *
#    from neurotools.jobs.ndecorator          import *
#except ImportError:
#    print('Skipping the neurotools.jobs package')


try:
    import h5py
except:
    print('could not locate h5py; support for hdf5 files missing')
    h5py = None
try:
    from neurotools.hdfmat                   import *
except ModuleNotFoundError:
    print('ModuleNotFoundError: No module named \'h5py\'; please install this to use neurotools.hdfmat')

# suppress verbose warning messages
nowarn()

from numpy.core.multiarray import concatenate as cat

# Last but not least 
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.pyplot import *
from scipy             import *
from scipy.special     import *
from scipy.linalg      import *
from numpy.random      import *
from numpy             import *
from pylab             import *

# Mess with matplotlib defaults
rcParams['figure.dpi']=120
plt.rcParams['image.cmap'] = 'parula'
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=[BLACK,RUST,TURQUOISE,OCHRE,AZURE])

