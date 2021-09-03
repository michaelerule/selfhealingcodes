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
from   neurotools.hdfmat import printmatHDF5, hdf2dict, getHDF
from   sklearn.decomposition import FactorAnalysis
from   scipy.linalg import solve,lstsq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   sklearn  import manifold
from   ppc_util import *
import ppc_trial
from neurotools.tools import find

# !!!! IMPORTANT: CHANGE THIS TO THE LOCATION OF THE DRICOLL PPC DATA ON 
# your system
DATAPATH = '/home/mer49/Workspace2/PPC_data/'
DATAPATH = os.path.expanduser(DATAPATH)
ensure_dir(DATAPATH)
path = DATAPATH

#memoize = functools.lru_cache(maxsize=None)
memoize = neurotools.jobs.ndecorator.memoize

def release_files(clear_cache=False):
    '''
    Ad-hoc way to deal with poor memory management. Not great. 
    
    Other Parameters
    ----------------
    clear_cache : bool, default `False`
        Whether to clear the RAM caches as well as the open files.
    '''
    import gc
    for obj in gc.get_objects():   # Browse through ALL objects
        try:
            if isinstance(obj, h5py.File):   # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass # Was already closed
        except:
            pass
    if clear_cache:
        clear_memoized()
    gc.collect()

def get_file(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    return 'm%02d_s%02d.mat'%(animal,session)

def get_data(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    return h5py.File(path+get_file(animal,session), 'r')

#@memoize
def get_FS(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    hdfdata = get_data(animal,session)
    FS = getHDF(hdfdata,'session_obj/timeSeries/frameRate') # Hz
    #hdfdata.close()
    #print('Sample rate is %f Hz'%FS)
    hdfdata.close()
    return FS

#@memoize
def get_duration(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    return get_nonneural(animal,session).shape[0]

#@memoize
def get_nonneural(animal,session):
    '''
    Get the non-neural timeseries (kinematics, mostly)
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    hdfdata = get_data(animal,session)
    result = getHDF(hdfdata,'session_obj/timeSeries/virmen/data')
    hdfdata.close()
    return result

#@memoize
def get_stability_index(animal,session):
    '''
    Stability index computed/provided by Laura

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    hdfdata = get_data(animal,session)
    labels = getHDF(hdfdata,'session_obj/confidenceLabel')
    labels[~isfinite(labels)] = 0
    labels = np.int32(labels)
    hdfdata.close()
    return labels

#@memoize
def get_confidence(animal,session):
    '''
    This uses the recorded confidence lable in the dataset
    This *should* agree with which neurons actually have
    data within a given session. (check this)

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    raise NotImplementedError("""The `get_confidence` function has been 
depricated, since it incorrectly used stability labels as a proxy for unit 
quality. Use the `good_units` and `good_units_index` functions instead.""") 

    hdfdata = get_data(animal,session)
    labels = getHDF(hdfdata,'session_obj/confidenceLabel')
    #hdfdata.close()
    labels[~isfinite(labels)] = 0
    hdfdata.close()
    return np.int32(labels)

@memoize
def get_recording_stability_index(animal,session):
    '''
    This uses the recorded confidence lable in the dataset
    This *should* agree with which neurons actually have
    data within a given session. (check this)

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    hdfdata = get_data(animal,session)
    labels = getHDF(hdfdata,'session_obj/confidenceLabel')
    #hdfdata.close()
    labels[~isfinite(labels)] = 0
    hdfdata.close()
    return np.int32(labels)

@memoize
def get_good_units(animal,session):
    '''
    This uses the recorded confidence lable in the dataset
    This *should* agree with which neurons actually have
    data within a given session. (check this)
    
    Note! This is not a confidence label, but rather
    a 'stability' label. 
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    raise NotImplementedError("""The `get_good_units` function has been 
depricated, since it incorrectly used stability labels as a proxy for unit 
quality. Use the `good_units` and `good_units_index` functions instead.""") 
    
    hdfdata = get_data(animal,session)
    labels  = getHDF(hdfdata,'session_obj/confidenceLabel')
    #hdfdata.close()
    labels  = np.int32(labels)
    hdfdata.close()
    return np.where(np.isfinite(labels) & (labels>1))[0]

@memoize
def get_number_confident_units(animal,session):
    '''
    This uses the recorded confidence lable in the dataset
    This *should* agree with which neurons actually have
    data within a given session. (check this)

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    raise NotImplementedError("""The `get_number_confident_units` function 
has been depricated, since it incorrectly used stability labels as a proxy 
for unit quality. Use the `good_units` and `good_units_index` functions 
instead.""") 
    return sum(get_confidence(animal,session)>1)

#@memoize
def get_timestamp(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[0]

#@memoize
def get_x(animal,session):
    '''
    x location in virtual maze
    units are meters

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[1]

#@memoize
def get_kinematics(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    kinematics:
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[1:6]

#@memoize
def get_y(animal,session):
    '''
    y location in virtual maze
    units are meters

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[2]

#@memoize
def get_theta(animal,session):
    '''
    Get head-direction view angle. Units are in degrees. 
    
    The Virmen system will wrap the angle if the animal is doing 
    corkscrews. This routine adds circular wrapping.

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return (nonneural.T[3]+180)%360-180

#@memoize
def get_theta_radians(animal,session):
    '''
    Head direction in radians
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    
    Returns
    -------
    '''
    return (get_theta(animal,session)*np.pi/180+3*pi)%(2*pi)-pi

#@memoize
def get_dx(animal,session):
    '''

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[4]

#@memoize
def get_dy(animal,session):
    '''
    y velocity. Units are meters per second

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[5]

#@memoize
def get_speed(animal,session):
    '''
    Speed computed from dx and dy.
    Units are meters per second

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    dx = get_dx(animal,session)
    dy = get_dy(animal,session)
    speed = (dx**2+dy**2)**0.5
    return speed

#@memoize
def get_type(animal,session):
    '''
    trial type (black right = 2, white left = 3)

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[6]

#@memoize
def get_reward(animal,session):
    '''
    Timeseries indicating reward
    0 = no reward
    1 = reward timepoint

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[7]

#@memoize
def get_nontrial(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[8]>0

#@memoize
def get_intrial(animal,session):
    '''
    Gets whether each time-point is in a trial or not. 
    This is based on the "in-trial" flag provided in the recording data.
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    return ~get_nontrial(animal,session)

@memoize
def get_cue(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    nonneural = get_nonneural(animal,session)
    return nonneural.T[9]

#@memoize
def get_subject_ids():
    '''
    Report a list of subjects that have data

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    #global path
    fs = os.listdir(path)
    alli = []
    for m in os.listdir(path):
        m = m.lower()
        if not m.endswith('mat'): continue
        m = m.split('.mat')[0]
        if not '_' in m: continue
        m = m.split('_')[0]
        try:
            alli.append(int(m[1:]))
        except:
            pass
    alli = sorted(list(np.unique(alli)))
    return alli

#@memoize
def get_session_ids(animal):
    '''
    Report a list of available sesson numbers for a given subject.
    
    Some invalid sessions have been hard-coded here to be removed.

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    prefix = 'm%02d_'%animal
    fs = os.listdir(path)
    ss = []
    for m in [f for f in fs if f.startswith(prefix)]:
        m = m.split('.mat')[0].split(prefix)[1]
        if m.startswith('s') and len(m)==3:
            s = int(m[1:])
            if not 'm%02d_s%02d.mat'%(animal,s) in os.listdir(path):
                continue
            ss.append(s)
    ss = set(ss)
    ss = sorted(list(ss))
    return ss

def get_days(a):
    days = []
    for s in get_session_ids(a):
        hdffile = get_data(a, s)
        dd = getHDF(hdffile,'session_obj/deltaDays')
        sn = getHDF(hdffile,'session_obj/sessionNumber')
        assert(int(sn)==s)
        days.append(int(dd[s-1]))
        hdffile.close()
    return days

@memoize
def good_units(animal,session):
    '''
    Rather than use the confidence label stored with the dataset,
    this routine actually checks which units have available data.
    This is pretty slow, so results are cached on disk.

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    y = get_dFF(animal,session)
    good = np.all(isfinite(y),axis=0)
    return good
        
def good_units_index(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    '''
    return np.where(good_units(animal,session))[0]

@memoize
def get_unit_availability_map_by_days(animal):
    sessions = get_session_ids(animal)
    unit_availability = None
    for i,s in enumerate(sessions):
        hdfdata = get_data(animal,s)
        Δd    = getHDF(hdfdata,'session_obj/deltaDays')
        day   = int(Δd[s-1])
        units = good_units(animal,s)
        if unit_availability is None:
            unit_availability = np.zeros((int(np.max(Δd)),len(units)))
        unit_availability[day-1,:] = units
        hdfdata.close()
    return unit_availability   

def get_session_days(animal):
    sessions = get_session_ids(animal)
    days=[]
    for i,s in enumerate(sessions):
        hdfdata = get_data(animal,s)
        Δd    = getHDF(hdfdata,'session_obj/deltaDays')
        day   = int(Δd[s-1])
        hdfdata.close()
        days.append(day)
    return days

def get_good_units_shared_across_sessions(animal):
    '''
    This is cached on disk

    Parameters
    ----------
    animal : int
        Which subject ID to use
        
    Returns
    -------
    '''
    cache = CACHEDIR+os.sep+'cache/good_units_shared_across_sessions/'
    ensure_dir(cache)
    filename = cache+'good_units_shared_across_sessions_m%02d.mat'%animal
    try:
        data = scipy.io.loadmat(filename)
        return np.squeeze(data['good_units'])
    except:
        print('Building cache for',filename)
        good = {}
        for s in get_session_ids(animal):
            good[s] = set(good_units_index(animal,s))
        shared = array(sorted(list(set.intersection(*good.values()))))
        savemat(filename,{'good_units':shared})
        return shared


@memoize
def get_number_of_units(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    N:int
        Number of available neurons in this session
    '''
    return len(good_units(animal,session))

@memoize
def get_units_in_common(animal,sessions):
    '''
    Get neurons in common to a set of sessions

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : list of ints
        Which session IDs to use


    Returns
    -------
    units: 
        List of unit numbers in common
    uidxs:
        Index into list of good units for each session
        
    '''
    unitmap  = {s:set(good_units_index(animal,s))      for s in sessions}
    units    = sorted(list(set.intersection(*unitmap.values())))
    uidxs    = [find([u in units for u in unitmap[s]]) for s in sessions]
    return units,uidxs

@memoize
def get_dFF(animal,session,units=None):
    '''
    Extract the 'dFF' variable from a PPC experiment
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    units : integer array-like
        List of units for which to return the dFF traces. Defaults to 
        returning all units (including ones with missing data). 
    
    Returns
    -------
    dFF : Ntimes x Nneurons array
        Raw dF/F calcium signals. First axis is time-points and second
        axis is neurons.
    '''
    hdfdata = get_data(animal,session)
    neural  = getHDF(hdfdata,'session_obj/timeSeries/calcium/data')
    dFF,decon_Ca = neural
    hdfdata.close()
    return dFF if units is None else dFF[:,np.int32(units)]

@memoize
def get_smoothed_dFF(animal,session,unit,Fmin,Fmax,zeromean=True):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    unit : int
        Which (0-indexed) unit ID to use
    Fmin : float
        Low-frequency cutoff for filtering
    Fmax : float
        High-frequency cutoff for filtering

    Returns
    -------
    dFF : Ntimes x Nneurons array
        Filtered dF/F calcium signals. 
        First axis is time-points and second axis is neurons.
    '''
    z  = get_dFF(animal,session,unit).ravel()
    FS = get_FS(animal,session)
    z  = bandpass_filter(z,fa=Fmin,fb=Fmax,Fs=FS)
    if zeromean:
        z = z-np.mean(z)
    # There is a memory management bug somewhere!
    release_files(clear_cache=True)
    return z

#@memoize
def get_good_dFF(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Returns
    -------
    dFF : Ntimes x N good neurons array
        Raw dF/F calcium signals for all valid neurons. 
        First axis is time-points and second axis is neurons.
    '''
    return get_dFF(animal,session)[:,good_units(animal,session)]

@memoize
def get_logF(animal,session,units):
    '''
    Get normalized log-fluorescence signals for the given subject, 
    session, and unit. 
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    units : list of ints
        Which (0-indexed) unit IDs to use
        
    Returns
    -------
    logF : length Ntimes array
        Normalized log-fluorescence signals. First axis is time-points
        and second axis is neurons.
    '''
    y = get_dFF(animal,session,units)
    z = np.log1p(np.maximum(-1.0+1e-6,y))
    assert(np.all(np.isfinite(z)))
    return z

@memoize
def get_smoothed_logF(animal,session,unit,Fmin,Fmax,zeromean=True):
    '''
    dF/F is (F-<F>)/<F>
    log(1+dFF) = log(F)-log(<F>)

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    unit : int
        Which (0-indexed) unit ID to use
    Fmin : float
        Low-frequency cutoff for filtering
    Fmax : float
        High-frequency cutoff for filtering
    '''
    z  = get_logF(animal,session,unit)
    #assert(np.all(np.isfinite(z)))
    FS = get_FS(animal,session)
    z = bandpass_filter(z,fa=Fmin,fb=Fmax,Fs=FS)
    #assert(np.all(np.isfinite(z)))
    if zeromean:
        z = z-np.mean(z)
    #assert(np.all(np.isfinite(z)))
    return z

@memoize
def get_all_smoothed_logF(animal,session,Fmin,Fmax,**kwargs):
    '''
    This will 

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    Fmin : float
        Low-frequency cutoff for filtering
    Fmax : float
        High-frequency cutoff for filtering
    '''
    # TODO: make more efficientg
    y = get_dFF(animal,session).copy()
    ok = good_units(animal,session)
    for ch in np.where(ok)[0]:
        y[:,ch] = get_smoothed_logF(animal,session,ch,Fmin,Fmax,**kwargs)
    return y

#@memoize
def get_deconvolved(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    '''
    hdfdata = get_data(animal,session)
    neural  = getHDF(hdfdata,'session_obj/timeSeries/calcium/data')
    dFF,decon_Ca = neural
    hdfdata.close()
    return decon_Ca

#@memoize
def get_good_deconvolved(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    '''
    spikes = get_deconvolved(animal,session)
    return spikes[:,good_units(animal,session)]

@memoize
def get_types(animal,session):
    '''
    names  = ['Other','Left, correct','Right, correct','Tee, left cue','Tee, right cue']
    colors = [BLACK,RUST,TURQUOISE,OCHRE,MOSS]
    colors = np.array(list(map(mpl.colors.to_rgb,colors)))

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    '''
    nonneural = get_nonneural(animal,session)
    x  = nonneural[:,1]
    y  = nonneural[:,2]
    theta  = nonneural[:,3]
    dx = nonneural[:,4]
    dy = nonneural[:,5]

    left_arm  = (y>4.5)&(x<0)
    right_arm = (y>4.5)&(x>0)
    tee       = (y<=4.5)&(abs(x)<0.01)
    other     = ~(left_arm|right_arm|tee)
    types     = int32(nonneural[:,6]-2)
    left0  = (types==0)&left_arm
    left1  = (types==1)&left_arm
    right0 = (types==0)&right_arm
    right1 = (types==1)&right_arm
    tee0   = (types==0)&tee
    tee1   = (types==1)&tee

    labels = int32(np.zeros(types.shape))
    labels[left0]  = 1
    labels[right1] = 2
    labels[tee0]   = 3
    labels[tee1]   = 4
    return labels

@memoize
def get_zscored_calcium_in_trials(animal,session,BIN=1,DECIMATE=1,use=None):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    use   = to_indices(good_units(animal,session) if use is None else use)
    FS   = get_FS(animal,session)
    Fmax = FS/BIN/2
    Fmin = None
    z = array([get_smoothed_dFF(animal,session,ch,Fmin,Fmax) for ch in use])
    x = zscore(z[:,get_intrial(animal,session)].T).T[:,::DECIMATE]
    return x

@memoize
def get_deconvolved_spikes_in_trials(animal,session,BIN=1,DECIMATE=1,use=None):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    use = to_indices(good_units(animal,session) if use is None else use)
    x  = get_deconvolved(animal,session)[:,use]
    z  = array([box_filter(xi,BIN) for xi in x.T]).T
    return z[get_intrial(animal,session)][::DECIMATE].T

@memoize
def get_zscored_kinematics_in_trials(animal,session,DECIMATE=1):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    b = get_kinematics(animal,session)
    b = zscore(b[:,get_intrial(animal,session)].T).T[:,::DECIMATE]
    return b

@memoize
def get_zscored_calcium(animal,session,BIN=1,use=None,Fmin=None,Fmax=None):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    use  = to_indices(good_units(animal,session) if use is None else use)
    FS   = get_FS(animal,session)
    if Fmax is None:
        Fmax = FS/BIN/2
    Fmin = None
    z = array([get_smoothed_dFF(animal,session,ch,Fmin,Fmax) for ch in use])
    return zscore(z.T).T

@memoize
def get_deconvolved_spikes(animal,session,BIN=1,use=None):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    use = to_indices(good_units(animal,session) if use is None else use)
    x  = get_deconvolved(animal,session)[:,use]
    z  = array([box_filter(xi,BIN) for xi in x.T]).T
    return z.T

@memoize
def get_zscored_kinematics(animal,session):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    '''
    return zscore(get_kinematics(animal,session).T).T

#@memoize
def get_fa(animal,session,
           DECIMATE=10,
           LAGS=1,
           LAGSHIFT=1,
           NFACTORS=None,
           use_spikes=False,
           use_behavior=False,
           remove_nontrial=True,
           **kwargs):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    if use_spikes:
        x = get_deconvolved_spikes(animal,session,**kwargs)
    else:
        x = get_zscored_calcium(animal,session,**kwargs)
    if use_behavior:
        b = get_zscored_kinematics(animal,session)
        x = np.concatenate([x,b])
    X = x.T

    '''
    # Blank out bad areas?
    intrial = get_intrial(animal,session)
    labels = ppc_trial.get_contextual_types(animal,session)[intrial][::DECIMATE]
    bad = labels<0
    fa.fit(X[~bad,:])
    '''

    if LAGS>1:
        # Expand X using time lags
        K = X.shape[1]
        T = X.shape[0]
        bigX = np.zeros((T,K*LAGS))
        for l in range(LAGS):
            bigX[l*LAGSHIFT:,l*K:(l+1)*K] = X[:T-l*LAGSHIFT,:]
            #bigX[l:,l*K:(l+1)*K] = X[:T-l,:]
        X = bigX

    insession = get_intrial(animal,session)
    if remove_nontrial:
        X = X[insession]
    X = X[::DECIMATE]

    fa    = FactorAnalysis(n_components=NFACTORS)
    Y     = fa.fit_transform(X)
    Sigma = fa.noise_variance_
    F     = fa.components_
    lmbda = diag(F.dot(F.T))
    order = argsort(lmbda)[::-1]
    lmbda = lmbda[order]
    F     = F[order,:]
    return X,Y,Sigma,F,lmbda,fa

#@memoize
def get_le(animal,session,
           use=None,
           NDIM=8,
           NNEIGHBOR=45,
           **kwargs):
    '''
    Compute Laplacian eigenmaps of neural activity for PPC datasets

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    # data loading
    if use is None:
        use = tuple(np.where(good_units(animal,session))[0])
    # Factor analysis whitening
    X,Y,Sigma,F,lmbda,fa = get_fa(animal,session,use=use,**kwargs)
    # Laplacian eigenmap
    se = manifold.SpectralEmbedding(n_components=NDIM,n_neighbors=NNEIGHBOR)
    '''
    # Blank out bad areas?
    intrial = get_intrial(animal,session)
    labels = ppc_trial.get_contextual_types(animal,session)[intrial][::DECIMATE]
    good   = labels>=0
    Z = se.fit_transform(Y[good,:])
    Z2 = np.zeros((Y.shape[0],NDIM))*np.nan
    Z2[good,:] = Z;
    Z = Z2;
    '''
    Z = se.fit_transform(Y)

    return X,Y,Z,Sigma,F,lmbda,fa

import neurotools.signal
from neurotools.nlab import autocorrelation,sexp,zscore,minimize_retry

def estimate_autocorrelation_decay(x,cutoff=0.2,doplot=True,lags=None):
    '''
    Estimate autocorrelation decay time using exponential model

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    x     = zscore(x)
    ac    = autocorrelation(x,lags=lags)
    decay = ac[len(ac)//2+2:]
    peak  = np.max(decay)
    t     = np.arange(len(decay))
    def objective(params):
        x0,tau = params
        predict = sexp(-t/tau)*x0
        return np.mean((decay-predict)**2)
    x0,tau = minimize_retry(objective,[peak,5])
    predict = sexp(-t/tau)*x0
    interval = int(-log(cutoff)*tau)+2
    if doplot:
        print('Predicted decay time is %0.2f samples'%tau)
        print('Suggest decimation by',interval)
        plot(decay,label='Autocorrelation')
        plot(predict,label='Model $\\tau=%0.1f$'%tau)
        #axvline(tau,color=BLACK,label='Tau')
        axvline(interval,color=RUST,label='Cutoff')
        legend(); simpleaxis()
    return tau,interval

def get_previous_trial_type(animal,session):
    '''
     0 = left
     1 = right
    -1 = unknown

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    '''
    labels    = ppc_trial.get_contextual_types(animal,session)
    left_cue  = np.array([ 0, 1, 2, 3, 4, 10,11,12,13,14])
    right_cue = np.array([ 5, 6, 7, 8, 9, 15,16,17,18,19])
    newlabels = np.int32(np.zeros(labels.shape)-1)
    for c in left_cue:
        newlabels[labels==c]=0
    for c in right_cue:
        newlabels[labels==c]=1
    return newlabels

def get_left_right_cue_labels(animal,session):
    '''
     0 = left
     1 = right
    -1 = unknown

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
    '''
    labels    = ppc_trial.get_contextual_types(animal,session)
    left_cue  = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    right_cue = np.array([10,11,12,13,14,15,16,17,18,19])
    newlabels = np.int32(np.zeros(labels.shape)-1)
    for c in left_cue:
        newlabels[labels==c]=0
    for c in right_cue:
        newlabels[labels==c]=1
    labels = newlabels
    return labels

def get_spatial_bins(animal,session,
    NXBINS = 7, # should be odd
    NYBINS = 10,
    doplot=False):
    '''
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    x = get_x(animal,session)
    y = get_y(animal,session)
    xs = ppc_trial.Trial.XSCALE*1.04
    ys = ppc_trial.Trial.YSCALE
    ys += ys/(NYBINS)
    NYBINS += 1
    x_bins = np.linspace(-xs,xs,NXBINS+1)
    y_bins = np.linspace(0,ys,NYBINS+1)
    if doplot:
        scatter(x,y,color=OCHRE,s=1)
        for px in x_bins:
            plot([px,px],[0,ys],color=BLACK,lw=1)
        for py in y_bins:
            plot([-xs,xs],[py,py],color=BLACK,lw=1)
        noaxis()
        force_aspect(0.5)
    x_centers = (x_bins[1:]+x_bins[:-1])/2
    y_centers = (y_bins[1:]+y_bins[:-1])/2
    p_names   = ['left','right']
    dx = np.digitize(x,x_bins)
    dy = np.digitize(y,y_bins)
    previous = get_previous_trial_type(animal,session)
    bin_tuples = np.array(list(zip(previous,dx,dy)))
    bin_set = list(map(tuple,np.unique(bin_tuples,axis=0)))
    # Get only times on the track with defined past-trial context
    middle  = (NXBINS+1)//2
    top     = NYBINS
    ok_bins = [(pr,bx,by) for (pr,bx,by) in bin_set if pr>=0 and (bx==middle or by==top)]
    # Construct mapping of bins
    bin_info = {-1:'unknown'}
    bin_ids  = np.zeros(x.shape,dtype=np.int32)-1
    for i,bin in enumerate(ok_bins):
        (pr,bx,by) = bin
        (npr,nbx,nby) = p_names[pr],x_centers[bx-1],y_centers[by-1]
        label = 'previous=%s x=%0.2f y=%0.2f'\
                %(p_names[pr],x_centers[bx-1],y_centers[by-1])
        bin_info[i] = ((pr,bx,by),(npr,nbx,nby),label)
        bin_ids[np.all(bin_tuples==bin,axis=1)]=i
        #print('bin %d is %s'%(i,label))
    return bin_ids,bin_info

def get_context_bins(animal,session,
    NXBINS = 5, # should be odd
    NYBINS = 10,
    doplot=False):
    '''
    Bin experimental timepoints based on (x,y) maze location,
    the left/right cue information,
    and whether the animal went left or right in the previous maze

    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    '''
    # Get (x,t) locations
    x  = get_x(animal,session)
    y  = get_y(animal,session)
    y -= np.min(y)
    # Define range for bins
    xs = np.max(np.abs(x))*(1+1e-2)
    ys = np.max(y)*(1+1e-2)
    ys += ys/(NYBINS-1)/2
    # Define bins
    x_bins = np.linspace(-xs,xs,NXBINS+1)
    y_bins = np.linspace(0,ys,NYBINS+1)
    # Show gridded locations
    if doplot:
        scatter(y,x,color=OCHRE,s=1)
        for px in x_bins:
            plot([0,ys],[px,px],color=BLACK,lw=1)
        for py in y_bins:
            plot([py,py],[-xs,xs],color=BLACK,lw=1)
        noaxis()
        force_aspect(2)
    # Bin timepoints based on spatial location
    dx = np.digitize(x,x_bins)
    dy = np.digitize(y,y_bins)
    # Get previous and cue labels
    # 0=left, 1=right, -1=unknown
    prev = get_previous_trial_type(animal,session)
    cue  = get_left_right_cue_labels(animal,session)
    # Form identifying tuples based on bins and context information
    # and get list of unique contexts
    bin_tuples = np.array(list(zip(prev,cue,dx,dy)))
    bin_set    = list(map(tuple,np.unique(bin_tuples,axis=0)))
    # Get only times on the track with defined past-trial context
    mid = (NXBINS+1)//2
    top = NYBINS
    ok_bins = []
    for (bp,bc,bx,by) in bin_set:
        is_good =\
            bp>=0 and bc>=0 and (             # previous direction and cue known
            bx==mid or                        # either in the T part of the maze
           (by==top and bx<=mid and bc==0) or # or, left branch on left cue
           (by==top and bx>=mid and bc==1))   # or, right branch on right cue
        if is_good:
            ok_bins.append((bp,bc,bx,by))
    # Construct mapping of bins
    bin_info = {-1:'unknown'}
    bin_ids  = np.zeros(x.shape,dtype=np.int32)-1
    # Get bin centers
    x_centers = (x_bins[1:]+x_bins[:-1])/2
    y_centers = (y_bins[1:]+y_bins[:-1])/2
    # Define previous trial and cue labels
    p_names   = ['L','R']
    c_names   = ['L','R']
    for i,bin in enumerate(ok_bins):
        (bp,bc,bx,by) = bin
        (pn,cn,xn,yn) = p_names[bp],c_names[bc],x_centers[bx-1],y_centers[by-1]
        label = 'prev=%s cue=%s x=%0.2f y=%0.2f'%(pn,cn,xn,yn)
        bin_info[i] = ((bp,bc,bx,by),(pn,cn,xn,yn),label)
        bin_ids[np.all(bin_tuples==bin,axis=1)]=i
        #print('bin %d is %s'%(i,label))
    return bin_ids,bin_info



def get_consecutive_recordings(animal,
                               MINNEURONS=200,
                               MINDURDAYS=4,
                               verbose=True):
    '''
    Identify consecutive spans of days for the given `animal`,
    with at least `MINNEURONS` in common and lasting at least 
    `MINDURDAYS` long.
    
    Parameters
    ----------
    animal: int
        which subject to use
        
    Other Parameters
    ----------------
    MINNEURONS: int
        Minimum number of neurons in common.
    MINDURDAYS: int
        Minimum duration in days.
    verbose: bool
        Print debugging info.
        
    Returns
    -------
    valid_spans : 
        dictionary mapping tuples of starting and ending session 
        numbers to the list of neurons that they share in common.
        
    '''
    # Get session information
    sessions = np.array(get_session_ids(animal))
    if verbose:
        print('Testing subject %d'%animal)
        print('  Available sessions',' '.join(map(str,sessions)))
    # Find all spans of sessions over consecutive days
    d = np.diff(sessions)
    L = len(sessions)
    spans = [(i,j) for i in range(L-1)
             for j in range(i+1,L)
             if all(d[i:j]==1)]
    spans = sessions[np.int32(spans)]
    # Get good units for all included sessions
    unitmap = {s:set(good_units_index(animal,s)) for s in sessions}
    # Get overlapping unit sets for all spans of sessions
    common = {(a,b):set.intersection(*[unitmap[s]
                                       for s in range(a,b+1)])
              for (a,b) in spans}
    # Get durations of valid spans of days
    durations = np.diff(spans).ravel()+1
    if verbose:
        print('  There are %2d spans at least %d days long; of these…'\
              %(sum(durations>=MINDURDAYS),MINDURDAYS))
    spans = spans[durations>=MINDURDAYS,:]
    # Get spans of days with enough units
    valid_spans = np.array([span for span in spans
                         if len(common[tuple(span)])>=MINNEURONS])
    # Hack to remove subsets
    daysets = [set(range(a,b+1)) for (a,b) in valid_spans]
    useme = []
    N = len(valid_spans)
    for i in range(N):
        if not any([daysets[i].issubset(daysets[j]) 
                    for j in range(N) if i!=j]):
            useme.append(valid_spans[i])
    valid_spans = useme
    if verbose:
        print('  There are %d spans with at least %d neurons:'\
              %(len(valid_spans),MINNEURONS))
        [print('  \t%02d-%02d (%d days, %3d neurons)'\
               %(a,b,b-a+1,len(common[a,b]))) 
         for a,b in valid_spans]
    return {k:sorted(list(common[k])) for k in map(tuple,valid_spans)}
