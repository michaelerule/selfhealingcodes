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
from   ppc_data_loader import *
import ppc_data_loader

matplotlib.rcParams['figure.dpi']=200

class Trial():
    # Location labels
    NONE_LABEL  = 0
    TEE_LABEL   = 1
    JOIN_LABEL  = 2
    LEFT_LABEL  = 3
    RIGHT_LABEL = 4

    # Cue type labels used in PPC data archives
    CUE_LEFT_DATASET  = 2
    CUE_RIGHT_DATASET = 3
    
    # Cue type labels we use
    CUE_LEFT          = 0
    CUE_RIGHT         = 1
    CUE_UNKNOWN       = -1

    get_time_for = [TEE_LABEL,JOIN_LABEL,LEFT_LABEL,RIGHT_LABEL]

    LABELNAMES = dict(zip(get_time_for,'Tee Join Left Right'.split()))

    CUENAMES = {CUE_LEFT:'Left',
                CUE_RIGHT:'Right',
                CUE_UNKNOWN:'?!'}

    CONTEXT_LABELS = { 
         0: '?, cue:←, prev:←',
         1: '↑, cue:←, prev:←',
         2: '⇄, cue:←, prev:←',
         3: '←, cue:←, prev:←',
         4: '→, cue:←, prev:←',
         5: '?, cue:←, prev:→',
         6: '↑, cue:←, prev:→',
         7: '⇄, cue:←, prev:→',
         8: '←, cue:←, prev:→',
         9: '→, cue:←, prev:→',
        10: '?, cue:→, prev:←',
        11: '↑, cue:→, prev:←',
        12: '⇄, cue:→, prev:←',
        13: '←, cue:→, prev:←',
        14: '→, cue:→, prev:←',
        15: '?, cue:→, prev:→',
        16: '↑, cue:→, prev:→',
        17: '⇄, cue:→, prev:→',
        18: '←, cue:→, prev:→',
        19: '→, cue:→, prev:→'}

    XSCALE = 0.15
    YSCALE = 4.5

    def __init__(self,*args,**kwargs):
        for k,v in kwargs:
            settr(self,k,v)
    def __repr__(self):
        return '\n'+'\n\t'.join(self.toStrings())
    def toStrings(self,recurse=True):
        cuename = Trial.CUENAMES[self.cue]
        s = ['Trial number %s:'%self.number,
            'sample start  : %s'%self.istart,
            'sample stop   : %s'%self.istop,
            'No. frames    : %s'%self.nsample,
            'Cue           : %s'%cuename,
            'correct?      : %s'%self.correct,
            'sample reward : %s'%self.rindex,
            ]
        for mark,name in Trial.LABELNAMES.items():
            if not hasattr(self,name): continue
            z = name+' index'
            z += ' '*(14-len(z))
            z += ': %s'%getattr(self,name)
            s.append(z)
        s.append('Reward index  : %s'%self.roffset)
        # Previous trial information
        if self.previous is None:
           s.append('(No previous trials)') 
        else:
            s.append('Previous trial: %d'%(self.previous.number))
            #if recurse:
            #    s.extend(['   %s'%s for s in self.previous.toStrings(False)])
            #else:
            #    s.append('...')
        return s


def location_labels(animal,session,
                    xscale=Trial.XSCALE,
                    xlimit=0.05,
                    yscale=Trial.YSCALE,
                    ylimit=0.05,
                    joinlimit=0.05,
                    doplot=False):
    '''
    Get coarse labels of location in T maze
    
    0 : nowhere
    1 : Tee
    2 : join
    3 : left arm
    4 : right arm
    '''
    x       = ppc_data_loader.get_x(animal,session)/xscale
    y       = ppc_data_loader.get_y(animal,session)/yscale
    tee     = np.abs(x)<xlimit
    y0      = np.median(y[~tee])
    join    = (x**2+(y-y0)**2)**0.5<joinlimit
    arms    = (np.abs(y-y0)<ylimit)&(x!=0)&(~join)
    tee    &= ~arms&(~join)
    left    = (x<0)&arms&(~join)
    right   = (x>0)&arms&(~join)
    nowhere = ~(left|right|tee|join)
    # Verify unique labeling
    assert (np.all(np.sum(np.int32([left,right,join,tee,nowhere]),axis=0)==1))
    if doplot:
        plt.scatter(x[tee]    ,y[tee]    ,label='Tee')
        plt.scatter(x[left]   ,y[left]   ,label='Left')
        plt.scatter(x[right]  ,y[right]  ,label='Right')
        plt.scatter(x[join]   ,y[join]   ,label='Join')
        plt.scatter(x[nowhere],y[nowhere],label='Nowhere')
        simpleaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        nice_legend()
    labels = np.zeros(x.shape,dtype='int32')
    labels[nowhere] = Trial.NONE_LABEL
    labels[tee]   = Trial.TEE_LABEL
    labels[join]  = Trial.JOIN_LABEL
    labels[left]  = Trial.LEFT_LABEL
    labels[right] = Trial.RIGHT_LABEL
    return labels

@memoize
def get_basic_trial_info(animal,session,
    verbose=False,
    JITTER =4,
    pad_edges = False):
    '''
    
    Extracts detailed trial statistics for the given animal and session.
    Use the `JITTER` parameter to specify how many time-points of mis-
    alignment to tolerate when defining a trial. 
    
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    verbose : bool, `False`
        print detailed logging info
    JITTER : int, 4
        expected No. samples for alignment errors
    pad_edges : boolean, True
        Should we treat blocks that start or stop at the beginning or end 
        of the signal as valid?
    
    Returns
    -------
    trials: list 
        List of trial objects

    '''
    # We'll store extracted trial info Objects here
    trials = []

    # Start with detecting trial edges as indicated in the data
    starts,stops = get_edges(ppc_data_loader.get_intrial(animal,session),pad_edges=pad_edges)

    # Get recorded trial-start timestamps
    t   = ppc_data_loader.get_timestamp(animal,session)

    # Get timepoints with reward flags from data
    rwd = ppc_data_loader.get_reward(animal,session)
    r   = np.where(rwd)[0]

    # Get recorded trial type
    c   = ppc_data_loader.get_type(animal,session)

    # Get Maze-location (coarse) timeseries
    l   = location_labels(animal,session)

    for i,(istart,istop) in enumerate(zip(starts,stops)):
        # new trial object
        newtrial = Trial()
        
        # Store time information
        # ( The stop time might be moved later to align with reward, so we 
        # set istop after finding reward signal )
        newtrial.number  = i
        newtrial.istart  = istart
        
        # Locate reward signal to see if correct trial
        # This does NOT use the trial-correct flag in the data!
        rintrial = (r>istart)&(r<=istop+JITTER)
        newtrial.correct = np.any(rintrial)
        
        # Add reward timestamp information
        if newtrial.correct:
            rindex = r[rintrial]
            # handle scenario where reward comes after trial end
            istop = max(istop,np.max(rindex))
            if len(rindex>1):
                rindex = np.min(rindex)
            newtrial.rindex  = rindex
            newtrial.roffset = newtrial.rindex - istart
        else:
            newtrial.rindex  = np.nan
            newtrial.roffset = np.nan
        
        # Set istop now
        # (might have been moved back if reward followed trial stop)
        newtrial.istop   = istop
        newtrial.nsample = istop-istart
        
        # Get location labels for this trial
        lt = l[istart:istop]
        newtrial.locations = lt
        lastt = -1
        for mark,name in Trial.LABELNAMES.items():
            matched = np.where(lt==mark)[0]
            matched = matched[matched>lastt]
            if len(matched)>0:
                markat = matched[0]
                lastt  = markat
            else:
                markat = np.nan
            setattr(newtrial,name,markat)
        
        # Check cue type
        # Convet (2,3)=(left,right) code from dataset to
        # (0,1)=(left,right)
        ct = c[istart+JITTER:istop-JITTER]
        cue=np.unique(ct)
        if (len(cue)!=1):
            if verbose:
                sys.stderr.write('Trial %d has ambiguous cue labels\n'%i)
            # Count occupancy in left/right arms
            nl  = np.sum(lt==Trial.LEFT_LABEL)
            nr  = np.sum(lt==Trial.RIGHT_LABEL)
            # Count frequency of each cue labels
            nlc = np.sum(ct==Trial.CUE_LEFT_DATASET)
            nrc = np.sum(ct==Trial.CUE_RIGHT_DATASET)
            behavior_cue = (nr>nl)==newtrial.correct
            # Check that inferred label matches majority
            if (nrc>nlc)==behavior_cue:
                if verbose:
                    sys.stderr.write(' Infer cue from behavior and reward signals\n')
                newtrial.cue = Trial.CUE_RIGHT if behavior_cue else Trial.CUE_LEFT
            else:
                if verbose:
                    sys.stderr.write(' Behavior disagrees with cue labels!!!\n')
                newtrial.cue = Trial.CUE_UNKNOWN
            
        else:
            if   cue[0]==Trial.CUE_LEFT_DATASET:
                newtrial.cue = Trial.CUE_LEFT
            elif cue[0]==Trial.CUE_RIGHT_DATASET:
                newtrial.cue = Trial.CUE_RIGHT
            else:
                if verbose:
                    sys.stderr.write('Bad cue label %s, trial %d!\n'%(cue,i))
                newtrial.cue = Trial.CUE_UNKNOWN
                
        # subject, session handles
        newtrial.animal  = animal
        newtrial.session = session
        
        # Create point to previous trial, if applicable
        newtrial.previous = trials[-1] if i>0 else None
        if verbose:
            print(newtrial)
        trials.append(newtrial)
           
    return trials


def get_detailed_trial_info(animal,session,
    verbose=False,
    JITTER =4,
    pad_edges = False):
    '''
    
    Extracts detailed trial statistics for the given animal and session.
    Use the `JITTER` parameter to specify how many time-points of mis-
    alignment to tolerate when defining a trial. 
    
    
    Parameters
    ----------
    animal : int
        Which subject ID to use
    session : int
        Which session ID to use
        
    Other Parameters
    ----------------
    verbose : bool, `False`
        print detailed logging info
    JITTER : int, 4
        expected No. samples for alignment errors
    pad_edges : boolean, True
        Should we treat blocks that start or stop at the beginning or end 
        of the signal as valid?
    
    Returns
    -------
    trials: list 
        List of trial objects

    '''
    
    # We'll store extracted trial info Objects here
    trials = []

    # Start with detecting trial edges as indicated in the data
    starts,stops = get_edges(ppc_data_loader.get_intrial(animal,session),pad_edges=pad_edges)

    # Get recorded trial-start timestamps
    t   = ppc_data_loader.get_timestamp(animal,session)

    # Get timepoints with reward flags from data
    rwd = ppc_data_loader.get_reward(animal,session)
    r   = np.where(rwd)[0]

    # Get recorded trial type
    c   = ppc_data_loader.get_type(animal,session)

    # Get Maze-location (coarse) timeseries
    l   = location_labels(animal,session)

    for i,(istart,istop) in enumerate(zip(starts,stops)):
        # new trial object
        newtrial = Trial()
        
        # Store time information
        # ( The stop time might be moved later to align with reward, so we 
        # set istop after finding reward signal )
        newtrial.number  = i
        newtrial.istart  = istart
        
        # Locate reward signal to see if correct trial
        # This does NOT use the trial-correct flag in the data!
        rintrial = (r>istart)&(r<=istop+JITTER)
        newtrial.correct = np.any(rintrial)
        
        # Add reward timestamp information
        if newtrial.correct:
            rindex = r[rintrial]
            # handle scenario where reward comes after trial end
            istop = max(istop,np.max(rindex))
            if len(rindex>1):
                rindex = np.min(rindex)
            newtrial.rindex  = rindex
            newtrial.roffset = newtrial.rindex - istart
        else:
            newtrial.rindex  = np.nan
            newtrial.roffset = np.nan
        
        # Set istop now
        # (might have been moved back if reward followed trial stop)
        newtrial.istop   = istop
        newtrial.nsample = istop-istart
        
        # Get location labels for this trial
        lt = l[istart:istop]
        newtrial.locations = lt
        lastt = -1
        for mark,name in Trial.LABELNAMES.items():
            matched = np.where(lt==mark)[0]
            matched = matched[matched>lastt]
            if len(matched)>0:
                markat = matched[0]
                lastt  = markat
            else:
                markat = np.nan
            setattr(newtrial,name,markat)
        
        # Check cue type
        # Convet (2,3)=(left,right) code from dataset to
        # (0,1)=(left,right)
        ct = c[istart+JITTER:istop-JITTER]
        cue=np.unique(ct)
        if (len(cue)!=1):
            if verbose:
                sys.stderr.write('Trial %d has ambiguous cue labels\n'%i)
            # Count occupancy in left/right arms
            nl  = np.sum(lt==Trial.LEFT_LABEL)
            nr  = np.sum(lt==Trial.RIGHT_LABEL)
            # Count frequency of each cue labels
            nlc = np.sum(ct==Trial.CUE_LEFT_DATASET)
            nrc = np.sum(ct==Trial.CUE_RIGHT_DATASET)
            behavior_cue = (nr>nl)==newtrial.correct
            # Check that inferred label matches majority
            if (nrc>nlc)==behavior_cue:
                if verbose:
                    sys.stderr.write(' Infer cue from behavior and reward signals\n')
                newtrial.cue = Trial.CUE_RIGHT if behavior_cue else Trial.CUE_LEFT
            else:
                if verbose:
                    sys.stderr.write(' Behavior disagrees with cue labels!!!\n')
                newtrial.cue = Trial.CUE_UNKNOWN
            
        else:
            if cue[0]==Trial.CUE_LEFT_DATASET:#2
                newtrial.cue = Trial.CUE_LEFT#0
            elif cue[0]==Trial.CUE_RIGHT_DATASET:#3
                newtrial.cue = Trial.CUE_RIGHT#1
            else:
                if verbose:
                    sys.stderr.write('Bad cue label %s, trial %d!\n'%(cue,i))
                newtrial.cue = Trial.CUE_UNKNOWN
        
        # Move data into the trial
        newtrial.x      = ppc_data_loader.get_x(animal,session)[istart:istop]
        newtrial.y      = ppc_data_loader.get_y(animal,session)[istart:istop]
        newtrial.dx     = ppc_data_loader.get_dx(animal,session)[istart:istop]
        newtrial.dy     = ppc_data_loader.get_dy(animal,session)[istart:istop]
        newtrial.theta  = ppc_data_loader.get_theta(animal,session)[istart:istop]
        newtrial.unitid = ppc_data_loader.good_units_index(animal,session)
        
        # subject, session handles
        newtrial.animal  = animal
        newtrial.session = session

        # Calcium signals and spikes
        newtrial.dFF    = ppc_data_loader.get_good_dFF(animal,session).T[:,istart:istop]
        newtrial.logF   = np.log1p(np.maximum(-1.0+1e-6,newtrial.dFF))
        newtrial.dlogF  = newtrial.logF - np.mean(newtrial.logF)
        newtrial.spikes = ppc_data_loader.get_good_deconvolved(animal,session).T[:,istart:istop]
        
        # Create point to previous trial, if applicable
        newtrial.previous = trials[-1] if i>0 else None
        if verbose:
            print(newtrial)
        trials.append(newtrial)
            
    return trials


def get_trials_with_context(animal,session,
    MINLEN = 40,
    MAXLEN = 200,
    **kwargs):
    '''
    CONTEXT_LABELS = { 
         0: '?, cue:←, prev:←',
         1: '↑, cue:←, prev:←',
         2: '⇄, cue:←, prev:←',
         3: '←, cue:←, prev:←',
         4: '→, cue:←, prev:←',
         5: '?, cue:←, prev:→',
         6: '↑, cue:←, prev:→',
         7: '⇄, cue:←, prev:→',
         8: '←, cue:←, prev:→',
         9: '→, cue:←, prev:→',
        10: '?, cue:→, prev:←',
        11: '↑, cue:→, prev:←',
        12: '⇄, cue:→, prev:←',
        13: '←, cue:→, prev:←',
        14: '→, cue:→, prev:←',
        15: '?, cue:→, prev:→',
        16: '↑, cue:→, prev:→',
        17: '⇄, cue:→, prev:→',
        18: '←, cue:→, prev:→',
        19: '→, cue:→, prev:→'}
    '''
    trials = get_basic_trial_info(animal,session,**kwargs)
    trials_with_context = []
    cl = []
    for tr in trials:
        # Trials must be correct trials with known cue
        if not tr.correct or tr.cue is Trial.CUE_UNKNOWN: 
            continue
        if tr.previous is None: 
            continue
        if not (tr.previous.correct and not tr.previous.cue is Trial.CUE_UNKNOWN): 
            continue
        if tr.nsample<MINLEN or tr.nsample>MAXLEN:
            continue
        # Location 0..4: ?, tee, junction, left, right
        # Cue:, 0,1 = left,right
        lt     = tr.locations
        past   = tr.previous.cue
        future = tr.cue
        assert(np.all(np.isfinite(lt)))
        assert(np.all(np.isfinite(past)))
        assert(np.all(np.isfinite(future)))
        tr.context = lt + 5 * (past + 2*future)
        cl.extend(tr.context)
        trials_with_context.append(tr)
    return trials_with_context


def get_contextual_types(animal,session):
    '''
    [get_intrial(animal,session)][::DECIMATE]

    CONTEXT_LABELS = { 
         0: '?, cue:←, prev:←',
         1: '↑, cue:←, prev:←',
         2: '⇄, cue:←, prev:←',
         3: '←, cue:←, prev:←',
         4: '→, cue:←, prev:←',
         5: '?, cue:←, prev:→',
         6: '↑, cue:←, prev:→',
         7: '⇄, cue:←, prev:→',
         8: '←, cue:←, prev:→',
         9: '→, cue:←, prev:→',
        10: '?, cue:→, prev:←',
        11: '↑, cue:→, prev:←',
        12: '⇄, cue:→, prev:←',
        13: '←, cue:→, prev:←',
        14: '→, cue:→, prev:←',
        15: '?, cue:→, prev:→',
        16: '↑, cue:→, prev:→',
        17: '⇄, cue:→, prev:→',
        18: '←, cue:→, prev:→',
        19: '→, cue:→, prev:→'}
    '''
    # Empty array same shape as labels
    labels = ppc_data_loader.get_types(animal,session)*0-1
    contextualized_trials = get_trials_with_context(animal,session)
    for tr in contextualized_trials:
        c = tr.context
        a,b = tr.istart,tr.istop
        labels[a:b]=c
    return labels


def align_trials(animal,sessions,y,
                 PSTIMERES = 1000,
                 CUE       = None,
                 PREV      = None,
                 **kwargs):
    '''
    Aligns a timeseries to trials.
    This uses pseudotime to register different trials into a common
    representation. 
    
    Parameters
    ----------
    animal:int
        Which subject to use
    sessions:list
        Which sessions to use
    y: list
        Nsessions x Nvariables x Ntimepoints set of covariates

    Other Parameters
    ----------------
    kwargs: extra keyword arguments are forwarded to 
        `get_trials_with_context`, which uses MINLEN and MAXLEN, 
        and forwarded any remaining keyword arguments on to
        `get_basic_trial_info`.
    

    Returns
    -------
    aligned : ???
    x2t : ???
    '''
    try:
        y = np.array(y)
    except ValueError:
        y = [np.array(yi) for yi in y]

    if len(np.shape(y))==2: # patch
        y = np.array(y)[None,...]
        
    NSESSIONS,NVARS,NTIMES = np.shape(y)
    if NVARS>NTIMES:
        raise ValueError('Signal input appears transposed;'
                         ' should be Nvariables x Ntimepoints')

    PDT      = 100/PSTIMERES
    VXMAX    = Trial.YSCALE + Trial.XSCALE
    x_sample = np.linspace(0,VXMAX,1000)
    t_sample = np.linspace(0,1,PSTIMERES)
    select_trials = [get_trials_with_context(animal,s,**kwargs) 
                     for s in sessions]
    LEFT  = 0
    RIGHT = 1
    if not CUE is None:
        select_trials = [[t 
                          for t in ts if t.cue==CUE]
                          for ts in select_trials]
    if not PREV is None:
        select_trials = [[t 
                          for t in ts if t.previous.cue==PREV]
                          for ts in select_trials]

    # Save X and Y data from each session in each trial object
    for (s,trials) in zip(sessions,select_trials):
        px = ppc_data_loader.get_x(animal,s)
        py = ppc_data_loader.get_y(animal,s)
        for t in trials:
            t.x = px[t.istart:t.istop]
            t.y = py[t.istart:t.istop]
            assert(all(t.x.shape==t.y.shape))

    # Merge all trial objects
    trials = sum(select_trials) if len(sessions)>1 else select_trials[0]
    K = np.array(list(map(len,select_trials)))

    x2t = [np.interp(x_sample,virtual_x(t.x,t.y),np.arange(t.nsample)) 
           for t in trials]
    x2t = unitscale(nanmean(x2t,axis=0))
    aligned  = []
    for tr in trials:
        vt = np.interp(virtual_x(tr.x,tr.y),x_sample,x2t)
        s  = np.where(array(sessions)==tr.session)[0][0]
        ty = y[s][:,tr.istart:tr.istop]
        vt = interp_all(t_sample,vt,ty).T
        if not np.all(np.isfinite(vt)):
            raise ValueError('Trial yielded NaN during alignment\n'+repr(tr))
        aligned.append(vt)
    aligned = array(aligned)
    NTRIALS = aligned.shape[0]
    return aligned,x2t


def extract_in_trial(x,animal,session,
    jitter=25,
    maxlen=200,
    dozscore=False,
    dozeromean=False):
    '''
    Extract sessions from continuous signal

    Parameters
    ----------
    x: array-like
        Signal from which to extract only good trials. 
        First axis should be timesteps
    animal: int
        Which subject to use
    session: int
        Which session to use

    Other Parameters
    ----------------
    jitter: int, default 25
        How much misalignment between trial and and reward to 
        tolerate (trials where the trial-end and reward markers 
        differ by more than `jitter` samples will be skipped)
    maxlen: int, detault 200
        Maximum session duration in samples. Overly long sessions
        indicate unusual behavior, even if they are 'correct'
    dozscore: bool, default False
        Whether to z-score each trial
    dozeromean: bool, default False
        Whether to zero-mean each trial
        
    Returns
    -------
    list of extracted trials

    '''
    trials = get_basic_trial_info(animal,session,pad_edges=False,JITTER=jitter)
    trials = [t          for t in trials if t.correct and t.nsample<=maxlen]
    starts = [t.istart   for t in trials]
    stops  = [t.istop    for t in trials]
    snips  = [x[a:b,...] for (a,b) in zip(starts,stops)]
    if dozscore:
        snips = [zscore(x,axis=0) for x in snips]
    if dozeromean:
        snips = [zeromean(x,axis=0) for x in snips]
    return snips



