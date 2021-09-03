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
from   scipy.linalg import solve,lstsq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ppc_data_loader
import ppc_trial

matplotlib.rcParams['figure.dpi']=200

def plot_pairs(Y,fig=None,colors=None,s=0.7,title=None):
    '''
    Scatter plot all pairs of coordinate axes as subplots
    '''
    if not fig is None and not type(fig) is matplotlib.figure.Figure:
        raise ValueError('Second argument fig should be a figure handle')
    if fig is None: 
        fig = plt.figure(figsize=(10,10))
    fig.set_facecolor('w')
    NDIM = Y.shape[1]
    if NDIM>Y.shape[0]:
        Y = Y.T
        NDIM = Y.shape[1]
    pairs = [(i,j) for i in range(NDIM) for j in range(i)]
    len(pairs)
    T = Y.shape[0]
    addcolorbar = False
    if colors is None:
        colors = linspace(0,100,T)
        addcolorbar = True
    NSUBPLOT = Y.shape[1]
    ax = fig.__ax if hasattr(fig,'__ax') else {}
    for isub,(i,j) in enumerate(pairs):
        #ax = fig.add_subplot(NSUBPLOT,NSUBPLOT,isub+1)
        if not (i,j) in ax:
            ax[i,j] = fig.add_subplot(NSUBPLOT,NSUBPLOT,i+j*NSUBPLOT)
        ax[i,j].scatter(Y[:,i],Y[:,j],s=s,c=colors)
        force_aspect()
        nox(); noy()
        if i==1+j:
            ylabel(j)
            xlabel(i)
        simpleaxis()
    if not title is None:
        suptitle(title)
    tight_layout()
    subplots_adjust(top=0.9)

    if addcolorbar:
        ax = ax[NDIM-1,0]
        sca(ax)
        good_colorbar(0,1,'viridis','Time (rescaled)')
        
    fig.__ax = ax
    return fig

def cross_factor_plot(f1,f2):
    '''
    Plot matrix image of f1.dot(f2.T)
    '''
    pairwise = abs(f1.dot(f2.T))
    print(pairwise.shape)
    vmin = int(10*np.min(pairwise)+0.5)/10
    vmax = int(10*np.max(pairwise)+0.5)/10
    img  = plt.imshow(pairwise,vmin=vmin,vmax=vmax);
    simpleaxis()
    plt.xlabel('Factor No., session 1')
    plt.ylabel('Factor No., session 2')
    cax=good_colorbar(vmin,vmax,'parula',title='$|x_1^{\\top} x_2|$')
    fudgey(ax=cax)
    plt.title('Factor similarity across sessions')

def leplot(animal,session,
           DECIMATE=10,
           NSHOW=None,
           FIGSCALE=0.8,
           color_mode='cue',
           s=1,
           **kwargs):
    '''
    Laplacian eigenmaps plot of neural activity for PPC datasets
    '''
    X,Y,Z,Sigma,F,lmbda,fa = ppc_data_loader.get_le(animal,session,DECIMATE=DECIMATE,**kwargs)
    if NSHOW is None: 
        NSHOW = NDIM

    intrial = ppc_data_loader.get_intrial(animal,session)
    # Color parts of session
    #labels = get_types(animal,session)[intrial][::DECIMATE]
    if color_mode=='x':
        x = ppc_data_loader.get_x(animal,session)/ppc_trial.Trial.XSCALE
        ctxt_colors,names = [],[]
        colors = np.array([np.interp(x,np.linspace(np.min(x),np.max(x),256),c) for c in parula_data.T]).T
        colors = colors[intrial][::DECIMATE]
        #y       = get_y(animal,session)/yscale
    if color_mode=='y':
        y = ppc_data_loader.get_y(animal,session)/ppc_trial.Trial.YSCALE
        ctxt_colors,names = [],[]
        colors = np.array([np.interp(y,np.linspace(np.min(y),np.max(y),256),c) for c in parula_data.T]).T
        colors = colors[intrial][::DECIMATE]
    if color_mode=='xy':
        x = ppc_data_loader.get_x(animal,session)/ppc_trial.Trial.XSCALE
        y = ppc_data_loader.get_y(animal,session)/ppc_trial.Trial.YSCALE
        l = ppc_data_loader.get_left_right_cue_labels(animal,session)
        z = (np.abs(y+np.abs(x))*0.5)**0.25
        z[l==0]*=-1
        #cmap = matplotlib.cm.Spectral
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('riley',[CRIMSON,OCHRE,INDEGO])
        colors = cmap((z+1)/2)    
        colors = colors[intrial][::DECIMATE]
        ctxt_colors = cmap([0,0.5,1])
        names = ['Left','Tee','Right']
        # Blank out bad trials?
        labels    = ppc_trial.get_contextual_types(animal,session)[intrial][::DECIMATE]
        colors[labels<0,-1]=0
    if color_mode=='location':
        labels = ppc_trial.location_labels(animal,session)[intrial][::DECIMATE]
        names  = ['Other','Tee','Junction','Left','Right']
        ctxt_colors = np.array(list(map(mpl.colors.to_rgb,[WHITE,OCHRE,MAUVE,RUST,AZURE])))
        ctxt_colors = np.concatenate([ctxt_colors,np.ones((ctxt_colors.shape[0],1))],axis=1)
        ctxt_colors[0,-1] = 0
        colors = ctxt_colors[labels,:]
    elif color_mode=='context':
        colors,ctxt_colors,names = le_contextual_colors(animal,session)
        colors = colors[intrial][::DECIMATE]
    elif color_mode=='cue':
        ctxt_colors = np.array(list(map(mpl.colors.to_rgb,[RUST,AZURE,WHITE])))
        ctxt_colors = np.concatenate([ctxt_colors,np.ones((ctxt_colors.shape[0],1))],axis=1)
        ctxt_colors[-1,-1] = 0
        labels = ppc_data_loader.get_left_right_cue_labels(animal,session)[intrial][::DECIMATE]
        colors = ctxt_colors[labels,:]
        names  = ['Left cue','Right cue']
    elif color_mode=='previous':
        ctxt_colors = np.array(list(map(mpl.colors.to_rgb,[OCHRE,TURQUOISE,WHITE])))
        ctxt_colors = np.concatenate([ctxt_colors,np.ones((ctxt_colors.shape[0],1))],axis=1)
        ctxt_colors[-1,-1] = 0
        labels = ppc_data_loader.get_previous_trial_type(animal,session)[intrial][::DECIMATE]
        colors = ctxt_colors[labels,:]
        names  = ['Previously went left','Previously went right']
                
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
    
    '''
    bad_dimensions = []    
    for i,z in enumerate(Z.T):
        z = z[np.isfinite(z)]
        a,b = np.min(z),np.max(z)
        m = (a+b)*0.5
        x = np.mean(z<m)
        if abs(x-0.5)>0.48:
            bad_dimensions.append(i)
    ok_dim = tuple(sorted(list(set(np.arange(NDIM))-set(bad_dimensions))))
    Z = Z[:,ok_dim]
    '''

    ok = colors[:,-1]!=0
    plot_pairs(Z[ok,:NSHOW],figure(figsize=(10*FIGSCALE,11*FIGSCALE),facecolor='w'),colors=colors[ok,:],s=s)

    # Legend
    subplot(NSHOW,NSHOW,1+NSHOW**2-NSHOW*2)
    for c,n in zip(ctxt_colors,names):
        scatter([-100],[-100],s=80,label=n,color=c)
    nox();noy();xlim(0,1);ylim(0,1);noaxis();nice_legend(loc=3,fontsize=14);
    subplots_adjust(wspace=0,hspace=0,top=1,bottom=0,left=0,right=1)
    
    return X,Y,Z,Sigma,F,lmbda,fa


def le_contextual_colors(animal,session):
    context_labels = ppc_trial.Trial.CONTEXT_LABELS
    # Check that there are 12 prevailing labels
    cl  = np.concatenate([tr.context for tr in ppc_trial.get_trials_with_context(animal,session)])
    #labels,cts  = np.unique(cl,return_counts=True)
    #please_show = cts>(len(cl)//100)
    #show_labels = sorted(labels[please_show])
    #assert len(show_labels)==12
    
    show_labels = array([1,2,3,6,7,8,11,12,14,16,17,19])

    # Define available colors (last color = not displayed)
    ctxt_colors = [OCHRE     , RUST, CRIMSON,
                   CHARTREUSE, RUST, CRIMSON,
                   MAUVE     , AZURE, TURQUOISE,
                   INDEGO    , AZURE, TURQUOISE,
                   WHITE]
    ctxt_colors = array(ctxt_colors)
    ctxt_colors = np.concatenate([ctxt_colors,np.ones((ctxt_colors.shape[0],1))],axis=1)
    ctxt_colors[-1,-1] = 0
    
    # Assign color codes to each timepoiont
    #print('Displaying the following context labels:')
    labels = ppc_trial.get_contextual_types(animal,session)
    color_idxs  = int32(zeros(labels.shape)-1)
    names = []
    for i,il in enumerate(show_labels):
        descr = context_labels[il]
        color = mpl.colors.rgb2hex(ctxt_colors[i])
        #print('\t%2d: %s (color: %s)'%(il,descr,color))
        color_idxs[labels==il] = i
        names.append(descr)

    remap = np.array([0,1,2,3,1,2,4,5,6,7,5,6])
    newnames = '↺ ↝ ↜ ↻ ↰ ← ↱ →'.split()
    newcolors = [OCHRE, CHARTREUSE, MAUVE, INDEGO, RUST, CRIMSON, AZURE, TURQUOISE]
    return ctxt_colors[color_idxs],newcolors,newnames


# Ad hoc merging some states for clarity
'''
	 1: ↑, cue:←, prev:← (color: #eea300) 
	 2: ⇄, cue:←, prev:← (color: #eb7a59)
	 3: ←, cue:←, prev:← (color: #b41d4d)
	 6: ↑, cue:←, prev:→ (color: #b59f1a)
	 7: ⇄, cue:←, prev:→ (color: #eb7a59)
	 8: ←, cue:←, prev:→ (color: #b41d4d)
	11: ↑, cue:→, prev:← (color: #8d5ccd)
	12: ⇄, cue:→, prev:← (color: #5aa0df)
	14: →, cue:→, prev:← (color: #00bac9)
	16: ↑, cue:→, prev:→ (color: #606ec3)
	17: ⇄, cue:→, prev:→ (color: #5aa0df)
	19: →, cue:→, prev:→ (color: #00bac9)
↶ ↝ ↜ ↷↺↻↷

	 1: ↑, cue:←, prev:← (color: #eea300) 0
	 2: ⇄, cue:←, prev:← (color: #eb7a59) 1
	 3: ←, cue:←, prev:← (color: #b41d4d) 2
	 6: ↑, cue:←, prev:→ (color: #b59f1a) 3 
	 7: ⇄, cue:←, prev:→ (color: #eb7a59) 1
	 8: ←, cue:←, prev:→ (color: #b41d4d) 2
	11: ↑, cue:→, prev:← (color: #8d5ccd) 4
	12: ⇄, cue:→, prev:← (color: #5aa0df) 5
	14: →, cue:→, prev:← (color: #00bac9) 6
	16: ↑, cue:→, prev:→ (color: #606ec3) 7 
	17: ⇄, cue:→, prev:→ (color: #5aa0df) 5
	19: →, cue:→, prev:→ (color: #00bac9) 6

(←)↑(←) ↰ ← (→)↑(←) (←)↑(→) ↱ → (←)↑(←)
'''



def xy_scalebars(dx=None,dy=None,xunits='',yunits='',xoffset=0,yoffset=0,xpadp=9,ypadp=12,
                 fontsize=12,
                 center=False,
                 color=BLACK,
                 show_labels=True,
                 **kwargs):
    '''
    '''
    xl,yl = plt.xlim(),plt.ylim()

    if dx is None:
        dx = pixels_to_xunits(150)
    if dy is None:
        dy = pixels_to_yunits(150)

    if gca().get_aspect()=='equal':
        hwx = hhx = hwy = hhy = pixels_to_yunits(15)
    else:
        hwx = pixels_to_yunits(15)
        hhx = pixels_to_xunits(15)
        hwy = pixels_to_xunits(15)
        hhy = pixels_to_yunits(15)
        
    x = xl[0]+xoffset*np.diff(xl)[0]
    y = yl[0]+yoffset*np.diff(yl)[0]
    if show_labels:
        plt.plot([x,x+dx],[y,y],color=color,**kwargs)
        plt.plot([x,x],[y,y+dy],color=color,**kwargs)
    else:
        #plt.arrow(x,y,dx,0,head_width=hwx,head_length=hhx,color=color,clip_on=False,**kwargs)
        #plt.arrow(x,y,0,dy,head_width=hwy,head_length=hhy,color=color,clip_on=False,**kwargs)
        w0 = array([x,y])
        wx = w0 + [dx,0]
        wy = w0 + [0,dy]
        annotate(None,wx,xytext=w0,arrowprops={'arrowstyle':"-|>",'facecolor':'k','shrinkA':0})
        annotate(None,wy,xytext=w0,arrowprops={'arrowstyle':"-|>",'facecolor':'k','shrinkA':0})

    px = neurotools.graphics.plot.pixels_to_xunits(xpadp)
    py = neurotools.graphics.plot.pixels_to_yunits(ypadp)
    
    text(x+0.5*dx*int(~~center),y-py,'%s %s'%(dx,xunits) if show_labels else xunits,
         horizontalalignment='center' if center else 'left',
         verticalalignment='top', wrap=True,
         fontsize=fontsize)
    text(x-px,y+0.5*dy*int(~~center),'%s %s'%(dy,yunits) if show_labels else yunits,
         horizontalalignment='right' if center else 'left',
         verticalalignment='center' if center else 'bottom',
         rotation=90, wrap=True,
         fontsize=fontsize)
    xlim(*xl); ylim(*yl)

def plot_shaded_trajectory_bundle(X,cmap=parula,lw=1,s=8,dotcolor=BLACK):
    '''
    Faster way to plot a lot of shaded line segments at once
    
    Parameters
    ----------
    X: Nlines x Npoints x 2
        Array of line segments
    cmap:
        color map, defualt parula
    '''
    K,T,_  = X.shape
    v = np.zeros((K*3-1,T,2))*np.nan
    v[0::3,:  ,:] = X
    v[1::3,:-1,:] = X[:,1:,:]
    v = v.transpose([1,2,0])
    for c,(x,y) in zip(cmap(np.linspace(0,1,T)),v):
        plt.plot(x,y,lw=lw,color=c)
    plt.scatter(*X[:,T//2,:].T,s=s,color=dotcolor,zorder=np.inf)

def plot_shaded_trajectory_bundle_3D(X,cmap=parula,lw=1,s=8,ax=None,dotcolor=BLACK):
    '''
    Faster way to plot a lot of shaded line segments at once
    
    Parameters
    ----------
    X: Nlines x Npoints x 2
        Array of line segments
    cmap:
        color map, defualt parula
    '''
    if ax is None:
        ax = plt.gca()
    K,T,_  = X.shape
    v = np.zeros((K*3-1,T,3))*np.nan
    v[0::3,:  ,:] = X
    v[1::3,:-1,:] = X[:,1:,:]
    v = v.transpose([1,2,0])
    for c,(x,y,z) in zip(cmap(np.linspace(0,1,T)),v):
        plt.plot(x,y,z,lw=lw,color=c)
    ax.scatter(*X[:,T//2,:].T,s=s,color=dotcolor,zorder=np.inf)

def hideaxis():
    '''
    Apply three commands `noaxis(); nox(); noy();`
    '''
    noaxis(); nox(); noy();

