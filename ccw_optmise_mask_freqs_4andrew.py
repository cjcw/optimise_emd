

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sb
import emd
import math
import scipy.signal as sig
import multiprocessing as mp

'''
import ccwBaseFunctions as ccw
mnfs=True
mntStr = None #'/mnt/ccwilliams' #'data@mrc.ox.ac.uk://mnt/ccwilliams'

if mntStr is None:
    mntStr_ = ['', '/mnfs'][mnfs]
else:
    mntStr_ = mntStr

outRootFolder = mntStr_+'/ripple2/data/ccwilliams_analysis/figure_2/tailorEMD/'
rootFolder = mntStr_+'/ripple2/data/ccwilliams_merged/'
eegHz=1250.
bsnms = ['mhb17-171222', 'mrr02-180503', 'mrr04-180520', 'mccw03-180219',          
         'mccw05-180525', 'mrr03-180602', 'mrr05-181125', 'mrr06-190904']
'''



## FUNCTIONS ##

def get_siftConfig(mask_freqs):
    sift_config = emd.sift.get_config('mask_sift')
    sift_config['mask_freqs'] = mask_freqs/sample_rate
    sift_config['max_imfs'] = len(mask_freqs)
    for k, x in zip(['mask_amp', 'mask_amp_mode', 'mask_step_factor', 'mask_type', 'ret_mask_freq', 'sift_thresh'], 
                    [mask_amp, mask_amp_mode, mask_step_factor, mask_type, ret_mask_freq, sift_thresh]):
        if x is not None:
            sift_config[k] = x
    return sift_config


def merge_maskFreqs(mainFreqs, maskFreqs2add, main_correction=1.):
    mask_freqs = list(np.copy(maskFreqs2add))
    for f in mainFreqs:
        mask_freqs.append(f*main_correction)
    mask_freqs = np.array(sorted(mask_freqs))[::-1]
    return mask_freqs


def get_modeMixScore(imfs, imfMixInds):
    corMat = np.abs(np.corrcoef(imfs[:, imfMixInds].T))
    return np.tril(corMat, k=-1).mean()


def get_fRanges(it_maskFreqs, it_mixScores, topN, mainImfInds):
    top_mixInds = np.where(np.argsort(np.argsort(it_mixScores.mean(axis=1)))<topN)[0]
    fRanges = np.column_stack([it_maskFreqs[:, mainImfInds][top_mixInds].min(axis=0),
                               it_maskFreqs[:, mainImfInds][top_mixInds].max(axis=0)])
    return fRanges


def get_rdm_freqs_from_ranges(fRanges, nTries=50):
    ''' randomly generates mask freqs within specified ranges '''
    optimised_fis = np.where(np.subtract(fRanges[:,0], fRanges[:,1])==0)[0]
    for fi in optimised_fis:
        fRanges[fi,1] = fRanges[fi,1]+1
    for tryi in range(nTries):
        freqs = np.array([np.random.choice(np.arange(fmin, fmax)) for fmin, fmax in fRanges])
        if all(np.diff(freqs)<=0): # if randomly selected freqs are in descending order
            break
        else:
            if tryi==(nTries-1):
                freqs = np.repeat(np.nan, fRanges.shape[0])
    return freqs

def run_optimisation(run_parallel=True):
    it_mixScores = []
    it_maskFreqs = []
    for opti in range(max_iterations):
        if opti==0:
            fRanges = None
        else:
            fRanges = get_fRanges(np.row_stack(it_maskFreqs), np.row_stack(it_mixScores), 
                                  topN, mainImfInds)
        
        it_maskFreqs_, it_mixScores_ = run_iteration(fRanges, traces, n_random, nMainFreqs,
                                                     maskFreqs2add, imfMixInds, freqs0, 
                                                     run_parallel)
        it_mixScores.append(it_mixScores_)
        it_maskFreqs.append(it_maskFreqs_)
        if opti>0 and np.sum(np.subtract(fRanges[:,1], fRanges[:,0])) <= (nMainFreqs*fInt): # if all freqs optimised
            break
    it_maskFreqs = np.row_stack(it_maskFreqs)
    it_mixScores = np.row_stack(it_mixScores)
    return it_maskFreqs, it_mixScores

def run_subIteration(fRanges):
    if fRanges is None:
        # randomly generate mask freq set without range specification
        rdmFreqs = np.array(sorted(np.random.choice(freqs0, nMainFreqs, replace=False))[::-1])
    else:
        # use fRange to generate mask freq set
        rdmFreqs = get_rdm_freqs_from_ranges(fRanges)
    
    mask_freqs = merge_maskFreqs(rdmFreqs, maskFreqs2add, main_correction=1.)
    sift_config = get_siftConfig(mask_freqs)
    trace_mixScores = []
    for trace in traces:
        imfs, _ = emd.sift.mask_sift(trace, **sift_config)
        trace_mixScores.append(get_modeMixScore(imfs, imfMixInds))
    #
    trace_mixScores = np.array(trace_mixScores)
    return mask_freqs, trace_mixScores

def run_iteration(fRanges, traces, n_random, nMainFreqs, maskFreqs2add, 
                  imfMixInds, freqs0, run_parallel):
    if run_parallel:
        nCPUs2use = mp.cpu_count()-1
    else:
        nCPUs2use = 1
    with mp.Pool(nCPUs2use) as p:
        itOutputs = p.map(run_subIteration, [fRanges]*n_random)
    it_maskFreqs = np.row_stack([itOutputsi[0] for itOutputsi in itOutputs])
    it_mixScores = np.row_stack([itOutputsi[1] for itOutputsi in itOutputs])
    
    return it_maskFreqs, it_mixScores



'''def run_iteration(first_iteration, traces, n_random, nMainFreqs, maskFreqs2add, 
                  imfMixInds, freqs0, fRanges):

    it_maskFreqs = []
    it_mixScores = []
    for iti in range(n_random):
        if first_iteration:
            # randomly generate mask freq set without range specification
            rdmFreqs = \
            np.array(sorted(np.random.choice(freqs0, nMainFreqs, replace=False))[::-1])
        else:
            # use fRange to generate mask freq set
            rdmFreqs = \
            get_rdm_freqs_from_ranges(fRanges)
        
        mask_freqs = merge_maskFreqs(rdmFreqs, maskFreqs2add, main_correction=1.)
        sift_config = get_siftConfig(mask_freqs)
        
        trace_mixScores = []
        for trace in traces:
            imfs, _ = emd.sift.mask_sift(trace, **sift_config)

            trace_mixScores.append(get_modeMixScore(imfs, imfMixInds))

        it_maskFreqs.append(mask_freqs)
        it_mixScores.append(np.array(trace_mixScores))
        
    it_maskFreqs = np.row_stack(it_maskFreqs)
    it_mixScores = np.row_stack(it_mixScores)
    
    return it_maskFreqs, it_mixScores'''

'''
def run_optimisation():
    
    it_mixScores = []
    it_maskFreqs = []

    for opti in range(max_iterations):
        if opti==0:
            first_iteration = True
            fRanges = None
        else:
            first_iteration = False
            fRanges = get_fRanges(np.row_stack(it_maskFreqs), np.row_stack(it_mixScores), topN, mainImfInds)
        
        it_maskFreqs_, it_mixScores_ = run_iteration(first_iteration, traces, n_random, nMainFreqs, 
                                                     maskFreqs2add, imfMixInds, freqs0, fRanges)
        
        it_mixScores.append(it_mixScores_)
        it_maskFreqs.append(it_maskFreqs_)
        
        if opti>0 and np.sum(np.subtract(fRanges[:,1], fRanges[:,0])) <= (nMainFreqs*fInt): # if all freqs optimised
            break

    it_maskFreqs = np.row_stack(it_maskFreqs)
    it_mixScores = np.row_stack(it_mixScores)
    #
    return it_maskFreqs, it_mixScores
'''





## INPUTS ##
traces = [ccw.loadLFPs(bsnm, ccw.getSeshLab4testStage(bsnm, 'rec', rootFolder), rootFolder, 'nac')[:3250]           for bsnm in bsnms[:2]] # list of LFPs to use for optimisation
sample_rate = 1250.

# relating to optimisation
max_iterations=6 # how many iterations to narrow freq ranges
n_random=200 # each iteration contains this many random sets of mask frequencies
topN=10 # narrow the frequency ranges based on the top N least-mixed 
fmin=1
fmax=70
fInt=1
imfMixInds=None # imf indices to consider mixing scores. None will be interpreted as all imfs

# relating to mask sifting
maskFreqs2add = np.array([300, 150, 0, 0])
nMainFreqs=5
# unless specified here, defaults will be used
mask_amp=3
mask_amp_mode='ratio_sig'
mask_step_factor=None
mask_type=None #'all'
ret_mask_freq=True
sift_thresh=None #1e-08




## WORK ##
if imfMixInds is None:
    imfMixInds = np.arange(len(maskFreqs2add)+nMainFreqs)
freqs0=np.arange(fmin, fmax+fInt)

n_above = len(np.where(maskFreqs2add>fmax)[0])
mainImfInds = np.arange(n_above, n_above+nMainFreqs)


it_maskFreqs, it_mixScores = run_optimisation()





it_maskFreqs.shape, it_mixScores.shape
plt.style.use(plt.style.available[-1])



optimised_maskFreqs = it_maskFreqs[it_mixScores.mean(axis=1).argmin()]
optimised_maskFreqs


plt.plot(it_mixScores[:,0])



plt.figure(figsize=(2, 10))
fCols = sb.color_palette('Spectral', nMainFreqs)
for fi, col in enumerate(fCols):
    plt.plot(it_maskFreqs[:,mainImfInds][:, fi], it_mixScores.mean(axis=1), 's', ms=5, color=col, lw=0, alpha=0.5)

