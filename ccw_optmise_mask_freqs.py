
import numpy as np
from scipy import stats
import emd
import multiprocessing # needs to be imported explicitely
import multiprocessing.pool # ...



class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess



def get_modeMixScore(imf, imfUseInds):
    """ get the mode mixing score for IMFs. This is the default loss function for the 
    optimisation, but can be set externally if a different optimisation goal is wanted """
    corMat = np.abs(np.corrcoef(imf[:, imfUseInds].T))
    corMat[np.tril_indices(corMat.shape[0], k=0)] = np.nan
    mixScore = np.nanmean(corMat)
    return mixScore


def run_subIteration(args):
    """ Randomly generates mask frequencies to apply mask-EMD to Xs and returns the 
    mask freqs and the mixing scores"""
    Xs, fRanges, freqs0, n_main_freqs, fixed_mask_freqs, mask_args, imfUseInds, sample_rate, lossFunc, nprocesses = args
    # randomly generate mask frequencies
    np.random.seed()
    if fRanges is None:
        # randomly generate mask freq set without range specification (1st iteration)
        
        rdmFreqs = np.array(sorted(np.random.choice(freqs0, n_main_freqs, replace=False))[::-1])
    else:
        # use fRange to randomly generate mask freq set within specified ranges (after 1st iteration)
        nTries=50
        optimised_fis = np.where(np.subtract(fRanges[:,0], fRanges[:,1])==0)[0]
        for fi in optimised_fis:
            fRanges[fi,1] = fRanges[fi,1]+1
        for tryi in range(nTries):
            rdmFreqs = np.array([np.random.choice(np.arange(fmin, fmax)) for fmin, fmax in fRanges])
            if all(np.diff(rdmFreqs)<=0): # if randomly selected freqs are in descending order
                break
            else:
                if tryi==(nTries-1):
                    rdmFreqs = np.repeat(np.nan, fRanges.shape[0])
    
    # get mask frequencies
    mask_freqs = list(np.copy(fixed_mask_freqs))
    for f in rdmFreqs:
        mask_freqs.append(f)
    mask_freqs = np.array(sorted(mask_freqs))[::-1]
    
    # get sift args
    sift_config = emd.sift.get_config('mask_sift')
    sift_config['mask_freqs'] = mask_freqs/sample_rate
    sift_config['max_imfs'] = len(mask_freqs)
    for k in mask_args:
        if mask_args[k] is not None:
            sift_config[k] = mask_args[k]
            
    mixScores_ = []
    for X in Xs:
        imf = emd.sift.mask_sift(X, **sift_config)
        mixScores_.append(lossFunc(imf, imfUseInds))
    #
    mixScores_ = np.array(mixScores_)
    return mask_freqs, mixScores_
    

def run_iteration(fRanges, Xs, n_random, n_main_freqs, fixed_mask_freqs, 
                  imfUseInds, sample_rate, freqs0, mask_args, lossFunc, nprocesses):
    
    pool = MyPool(nprocesses)
    args = (Xs, fRanges, freqs0, n_main_freqs, fixed_mask_freqs, mask_args, imfUseInds, sample_rate, lossFunc, nprocesses)
    it_outputs = pool.map(run_subIteration, [args for i in range(n_random)])
    pool.close()
    pool.join()

    it_mask_freqs = np.row_stack([it_outputsi[0] for it_outputsi in it_outputs])
    it_mix_scores = np.row_stack([it_outputsi[1] for it_outputsi in it_outputs])

    return it_mask_freqs, it_mix_scores
    
def get_fRanges(it_maskFreqs, it_mixScores, top_n, mainImfInds):
    top_mixInds = np.where(np.argsort(np.argsort(it_mixScores.mean(axis=1)))<top_n)[0]
    fRanges = np.column_stack([it_maskFreqs[:, mainImfInds][top_mixInds].min(axis=0),
                               it_maskFreqs[:, mainImfInds][top_mixInds].max(axis=0)])
    return fRanges

def optimise_mask_freqs(Xs, sample_rate, freq_lim, freq_int, n_main_freqs, fixed_mask_freqs, 
                        imfUseInds=None, lossFunc=get_modeMixScore,
                        max_iterations=6, n_random=200, top_n=10, nprocesses=1, 
                        mask_amp=1, mask_amp_mode='ratio_imf', sift_thresh=1e-08, nphases=4, 
                        imf_opts={}, envelope_opts={}, extrema_opts={}):
    
    """ 
    Find the set of mask frequencies which yeild IMFs with the lowest loss function output.  
    
    Parameters
    ----------
    Xs : list
        each element is a 1D array containing a sample time-series used to tune the optimisation
    sample_rate : float
        the sampling rate for all the data provided in Xs
    freq_lim : tuple
        of length 2, corresponding to the minimum and maximum frequencies (in Hz) for the initial 
        generation mask frequencies in the first iteration. (This range converges for subsequent iterations)
    freq_int : float
        the stepsize between frequencies from which to randomly generate
    n_main_freqs : int
        the number of mask frequencies to be randomly generated
    fixed_mask_freqs : ndarray
        1D array specifying the mask frequencies (in Hz) which are to be fixed for each subiteration. These 
        need to be outside the freq_lim range. 
    imfUseInds : ndarray | None
        1D array specifying the indices of the extracted IMFs to be used for the loss function calculation.
        If None, all IMFs will be used.
    lossFunc : function
        Takes the IMFs from each sub-iteration and imfUseInds as argmuents.
        Returns a float which is to be minimised by the optimisation.
    max_iterations : int
        the maximum number of iterations to run. (Default 6)
    n_random : int
        the number of random sets of mask frequencies to generate per iteration. (Default 200)
    top_n : int
        after each iteration, the frequency ranges will narrow based on the current top_n sets
        of mask frequencies. Must be smaller that n_random
    nprocesses : int
        number of processes to run the optimisation in parallel (No need to exceed n_random)
    
    # args specifically relating to mask sifting:
    mask_amp : scalar or array_like
        Amplitude of mask signals as specified by mask_amp_mode. If scalar the
        same value is applied to all IMFs, if an array is passed each value is
        applied to each IMF in turn (Default value = 1)
    mask_amp_mode : {'abs','ratio_imf','ratio_sig'}
        Method for computing mask amplitude. Either in absolute units ('abs'),
        or as a ratio of the standard deviation of the input signal
        ('ratio_sig') or previous imf ('ratio_imf') (Default value = 'ratio_imf')
    sift_thresh : scalar
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    nphases : int
        the number of equally-spaced phases (from 0 - 2pi) to start each mask freqs. The average 
        of all these mask sifts will be taken.
        
    
    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema
    
    
    Returns
    -------
    it_mask_freqs : ndarray
        [nIterations x nMaskFreqs]
    it_mix_scores : ndarray
        [nIterations x len(Xs)]
    optimised_mask_freqs : ndarray
        1D array of the mask freqs which yeilded the lowest lossFunc score 
        across Xs 
    mainImfInds : ndarray
        1D array of the indices corresponding to the IMFs which would be obtained by the non-fixed mask frequencies
    
    """
    
    mask_args = {'mask_amp' : mask_amp,
                 'mask_amp_mode' : mask_amp_mode,
                 'sift_thresh' : sift_thresh, 
                 'nphases' : nphases,
                 'imf_opts' : imf_opts,
                 'envelope_opts' : envelope_opts,
                 'extrema_opts' : extrema_opts
                }
    
    
    ### WORK ###
    if imfUseInds is None:
        imfUseInds = np.arange(len(fixed_mask_freqs)+n_main_freqs)
    fmin, fmax = freq_lim
    freqs0=np.arange(fmin, fmax+freq_int)
    
    n_above = len(np.where(fixed_mask_freqs>fmax)[0])
    mainImfInds = np.arange(n_above, n_above+n_main_freqs)
    
    it_mix_scores = []
    it_mask_freqs = []
    for opti in range(max_iterations):
        if opti==0:
            fRanges = None
        else:
            fRanges = get_fRanges(np.row_stack(it_mask_freqs), np.row_stack(it_mix_scores), 
                                  top_n, mainImfInds)
        
        it_mask_freqs_, it_mix_scores_ = run_iteration(fRanges, Xs, n_random, n_main_freqs, fixed_mask_freqs, 
                                                       imfUseInds, sample_rate, freqs0, mask_args, lossFunc, nprocesses)
        
        it_mix_scores.append(it_mix_scores_)
        it_mask_freqs.append(it_mask_freqs_)
        if opti and np.sum(np.subtract(fRanges[:,1], fRanges[:,0])) <= (n_main_freqs*freq_int): # if all freqs optimised
            converged = True
            break
        elif opti == (max_iterations-1):
            print('Warning: optimisation did not converge. Consider increasing: max_iterations/n_random')
            converged = False
    it_mask_freqs = np.row_stack(it_mask_freqs)
    it_mix_scores = np.row_stack(it_mix_scores)
    optimised_mask_freqs = it_mask_freqs[it_mix_scores.mean(axis=1).argmin()]
    return it_mask_freqs, it_mix_scores, optimised_mask_freqs, mainImfInds, converged



# Example:
if False:
    Xs = [ccw.loadLFPs(bsnm, ccw.getSeshLab4testStage(bsnm, 'rec', rootFolder), rootFolder, 'nac')[:12500]           for bsnm in bsnms[:2]] # list of LFPs to use for optimisation
    sample_rate=1250.
    freq_lim=(1, 70)
    freq_int=1.
    n_main_freqs=5
    fixed_mask_freqs = np.array([300, 150, 0, 0])
    imfUseInds=None # imf indices to use to compute mixing scores. None will be interpreted as all imfs
    #
    n_random=50
    nprocesses=4
    top_n=10
    max_iterations=2



    it_mask_freqs, it_mix_scores, optimised_mask_freqs, mainImfInds, converged = \
    optimise_mask_freqs(Xs, sample_rate, freq_lim, freq_int, n_main_freqs, fixed_mask_freqs, 
    top_n=top_n, n_random=n_random, nprocesses=nprocesses, max_iterations=max_iterations)

    import seaborn as sb
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.plot(it_mix_scores[:,0])

    plt.subplot(122)
    fCols = sb.color_palette('Spectral', len(mainImfInds))
    for fi, col in enumerate(fCols):
        plt.plot(it_mask_freqs[:,mainImfInds][:, fi], it_mix_scores.mean(axis=1), 's', ms=5, color=col, lw=0, alpha=0.5)
    #plt.xscale('log')

