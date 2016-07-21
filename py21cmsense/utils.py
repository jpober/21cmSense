#! usr/bin/env python
from scipy.interpolate import interp2d
import numpy as np, re
'''
    Python module used to accompany 21cmSense.
'''

def load_noise_files(files=None,verbose=False,polyfit_deg=3):
    '''
    Loads input 21cmsense files and clips nans and infs
    reduces k-range to [.01,.51] hMpc^-1
    returns freqs, ks, noise_values
    '''
    if files is None:
        if verbose: print 'No files Input'
        if verbose: print 'Exiting'
        return 0,'',''

    if not files:
        if verbose: print 'No Files Input'
        return 0,'',''

    #wrap single files for iteration
    flag=False
    if len(np.shape(files)) ==0: files = [files]; flag=True
    files.sort()
    re_f = re.compile('(\d+\.\d+)')#glob should load them in increasing freq order

    noises = []
    noise_ks = []
    noise_freqs = []
    nk_grid = np.linspace(0,1)*0.5+0.01

    for noisefile in files:
        noise = np.load(noisefile)['T_errs']
        noise_k = np.load(noisefile)['ks']

        bad = np.logical_or(np.isinf(noise),np.isnan(noise))
        noise = noise[np.logical_not(bad)]
        noise_k = noise_k[np.logical_not(bad)]

        #keep only the points that fall in our desired k range
        good_k = noise_k < nk_grid.max()
        noise = noise[noise_k<nk_grid.max()]
        noise_k = noise_k[noise_k<nk_grid.max()]
        if verbose: print noisefile,np.max(noise),

        ##adds values at k=0 to help interpolator function
        ##noise approaches infinite at low K
        noise_k = np.insert(noise_k,0,0)
        noise = np.insert(noise,0, np.min([1e+3,np.median(noise)]))

        tmp = np.polyfit(noise_k,noise,polyfit_deg,full=True)
        noise = np.poly1d(tmp[0])(nk_grid)
        noises.append(noise)
        noise_ks.append(nk_grid)

        small_name = noisefile.split('/')[-1].split('.npz')[0].split('_')[-1]
        f = float(re_f.match(small_name).groups()[0])*1e3 #sensitivity freq in MHz
        if verbose: print f
        noise_freqs.append(f)

    if flag:
        noise_freqs = np.squeeze(noise_freqs)
        noise_ks = np.squeeze(noise_ks)
        noises = np.squeeze(noises)
    return noise_freqs, noise_ks, noises


def noise_interp2d(noise_freqs=None,noise_ks=None,noises=None,
        interp_kind='linear', verbose=False ,**kwargs):
    '''
    Builds 2d interpolator from loaded data, default interpolation: linear
    interpolator inputs k (in hMpci), freq (in MHz)n
    '''
    if noise_freqs is None:
        if verbose: print 'Must Supply frequency values'
        return 0
    if noise_ks is None:
        if verbose: print 'Must supply k values'
        return 0
    if noises is None:
        if verbose: print 'Must supply T_errs'
        return 0

    noise_k_range = [np.min(np.concatenate(noise_ks)),np.max(np.concatenate(noise_ks))]

    if np.min(noise_k_range) == np.max(noise_k_range):
        if verbose: print 'K range only contains one value'
        if verbose: print 'Exiting'
        return 0

    nk_count = np.mean([len(myks) for myks in noise_ks])
    nks = np.linspace(noise_k_range[0],noise_k_range[1],num=nk_count)
    noise_interp = np.array([np.interp(nks,noise_ks[i],noises[i]) for i in range(len(noises))])
    NK,NF = np.meshgrid(nks,noise_freqs)
    noise_interp = interp2d(NK, NF, noise_interp, kind=interp_kind,
            **kwargs)

    return noise_interp
