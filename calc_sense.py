#! /usr/bin/env python
'''
Calculates the expected sensitivity of a 21cm experiment to a given 21cm power spectrum.  Requires as input an array .npz file created with mk_array_file.py.
'''
import aipy as a, numpy as n, optparse, sys, capo as C
from scipy import interpolate

o = optparse.OptionParser()
o.set_usage('calc_sense.py [options] *.npz')
o.set_description(__doc__)
o.add_option('-m', '--model', dest='model', default='mod',
    help="The model of the foreground wedge to use.  Three options are 'pess' (all k modes inside horizon + buffer are excluded, and all baselines are added incoherently), 'mod' (all k modes inside horizon + buffer are excluded, but all baselines within a uv pixel are added coherently), and 'opt' (all modes k modes inside the primary field of view are excluded).  See Pober et al. 2014 for more details.")
o.add_option('-b', '--buff', dest='buff', default=0.1, type=float,
    help="The size of the additive buffer outside the horizon to exclude in the pessimistic and moderate models.")
o.add_option('-f', '--freq', dest='freq', default=.135, type=float,
    help="The center frequency of the observation in GHz.  If you change from the default, be sure to use a sensible power spectrum model from that redshift.  Note that many values in the code are calculated relative to .150 GHz and are not affected by changing this value.")
o.add_option('--eor', dest='eor', default='ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2',
    help="The model epoch of reionization power spectrum.  The code is built to handle output power spectra from 21cmFAST.")
o.add_option('--ndays', dest='ndays', default=180., type=float,
    help="The total number of days observed.  The default is 180, which is the maximum a particular R.A. can be observed in one year if one only observes at night.  The total observing time is ndays*n_per_day")
o.add_option('--n_per_day', dest='n_per_day', default=6., type=float.=,
    help="The number of good observing hours per day.  This corresponds to the size of a low-foreground region in right ascension for a drift scanning instrument.  The total observing time is ndays*n_per_day")

opts, args = o.parse_args(sys.argv[1:])

#====================OBSERVATION/COSMOLOGY PARAMETER VALUES====================

#Load in data from array file; see mk_array_file.py for definitions of the parameters
array = n.load(args[0])
name = array['name']
obs_duration = array['obs_duration']
dish_size_in_lambda = array['dish_size_in_lambda']
Trx = array['Trx']
t_int = array['t_int']
if opts.model == 'pess':
    uv_coverage = array['uv_coverage_pess']
else:
    uv_coverage = array['uv_coverage']

h = 0.7
B = .008 #largest bandwidth allowed by "cosmological evolution", i.e., the maximum line of sight volume over which the universe can be considered co-eval
z = C.pspec.f2z(opts.freq)

dish_size_in_lambda = dish_size_in_lambda*(opts.freq/.150) # linear frequency evolution, relative to 150 MHz
first_null = 1.22/dish_size_in_lambda #for an airy disk, even though beam model is Gaussian
bm = 1.13*(2.35*(0.45/dish_size_in_lambda))**2
nchan = int((B/.1)*1024) #assumes 100 MHz of bandwidth are cut into 1024 channels
kpls = C.pspec.dk_deta(z) * n.fft.fftfreq(nchan,B/nchan)

Tsky = 60e3 * (3e8/(opts.freq*1e9))**2.55  # sky temperature in mK
n_lstbins = opts.n_per_day*60./obs_duration

#===============================EOR MODEL===================================

#You can change this to have any model you want, as long as mk, mpk and p21 are returned

#This is a dimensionless power spectrum, i.e., Delta^2
modelfile = 'ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2'
model = n.loadtxt(modelfile)
mk, mpk = model[:,0]/h, model[:,1] #k, Delta^2(k)
#note that we're converting from Mpc to h/Mpc

#interpolation function for the EoR model
p21 = interpolate.interp1d(mk, mpk, kind='linear')

#==============================OTHER FUNCTIONS===============================

#A function used for binning
def find_nearest(array,value):
    idx = (n.abs(array-value)).argmin()
    return idx

#=================================MAIN CODE===================================

#set up blank arrays/dictionaries
kprs = []
#sense will include sample variance, Tsense will be Thermal only
sense, Tsense = {}, {}
    
uv_coverage *= t_int
SIZE = uv_coverage.shape[0]

# Cut unnecessary data out of uv coverage
# Get rid of auto-correlations
uv_coverage[SIZE/2,SIZE/2] = 0
# Cut out 1/2 uv plane (not statistically independent)
uv_coverage[:,:SIZE/2] = 0
uv_coverage[SIZE/2:,SIZE/2] = 0

#loop over uv_coverage to calculate k_pr
nonzero = n.where(uv_coverage > 0)
for iu,iv in zip(nonzero[1], nonzero[0]):
   u, v = (iu - SIZE/2) * dish_size_in_lambda, (iv - SIZE/2) * dish_size_in_lambda
   umag = n.sqrt(u**2 + v**2)
   kpr = umag * C.pspec.dk_du(z)
   kprs.append(kpr)
   #calculate horizon limit for baseline of length umag
   if opts.model in ['mod','pess']: hor = C.pspec.dk_deta(z) * umag/opts.freq + opts.buff
   elif opts.model in ['opt']: hor = C.pspec.dk_deta(z) * (umag/opts.freq)*n.sin(first_null/2)
   else: print '%s is not a valid foreground model; Aborting...' % opts.model; sys.exit()
   if not sense.has_key(kpr): 
       sense[kpr] = n.zeros_like(kpls)
       Tsense[kpr] = n.zeros_like(kpls)
   for i, kpl in enumerate(kpls):
       #exclude k_parallel modes contaminated by foregrounds
       if n.abs(kpl) < hor: continue
       k = n.sqrt(kpl**2 + kpr**2)
       if k < min(mk): continue
       #don't include values beyond the interpolation range (no sensitivity anyway)
       if k > n.max(mk): continue
       tot_integration = uv_coverage[iv,iu] * opts.ndays
       delta21 = p21(k)
       Tsys = Tsky + Trx
       bm2 = bm/2. #beam^2 term calculated for Gaussian; see Parsons et al. 2014
       bm_eff = bm**2 / bm2 # this can obviously be reduced; it isn't for clarity
       scalar = C.pspec.X2Y(z) * bm_eff * B * k**3 / (2*n.pi**2)
       Trms = Tsys / n.sqrt(2*(B*1e9)*tot_integration)
       #add errors in inverse quadrature
       sense[kpr][i] += 1./(scalar*Trms**2 + delta21)**2
       Tsense[kpr][i] += 1./(scalar*Trms**2)**2

#bin the result in 1D
delta = C.pspec.dk_deta(z)*(1./B) #default bin size is given by bandwidth
kmag = n.arange(delta,n.max(mk),delta)

kprs = n.array(kprs)
sense1d = n.zeros_like(kmag)
Tsense1d = n.zeros_like(kmag)
for ind, kpr in enumerate(sense.keys()):
    #errors were added in inverse quadrature, now need to invert and take square root to have error bars; also divide errors by number of indep. fields
    sense[kpr] = sense[kpr]**-.5 / n.sqrt(n_lstbins)
    Tsense[kpr] = Tsense[kpr]**-.5 / n.sqrt(n_lstbins)
    for i, kpl in enumerate(kpls):
        k = n.sqrt(kpl**2 + kpr**2)
        if k > n.max(mk): continue
        #add errors in inverse quadrature for further binning
        sense1d[find_nearest(kmag,k)] += 1./sense[kpr][i]**2
        Tsense1d[find_nearest(kmag,k)] += 1./Tsense[kpr][i]**2

#invert errors and take square root again for final answer
for ind,kbin in enumerate(sense1d):
    sense1d[ind] = kbin**-.5
    Tsense1d[ind] = Tsense1d[ind]**-.5

#save results to output npz
n.savez('%s_%s_%.3f.npz' % (name,opts.model,opts.freq),ks=kmag,errs=sense1d,T_errs=Tsense1d)

#calculate significance with least-squares fit of amplitude
A = p21(kmag)
M = p21(kmag)
wA, wM = A * (1./sense1d), M * (1./sense1d)
wA, wM = n.matrix(wA).T, n.matrix(wM).T
amp = (wA.T*wA).I * (wA.T * wM)
#errorbars
Y = n.float(amp) * wA
dY = wM - Y
s2 = (len(wM)-1)**-1 * (dY.T * dY)
X = n.matrix(wA).T * n.matrix(wA)
err = n.sqrt((1./n.float(X)))
print 'total snr = ', amp/err
