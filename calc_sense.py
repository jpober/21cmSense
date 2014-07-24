#! /opt/local/bin python
import aipy as a, numpy as n, pylab as p
import capo as C
import sys
from scipy import interpolate

params = {'legend.fontsize': 10}
p.rcParams.update(params)

#functions for binning
def near(a,b,rtol=1e-5,atol=1e-8):
    try:
        return n.abs(a-b)<(atol+rtol*n.abs(b))
    except TypeError:
        print "Error in NEAR!"
        return False

def find_nearest(array,value):
    idx = (n.abs(array-value)).argmin()
    return idx

#array = 'hera' #547
#array = 'hera19'
#array = 'hera37'
#array = 'hera61'
#array = 'hera91'
#array = 'hera127'
#array = 'hera169'
#array = 'hera217'
#array = 'hera271'
#array = 'hera331'
#array = 'hera397'
#array = 'hera469'
#array = 'paper'
array = 'mwa'
#array = 'lofar'
#array = 'skalo1'

#model = 'pess'
model = 'mid'
#model = 'opt'

#telescope varying dictionaries
N_2HR_LSTBINS_DICT = {'hera': 9, 'hera19': 9, 'hera37': 9, 'hera61': 9, 'hera91': 9, 'hera127': 9, 'hera169': 9, 'hera217': 9, 'hera271': 9, 'hera331': 9, 'hera397': 9, 'hera469': 9, 'paper': 4, 'mwa': 3, 'lofar': 24, 'skalo1': 24}
#paper observed for 6 hours per day in your publication
dish_size_in_lambda_dict = {'hera': 7, 'hera19': 7, 'hera37': 7, 'hera61': 7, 'hera91': 7, 'hera127': 7, 'hera169': 7, 'hera217': 7, 'hera271': 7, 'hera331': 7, 'hera397': 7, 'hera469': 7, 'paper': 1.5, 'mwa': 2.65, 'lofar': 15.4, 'skalo1': 17.5} 
# note you published your table with paper = 2, mwa = 2.65
# fixing the paper collecting area to 1.5 gives: pess = 1.17, mod = 2.02, opt = 4.82  
FILENAME_DICT = {
'hera': '/Users/jpober/science/hera/uvcov/hex547_delta.npz',
'hera19': '/Users/jpober/science/hera/uvcov/hex19.npz',
'hera37': '/Users/jpober/science/hera/uvcov/hex37.npz',
'hera61': '/Users/jpober/science/hera/uvcov/hex61.npz',
'hera91': '/Users/jpober/science/hera/uvcov/hex91.npz',
'hera127': '/Users/jpober/science/hera/uvcov/hex127.npz',
'hera169': '/Users/jpober/science/hera/uvcov/hex169.npz',
'hera217': '/Users/jpober/science/hera/uvcov/hex217.npz',
'hera271': '/Users/jpober/science/hera/uvcov/hex271.npz',
'hera331': '/Users/jpober/science/hera/uvcov/hex331.npz',
'hera397': '/Users/jpober/science/hera/uvcov/hex397.npz',
'hera469': '/Users/jpober/science/hera/uvcov/hex469.npz',
#'paper': '/Users/jpober/science/hera/uvcov/paper_11x12.npz',
'paper': '/Users/jpober/science/hera/uvcov/paper_11x12_1.5m.npz',
#'paper': '/Users/jpober/science/hera/uvcov/paper32_dcj_newfix.npz',
#'paper': '/Users/jpober/science/hera/uvcov/paper64_arp.npz',
#'paper': '/Users/jpober/science/hera/uvcov/paper128_dcj_3bl.npz',
#'paper': '/Users/jpober/science/hera/uvcov/paper128_dcj.npz',
'mwa': '/Users/jpober/science/hera/uvcov/mwa128.npz',
'lofar': '/Users/jpober/science/hera/uvcov/lofar_core.npz',
'skalo1': '/Users/jpober/science/hera/uvcov/skalo1.npz'
}

#mean temperature function
def mean_temp(z):
    return 28. * ((1.+z)/10.)**.5 #mK

#redefined capo functions
def k3pk_from_Trms(list_of_Trms, list_of_Twgt, k=.3, fq=.150, B=.001, bm=.05):
    z = C.pspec.f2z(fq)
    bm2 = bm/2. #analytic
    bm_eff = bm**2 / bm2 # this can obviously be reduced; it isn't for clarity
    scalar = C.pspec.X2Y(z) * bm_eff * B * k**3 / (2*n.pi**2)
    Trms2_sum, Trms2_wgt = 0, 0
    if len(list_of_Trms) == 1: Trms2_sum,Trms2_wgt = n.abs(list_of_Trms[0])**2, n.abs(list_of_Twgt[0])**2
    else:
        for i,(Ta,Wa) in enumerate(zip(list_of_Trms, list_of_Twgt)):
            for Tb,Wb in zip(list_of_Trms[i+1:], list_of_Twgt[i+1:]):
                Trms2_sum += Ta * n.conj(Tb)
                Trms2_wgt += Wa * n.conj(Wb)
    #print 'A', scalar * Trms2_sum / Trms2_wgt
    return scalar * Trms2_sum / Trms2_wgt, Trms2_wgt

def k3pk_sense_vs_t(t, k=.3, fq=.150, B=.001, bm=.05, Tsys=500e3):
    Trms = Tsys / n.sqrt(2*(B*1e9)*t) # This is the correct equation for a DUAL-pol, cross-correlation measurement
    return k3pk_from_Trms([Trms], [1.], k=k, fq=fq, B=B, bm=bm)[0]


#telescope and observation parameters
h = 0.7
#fq = .168 #danny wants 126, 150, 160, 165, 170
fq = C.pspec.z2f(9.5)
B = .010
Trx = 1e5 # ARP model
#Trx = 3e5 #DCJ model
#Tsky = 10e3 * (fq/.750)**-2.5 BAOBAB model
#Tsky = 4.5e5*(fq/.160)**-2.6 # ARP model
#Tsky = 2.5e5*(fq/.160)**-2.6 # DCJ model
Tsky = 60e3 * (3e8/(fq*1e9))**2.55  # LOFAR formula
Tsys = Trx + Tsky #in mK
print "Tsys = ", Tsys, "mK"
z = C.pspec.f2z(fq)
WINDOW = 'kaiser3'
NDAYS = 180 #ignoring days when sun is up
#NDAYS = 40 #psa32_dcj data / psa64_arp ???
N_2HR_LSTBINS = N_2HR_LSTBINS_DICT[array]
WINDOW_SENSE = 1.
#dish_size_in_lambda = 5.51 #using PAPER primary beam model
#dish_size_in_lambda = 7 # frequency independent beam
dish_size_in_lambda = dish_size_in_lambda_dict[array]*(fq/.150) # linear frequency evolution
FWHM = 2.35*(0.45/dish_size_in_lambda)
bm = 1.13*(2.35*(0.45/dish_size_in_lambda))**2

nchan = int((8/100.)*1024)
kpls = C.pspec.dk_deta(z) * n.fft.fftfreq(nchan,B/nchan)

#model EoR signal
#filename = '/Users/jpober/science/pspec/lidz_mcquinn_k3pk/power_21cm_z7.32.dat'
filename = '/Users/jpober/science/21cmFAST_v1.01/oldruns/nextgen/vanilla/ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2'
#d = n.array([map(float, L.split()) for L in open(filename).readlines()])
#mk, mpk = d[:,0], d[:,1]
#mdelta2 = mpk * mean_temp(z)**2 * mk**3 * (2.*n.pi**2.)**-1.

d = n.loadtxt(filename)
mk, mpk = d[:,0]/h, d[:,1]
#interpolate model over ks
p21 = interpolate.interp1d(mk, mpk, kind='linear')
mdelta2 = p21(mk)
 
#interpolate model over ks
#p21 = interpolate.interp1d(mk, mpk, kind='linear')

#FILENAME = '/Users/jpober/science/hera/uvcov/hex547_30min.bms.fits'
#FILENAME = '/Users/jpober/science/hera/uvcov/hex547_40min_dillon.bms.fits'
#FILENAME = '/Users/jpober/science/hera/uvcov/hex547_40min.bms.fits'
#FILENAME = '/Users/jpober/science/hera/uvcov/hex547_delta.npz'
#d,kwds = a.img.from_fits(FILENAME)
data = n.load(FILENAME_DICT[array])

#print 'uv plane sum = ', n.sum(d.flatten())

#f_f0 = n.sum(d.flatten()**2)/n.sum(d.flatten())
#print 'f/f0 = ', f_f0

allks = []
kprs = []
sense, Tsense = {}, {}

if model in ['pess']: planes = ['quadsum']
if model in ['mid','opt']: planes = ['sum']

mod_2_title = {
'opt': 'Optimistic Foreground Removal',
'mid': 'Moderate Foreground Removal',
'pess': 'Pessimistic Foreground Removal'
}

#if True:
    #planes = ['sum']
    #planes = ['quadsum']
    #planes = [data.files[-1]]
    #planes = data.files[-10:-1]
#else: 
    #planes = data.files
    #planes.remove('sum')
    #planes.remove('quadsum')

tkpr,tkpl,tsense = [],[],[]
for cnt, uvplane in enumerate(planes):
    print 'working on plane %i of %i' % (cnt+1, len(planes))
    d = data[uvplane]
    t_int = 60. # seconds of integration per count in sampling
    #t_int = 60.*1.5 # ARP uses this 1.5 to remove negative fringe rates
    d = d.squeeze() * t_int
    SIZE = d.shape[0]
    #if SIZE % 2 != 0: d = a.img.recenter(d,(1,1)) #unecessary if using gen_uv_cov
    # Get rid of auto-correlations
    d[SIZE/2,SIZE/2] = 0
    # Cut out 1/2 uv plane (not statistically independent); also cut out half the north-south baselines
    d[:,:SIZE/2] = 0
    d[SIZE/2:,SIZE/2] = 0
    # Get rid of north-south baselines??? -- NO
    #d[:,SIZE/2] = 0
   
    #plot the uv coverage
    if True:
        p.imshow(d,interpolation='nearest',aspect='auto')
        p.colorbar() 
        p.show()
 
    #loop over d to calculate k_pr
    res = dish_size_in_lambda #size of uv pixel in wavelengths
    nonzero = n.where(d > 1e-4) #hack for speed
    for iu,iv in zip(nonzero[1], nonzero[0]):
       u, v = (iu - SIZE/2) * res, (iv - SIZE/2) * res
       umag = n.sqrt(u**2 + v**2)
       kpr = umag * C.pspec.dk_du(z)
       kprs.append(kpr)
       if model in ['mid','pess']: hor = C.pspec.dk_deta(z) * umag/fq + 0.1 #horizon and an additive term
       if model in ['opt']: hor = C.pspec.dk_deta(z) * (umag/fq)*n.sin(FWHM/2) #primary beam

       #if umag < 20: print umag, kpr, hor
       if not sense.has_key(kpr): 
           sense[kpr] = n.zeros_like(kpls)
           Tsense[kpr] = n.zeros_like(kpls)
       for i, kpl in enumerate(kpls):
           #exclude k_parallel modes contaminated by foregrounds
           if n.abs(kpl) < hor: continue
           k = n.sqrt(kpl**2 + kpr**2)
           allks.append(k)
           if k < min(mk): continue
           #don't include values beyond the interpolation range (no sensitivity anyway)
           if k > n.max(mk): continue
           #if k > 3: continue
           noise = d[iv,iu] * NDAYS
           #delta21 = p21(k) * mean_temp(z)**2 * k**3 * (2.*n.pi**2.)**-1.
           delta21 = p21(k)
           #k3pk_sense scripts modified to accept non-polynomial beam
           #sense[kpr][i] += 1./(k3pk_sense_vs_t(noise,k=k,Tsys=Tsys,bm=bm,B=B) + delta21)**2
           Tsense[kpr][i] += 1./(k3pk_sense_vs_t(noise,k=k,Tsys=Tsys,bm=bm,B=B))**2
           bm2 = bm/2. #analytic
           bm_eff = bm**2 / bm2 # this can obviously be reduced; it isn't for clarity
           scalar = C.pspec.X2Y(z) * bm_eff * B * k**3 / (2*n.pi**2)
           Trms = Tsys / n.sqrt(2*(B*1e9)*noise)
           #print 'B', scalar * Trms**2
           sense[kpr][i] += 1./(scalar*Trms**2 + delta21)**2
           tkpr.append(kpr)
           tkpl.append(kpls[i])
           tsense.append(delta21/(scalar*Trms**2)**2)

if False:
    #2D sense plot
    #kprs = sense.keys()
    kprs = n.arange(0,0.3,.01)
    sense2d = n.zeros((len(kprs),len(kpls)))
    cen = len(kpls)/2 + 1
    #print kpls
    pdf,cdf = [],[]
    for kpr in sense.keys():
        ind = find_nearest(kprs,kpr)
        sense2d[ind] += n.append(sense[kpr][cen:], sense[kpr][:cen])

    for ind, kpr in enumerate(kprs):
        sense2d[ind] = sense2d[ind]**-.5 / n.sqrt(N_2HR_LSTBINS)
 
    p.imshow(n.log10(sense2d.T),aspect='auto',interpolation='nearest')
    p.colorbar()
    p.show()
 
    for ind, kpr in enumerate(kprs):
        #if ind == 0: continue
        pdf.append(2./sense2d[ind,cen+1]**2)
        cdf.append(n.sum(n.array(pdf)))
        #print pdf, cdf
   
    cdf /= n.sum(pdf) 
    p.plot(kprs,cdf)

    #p.imshow(n.log10(sense2d).T,interpolation='nearest',aspect='auto',vmax=4,extent=[0,0.3,n.min(kpls),n.max(kpls)])
    #p.imshow(n.log10(1./sense2d**2).T,interpolation='nearest',aspect='auto',vmin=-1,extent=[0,0.3,n.min(kpls),n.max(kpls)])
    #p.colorbar()
    #n.savez('weights.npz',kprs=kprs,pdf=pdf,cdf=cdf)
    p.show()
    #sys.exit()

        
#for kpr in sense.keys():
#    print sense[kpr]

#bin in kmag
#delta = 0.06322 #resolution of 8 MHz band
delta = C.pspec.dk_deta(z)*(1./B)
kmag = n.arange(delta,n.max(mk),delta)
#kmag = n.array(josh_bins)
#print kmag


allks = n.array(allks)
kprs = n.array(kprs)
#print n.min(n.array(allks))
if False:
    p.subplot(121)
    khist = p.hist(allks,bins=kmag)
    dbin = (khist[1][1] - khist[1][0])/2
    p.subplot(122)
    p.semilogy(khist[1][:-1],khist[0].astype(float)**-2)
    p.show()
if False:
    khist = p.hist(kprs,bins=20)
    p.show()

sense1d = n.zeros_like(kmag)
Tsense1d = n.zeros_like(kmag)
for ind, kpr in enumerate(sense.keys()):
    #if ind == 294: print ind, sense[kpr]
    sense[kpr] = sense[kpr]**-.5 / n.sqrt(N_2HR_LSTBINS)
    Tsense[kpr] = Tsense[kpr]**-.5 / n.sqrt(N_2HR_LSTBINS)
    #if ind == 294: print ind, sense[kpr]
    for i, kpl in enumerate(kpls):
        #print kpl
        k = n.sqrt(kpl**2 + kpr**2)
        mu = kpl/k
        #sense[kpr][i] /= rsd(mu)
        if k > n.max(mk): continue
        if False:
            roundk = n.round(k/delta) * delta
            if roundk == 0: continue
            sense1d[n.where(near(kmag,roundk))] += 1./sense[kpr][i]**2
            Tsense1d[n.where(near(kmag,roundk))] += 1./Tsense[kpr][i]**2
        else:
            sense1d[find_nearest(kmag,k)] += 1./sense[kpr][i]**2
            Tsense1d[find_nearest(kmag,k)] += 1./Tsense[kpr][i]**2
        #if n.where(near(kmag,roundk))[0] == [0]: print k, roundk, n.where(near(kmag,roundk))

for ind,kbin in enumerate(sense1d):
    sense1d[ind] = kbin**-.5
    Tsense1d[ind] = Tsense1d[ind]**-.5

#print kmag, sense1d
#p.plot(kmag,Tsense1d); p.show()

ax = p.subplot(111)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
p.plot(mk, mdelta2, 'gray', linewidth=6)
p.errorbar(kmag,p21(kmag), yerr=sense1d,fmt='o',color='b')
#p.errorbar(kmag,p21(kmag), yerr=Tsense1d,fmt='o',color='r')
fg = n.where(sense1d == n.inf)
#p.plot(kmag[fg], p21(kmag[fg]),marker='o',ls='', color='k')
#p.plot(kmag,Tsense1d,ls='steps--',color='gray')
p.ylabel(r'$\Delta^2(k)\ [{\rm mK}^2]$', size=18)
p.xlabel(r'$k\ [h\ {\rm Mpc}^{-1}]$', size=18)
ax.set_xlim(.05, 1.)
ax.set_xlim(n.min(kmag), 1.)
#ax.set_ylim(4,1e2)
ax.set_ylim(10,50)
#ax.yaxis.set_minor_formatter(FormatStrFormatter('%1.f'))
#ax.set_yticks([10,20,30,40,50])
#ax.set_yticklabels([10,20,30,40,50])

if model in ['mid','pess']:
    pkmag = kmag
    psense1d = sense1d.copy()
    psense1d[fg] = 500 * n.ones_like(psense1d[fg])
if model in ['opt']:
    valid = n.where(kmag > .2,1,0)
    invalid = n.where(kmag < .2,1,0)

    pkmag = kmag.compress(valid)
    psense1d = sense1d.compress(valid)

    ikmag = kmag.compress(invalid) 
    isense1d = sense1d.compress(invalid)

    sense_interp = interpolate.interp1d(ikmag, isense1d, kind='linear')
    interp_range = n.arange(n.min(ikmag),n.max(ikmag),.01)
    pkmag = n.append(interp_range,pkmag)
    psense1d = n.append(sense_interp(interp_range),psense1d)
ax.fill_between(pkmag,p21(pkmag)+psense1d,p21(pkmag)-psense1d,color='gray',alpha=.33)
p.grid()
p.title(mod_2_title[model],size=18)

sign = 1
for ind, k in enumerate(kmag):
    if k > 1: continue
    pk = p21(k) #* mean_temp(z)**2 * k**3 * (2*n.pi**2.)**-1
    if sign == 1: yoffset = 0.075
    else: yoffset = 0.05
    dx1, dy1 = 1 - 0.08, 1 - sign*(yoffset+0.075)
    dx2, dy2 = 1 - 0.08, 1 - sign*(yoffset)
    if True:
        sigma = pk / sense1d[ind]
        if dy2*(pk - sign*sense1d[ind]) < 10: continue
        p.text(dx2*k, dy2*(pk - sign*sense1d[ind]), '%.1f' % sigma, color='k')
    else:
        sigma1 = sense1d[ind]
        sigma2 = Tsense1d[ind]
        if dy2*(pk - sign*sense1d[ind]) < 10: continue
        if dy2*(pk - sign*sense1d[ind]) > 50: continue
        p.text(dx1*k , dy1*(pk - sign*sense1d[ind]), '%.1f' % (sigma1-sigma2),color='b')
        p.text(dx2*k , dy2*(pk - sign*sense1d[ind]), '%.1f' % sigma2,color='r')
        sign *= -1 

p.show()

n.savez('%s_%s_%.3f.npz' % (array,model,fq),ks=kmag,errs=sense1d,T_errs=Tsense1d)

if True:
    ks,sense = kmag, sense1d
    #least squares to estimate significance
    A = p21(ks) #* ks**3 * (2.*n.pi**2)**-1
    M = p21(ks) #* ks**3 * mean_temp(z)**2 * (2.*n.pi**2)**-1
    #sense *=10
    wA, wM = A * (1./sense), M * (1./sense)
    #print wA, wM
    wA, wM = n.matrix(wA).T, n.matrix(wM).T
    amp = (wA.T*wA).I * (wA.T * wM)
    #errorbars
    Y = n.float(amp) * wA
    dY = wM - Y
    s2 = (len(wM)-1)**-1 * (dY.T * dY)
    X = n.matrix(wA).T * n.matrix(wA)
    #print X
    #err = s2*(1./n.float(X))
    err = n.sqrt((1./n.float(X)))
    print 'snr = ', amp/err
