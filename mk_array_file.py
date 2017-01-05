#! /usr/bin/env python
'''
Creates an array file for use by calc_sense.py.  The main product is the uv coverage produced by the array during the time it takes the sky to drift through the primary beam; other array parameters are also saved.  Array specific information comes from an aipy cal file.  If opts.track is set, produces the uv coverage
for the length specified instead of that set by the primary beam.'''
import aipy as a, numpy as n
import optparse, sys

o = optparse.OptionParser()
o.set_usage('mk_array_file.py -C [calfile]')
a.scripting.add_standard_options(o,cal=True)
o.add_option('--track', dest='track', default=None, type=float,
    help="If set, calculate sensitivity for a tracked observation of this duration in hours; otherwise, calculate for a drift scan.")
o.add_option('--bl_min', dest='bl_min', default=0., type=float,
    help="Set the minimum baseline (in meters) to include in the uv plane.")
o.add_option('--bl_max', dest='bl_max', default=None, type=float,
    help="Set the maximum baseline (in meters) to include in the uv plane.  Use to exclude outriggers with little EoR sensitivity to speed up calculation.")
o.add_option('-f', '--freq', dest='freq', default=0.135, type=float,
    help="The central frequency of the observation in GHz.  Default is 0.135 GHz, corresponding to z = 9.5, which matches the default model power spectrum used in calc_sense.py.")
opts, args = o.parse_args(sys.argv[1:])

#============================SIMPLE GRIDDING FUNCTION=======================

def beamgridder(xcen,ycen,size):
    crds = n.mgrid[0:size,0:size]
    cen = size/2 - 0.5 # correction for centering
    xcen += cen
    ycen = -1*ycen + cen
    beam = n.zeros((size,size))
    if round(ycen) > size - 1 or round(xcen) > size - 1 or ycen < 0. or xcen <0.: 
        return beam
    else:
        beam[round(ycen),round(xcen)] = 1. #single pixel gridder
        return beam

#==============================READ ARRAY PARAMETERS=========================

#load cal file
aa = a.cal.get_aa(opts.cal,n.array([.150]))
nants = len(aa)
prms = aa.get_arr_params()
if opts.track:
    obs_duration=60.*opts.track
    name = prms['name']+'track_%.1fhr' % opts.track
else:
    obs_duration = prms['obs_duration']*(0.15/opts.freq) #scales observing time linearly with frequency to account for change in beam FWHM
    name = prms['name']+'drift'; print name
dish_size_in_lambda = prms['dish_size_in_lambda']

#==========================FIDUCIAL OBSERVATION PARAMETERS===================

#while poor form to hard code these to arbitrary values, they have very little effect on the end result

#observing time
t_int = 60. #how many seconds a single visibility has integrated
cen_jd = 2454600.90911
start_jd = cen_jd - (1./24)*((obs_duration/t_int)/2)
end_jd = cen_jd + (1./24)*(((obs_duration-1)/t_int)/2)
times = n.arange(start_jd,end_jd,(1./24/t_int))
print 'Observation duration:', start_jd, end_jd

ref_fq = .150

#================================MAIN CODE===================================

cnt = 0
uvbins = {}

cat = a.src.get_catalog(opts.cal,'z') #create zenith source object
aa.set_jultime(cen_jd)
obs_lst = aa.sidereal_time()
obs_zen = a.phs.RadioFixedBody(obs_lst,aa.lat)
obs_zen.compute(aa) #observation is phased to zenith of the center time of the drift 

#find redundant baselines
bl_len_min = opts.bl_min / (a.const.c/(ref_fq*1e11)) #converts meters to lambda
bl_len_max = 0.
for i in xrange(nants):
    print 'working on antenna %i of %i' % (i, len(aa))
    for j in xrange(nants):
        if i == j: continue #no autocorrelations
        u,v,w = aa.gen_uvw(i,j,src=obs_zen)
        bl_len = n.sqrt(u**2 + v**2)
        if bl_len > bl_len_max: bl_len_max = bl_len
        if bl_len < bl_len_min: continue
        uvbin = '%.1f,%.1f' % (u,v)
        cnt +=1
        if not uvbins.has_key(uvbin): uvbins[uvbin] = ['%i,%i' % (i,j)]
        else: uvbins[uvbin].append('%i,%i' % (i,j))
print 'There are %i baseline types' % len(uvbins.keys())

print 'The longest baseline is %.2f meters' % (bl_len_max*(a.const.c/(ref_fq*1e11))) #1e11 converts from GHz to cm
if opts.bl_max: 
    bl_len_max = opts.bl_max / (a.const.c/(ref_fq*1e11)) #units of wavelength
    print 'The longest baseline being included is %.2f m' % (bl_len_max*(a.const.c/(ref_fq*1e11)))

#grid each baseline type into uv plane
dim = n.round(bl_len_max/dish_size_in_lambda)*2 + 1 # round to nearest odd
uvsum,quadsum = n.zeros((dim,dim)), n.zeros((dim,dim)) #quadsum adds all non-instantaneously-redundant baselines incoherently
for cnt, uvbin in enumerate(uvbins):
    print 'working on %i of %i uvbins' % (cnt+1, len(uvbins))
    uvplane = n.zeros((dim,dim))
    for t in times:
        aa.set_jultime(t)
        lst = aa.sidereal_time()
        obs_zen.compute(aa)
        bl = uvbins[uvbin][0]
        nbls = len(uvbins[uvbin])
        i, j = bl.split(',')
        i, j = int(i), int(j)
        u,v,w = aa.gen_uvw(i,j,src=obs_zen)
        _beam = beamgridder(xcen=u/dish_size_in_lambda,ycen=v/dish_size_in_lambda,size=dim)
        uvplane += nbls*_beam
        uvsum += nbls*_beam
    quadsum += (uvplane)**2

quadsum = quadsum**.5

print "Saving file as %s_blmin%0.f_blmax%0.f_%.3fGHz_arrayfile.npz" % (name, bl_len_min, bl_len_max, opts.freq) 

n.savez('%s_blmin%0.f_blmax%0.f_%.3fGHz_arrayfile.npz' % (name, bl_len_min, bl_len_max, opts.freq),
uv_coverage = uvsum,
uv_coverage_pess = quadsum,
name = name,
obs_duration = obs_duration,
dish_size_in_lambda = dish_size_in_lambda,
Trx = prms['Trx'],
t_int = t_int,
freq=opts.freq
)
