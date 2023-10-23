import aipy as a, numpy as n, os
import csv

class AntennaArray(a.pol.AntennaArray):
    def __init__(self, *args, **kwargs):
        a.pol.AntennaArray.__init__(self, *args, **kwargs)
        self.array_params = {}
    def get_ant_params(self, ant_prms={'*':'*'}):
        prms = a.fit.AntennaArray.get_params(self, ant_prms)
        for k in ant_prms:
            top_pos = n.dot(self._eq2zen, self[int(k)].pos)
            if ant_prms[k] == '*':
                prms[k].update({'top_x':top_pos[0], 'top_y':top_pos[1], 'top_z':top_pos[2]})
            else:
                for val in ant_prms[k]:
                    if   val == 'top_x': prms[k]['top_x'] = top_pos[0]
                    elif val == 'top_y': prms[k]['top_y'] = top_pos[1]
                    elif val == 'top_z': prms[k]['top_z'] = top_pos[2]
        return prms
    def set_ant_params(self, prms):
        changed = a.fit.AntennaArray.set_params(self, prms)
        for i, ant in enumerate(self):
            ant_changed = False
            top_pos = n.dot(self._eq2zen, ant.pos)
            try:
                top_pos[0] = prms[str(i)]['top_x']
                ant_changed = True
            except(KeyError): pass
            try:
                top_pos[1] = prms[str(i)]['top_y']
                ant_changed = True
            except(KeyError): pass
            try:
                top_pos[2] = prms[str(i)]['top_z']
                ant_changed = True
            except(KeyError): pass
            if ant_changed: ant.pos = n.dot(n.linalg.inv(self._eq2zen), top_pos)
            changed |= ant_changed
        return changed 
    def get_arr_params(self):
        return self.array_params
    def set_arr_params(self, prms):
        for param in prms:
            self.array_params[param] = prms[param]
            if param == 'dish_size_in_lambda':
                FWHM = 1.037*(1/prms[param]) #FWHM in radians = 1.03*(1/dish_size_in_lambda)
                self.array_params['obs_duration'] = 24*60.*FWHM / (2*a.const.pi) # minutes it takes the sky to drift through beam FWHM = 24*60('minutes in a day')*FWHM('in radians')/(2*pi)('radians covered by sky in a day')
            if param == 'antpos':
                bl_lens = n.sum(n.array(prms[param])**2,axis=1)**.5
        return self.array_params

#===========================ARRAY SPECIFIC PARAMETERS==========================

#Set antenna positions here; antpos should just be a list of [x,y,z] coords in light-nanoseconds
x = []
y = []
z = []
antpos = [] #initialize arrays

with open('calibration_files/antpos_for_other_arrays/mwa_compact_antpos.csv') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
	x.append(row[4])
        y.append(row[5])
        z.append(row[3])

for i in xrange(len(x)): #to convert string values to float.
    if i > 0:
        ax = float(x[i]) * 1e2 / a.const.len_ns
	by = float(y[i]) * 1e2 / a.const.len_ns
	cz = float(z[i]) * 1e2 / a.const.len_ns
	antpos.append((ax, by, cz))

#Set other array parameters here
prms = {
    'name': os.path.basename(__file__)[:-3], #remove .py from filename
    'loc': ('-26:42:11.95',  '116:40:14.93'), #The MWA is located within the Murchison Radio-astronomy Observatory (MRO) in the Shire of Murchison in Western Australia (WA).
    'antpos': antpos,
    'beam': a.fit.Beam2DGaussian,
    'dish_size_in_lambda': 4.0 * (150.0*1e6) / (a.const.c*1e-2), #in units of wavelengths at 150 MHz; this will also define the observation duration
    'Trx': 1e5 #receiver temp in mK
}

#=======================END ARRAY SPECIFIC PARAMETERS==========================

def get_aa(freqs):
    '''Return the AntennaArray to be used for simulation.'''
    location = prms['loc']
    antennas = []
    nants = len(prms['antpos'])
    for i in range(nants):
        beam = prms['beam'](freqs, xwidth=(0.45/prms['dish_size_in_lambda']), ywidth=(0.45/prms['dish_size_in_lambda'])) #as it stands, the size of the beam as defined here is not actually used anywhere in this package, but is a necessary parameter for the aipy Beam2DGaussian object
        antennas.append(a.fit.Antenna(0, 0, 0, beam))
    aa = AntennaArray(prms['loc'], antennas)
    p = {}
    for i in range(nants):
        top_pos = prms['antpos'][i]
        p[str(i)] = {'top_x':top_pos[0], 'top_y':top_pos[1], 'top_z':top_pos[2]}
    aa.set_ant_params(p)
    aa.set_arr_params(prms) 
    return aa

def get_catalog(*args, **kwargs): return a.src.get_catalog(*args, **kwargs)
