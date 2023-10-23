'''Takes input as *.npz file and shows a plot of Sensitivity data. Do not run before you have already run mk_array_file.py and subsequent calc_sense.py'''

import numpy as n
import optparse, sys
from matplotlib import pyplot as plt

o = optparse.OptionParser()
o.set_usage('plot_sense.py [options] *.npz')
o.set_description(__doc__)
opts, args = o.parse_args(sys.argv[1:])

#Load in data from array file

array = n.load(args[0])
k_s = array['ks']
sense = array['errs']
Tsense = array['T_errs']
P21 = array['ps21']
	
#Write array to csv file

l = len(k_s)
with open('sensitivity.csv','w+') as f:
    f.write('k, Power Spectrum (interpolated), Full Sensitivity, Sensitivity [Thermal Noise Only]\n')
    for i in xrange(l):
        f.write('%f, %f, %f, %f\n' % (k_s[i], P21[i], sense[i], Tsense[i]))

#Plotting

plt.figure(figsize = (12,4))

plt.plot(k_s, sense, 'y--', drawstyle = 'steps', label = 'Full Sensitivity')
plt.plot(k_s, Tsense, 'r:', drawstyle = 'steps', label = 'Sensitivity [Thermal Noise Only]')
plt.plot(k_s, P21, 'c-', label = r'$\Delta^{2}_{21}(k)$')
plt.xscale('log')
plt.yscale('log')
plt.title('Sensitivity')
plt.xlabel(r'$k\ [h\ Mpc^{-1}]$')
plt.ylabel(r'$\delta\Delta^{2}(k)\ [mK^{2}]$')
plt.legend()
plt.savefig('sensitivity_plot.png', dpi = 600)

plt.show()
