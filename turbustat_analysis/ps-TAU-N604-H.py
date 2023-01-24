#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import json
from astropy.io import fits
from astropy.utils.misc import JsonCustomEncoder
import astropy.units as u
from matplotlib import pyplot as plt
import seaborn as sns
import sys
import turbustat.statistics as tss
from turbustat.statistics import PowerSpectrum
from turbustat.io.sim_tools import create_fits_hdu
import bfunc





datapath_obs= Path(open("path-observations.txt", "r").read()).expanduser()
datapath_data = Path(open("path-results.txt", "r").read()).expanduser()


name = 'TAU-N604-H'
name_file = 'TAURUS-604-Ha-RV-mod.fits'
distance = 840000 #parsecs
pix = 0.26 #arcsec 


data = json.load(open(str(datapath_data) + '/' + name + ".json"))
rad_vel =fits.open(datapath_obs / name_file)


rad_vel.info()


hdr = rad_vel[0].header


hdr ['CDELT1'] = (-pix / (60*60), '[deg] Coordinate increment at reference point')
hdr ['CDELT2'] = (pix / (60*60), '[deg] Coordinate increment at reference point')
hdr['CUNIT1']  = ('deg' , 'Units of coordinate increment and value' )      
hdr['CUNIT2']  = ('deg' , 'Units of coordinate increment and value'  )
hdr['CTYPE1']  = ('RA---CAR', 'Right ascension, plate caree projection  ')
hdr['CTYPE2']  = ('DEC--CAR', 'Declination, plate caree projection   ')


hdr


rad_vel.info()


##nan values to mean velocity values
vmed = np.nanmedian(rad_vel[0].data)
m = np.isfinite(rad_vel[0].data)
rad_vel[0].data[~m] = vmed


vv = rad_vel[0].data.astype(float)
##load  thecorrelation length and seeing derived from the fit
r0 = data["results_2sig"]['r0'][0] #pc
s0 = data["results_2sig"]['s0'][0] #pc
m = data["results_2sig"]['m'][0] #-
sig2 = data["results_2sig"]['sig2'][0] #km^2/s^2
noise = data["results_2sig"]['noise'][0] #km^2/s^2
box_size = data["properties"]['box_size']
r0,s0,m,sig2,box_size


data["properties"]['box_size']



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

sns.heatmap(vv, cmap="RdBu_r",cbar_kws={'label': 'km/s'})
ax.set_facecolor('xkcd:gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')


##img_hdu = create_fits_hdu(img, pixel_scale, beamfwhm, imshape, restfreq, bunit)
#img_hdu = create_fits_hdu(vv,pix*u.arcsec,1 * u.arcsec, vv.shape, 1 * u.Hz, u.K)
#img_hdu.header


# Spatial Power Spectrum

#pspec = PowerSpectrum(img_hdu, distance=distance* u.pc) 


pspec = PowerSpectrum(vv, header = hdr, distance=distance* u.pc) 
#pspec.run(verbose=True, xunit=u.pc**-1)
pspec.run(verbose=True, xunit=u.pc**-1, low_cut=(r0*u.pc)**-1, high_cut=(s0*u.pc)**-1)


pspec.slope


(r0*u.pc)**-1,(s0*u.pc)**-1,0.01*(u.pc)**-1


np.log10(1/s0)*(u.pc)**-1,-1.75*(u.pc)**-1,-1.0*(u.pc)**-1


pspec = PowerSpectrum(vv, header = hdr, distance=distance * u.pc) 

pspec.run(verbose=True, xunit=(u.pc)**-1, low_cut=0.01*(u.pc)**-1, high_cut=(1/s0)*(u.pc)**-1,
          fit_kwargs={'brk': (1/r0)*(u.pc)**-1, 'log_break': False}, fit_2D=False)  


# Delta-Variance

dvar = tss.DeltaVariance(vv, header = hdr, distance=distance* u.pc,nlags=50)


plt.figure(figsize=(14, 8))
dvar.run(verbose=True, boundary="fill",xunit=u.pc, xlow=s0*u.pc, xhigh=r0*u.pc)


#sns.set_context("talk", font_scale=1.1)
#plt.style.use(["seaborn-poster",])


fig, (ax) = plt.subplots(
    1,
    1,
    sharey=False,
    figsize=(10, 10),
)

##spatial power spectra
ax.scatter(pspec.freqs,pspec.ps1D)

#yy1 = pspec.ps1D+pspec.ps1D_stddev
#yy2 = pspec.ps1D-pspec.ps1D_stddev
#ax.fill_between(pspec.freqs, yy1, yy2, alpha = 0.15, zorder = 0, color = 'b')

ax.axvline(1/r0, c="b")
ax.axvline(1/s0, c="k")

#xgrid = np.linspace(1/r0,1/s0, 200)
#ax.plot(xgrid, (10**4.6)*xgrid**pspec.slope, '-', c="k")

ax.set(xscale='log', yscale='log', 
       xlabel='log spatial frequency $k$,1/pc',
       ylabel=r'log $P(k)_2,\ \mathrm{-}$'
      )


fig, (axx) = plt.subplots(
    1,
    1,
    sharey=False,
    figsize=(10, 10),
)

##delta-variance
axx.scatter(dvar.lags,dvar.delta_var,alpha = 0.75, color = 'k', zorder = 0)
axx.plot(dvar.lags,dvar.delta_var,alpha = 0.75, color = 'k', zorder = 0)
yy1 = dvar.delta_var+dvar.delta_var_error
yy2 = dvar.delta_var-dvar.delta_var_error
axx.fill_between(dvar.lags, yy1, yy2, alpha = 0.15, zorder = 0, color = 'k')

##detlta-fit
xgrid = np.linspace(s0,r0,100)
axx.plot(xgrid,(10**(-0.86))*(xgrid**dvar.slope), color = 'r', alpha = 0.75, linewidth = 2.5)


##observational structure function
b_sigo = np.array(data['B'])/data['preres']['sig2']
b_sigd = np.array(data['B'])/sig2
b_m = ( b_sigo + b_sigd ) / 2
axx.scatter(data['r'],b_m, color = 'b', alpha = 0.75)
axx.fill_between(data['r'], b_sigo , b_sigd, alpha = 0.15, zorder = 0, color = 'b')


##model structure function
rgrid = np.linspace(np.array(data['r']).min(),np.array(data['r']).max(),100)
axx.plot(rgrid, bfunc.bfunc04s(rgrid, r0, sig2, m, s0, noise, box_size)/sig2, color="orange",  linewidth = 2.5)


axx.set(xscale='log', yscale='log', 
       xlabel='log lags,pc',
       ylabel=r'log $\Delta, \mathrm{-}$'
      )

axx.axvline(r0, c="k", linestyle = '--')
axx.axvline(s0, c="k", linestyle = ':')


dvar


get_ipython().system('jupyter nbconvert --to script --no-prompt ps-TAU-N604-H.ipynb')

