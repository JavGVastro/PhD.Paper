#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()


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
import pandas as pd
from rebin_utils import downsample, oversample
from astropy.modeling import models, fitting
import statsmodels.api as sm
from scipy.stats import linregress
fitter = fitting.LevMarLSQFitter()


#Info for images and exports
name_plt = 'Orion core'
name_exp = 'Orion'
element = 'N'
em_line = '[NII]'


datapath_res = Path(open("path-results.txt", "r").read()).expanduser()


name_data = 'KPech-Orion-N' 
distance = 440 #pc
pix = 0.534 #arcsec/pix


pc = distance*(2*np.pi) / (360 * 60 * 60) #arcsec to parsecs
corr = pix*pc 
corr


data = json.load(open(str(datapath_res) + '/' + name_data + ".json"))
sb = np.array(data['observations']["sb"])
vv = np.array(data['observations']["vv"])
#ss = np.array(data['observations']["ss"])


## Replace spurious values in the arrays
mm = ~np.isfinite(sb*vv) | (sb < 0.0)

sb[mm] = 0.0
vv[mm] = np.nanmean(vv)
#ss[m] = 0.0
sb /= sb.max()

good = (~mm) & (sb > 0.001)


trim = (slice(0, 550), slice(0, 350))
vv = vv[trim]
sb = sb[trim]


fig, ax = plt.subplots(figsize=(12, 12))



plt.imshow(sb, cmap='magma')

cbar = plt.colorbar()
#cbar.set_label('km/s', rotation=270, labelpad=15)  

ax.set_xlabel('X')
ax.set_ylabel('Y')


plt.gca().invert_yaxis()


fig, ax = plt.subplots(figsize=(12, 12))

plt.imshow(vv, cmap='RdBu_r')

cbar = plt.colorbar()
cbar.set_label('km/s', rotation=270, labelpad=15)  

ax.set_xlabel('X')
ax.set_ylabel('Y')


plt.gca().invert_yaxis()


##Open results.json file
##Path and name
##Load results
data = json.load(open(str(datapath_res) + '/' + name_data + ".json"))
##load  parameters derived from the fit
r0 = data["results_2sig"]['r0'][0] #pc
s0 = data["results_2sig"]['s0'][0] #pc
m = data["results_2sig"]['m'][0] #-
mer = data["results_2sig"]['m'][1] #-
sig2 = data["results_2sig"]['sig2'][0] #km^2/s^2
noise = data["results_2sig"]['noise'][0] #km^2/s^2
box_size = data["properties"]['box_size']
r0,s0,m,mer,sig2,box_size


# mf = ~np.isfinite(sb) | (sb < 0.0)
# sb[mf] = 0.0
# sb /= sb.max()
# sb /= np.nanmean(sb)

# ##nan values to mean velocity values
# vmed = np.nanmedian(vv)
# mv = np.isfinite(vv)
# vv[~mv] = vmed

new_hdul = fits.HDUList()
new_hdul.append(fits.PrimaryHDU())
new_hdul.append(fits.ImageHDU(sb))
new_hdul.append(fits.ImageHDU(vv))


hdr = new_hdul[0].header


hdr ['CDELT1'] = (-pix / (60*60), '[deg] Coordinate increment at reference point')
hdr ['CDELT2'] = (pix / (60*60), '[deg] Coordinate increment at reference point')
hdr['CUNIT1']  = ('deg' , 'Units of coordinate increment and value' )      
hdr['CUNIT2']  = ('deg' , 'Units of coordinate increment and value'  )
hdr['CTYPE1']  = ('RA---CAR', 'Right ascension, plate caree projection  ')
hdr['CTYPE2']  = ('DEC--CAR', 'Declination, plate caree projection   ')
hdr['targname']  = ('Orion', 'Target name   ')
hdr['distance']  = (distance, 'Distance to target   ')
hdr['pix'] = (pix, 'arcsec.pixel^{-1}')


sb = new_hdul[1].data.astype(float)
vv = new_hdul[2].data.astype(float)


new_hdul.info()


distance


##Tutorial data
##img_hdu = create_fits_hdu(img, pixel_scale, beamfwhm, imshape, restfreq, bunit)
#img_hdu = create_fits_hdu(vv,pix*u.arcsec,1 * u.arcsec, vv.shape, 1 * u.Hz, u.K)
#img_hdu.header
#pspec = PowerSpectrum(img_hdu, distance=distance* u.pc) 


# Spatial Power Spectrum

plt.figure(figsize=(14, 8))
pspec = PowerSpectrum(vv, header = hdr, distance = distance* u.pc) 
pspec.run(verbose=True, xunit = u.pc**-1, low_cut=(r0*u.pc)**-1, high_cut=(s0*u.pc)**-1)


from turbustat.statistics.apodizing_kernels import    (CosineBellWindow, TukeyWindow, HanningWindow, SplitCosineBellWindow)

taper = HanningWindow()
taper3 = SplitCosineBellWindow(alpha=0.85, beta=0.55)
shape = (550, 350)
tap = taper3(shape)
plt.imshow(tap, cmap='viridis', origin='lower')  


##NOTE 1: IDK why but the fit is done in pixel units despite introducing xunit as parsec. To compensate for this I need 
##to introduce a corr factor. To kinda avoid this I repeat the the WLS using the correct 'x' units and using 
##the derived new fit parameters in the comparison plots. 
##NOTE 2: this is no problem for NGC 604 since the correction factor is similar to 1 since 
## 0.26 (arcsec/pix) * 4.07 (pc/arsec)= 1.05

plt.figure(figsize=(14, 8))
pspec = PowerSpectrum(vv, header = hdr, distance = distance* u.pc) 
pspec.run(verbose=True, xunit = u.pc**-1, low_cut=(0.5*r0*u.pc)**-1, high_cut=(1.25*s0*u.pc)**-1,
           apodize_kernel='splitcosinebell', alpha=0.85, beta=0.55)


##NOTE 1: IDK why but the fit is done in pixel units despite introducing xunit as parsec. To compensate for this I need 
##to introduce a corr factor. To kinda avoid this I repeat the the WLS using the correct 'x' units and using 
##the derived new fit parameters in the comparison plots. 
##NOTE 2: this is no problem for NGC 604 since the correction factor is similar to 1 since 
## 0.26 (arcsec/pix) * 4.07 (pc/arsec)= 1.05

plt.figure(figsize=(14, 8))
pspec1 = PowerSpectrum(vv, header = hdr, distance=distance * u.pc) 
pspec1.run(verbose=True, xunit=(u.pc)**-1, low_cut=5*(u.pc)**-1, high_cut=(1/s0)*(u.pc)**-1,
          fit_kwargs={'brk': (1/r0)*(u.pc)**-1, 'log_break': False}, fit_2D=False)  


# Delta-Variance

##NOTE 1: IDK why but the fit is done in pixel units despite introducing xunit as parsec. To compensate for this I need 
##to introduce a corr factor. To kinda avoid this I repeat the the WLS using the correct 'x' units and using 
##the derived new fit parameters in the comparison plots. 
##NOTE 2: this is no problem for NGC 604 since the correction factor is similar to 1 since 
## 0.26 (arcsec/pix) * 4.07 (pc/arsec)

dvar = tss.DeltaVariance(vv, header = hdr, distance=distance* u.pc,nlags=50)
plt.figure(figsize=(8, 8))
dvar.run(verbose=True, boundary="fill",xunit=u.pc, xlow=s0*u.pc, xhigh=r0*u.pc)


##Plots
sns.set_context("talk", font_scale=1.1)
#plt.style.use(["seaborn-poster",])


x = np.array(pspec.freqs*(corr**-1))
y = np.array(pspec.ps1D)
y_er = np.array(pspec.ps1D_stddev)

log_x = np.log10(x)
log_y = np.log10(y)
log_y_er = np.log10(y_er)


i = 10
f = len(pspec.ps1D)-1
#f = 36


intr=linregress(x[i:f], y[i:f])
intr.slope,intr.stderr,intr.intercept,intr.rvalue


intrl=linregress(log_x[i:f], log_y[i:f])
intrl.slope,intrl.stderr,intrl.intercept,intrl.rvalue


fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(x,y)
#ax.plot(x,y)

xgrid = np.linspace(x[i],x[f])
#ax.plot(xgrid, xgrid*intr.slope + intr.intercept, color = 'k')
ax.plot(xgrid, 10**(intrl.intercept)*(xgrid**intrl.slope), color ='r', marker = '.')


ax.set(xscale = 'log', yscale = 'log')


x,y,z=log_x[i:f],log_y[i:f],log_y_er[i:f]
X = sm.add_constant(x)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


x,y,z=log_x[i:f],log_y[i:f],log_y_er[i:f]
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=z)
resultsw = model.fit()
print(resultsw.summary())








fig, (ax) = plt.subplots(
    1,
    1,
    sharey=False,
    figsize=(10, 10),
)

##spatial power spectra
ax.scatter(pspec.freqs*(corr**-1),pspec.ps1D, color = 'k',label = 'turbustat')

##spatial power spectra errors
yy1 = pspec.ps1D+pspec.ps1D_stddev
yy2 = np.sqrt((pspec.ps1D-pspec.ps1D_stddev)**2)
ax.fill_between(pspec.freqs*(corr**-1), yy1, yy2, alpha = 0.15, zorder = 0, color = 'k')

##ps-fit
xgrid = np.logspace(np.log10(1/r0),np.log10(1/s0),100)
ax.plot(xgrid,(10**(results.params[0]))*(xgrid**results.params[1]), color = 'r', alpha = 0.85, linewidth = 2.5)

##seeing and corelation length
ax.axvline(1/r0, c="k", linestyle = '--')
ax.axvline(1/s0, c="k", linestyle = ':')

##annotations
ax.text(.1, .10,'m$_{ps}$ =' + str(np.round(results.params[1],2)) + '$\pm$' + str(np.round(results.bse[1],2)), transform=ax.transAxes)

ax.annotate('r$_0$', xy=(1/r0, 1e8),  xycoords='data',
           xytext=(0.5, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.02),
            horizontalalignment='right', verticalalignment='top',
            )

ax.annotate('s$_0$', xy=(1/s0, 1e8),  xycoords='data',
            xytext=(0.90, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.02),
            horizontalalignment='right', verticalalignment='top',
            )

plt.title(name_plt + ' '+ em_line)
plt.legend(loc = 0)

##config
ax.set(xscale='log', yscale='log', 
       xlabel='log spatial frequency $k$,1/pc',
       ylabel=r'log $P(k)_2,\ \mathrm{-}$'
      )

plt.savefig('Imgs//'+ 'ps_'+ name_exp + element +  '.pdf', bbox_inches='tight')




x = np.array(dvar.lags*corr)
#x = np.array(dvar.lags)
y = np.array(dvar.delta_var)
y_er = np.array(dvar.delta_var_error)

log_x = np.log10(x)
log_y = np.log10(y)
log_y_er = np.log10(y_er)


i = 0
#f = len(dvar.lags)-1
f = 36


intr=linregress(x[i:f], y[i:f])
intr.slope,intr.stderr,intr.intercept,intr.rvalue


intrl=linregress(log_x[i:f], log_y[i:f])
intrl.slope,intrl.stderr,intrl.intercept,intrl.rvalue


fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(x,y)
#ax.plot(x,y)

xgrid = np.linspace(x[i],x[f])
ax.plot(xgrid, xgrid*intr.slope + intr.intercept, color = 'k')
ax.plot(xgrid, 10**(intrl.intercept)*(xgrid**intrl.slope), color ='r')


ax.set(xscale = 'log', yscale = 'log')





x,y,z=log_x[i:f],log_y[i:f],log_y_er[i:f]
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())





fig, (axx) = plt.subplots(
    1,
    1,
    sharey=False,
    figsize=(10, 10),
)

##delta-variance
axx.scatter(dvar.lags*corr,dvar.delta_var,alpha = 0.75, color = 'k', zorder = 0, label = 'turbustat')
axx.plot(dvar.lags*corr,dvar.delta_var,alpha = 0.75, color = 'k', zorder = 0)
yy1 = dvar.delta_var+dvar.delta_var_error
yy2 = dvar.delta_var-dvar.delta_var_error
axx.fill_between(dvar.lags*corr, yy1, yy2, alpha = 0.15, zorder = 0, color = 'k')

##delta-fit
xgrid = np.linspace(s0,r0,100)
#axx.plot(xgrid,10**(-1.54)*(xgrid**dvar.slope), color = 'r', alpha = 0.75, linewidth = 2.5)
axx.plot(xgrid, 10**(results.params[0])*(xgrid**results.params[1]), color ='r')


#xgrid = np.logspace(np.log10(s0),np.log10(r0),100)
#axx.plot(xgrid,-1.54+(xgrid*dvar.slope), color = 'r', alpha = 0.75, linewidth = 2.5)


##observational structure function: average between sig obs and sig derived normalized
b_sigo = np.array(data['B'])/data['preres']['sig2']
b_sigd = np.array(data['B'])/sig2
b_m = ( b_sigo + b_sigd ) / 2
axx.scatter(data['r'],b_m, color = 'b', alpha = 0.75, marker = 'o')
axx.fill_between(data['r'], b_sigo , b_sigd, alpha = 0.15, zorder = 0, color = 'b')


##model structure function
rgrid = np.linspace(np.array(data['r']).min(),np.array(data['r']).max(),100)
axx.plot(rgrid, bfunc.bfunc00s(rgrid, r0, sig2, m)/sig2, color="green",  linewidth = 2.5)
axx.plot(rgrid, bfunc.bfunc04s(rgrid, r0, sig2, m, s0, noise, box_size)/sig2, color="orange",  linewidth = 2.5)

##annotations
axx.text(.65, .10,'m$_{Δv}$ =' + str(np.round(results.params[1],2)) + '$\pm$' + str(np.round(results.bse[1],2)), transform=ax.transAxes)
axx.text(.65, .15,'m =' + str(np.round(m,2)) + '$\pm$' + str(np.round(mer,2)), transform=ax.transAxes)

axx.annotate('r$_0$', xy=(r0, 3),  xycoords='data',
           xytext=(0.55, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.02),
            horizontalalignment='right', verticalalignment='top',
            )

axx.annotate('s$_0$', xy=(s0, 3),  xycoords='data',
            xytext=(0.25, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.02),
            horizontalalignment='right', verticalalignment='top',
            )

##config
axx.set(xscale='log', yscale='log', 
        xlabel='log lag, pc', 
        ylabel=r'log $σ, \mathrm{-}$')

axx.axvline(r0, c="k", linestyle = '--')
axx.axvline(s0, c="k", linestyle = ':')

plt.title(name_plt + ' '+ em_line)
plt.legend(loc = 4)

plt.savefig('Imgs//'+ 'sf_'+ name_exp + element +  '.pdf', bbox_inches='tight')














#sb = fits.open(datapath_obs / 'TAURUS-604-Ha-Flux.fits')[0].data.astype(float)
#mf = ~np.isfinite(sb) | (sb < 0.0)
#sb[mf] = 0.0
#sb /= sb.max()
#sb /= np.nanmean(sb)


# PDF/CFD

#header=0
input_data = (sb,hdr) 


pdf_mom0 = tss.PDF(input_data, min_val=0.0, bins=40, normalization_type= "normalize_by_mean" )
plt.figure(figsize=(12, 6))
pdf_mom0.run(verbose=True)


from scipy.stats import lognorm
import seaborn as sns
sns.set_color_codes()
sns.set_context("talk")


LN = lognorm(s=1.0, scale=np.exp(1.0))


x = np.logspace(-2.0, 2.0, 300)
fig, ax = plt.subplots()
ax.plot(x, LN.pdf(x))
ax.set(xscale="log")


len(pdf_mom0.bins)


fig, ax = plt.subplots()
s, scale = pdf_mom0.model_params
LN = lognorm(s=s, scale=scale)
x = pdf_mom0.bins
ax.plot(x, x*pdf_mom0.pdf)
ax.plot(x, x*LN.pdf(x))
ax.set(
    xlabel="intensity",
    ylabel="PDF",
    xscale="log",
    ylim=[0, None],
);


pdf_mom0.model_params


fig, ax = plt.subplots()
ax.plot(x, x*pdf_mom0.pdf)
ax.plot(x, x*LN.pdf(x))
ax.set(
    xlabel="intensity",
    ylabel="PDF",
    xscale="log",
    yscale="log",
    ylim=[1e-3, 1.0],
);


wpdf_mom0 = tss.PDF(input_data, min_val=0.0, weights=sb)
plt.figure(figsize=(12, 6))
wpdf_mom0.run(verbose=True)


fig, ax = plt.subplots()
s, scale = wpdf_mom0.model_params
LN = lognorm(s=s, scale=scale)
x = wpdf_mom0.bins
ax.plot(x, x*wpdf_mom0.pdf)
ax.plot(x, x*LN.pdf(x))
ax.set(
    xlabel="intensity",
    ylabel="PDF",
    xscale="log",
    ylim=[0, None],
);





wpdf_mom0.model_params


m1 = np.isfinite(sb) & (sb > 0.0)
sns.histplot(x=np.log(sb[m1]), kde=False, weights=sb[m1].astype(float), bins=100)


H, edges = np.histogram(np.log(sb[m1]), weights=sb[m1], bins=100, range=[-4.0, 2.5], density=True)


fig, ax = plt.subplots()
centers = 0.5*(edges[:-1] + edges[1:])
ax.plot(centers, H)
LN = lognorm(s=0.75,scale=1.8)
ax.plot(centers, np.exp(centers)*LN.pdf(np.exp(centers)))
ax.set(
    xlabel="$\ln (S/S_0)$",
    ylabel="PDF",
#    yscale="log",
#    ylim=[1e-3, 1.0],
)


cdf = np.cumsum(H)*(centers[1] - centers[0])
fit = np.exp(centers)*LN.pdf(np.exp(centers))
cdf_fit = np.cumsum(fit)*(centers[1] - centers[0])

fig, ax = plt.subplots()
ax.plot(centers, cdf)
ax.plot(centers, cdf_fit)
ax.set(
    xlabel="$\ln (S/S_0)$",
    ylabel="CDF",
#    yscale="log",
#    ylim=[1e-3, 1.0],
)


fig, ax = plt.subplots()
ax.plot(centers, cdf/(1 - cdf))
ax.plot(centers, cdf_fit/(1 - cdf_fit))
ax.set(
    xlabel="$\ln (S/S_0)$",
    ylabel="CDF / (1 $-$ CDF)",
    yscale="log",
    ylim=[1e-3/3, 3e3],
)


#from sbfluct import sbfluct
#sbfluct(sb, 1e-8, 2, "604")



#sb = fits.open(datapath_obs / flux_map)[0].data.astype(float)
fmin = 1e-8
kmax = 4


resamples = [2, 4, 8, 16, 32, 64]


m = (sb > fmin) & np.isfinite(sb)
w = np.ones_like(sb)


for n in resamples[: kmax]:
    [sb,], m, w = downsample([sb,],
        m,
        weights=w,
        mingood=1,
        )
    
sb /= np.mean(sb[m])


smin, smax = -3.1, 3.1


H, edges, patches = plt.hist(
            np.log(sb[m]),
            # weights=s[m],
            density=True,
            bins=40,
            range=[-3.1, 3.1],)


# Calculate bin centers
x = 0.5 * (edges[:-1] + edges[1:])
# Fit Gaussian
g = models.Gaussian1D(amplitude=H.max(), stddev=0.5)
core = H > 0.01
g = fitter(g, x, H)
sigS = np.sqrt(np.exp(g.stddev.value ** 2) - 1)
esigS = 0.5 * np.sqrt(np.exp(g.mean.value ** 2) - 1)
xx = np.linspace(smin, smax, 200)

plt.plot(xx, g(xx), "orange", lw=2)


X = np.exp(xx)
Xmean = np.average(X, weights=g(xx))
Xvariance = np.average((X - Xmean) ** 2, weights=g(xx))
eps_rms = np.sqrt(Xvariance) / Xmean
eps_rms 


##turbustat results
x = pdf_mom0.bins 
s, scale = pdf_mom0.model_params
LN = lognorm(s=s, scale=scale)

Xmean = np.average(x, weights=x*LN.pdf(x))
Xvariance = np.average((x - Xmean) ** 2, weights=x*LN.pdf(x))
eps_rms_t = np.sqrt(Xvariance) / Xmean
eps_rms_t 


fig, ax = plt.subplots(figsize=(8, 6))


##pdf
H, edges, patches = ax.hist(
            np.log(sb[m]),
            # weights=s[m],
            density=True,
            bins=40,
            range=[-3.1, 3.1],)

##lognorm fit
ax.plot(xx, g(xx), "orange", lw=2)


##pdf turbustat and fit
ax.scatter(np.log(x), x*pdf_mom0.pdf, color = 'k', zorder = 2, label = 'turbustat')
ax.plot(np.log(x), x*LN.pdf(x), color = 'r')

ax.text(.6, 0.9,r"$\langle \delta S^2 \rangle^{1/2} / S_0$ = " + str(np.round(eps_rms ,2)), transform=ax.transAxes, color = 'orange')
ax.text(.6, 0.8,r"$\langle \delta S^2 \rangle^{1/2} / S_0$ = " + str(np.round(eps_rms_t ,2)), transform=ax.transAxes, color = 'r')

plt.title(name_plt + ' '+ em_line)
ax.legend(loc = 2)

ax.set(
    xlabel="$\ln (S/S_0)$",

)

plt.savefig('Imgs//'+ 'bf_'+ name_exp + element +  '.pdf', bbox_inches='tight')


get_ipython().system('jupyter nbconvert --to script --no-prompt ts-KPech-Orion-N.ipynb')


print("--- %s seconds ---" % (time.time()-start_time))

