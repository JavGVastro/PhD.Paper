#!/usr/bin/env python
# coding: utf-8


import time
start_time=time.time()
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import linmix
from scipy.stats import pearsonr
import math
import itertools
import json
from logerr import logify
from results import loadresults


sns.set_context("talk")


# Load Results and errors

res = loadresults()
res


sig = res['sig [km/s]']
sig_er = res['siger']
X = logify(sig, sig_er)[0]
Xe = logify(sig, sig_er)[1]


# Luminosity values from literature

# 1. Kennicut 1984: https://articles.adsabs.harvard.edu//full/1984ApJ...287..116K/0000122.000.html 
# 2. \
# Orion : \
# Carina : \
# Smith & Brooks 2007 https://academic.oup.com/mnras/article/379/4/1279/996059
# 30 Dor : \
# Bestenleher et al https://academic.oup.com/mnras/article/499/2/1918/5905414 \
# M8,346: \
# Kennicut 1984 https://articles.adsabs.harvard.edu//full/1984ApJ...287..116K/0000122.000.html \
# HX,HV,N604,N595: \
# Bosch et al. 2002 Table 11- https://academic.oup.com/mnras/article/329/3/481/1031037
# 

L_data=pd.DataFrame()
L_data['Region'] = ['Orion','M8','Carina','30 Dor','346','H X', 'H V', '595', '604']
L_data['L(Ha) [erg/s]$^1$'] = [1e37,3e37,6e38,1.5e40,6e38,4e38,7.5e38,2.3e39,4.5e39]
L_data['log L(Ha)$^1$ [erg/s]'] = np.log10(L_data['L(Ha) [erg/s]$^1$'] )
L_data['log L(Ha)$^2$ [erg/s]'] = [37.18,37.47,39.01,39.46,38.77,38.21,38.3,38.95,39.42]
L_data['log Q(H)$^2$ [photos/s]'] = [49.12,0,50.95,51.4,0,0,0,0,0]
L_data['conv_fact']=(10**L_data['log Q(H)$^2$ [photos/s]'])/(10**L_data['log L(Ha)$^2$ [erg/s]'])
L_data['log L(Ha)$^3$ [erg/s]'] = (L_data['log L(Ha)$^1$ [erg/s]']+L_data['log L(Ha)$^2$ [erg/s]'])/2

L_data.round(2)


Y1 = L_data['log L(Ha)$^1$ [erg/s]']
Y1e = logify(Y1, Y1*.1)[1]
Y2 = L_data['log L(Ha)$^2$ [erg/s]']
Y2e = logify(Y2, Y2*.1)[1]
Y3 = L_data['log L(Ha)$^3$ [erg/s]']
Y3e = Y3-Y1

fig, ax = plt.subplots(figsize=(10,10))
ax.errorbar(X, Y3, xerr=Xe, yerr=Y3e, ls=" ", elinewidth=0.4, alpha=1.0, c="k")

marker=itertools.cycle(('o','o','o','s','s','^','^','x','x'))
for i in range(len(L_data)):
    ax.scatter(X[i], Y1[i], marker=next(marker), s=250,zorder=0, c ='r', alpha=0.45)
for i in range(len(L_data)):    
    ax.scatter(X[i], Y2[i], marker=next(marker), s=250,zorder=0, c ='b', alpha=0.45)
#for i in range(len(L_data)):    
#    ax.scatter(X[i], Y3[i], marker=next(marker), s=250,zorder=0, c ='purple', alpha=0.65)
    
ax.set(ylabel='Log(L$_{Hα}$) [erg/s]', xlabel='$σ_{pos}$ [km s$^{-1}$]')


Y=Y3
Ye=Y3e
lm = linmix.LinMix(X, Y, Xe, Ye, K=2)
lm.run_mcmc(silent=True)

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)


vmin, vmax = 0.2, 1.4
xgrid = np.linspace(vmin, vmax, 200)


fig, ax = plt.subplots(figsize=(10, 10))

ax.errorbar(X, Y, xerr=Xe, yerr=Ye, ls=" ", elinewidth=0.4, alpha=1.0, c="k")

marker=itertools.cycle(('o','o','o','s','s','^','^','x','x'))
#for i in [0,1,2,3,4,6,8]:
for i in range(len(L_data)):
    ax.scatter(X[i], Y[i], marker=next(marker), s=250,zorder=5, c ='k', alpha=0.5)

# The original fit
ax.plot(xgrid, dfchain["alpha"].mean() + xgrid*dfchain["beta"].mean(), 
        '-', c="k")
for samp in lm.chain[::25]:
    ax.plot(xgrid, samp["alpha"] + xgrid*samp["beta"], 
        '-', c="r", alpha=0.2, lw=0.1)
    
ax.text(.05, .95,'log L(H) = (' 
        + str(np.round(dfchain["beta"].mean(),3)) + '$\pm$' + str(np.round(dfchain["beta"].std(),3))
        + ')log $\sigma$+('
        + str(np.round(dfchain["alpha"].mean(),3)) + '$\pm$' + str(np.round(dfchain["alpha"].std(),3))
        + ')',  color='k', transform=ax.transAxes)
    
ax.set(
    ylim=[37, 40], xlim=[0.2, 1.35],
    ylabel=r"log L(H) [erg s^-1]", xlabel=r"log $\sigma$ [km/s]",
)


['log L(H)','log $\sigma$',np.round(dfchain["beta"].mean(),2),np.round(dfchain["beta"].std(),2),
       np.round(dfchain["alpha"].mean(),2),np.round(dfchain["alpha"].std(),2),
      np.round(pearsonr(X, Y3)[0],2),np.round(pearsonr(X, Y3)[1],3)]


# Change variables

lm = linmix.LinMix(Y, X, Ye, Xe, K=2)
lm.run_mcmc(silent=True)

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
['log L(H)','log $\sigma$',np.round(dfchain["beta"].mean(),2),np.round(dfchain["beta"].std(),2),
       np.round(dfchain["alpha"].mean(),2),np.round(dfchain["alpha"].std(),2),
      np.round(pearsonr(Y, X)[0],2),np.round(pearsonr(Y, X)[1],3)]


# Other L

lm = linmix.LinMix(X, Y1, Xe, Y1e, K=2)
lm.run_mcmc(silent=True)

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
['log L(H)','log $\sigma$',np.round(dfchain["beta"].mean(),2),np.round(dfchain["beta"].std(),2),
       np.round(dfchain["alpha"].mean(),2),np.round(dfchain["alpha"].std(),2),
      np.round(pearsonr(X, Y1)[0],2),np.round(pearsonr(X, Y1)[1],3)]


lm = linmix.LinMix(X, Y2, Xe, Y2e, K=2)
lm.run_mcmc(silent=True)

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
['log L(H)','log $\sigma$',np.round(dfchain["beta"].mean(),2),np.round(dfchain["beta"].std(),2),
       np.round(dfchain["alpha"].mean(),2),np.round(dfchain["alpha"].std(),2),
      np.round(pearsonr(X, Y2)[0],2),np.round(pearsonr(X, Y2)[1],3)]


# +Previous work

path_previous = 'data-previous-scaling-relations'

Fer=pd.read_csv(path_previous+'//Fernandez2018.csv')


#sig = res['sig [km/s]']
#sig_er = res['siger']
#X = logify(sig, sig_er)[0]
#Xe = logify(sig, sig_er)[1]
#siglos_er=sig*(1.03)+7.3


siglos=sig*(1.03)+7.3
siglos_er=(sig_er/sig)*siglos
X2 = logify(siglos, siglos_er)[0]
X2e = logify(siglos, siglos_er)[1]


Y3b=Y3-0.44


lm = linmix.LinMix(X2, Y3b, X2e, Y3e, K=2)
lm.run_mcmc(silent=True)

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
['log L(H)','log $\sigma$',np.round(dfchain["beta"].mean(),2),np.round(dfchain["beta"].std(),2),
       np.round(dfchain["alpha"].mean(),2),np.round(dfchain["alpha"].std(),2),
      np.round(pearsonr(X2, Y3)[0],2),np.round(pearsonr(X2, Y3)[1],3)]


fig, ax=plt.subplots(figsize=(9,9))


plt.scatter(Fer.sig,Fer.L,label='Fernandez 2018',marker='P',alpha=0.95,color='red',s=200)

marker=itertools.cycle(('o','o','o','s','s','^','^','x','x'))
for i in range(len(L_data)):
    ax.scatter(X2[i], Y3b[i], marker=next(marker), label='This work',alpha=0.95,color='blue',s=200)
    
ax.errorbar(X2, Y3b, xerr=X2e, yerr=Y3e, ls=" ", elinewidth=0.4, alpha=1.0, c="b")


#plt.yscale('log')
ax.set(ylabel='Log(L$_{Hβ}$) [erg/s]', xlabel='log $σ_{los}$ [km s$^{-1}$]')

vmin, vmax = 0.9, 1.5
xgrid = np.linspace(vmin, vmax, 200)
ax.plot(xgrid, (33.25) + xgrid*(5.02), '-', c="r")
ax.plot(xgrid, (31.18) + xgrid*(6.11), '-', c="b")



#ax.text(.05, .95,'log L(H) = (' 
#        + str(np.round(dfchain["beta"].mean(),3)) + '$\pm$' + str(np.round(dfchain["beta"].std(),3))
#        + ')log $\sigma$+('
#        + str(np.round(dfchain["alpha"].mean(),3)) + '$\pm$' + str(np.round(dfchain["alpha"].std(),3))
#        + ')',  color='k', transform=ax.transAxes)

#plt.legend()
#ax.set(
#    ylim  = [37.5, 42],
#    xlim  = [0.9, 1.5],
#)


print("--- %s seconds ---" % (time.time()-start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt luminosity-H.ipynb')

