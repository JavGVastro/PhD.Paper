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


res['siglos [km/s]']=[9.9,11.2,18.6,21.7,9.6,10.0,9.8,16.5,17.5]
res['sigloser']=[1.2,1.6,3.3,2.2,1.0,0.02,0.03,0.1,0.3]


# Fine-structure splitting value removed

res['siglos [km/s]'] = (res['siglos [km/s]']**2+10.23)**0.5


sig = res['sig [km/s]']
sig_er = res['siger']
X = logify(sig, sig_er)[0]
Xe = logify(sig, sig_er)[1]


siglos = res['siglos [km/s]']
siglos_er = res['sigloser']
Z = logify(siglos, siglos_er)[0]
Ze = logify(siglos, siglos_er)[1]


# Luminosity values from literature

# 1. Kennicut 1984: https://articles.adsabs.harvard.edu//full/1984ApJ...287..116K/0000122.000.html 
# 2. \
# Orion : \
# Carina : \
# Smith & Brooks 2007 https://academic.oup.com/mnras/article/379/4/1279/996059 \
# 30 Dor : \
# Bestenleher et al https://academic.oup.com/mnras/article/499/2/1918/5905414 \
# M8,346: \
# Kennicut 1984 https://articles.adsabs.harvard.edu//full/1984ApJ...287..116K/0000122.000.html \
# HX,HV,N604,N595: \
# Bosch et al. 2002 Table 11- https://academic.oup.com/mnras/article/329/3/481/1031037
# 3. \
# Mean 1 and 2
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


# Comparison Previous work

path_previous = 'data-previous-scaling-relations'

Fer = pd.read_csv(path_previous + '//Fernandez2018.csv')
TM81 = pd.read_csv(path_previous + '//TM1981.csv')
M87 = pd.read_csv(path_previous + '//M1987.csv')
Hp86 = pd.read_csv(path_previous + '//Hp1986.csv')





siglos=sig*(1.03)+7.3
siglos_er=(sig_er/sig)*siglos
X2 = logify(siglos, siglos_er)[0]
X2e = logify(siglos, siglos_er)[1]


Y1b = Y1-0.45
Y2b = Y2-0.45
Y3b = Y3-0.45


lm = linmix.LinMix(X2, Y3b, X2e, Y3e, K=2)
lm.run_mcmc(silent=True)

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
['log L(H)','log $\sigma$',np.round(dfchain["beta"].mean(),2),np.round(dfchain["beta"].std(),2),
       np.round(dfchain["alpha"].mean(),2),np.round(dfchain["alpha"].std(),2),
      np.round(pearsonr(X2, Y3)[0],2),np.round(pearsonr(X2, Y3)[1],3)]


lm = linmix.LinMix(Z, Y3b, Ze, Y3e, K=2)
lm.run_mcmc(silent=True)

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
['log L(H)','log $\sigma$',np.round(dfchain["beta"].mean(),2),np.round(dfchain["beta"].std(),2),
       np.round(dfchain["alpha"].mean(),2),np.round(dfchain["alpha"].std(),2),
      np.round(pearsonr(X2, Y3)[0],2),np.round(pearsonr(X2, Y3)[1],3)]


fig, ax=plt.subplots(figsize=(10,10))

#ax.axvline(1.27, color="red",linestyle="dotted")
#ax.axvline(1.25, color="orange", linestyle="dotted")
#ax.axvline(np.log10(16.8), color="b", linestyle="dotted")

#ax.axvline(Z[3], color="purple", linestyle="dashed")
#ax.axvline(1.24, color="orange", linestyle="dashed")
#ax.axvline(np.log10(22.4), color="b", linestyle="dashed")

#ax.axhline(38.8 + 0.22, color="b", linestyle="dotted")
#ax.axhline(38.75 + 0.05, color="orange", linestyle="dotted")
#ax.axhline(39.22, color="red", linestyle="dotted")
#ax.axhline(39.2 + 0.32, color="b", linestyle="dashed")

plt.scatter(np.log10(TM81.sig),TM81.L+TM81.C,label='TM 81',marker='.',alpha=0.85,color='orange',s=200)
plt.scatter(np.log10(Hp86.sig),Hp86.L-0.45,label='Hp 86',marker='.',alpha=0.65,color='pink',s=200)
plt.scatter(np.log10(M87.sig),M87.L+M87.C,label='M 87',marker='.',alpha=0.65,color='blue',s=200)
plt.scatter(Fer.sig,Fer.L,label='F 18',marker='.',alpha=0.75,color='red',s=200)
#plt.scatter(Z,Y1b,label='TM 81',marker='o',alpha=0.65,color='purple',s=200)


marker=itertools.cycle(('o','o','o','s','s','^','^','x','x'))
#for i in range(len(L_data)):
#    ax.scatter(Z[i], Y2b[i], marker=next(marker),alpha=0.5,color='green',s=100)
    
#for i in range(len(L_data)):
#    ax.scatter(Z[i], Y1b[i], marker=next(marker),alpha=0.5,color='purple',s=200)
    

ax.set(ylabel='Log(L$_{Hβ}$) [erg/s]', xlabel='log $σ_{los}$ [km s$^{-1}$]')

vmin, vmax = 0.9, 1.5
xgrid = np.linspace(vmin, vmax, 200)
#ax.plot(xgrid, (33.25) + xgrid*(5.02), '-', c="r")
#ax.plot(xgrid, (32.65) + xgrid*(4.97), '-', c="g")


plt.legend()
#ax.set(
#    ylim  = [37.5, 42],
#    xlim  = [0.9, 1.5],


fig, ax=plt.subplots(figsize=(10,10))

ax.axvline(Z[8], color='k', linestyle="dashed",alpha=0.5,linewidth=0.75)
ax.axvline(Z[5], color='k', linestyle="dashed",alpha=0.5,linewidth=0.75)
ax.axvline(Z[6], color='k', linestyle="dashed",alpha=0.5,linewidth=0.75)
ax.axvline(Z[7], color='k', linestyle="dashed",alpha=0.5,linewidth=0.75)

#604
ax.scatter(np.log10(TM81.sig[0]),TM81.L[0]+TM81.C[0],label='TM81',color = 'orange')
ax.scatter(np.log10(Hp86.sig[0]),Hp86.L[0]-0.45,label='Hp86',color = 'pink')
ax.scatter(np.log10(M87.sig[0]),M87.L[0]+M87.C[0],label='M87',color = 'blue')
ax.scatter(Fer.sig[10],Fer.L[10],label='Fer',color = 'red')
ax.errorbar(Fer.sig[10], Fer.L[10], xerr=Fer.sigerr[10], yerr=Fer.err[10], ls=" ", elinewidth=0.5, alpha=1.0, c="r")
ax.scatter(Z[8], Y1b[8], marker='.',alpha=0.5,color='purple',s=200,label='K84')
ax.errorbar(Z[8], Y1b[8], yerr=0.25, ls=" ", elinewidth=0.5, alpha=1.0, c="purple")
ax.scatter(Z[8], Y2b[8], marker='.',alpha=0.5,color='green',s=200,label='B02')
#ax.errorbar(Z[8], Y2b[8], yerr=0.25, ls=" ", elinewidth=0.5, alpha=1.0, c="green")
ax.scatter(Z[8], 38.99, marker='.',alpha=1.0,color='k',s=250)
ax.errorbar(Z[8], 38.99, yerr=0.08, ls=" ", elinewidth=2.25, alpha=1.0, c='k')


#595
ax.scatter(np.log10(TM81.sig[1]),TM81.L[1]+TM81.C[1],color = 'orange', marker='^')
ax.scatter(np.log10(M87.sig[1]),M87.L[1]+M87.C[1],color = 'blue', marker='^')
ax.scatter(np.log10(Hp86.sig[1]),Hp86.L[1]-0.45,color = 'pink', marker='^')
ax.scatter(Fer.sig[9],Fer.L[9],color = 'red', marker='^')
ax.errorbar(Fer.sig[9], Fer.L[9], xerr=Fer.sigerr[9], yerr=Fer.err[9], ls=" ", elinewidth=0.75, alpha=1.0, c="r", marker='^')
ax.scatter(Z[7], Y1b[7], marker='^',alpha=0.5,color='purple')
ax.errorbar(Z[7], Y1b[7], yerr=0.25, ls=" ", elinewidth=0.75, alpha=1.0, c="purple")
ax.scatter(Z[7], Y2b[7], marker='^',alpha=0.5,color='green')
ax.scatter(Z[7], 38.66, marker='^',alpha=1.0,color='k')
ax.errorbar(Z[7], 38.66, yerr=0.12, ls=" ", elinewidth=2.25, alpha=1.0, c='k')


#30Dor
ax.scatter(np.log10(TM81.sig[17]),TM81.L[17]+TM81.C[17],color = 'orange', marker='s')
ax.scatter(np.log10(M87.sig[26]),M87.L[26]+M87.C[26],color = 'blue', marker='s')
ax.scatter(Z[3], Y1b[3], marker='s',alpha=0.5,color='purple')
ax.errorbar(Z[3], Y1b[3], yerr=0.25, ls=" ", elinewidth=0.75, alpha=1.0, c="purple")

#HX
ax.scatter(np.log10(TM81.sig[10]),TM81.L[10]+TM81.C[10],color = 'orange', marker='x')
ax.scatter(np.log10(M87.sig[24]),M87.L[24]+M87.C[24],color = 'blue', marker='x')
ax.scatter(np.log10(Hp86.sig[40]),Hp86.L[40]-0.45,color = 'pink', marker='x')
ax.scatter(Z[5], Y2b[5], marker='x',alpha=0.5,color='green')
ax.scatter(Z[5], Y1b[5], marker='x',alpha=0.5,color='purple')
ax.errorbar(Z[5], Y1b[5], yerr=0.25, ls=" ", elinewidth=0.75, alpha=1.0, c="purple")
ax.scatter(Z[5], 37.91, marker='x',alpha=1.0,color='k')
ax.errorbar(Z[5], 37.91, yerr=0.14, ls=" ", elinewidth=2.25, alpha=1.0, c='k')


#HV
ax.scatter(np.log10(TM81.sig[11]),TM81.L[11]+TM81.C[11],color = 'orange', marker='d')
ax.scatter(np.log10(M87.sig[25]),M87.L[25]+M87.C[25],color = 'blue', marker='d')
ax.scatter(np.log10(Hp86.sig[41]),Hp86.L[41]-0.45,color = 'pink', marker='d')
ax.scatter(Z[6], Y2b[6], marker='d',alpha=0.5,color='green')
ax.scatter(Z[6], Y1b[6], marker='d',alpha=0.5,color='purple')
ax.errorbar(Z[6], Y1b[6], yerr=0.25, ls=" ", elinewidth=0.75, alpha=1.0, c="purple")
ax.scatter(Z[6], 37.96, marker='d',alpha=1.0,color='k')
ax.errorbar(Z[6], 37.96, yerr=0.14, ls=" ", elinewidth=2.25, alpha=1.0, c='k')



ax.set(ylabel='Log(L$_{Hβ}$) [erg/s]', xlabel='log $σ_{los}$ [km s$^{-1}$]')

ax.set(
    ylim  = [37.5, 40],
    xlim  = [0.9, 1.5])

plt.legend()





fig, ax=plt.subplots(figsize=(9,9))


plt.scatter(Fer.sig,Fer.L,label='Fernandez 2018',marker='P',alpha=0.95,color='red',s=200)

marker=itertools.cycle(('o','o','o','s','s','^','^','x','x'))
for i in range(len(L_data)):
    ax.scatter(X2[i], Y3b[i], marker=next(marker), label='This work',alpha=0.35,color='blue',s=100)
    
for i in range(len(L_data)):
    ax.scatter(Z[i], Y3b[i], marker=next(marker), label='This work',alpha=0.95,color='green',s=200)
    
ax.errorbar(X2, Y3b, xerr=X2e, yerr=Y3e, ls=" ", elinewidth=0.2, alpha=1.0, c="b")
ax.errorbar(Z, Y3b, xerr=Ze, yerr=Y3e, ls=" ", elinewidth=0.4, alpha=1.0, c="g")


#plt.yscale('log')
ax.set(ylabel='Log(L$_{Hβ}$) [erg/s]', xlabel='log $σ_{los}$ [km s$^{-1}$]')

vmin, vmax = 0.9, 1.5
xgrid = np.linspace(vmin, vmax, 200)
ax.plot(xgrid, (33.25) + xgrid*(5.02), '-', c="r")
ax.plot(xgrid, (32.65) + xgrid*(4.97), '-', c="g")



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

