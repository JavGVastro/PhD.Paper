#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
import json


datapath_names = Path(open("path-name-list.txt", "r").read()).expanduser()





samples=pd.read_csv(str(datapath_names) +'//sample-names.csv',header=None)
samples


Names=pd.read_csv(str(datapath_names) +'//formal-names.csv',header=None)
Names


# Load results

datapath_res = Path(open("path-results.txt", "r").read()).expanduser()


data = {}
Results = {}

for i in range(len(samples)):
    data[samples[0][i]] = json.load(open(str(datapath_res) + '/' + samples[0][i] + ".json"))


for i in range(len(samples)):
    print(str(samples[0][i]) + ':',
          ' r0 = ' + str(np.round(data[samples[0][i]]['results_2sig']['r0'][0],4)) + ' pc,',
          ' s0 = ' + str(np.round(data[samples[0][i]]['results_2sig']['s0'][0],4)) + ' pc,',
    ' m = ' + str(np.round(data[samples[0][i]]['results_2sig']['m'][0],4)) + ',',
    ' sig2 = ' + str(np.round(data[samples[0][i]]['results_2sig']['sig2'][0],4)) + ' (km/s)^2,',
    ' noise = ' + str(np.round(data[samples[0][i]]['results_2sig']['noise'][0],4)) + ' (km/s)^2')


# Results and confidence intervals

# Create the columns for the values

#sigma
sig = [[0]*(1) for i in range(len(samples))]
siger = [[0]*(1) for i in range(len(samples))]

#velocity dispersion with 2-sig intervals
sig2 = [[0]*(1) for i in range(len(samples))]
#sig2er = [[0]*(1) for i in range(len(samples))]
sig2s2 = [[0]*(1) for i in range(len(samples))]
sig2s2p = [[0]*(1) for i in range(len(samples))]
sig2s2m = [[0]*(1) for i in range(len(samples))]

#correlation length with 2-sig intervals
r0 = [[0]*(1) for i in range(len(samples))]
#r0er = [[0]*(1) for i in range(len(samples))]
r0s2 = [[0]*(1) for i in range(len(samples))]
r0s2p = [[0]*(1) for i in range(len(samples))]
r0s2m = [[0]*(1) for i in range(len(samples))]

#power-law
m = [[0]*(1) for i in range(len(samples))]
#mer = [[0]*(1) for i in range(len(samples))]
ms2 = [[0]*(1) for i in range(len(samples))]
ms2p = [[0]*(1) for i in range(len(samples))]
ms2m = [[0]*(1) for i in range(len(samples))]

#noise with 2-sig intervals
bn = [[0]*(1) for i in range(len(samples))]
#ner = [[0]*(1) for i in range(len(samples))]
bns2 = [[0]*(1) for i in range(len(samples))]
bns2p = [[0]*(1) for i in range(len(samples))]
bns2m = [[0]*(1) for i in range(len(samples))]

#seeing with 2-sig intervals
s0 = [[0]*(1) for i in range(len(samples))]
#s0er = [[0]*(1) for i in range(len(samples))]
s0s2 = [[0]*(1) for i in range(len(samples))]
s0s2p = [[0]*(1) for i in range(len(samples))]
s0s2m = [[0]*(1) for i in range(len(samples))]

pc = [[0]*(1) for i in range(len(samples))]
box_size = [[0]*(1) for i in range(len(samples))]


# Results to empty columns

for i in range(len(samples)):    
    
    sig2[i] = data[samples[0][i]]['results_2sig']['sig2'][0]
    sig2s2p[i] = data[samples[0][i]]['results_2sig']['sig2'][1]
    sig2s2m[i] = data[samples[0][i]]['results_2sig']['sig2'][2]
    
    r0[i]    = data[samples[0][i]]['results_2sig']['r0'][0]
    r0s2p[i] = data[samples[0][i]]['results_2sig']['r0'][1]
    r0s2m[i] = data[samples[0][i]]['results_2sig']['r0'][2]
    
    m[i]    = data[samples[0][i]]['results_2sig']['m'][0]
    ms2p[i] = data[samples[0][i]]['results_2sig']['m'][1]
    ms2m[i] = data[samples[0][i]]['results_2sig']['m'][2]
    
    bn[i]    = data[samples[0][i]]['results_2sig']['noise'][0]
    bns2p[i] = data[samples[0][i]]['results_2sig']['noise'][1]
    bns2m[i] = data[samples[0][i]]['results_2sig']['noise'][2]
    
    s0[i]    = data[samples[0][i]]['results_2sig']['s0'][0]
    s0s2p[i] = data[samples[0][i]]['results_2sig']['s0'][1]
    s0s2m[i] = data[samples[0][i]]['results_2sig']['s0'][2]
    
    box_size[i] = data[samples[0][i]]['properties']['box_size']
    pc[i] = data[samples[0][i]]['properties']['pc']


#pc[8]=pc[8]/60


# Crate table

td = pd.DataFrame(
    {
       "A": sig2,
       "B": m,
       "C": r0,
       "E": bn,
       "F": s0,
        "G":  box_size,
        "H": np.array(box_size)/np.array(r0),
        "I": np.array(r0)/np.array(s0),
    },
)


#td=td.sort_values( by='A', ascending=False)


SFres=td[['A','E','F','C','B','G','H','I']].copy()
SFres.rename(columns={'A':'$\sigma^2$ [km$^2$/s$^2$]',
                      'E':'$B_{noise}$ [km$^2$/s$^2$]',
                      'B':'$m$',
                      'C':'$r_0$ [pc]',
                      'F':'$s0$ (rms) [pc]',
                      'G':'L$_{box}$ [pc]',
                      'H':'L$_{box} / r_0$',
                      'I':'$r_0$/s0  ',},
                      inplace=True)

             
SFres.insert(loc=0, column='Region', value=Names)
SFres.round(4)


# Crate table with confidence intervals

s0f = pd.DataFrame(
    {
        "s0 [RMS]":s0,
        "s0+[RMS]": s0s2p,
        "s0-[RMS]": s0s2m,  
       "s0 [FWHM]": np.array(s0)*2.35/np.array(pc),
       "s0- [FWHM]": np.array(s0s2m)*2.35/np.array(pc),
       "s0+ [FWHM]": np.array(s0s2p)*2.35/np.array(pc),
        "bn ":bn,
        "bn+": bns2p,
        "bn- ": bns2m,     
    }
)

s0f.insert(loc=0, column='Region', value=Names)


s0f.round(4)


s1f = pd.DataFrame(
    {
        "sig2":sig2,
        "sig2+": sig2s2p,
        "sig2-": sig2s2m,
        "r0":r0,
        "r0+": r0s2p,
        "r0-": r0s2m,
        "m":m,
        "m+": ms2p,
        "m-": ms2m,
       
    }
)

s1f.insert(loc=0, column='Region', value=Names)


s1f.round(4)


results_paper = s1f.drop([1,6,7,8,10,12,13,15,16,17,19,20,22,23])


results_paper.sort_values(by=['sig2'], ascending=False)


xx= s0f.drop([1,6,7,8,10,12,13,15,16,17,19,20,22,23])


xx


fig, ax = plt.subplots(figsize=(7, 7))

errorsig2 = [s1f['sig2-'],s1f['sig2+']]
errorr0 = [s1f['r0-'],s1f['r0+']]

#ax.errorbar(s1f.sig2,s1f.r0, xerr = errorsig2, yerr = errorr0,  marker='o', linestyle=' ', markersize='7')

for xp, yp, m in zip(s1f.sig2, s1f.r0, marker_sample):
    plt.scatter(xp, yp, marker=m, s=150, color = 'blue')

ax.set(
    xscale="log",
    yscale="log",
    ylabel="r0 [pc]",
    xlabel=r"sig2 [km$^{2}$/s$^{2}$]",
)


print("--- %s seconds ---" % (time.time() - start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt results-compiler.ipynb')

