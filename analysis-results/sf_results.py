#!/usr/bin/env python
# coding: utf-8

from pathlib import Path


import time
start_time=time.time()
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import lmfit
import statsmodels.api as sm
import linmix
from scipy.stats import pearsonr
import pickle
import math
import itertools
import json
from logerr import logify


# Load Table with physical properties

physical_data = pd.read_table('property-regions-data.csv', delimiter=',')
#physical_data = physical_data.drop(physical_data .index[[5]])


physical_data 


# Path names

datapath_names = Path(open("path-name-list.txt", "r").read()).expanduser()


samples=pd.read_csv(str(datapath_names) +'//sample-names-corr.csv',header=None)
Names=pd.read_csv(str(datapath_names) +'//formal-names-corr.csv',header=None)


# Load Results

datapath_res = Path(open("path-results.txt", "r").read()).expanduser()





data = {}
Results = {}

for i in range(len(samples)):
    data[samples[0][i]] = json.load(open(str(datapath_res) + '/' + samples[0][i] + ".json"))


#r_2 = np.interp(2*data[samples[0][i]]['preres']['sig2'], B, r, period = 360)
#r_2


i = 1
B = data[samples[0][i]]['B']
r = data[samples[0][i]]['r']

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(r,B)
ax.axhline(np.array(data[samples[0][i]]['B']).max(), linestyle = ':')
ax.axhline(2*data[samples[0][i]]['preres']['sig2'])
#ax.axvline(15)
ax.axvline(22)
ax.axvline(30)
ax.axvline(55)
ax.axvline(110)

ax.set(xscale='log', yscale='log', 
       xlabel='separation,pc',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )


data[samples[0][i]]['preres'],np.array(data[samples[0][i]]['B']).max()


fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(r[25:],B[25:])
#ax.axhline(np.array(data[samples[0][i]]['B']).max())
ax.axhline(2*data[samples[0][i]]['preres']['sig2'])
#ax.axvline(2.6)
#ax.axvline(4)

ax.set(xscale='log', yscale='log', 
       xlabel='separation,pc',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )


np.array(B[25:]).std()


#data[samples[0][i]]['r'][36:43]


data[samples[0][i]]['B'],data[samples[0][i]]['r']


x1 = 1.05
y1 = 2.5
x2 = 10
y2 = 44
x3 = 21
y3 = 2*y2

(np.log10(y2)-np.log10(y1))/(np.log10(x2)-np.log10(x1)),(np.log10(y3)-np.log10(y1))/(np.log10(x3)-np.log10(x1))








get_ipython().system('jupyter nbconvert --to script --no-prompt sf_results.ipynb')

