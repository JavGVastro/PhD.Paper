# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import time
start_time=time.time()

# +
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import interpolate
import pickle
import json
from pathlib import Path

import strucfunc
from strucfunc import sosflog
import bfunc
# -

reg = 'Car'
line = 'CarC'

pickle_in = open(Path('VFL') / f'{reg}.pkl', "rb")
VF = pickle.load(pickle_in)
data = VF[line]

data.RV.var()

sig = data.RV.std()
sig2 = data.RV.var()

box_size = np.sqrt((data.X.max()-data.X.min())*(data.Y.max()-data.Y.min()))*VF['pc']*3600

table = sosflog(data, 0.05, 3600)

table

s = 0.5 * (table[('s', 'min')] + table[('s', 'max')])*VF['pc']
e_s = 0.5 * (table[('s', 'max')] - table[('s', 'min')])*VF['pc']
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')] / np.sqrt(ng)

r0 = np.interp(sig2, b2, s,period = 360)
r0

r1 = np.interp(sig, b2, s,period = 360)
r1

r2 = np.interp(2*sig2, b2, s, period = 360)
r2

# +
#x = s
#y = b2 - sig2
#tck=interpolate.splrep(x,y,s=0)
#grid=np.linspace(x.min(),x.max(),num=len(x))
#ynew=interpolate.splev(grid,tck,der=0)
#inter=pd.DataFrame([grid,ynew]).T
#SFr=interpolate.sproot(tck)
#SFr
# -



rgrid = np.linspace(s[0], s[:-1])

s

# +
m = 0.9
noise = 2.0
s0 = VF['s0']

fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(s, b2, yerr=e_b2, xerr=e_s, fmt='o', alpha=0.4,color="blue")

ax.axhline(sig2, ls='-')
ax.axvline(r0, ls='-')

ax.axhline(2*sig2, ls=':')
ax.axvline(r2, ls=':')

ax.axhline(sig, ls=':')
ax.axvline(r1, ls=':')

ax.plot(rgrid, bfunc.bfunc00s(rgrid, r0, sig2, m), color="0.8")
ax.plot(rgrid, bfunc.bfunc03s(rgrid, r0, sig2, m, s0, noise), color="red")
ax.plot(rgrid, bfunc.bfunc04s(rgrid, r0, sig2, m, s0, noise, box_size), color="black")


ax.set(xscale='log', yscale='log', 
       xlabel='separation,pc',
       ylabel=r'$b^2,\ \mathrm{km^2\ s^{-2}}$'
      )
None

sig2, r0, m, VF['s0'], noise
# -

table



Car = { 'VF' : data,
        'SF' : table,
        's' : s,
        'b2' : b2,
        'sig2' : sig2,
        'pc' : VF['pc'],
        'r0' : r0,
        'r1' : r1,
        'r2' : r2,
        's0' : VF['s0'],
        'm' : m,
        'box_size': box_size}

f = open(Path('Results') / f'SF{line}.pkl', "wb")
pickle.dump(Car,f)
f.close()

# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(MyEncoder, self).default(obj)

# jsonfilename = f'SFresults//' + line +'.json'
# with open(jsonfilename, "w") as f:
#     json.dump(Car, fp=f, indent=3, cls=MyEncoder)
# print(jsonfilename, end="")

print("--- %s seconds ---" % (time.time()-start_time))
