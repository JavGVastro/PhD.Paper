#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()


from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import interpolate
import pickle
import json

import strucfunc
from strucfunc import sosflog
import bfunc


reg = 'Car'
line = 'CarC'


text_file_0 = open("path-data.txt", "r")
path_data = text_file_0.read()


datapath_data = Path(path_data).expanduser()


name = 'FLA-Car-N'


data_in = json.load(open(str(datapath_data) + '/' + name + "-l.json"))


data=pd.DataFrame(data_in[name])


data.columns=['X','Y','RV','I','Sig']





data.RV.var()


sig = data.RV.std()
sig2 = data.RV.var()


box_size = np.sqrt((data.X.max()-data.X.min())*(data.Y.max()-data.Y.min()))*data_in['pc']*3600
box_size


table = sosflog(data,0.05,3600)


table


s = 0.5 * (table[('s', 'min')] + table[('s', 'max')])*data_in['pc']
e_s = 0.5 * (table[('s', 'max')] - table[('s', 'min')])*data_in['pc']
b2 = table[('dV2', 'mean')]
ng = table[('dV2', 'count')]
e_b2 = table[('dV2', 'std')] / np.sqrt(ng)


r0 = np.interp(sig2, b2[1:35], s[1:35])
r0


r1 = np.interp(sig, b2[1:35], s[1:35])
r1


r2 = np.interp(2*sig2, b2[1:35], s[1:35])
r2


m = 0.9
noise = 2.0
s0 = data_in['s0']
rgrid = np.linspace(s[0], s[:-1])

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

sig2, r0, m, s0, noise


table


results = {
    'sig2' : sig2,
        'r0' : r0,
        'r1' : r1,
        'r2' : r2,
          } 


data_export = {   
        's' : np.array(s),
        'b2' : np.array(b2),
        'pc' : data_in['pc'],
#        'pix' : data_in['pix'],
        's0' : data_in['s0'],
        'box_size': box_size,
         'results':results,
         'SF' : np.array(table),
        'VF' : data_in,
}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


jsonfilename ="sf-" + name +".json"
with open(jsonfilename, "w") as f:
    json.dump(data_export, fp=f, indent=3, cls=MyEncoder)


print("--- %s seconds ---" % (time.time()-start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt sf-FLA-Car-N.ipynb')

