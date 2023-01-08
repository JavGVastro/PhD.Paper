#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()


import sys
from pathlib import Path
import json
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Input path

# observations folder location

text_file_0 = open("path-observations.txt", "r")
path_obs = text_file_0.read()


datapath_obs = Path(path_obs).expanduser()


# results folder location

text_file_1 = open("path-results.txt", "r")
path_res = text_file_1.read()


datapath_res= Path(path_res).expanduser()


# files names

data_file = 'KPech-Orion-N-'


name_export='KPech-Orion-N'


flux_in = data_file + 'sum.fits'
radial_velocity_in = data_file + 'mean.fits'
#sigma_in = data_file + 'sigma.fits'


flux=fits.open(datapath_obs / flux_in)
rad_vel=fits.open(datapath_obs / radial_velocity_in)
#sigma=fits.open(datapath_obs / sigma_in)


# Input data of the region

dist = 410 #parsecs
pix = 0.534 #arcsec
seeing = 0.9 #seeing arcsec FWHM


pc = dist*(2*np.pi) / (360 * 60 * 60)
s0 = (seeing*pc)/2.355 #seeing pc RMS
pc,s0


flux.info()


sb = flux[0].data.astype(float)
vv = rad_vel[0].data.astype(float)
#ss = sigma["DATA"].data.astype("float")


## Replace spurious values in the arrays
m = ~np.isfinite(sb*vv) | (sb < 0.0)

sb[m] = 0.0
vv[m] = np.nanmean(vv)
#ss[m] = 0.0
sb /= sb.max()

good = (~m) & (sb > 0.001)


trim = (slice(0, 513), slice(0, 355))


fig, ax = plt.subplots(figsize=(12, 12))


dataI=sb[trim]

plt.figure(1)
plt.imshow(dataI, cmap='inferno')

cbar = plt.colorbar()
plt.clim(0.001,1)
cbar.set_label(' ', rotation=270, labelpad=15)  

ax.set_xlabel('X')
ax.set_ylabel('Y')


ax.text(0.9, 0.1, '10 pc',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)
    
plt.axhline(y=50, xmin=0.59, xmax=0.925, linewidth=2, color = 'k')


plt.gca().invert_yaxis()


dffx=pd.DataFrame(sb[trim])


dffx=dffx.stack().reset_index().rename(columns={'level_0':'X', 'level_1':'Y', 0:'I'})


fig, ax = plt.subplots(figsize=(12, 12))


dataRV=vv[trim]

plt.figure(1)
plt.imshow(dataRV, cmap='RdBu_r')

cbar = plt.colorbar()
#plt.clim(225,350)
cbar.set_label('km/s', rotation=270, labelpad=15)  

ax.set_xlabel('X')
ax.set_ylabel('Y')


#ax.text(0.9, 0.1, '10 pc',
#        verticalalignment='bottom', horizontalalignment='right',
#        transform=ax.transAxes,
#        color='black', fontsize=20)
    
#plt.axhline(y=50, xmin=0.59, xmax=0.925, linewidth=2, color = 'k')


plt.gca().invert_yaxis()





RV=pd.DataFrame(vv[trim])


RV=RV.stack().reset_index().rename(columns={'level_0':'X', 'level_1':'Y', 0:'RV'})


# fig, ax = plt.subplots(figsize=(12, 12))
# 
# 
# dataS=ss[trim]
# 
# plt.figure(1)
# plt.imshow(dataS, cmap='magma')
# 
# cbar = plt.colorbar()
# #plt.clim(225,350)
# cbar.set_label(' ', rotation=270, labelpad=15)  
# 
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# 
# 
# ax.text(0.9, 0.1, '10 pc',
#         verticalalignment='bottom', horizontalalignment='right',
#         transform=ax.transAxes,
#         color='black', fontsize=20)
#     
# plt.axhline(y=50, xmin=0.59, xmax=0.925, linewidth=2, color = 'k')
# 
# 
# plt.gca().invert_yaxis()

# dsig=pd.DataFrame(ss[trim])

# dsig=dsig.stack().reset_index().rename(columns={'level_0':'X', 'level_1':'Y', 0:'Sig'})




data=RV
data['I']=dffx.I
#data['Sig']=dsig.Sig
data.describe()


mI=data.I>0.001


data[mI].describe()


#sns.displot(RV[0]-RV[0].mean(),bins=100)
sns.displot(data[mI].RV,bins=100)

#plt.xlim(200,350)

#plt.text(0.75, 1.15,'n ='+str(RV[0].count()), ha='center', va='center', transform=ax.transAxes, color='k')
#plt.text(0.80, 0.82,'$μ$ ='+str(np.round(RV[0].mean(),2))+' km/s', ha='center', va='center', transform=ax.transAxes, color='k')
#plt.text(0.85, 1.25,'$σ^{2}$ ='+str(np.round(RV[0].var(),2))+' km$^{2}$/s$^{2}$', ha='center', va='center', transform=ax.transAxes, color='k')

plt.title('Orion')

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]="15"
plt.xlabel('Radial velocity [km/s]')


data_export_list = {
        'name': name_export, 
        'pc' : pc,
        's0' : s0,
        'pix' : pix,
         name_export : np.array(data[mI]),
      }
data_export_list



data_export_matrix = {
       'name': name_export, 
       'pc' : pc,
       's0' : s0,
       'pix' : pix,
       'sb' : flux[0].data.astype(float),
       'vv' : rad_vel[0].data.astype(float),
#       'ss' : sigma["DATA"].data.astype("float")

      }

data_export_matrix


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


jsonfilename =name_export +"-m.json"
with open(datapath_res/jsonfilename, "w") as f:
    json.dump(data_export_matrix, fp=f, indent=3, cls=MyEncoder)


jsonfilename =name_export +"-l.json"
with open(datapath_res/jsonfilename, "w") as f:
    json.dump(data_export_list, fp=f, indent=3, cls=MyEncoder)


print("--- %s seconds ---" % (time.time()-start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt otv-KPech-Orion-N.ipynb')

