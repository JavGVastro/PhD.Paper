#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()


from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
from scipy import interpolate
import numpy as np
import pandas as pd
import json

from astropy.io import fits
import astropy.units as u

#plt.rcParams["font.family"]="Times New Roman"
#plt.rcParams["font.size"]="20"


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

data_file = 'Hanel-EON-O-RV'


name_export='Hanel-EON-O'


# Input data of the region

dist = 410 #parsecs
pix = 1.0 #arcmin 
seeing = 0.9 #seeing arcsec FWHM


pc = dist*(2*np.pi) / (360 * 60) #arcsec to parsecs
s0 = (seeing*pc)/2.355 #seeing pc RMS
pc,s0


orion=pd.read_table(str(datapath_obs)+ '/' +data_file+'.csv', delimiter=',',header=None)
orion.describe()


#orion[orion == 25] = 'nan' 


fig, ax = plt.subplots(figsize = (8,6))

sns.heatmap(orion,cmap='RdBu_r',cbar_kws={'label': 'km/s'})

plt.scatter(13.5, 13.5, marker='+', color='yellow', s=150)
ax.set(xlabel='arcmin', ylabel='arcmin')
ax.text(0.93, 0.81, '1.3 pc',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)
plt.axhline(y=2, xmin=0.68, xmax=0.96, linewidth=2, color = 'k')
plt.text(14.5, 14.5, '$θ^{1}Ori\ C$', fontsize=20, color='yellow')

plt.show()


# Matrix To list

data=orion.stack().reset_index().rename(columns={'level_0':'X', 'level_1':'Y', 0:'RV'})
data.describe()


m=data.RV<25
data=data[m]


data[m].describe()


plt.style.use([
    "seaborn-poster",
])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

datal=data[m].copy()

dataH_rv=(datal.round(2)).pivot(index='X', columns='Y', values='RV')
sns.heatmap(dataH_rv, cmap="RdBu_r",cbar_kws={'label': 'km/s'})
plt.title('Orion')
ax.set_facecolor('xkcd:gray')


#plt.savefig('Imgs//VF//N604.pdf', bbox_inches='tight')


# Fits file

#hdu = fits.PrimaryHDU(dataH_rv)
#hdu.writeto(str(datapath_obs)+ '/' +data_file + '.fits')


sns.displot(data[m].RV-data[m].RV.mean(),bins=25)
plt.xlim(-10,10)

plt.text(0.45, 0.86,'n ='+str(data.RV.count()), ha='center', va='center', transform=ax.transAxes, color='k')
plt.text(0.45, 0.78,'$μ$ ='+str(np.round(data.RV.mean(),2))+' km/s', ha='center', va='center', transform=ax.transAxes, color='k')
plt.text(0.45, 0.68,'$σ^{2}$ ='+str(np.round(data.RV.var(),2))+' km$^{2}$/s$^{2}$', ha='center', va='center', transform=ax.transAxes, color='k')

plt.title('EON')

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]="15"

plt.xlabel('Radial velocity [km/s]')


# Export data

data_export_list = {
        'name': name_export, 
        'pc' : pc,
        's0' : s0,
        'pix' : pix,
         name_export : np.array(data[m]),
      }
data_export_list


data_export_matrix = {
       'name': name_export, 
       'pc' : pc,
       's0' : s0,
       'pix' : pix,
#       'sb' :np.array(dataH_f),
       'vv' : np.array(dataH_rv),
#       'ss' : np.array(dataH_s),

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


get_ipython().system('jupyter nbconvert --to script --no-prompt otv-Hanel-EON-O.ipynb')

