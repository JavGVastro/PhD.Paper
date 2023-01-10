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
#from astropy.table import Table
#import itertools


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

data_file = 'TAURUS-604-O-'


name_export='TAU-N604-O'


flux_in = data_file + 'Flux.fits'
radial_velocity_in = data_file + 'RV.fits'
sigma_in = data_file + 'Sigma.fits'


flux=fits.open(datapath_obs / flux_in)
rad_vel=fits.open(datapath_obs / radial_velocity_in)
sigma=fits.open(datapath_obs / sigma_in)


# Input data of the region

distance = 840000 #parsecs
pix = 0.26 #arcsec 
seeing = 0.9 #seeing arcsec FWHM


pc = distance*(2*np.pi) / (360 * 60 * 60) #arcsec to parsecs
s0 = (seeing*pc)/2.355 #seeing pc RMS
pc,s0


# Flux map

fig, ax = plt.subplots(figsize=(5, 5))

image_data=flux[0].data

plt.imshow(image_data, cmap='inferno')

ax.set_xlabel('pixels')
ax.set_ylabel('pixels')

cbar = plt.colorbar()
cbar.set_label('Flux', rotation=270, labelpad=15) 

plt.title('OIII Flux')


# Matrix to List

flx=flux[0].data
df=pd.DataFrame(flx)
dffx=df.stack().reset_index().rename(columns={'level_0':'X', 'level_1':'Y', 0:'I'})
dffx.describe()


# Radial velocity map

fig, ax = plt.subplots(figsize=(5, 5))

image_data=rad_vel[0].data

plt.imshow(image_data, cmap='viridis')

ax.set_xlabel('pixels')
ax.set_ylabel('pixels')

cbar = plt.colorbar()
cbar.set_label('km/s', rotation=270, labelpad=15)  
plt.clim(-25,25) 

plt.title('OIII Radial Velocity')


# Matrix to List

vel=rad_vel[0].data
df=pd.DataFrame(vel)
dfvr=df.stack().reset_index().rename(columns={'level_0':'X', 'level_1':'Y', 0:'RV'})
dfvr.describe()


# Sigma map

fig, ax = plt.subplots(figsize=(5, 5))

image_data=sigma[0].data

plt.imshow(image_data, cmap='magma')

ax.set_xlabel('X coordintate')
ax.set_ylabel('Y coordintate')

cbar = plt.colorbar()
cbar.set_label('km/s', rotation=270, labelpad=15)  

#plt.gca().invert_yaxis()
plt.title('OIII σ')


# Matrix to list

sig=sigma[0].data
df=pd.DataFrame(sig)
dsig=df.stack().reset_index().rename(columns={'level_0':'X', 'level_1':'Y', 0:'Sig'})
dsig.describe()


# Merge previous lists

data=dfvr
data['I']=dffx.I
data['Sig']=dsig.Sig
data.describe()


sns.pairplot(data, 
             vars=["I","RV","Sig"], 
             diag_kind='hist',  
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# Clean data

#mI=(data.I>data.I.mean()+0.25*data.I.std())#&(data.Sig>data.Sig.mean()+0.5*data.Sig.std())
mI=(data.I>1)#&(data.Sig>1)
#mI=(data.Sig>5)
data2=data[mI]


data2.describe()


sns.pairplot(data2, 
             vars=["I","RV","Sig"], 
             diag_kind='hist',  
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none',color='green'),
             diag_kws=dict(bins=20, color="green"),
            )


data.describe()


# standars errors

data2.sem()


# Clean maps and other results

plt.style.use([
    "seaborn-poster",
])

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot()

datal=data2.copy()
#datal.I=np.log10(datal.I)

datal.X=datal.X.astype(int)
dataH_f=(datal.round(2)).pivot(index='X', columns='Y', values='I')
sns.heatmap(dataH_f, cmap="inferno",xticklabels='auto',cbar_kws={'label': 'Flux'})
#plt.title('H$_{α}$ Flux')
plt.title('NGC 604')

#plt.savefig('Imgs//Flux//N604.pdf', bbox_inches='tight')


plt.style.use([
    "seaborn-poster",
])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

datal=data2.copy()

dataH_rv=(datal.round(2)).pivot(index='X', columns='Y', values='RV')
sns.heatmap(dataH_rv, cmap="RdBu_r",cbar_kws={'label': 'km/s'})
plt.title('NGC 604')
ax.set_facecolor('xkcd:gray')

plt.axhline(y=20, xmin=0.05, xmax=0.39, linewidth=2, color = 'k')

ax.text(0.32, 0.9, '60 pc',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)

#plt.savefig('Imgs//VF//N604.pdf', bbox_inches='tight')


plt.style.use([
    "seaborn-poster",
])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

datal=data2.copy()

dataH_s=(datal).pivot(index='X', columns='Y', values='Sig')

sns.heatmap(dataH_s, cmap="magma",cbar_kws={'label': 'km/s'})
plt.title('H$_{α}$ σ')
#plt.gca().invert_yaxis()
#plt.savefig('TAURUS/Imgs/A/'+reg+'SH.png')


# Sigma disp - deconvolution

data2['SigDisp']=(data2.Sig**2-2.28**2-13.4**2)**0.5


data2.describe()


fig, ax = plt.subplots()
plt.scatter(np.log10(data2.I),data2.SigDisp, alpha=0.04, color='k',label='NGC 604')

ax.set_xlabel('Log I')
ax.set_ylabel('$σ_{disp}$ [km/s]')
plt.legend()

fig, ax = plt.subplots()
plt.scatter(np.log10(data2.I),data2.RV, alpha=0.04, color='k',label='NGC 604')

ax.set_xlabel('Log I')
ax.set_ylabel('centroid velocity [km/s]')
plt.legend()

fig, ax = plt.subplots()
plt.scatter(data2.RV,data2.SigDisp, alpha=0.04, color='k',label='NGC 604')

ax.set_xlabel('centroid velocity [km/s]')
ax.set_ylabel('$σ_{disp}$ [km/s]')
plt.legend()


#fig, ax = plt.subplots()

sns.displot(data2.RV-data2.RV.mean(),bins=25)
plt.xlim(-25,25)

#plt.text(0.65, 1.20,'n ='+str(data.RV.count()), ha='center', va='center', transform=ax.transAxes, color='k')
#plt.text(0.70, 0.82,'$μ$ ='+str(np.round(data.RV.mean(),2))+' km/s', ha='center', va='center', transform=ax.transAxes, color='k')
#plt.text(0.65, 1.3,'$σ^{2}$ ='+str(np.round(data.RV.var(),2))+' km$^{2}$/s$^{2}$', ha='center', va='center', transform=ax.transAxes, color='k')

plt.title('NGC 604')

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]="15"

plt.xlabel('Radial velocity [km/s]')


# Export data

data_export_list = {
        'name': name_export, 
        'pc' : pc,
        's0' : s0,
        'pix' : pix,
         name_export : np.array(data2),
      }
data_export_list


data_export_matrix = {
       'name': name_export, 
       'pc' : pc,
       's0' : s0,
       'pix' : pix,
       'sb' :np.array(dataH_f),
       'vv' : np.array(dataH_rv),
       'ss' : np.array(dataH_s),

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


get_ipython().system('jupyter nbconvert --to script --no-prompt otv-TAU-N604-O.ipynb')

