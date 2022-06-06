#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from astropy.io import fits
import astropy.units as u
from astropy.table import Table
import json
#plt.rcParams["font.family"]="Times New Roman"
#plt.rcParams["font.size"]="17"


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

data_file = 'FLA-Car'


name_export='FLA-Car-S'


dist = 2130 #parsecs
pc = dist*(2*np.pi) / (360 * 60 * 60)
s0=(0.9*pc)
pc,s0


damiani_tab1_file = str(datapath_obs) +'/' + data_file + ".fits"
tab = Table.read(damiani_tab1_file)
tab


df = tab.to_pandas()
df.describe()


m= tab["[SII]2sigmab"] > 5


fig, [axb, axr, axd] = plt.subplots(3, 1, sharex=True)
axb.hist(tab["[SII]2RVb"][m], label='Blue comp')
axr.hist(tab["[SII]2RVr"][m], color='r', label='Red comp')
axd.hist(tab["[SII]2RVr"][m] - tab["[SII]2RVb"][m], color='g', label='Delta')
for ax in axb, axr, axd:
    ax.legend()
axd.set(xlabel='Velocity')


xxx


df = df.assign(Ha_dV=df["[SII]2RVr"] - df["[SII]2RVb"])
df = df.assign(Ha_close=(df['Ha_dV'] < 15.0).astype('S5') )
df = df.assign(Ha_rb_ratio=np.log10(df['[SII]2Nr']/df['[SII]2Nb']))


# Blue component

sns.pairplot(df[m], 
             vars=["[SII]2RVb", "[SII]2Nb", "[SII]2sigmab"], 
             diag_kind='hist', hue="Ha_close", 
             plot_kws=dict(alpha=0.2, s=10, edgecolor='none'),
             diag_kws=dict(bins=20),
            )


# fig, ax = plt.subplots()
# plt.scatter(np.log10(df.HaNb),df.Hasigmab, alpha=0.1, color='k', label='CarB')
# 
# ax.set_xlabel('Log I')
# ax.set_ylabel('$σ_{LOS}$ [km/s]')
# plt.legend()
# 
# fig, ax = plt.subplots()
# plt.scatter(np.log10(df.HaNb),df.HaRVb, alpha=0.1, color='k', label='CarB')
# 
# ax.set_xlabel('Log I')
# ax.set_ylabel('centroid velocity [km/s]')
# plt.legend()
# 
# fig, ax = plt.subplots()
# plt.scatter(df.HaRVb,df.Hasigmab, alpha=0.1, color='k', label='CarB')
# 
# ax.set_ylabel('$σ_{LOS}$ [km/s]')
# ax.set_xlabel('centroid velocity [km/s]')
# plt.legend()
# 
# plt.rcParams["font.size"]="17"
# 
# #fig.savefig('CarinaBlue.pdf', bbox_inches='tight')

# Red Component

# mask = df['Hasigmar'] > 35.0
# df = df[~mask]
# 

# df.dropna(inplace=True)

# sns.pairplot(df, 
#              vars=["HaRVr", "HaNr", "Hasigmar"], 
#              diag_kind='hist', hue="Ha_close",
#              plot_kws=dict(alpha=0.3, s=10, edgecolor='none'),
#              diag_kws=dict(bins=20),
#             )

# fig, ax = plt.subplots()
# plt.scatter(np.log10(df.HaNr),df.Hasigmar, alpha=0.1, color='k', label='CarR')
# 
# ax.set_xlabel('Log I')
# ax.set_ylabel('$σ_{LOS}$ [km/s]')
# 
# plt.legend()
# 
# fig, ax = plt.subplots()
# plt.scatter(np.log10(df.HaNr),df.HaRVr, alpha=0.1, color='k', label='CarR')
# 
# plt.legend()
# 
# ax.set_xlabel('Log I')
# ax.set_ylabel('centroid velocity [km/s]')
# 
# fig, ax = plt.subplots()
# plt.scatter(df.HaRVr,df.Hasigmar, alpha=0.1, color='k', label='CarR')
# 
# ax.set_ylabel('$σ_{LOS}$ [km/s]')
# ax.set_xlabel('centroid velocity [km/s]')
# plt.legend()
# 
# plt.rcParams["font.size"]="17"
# 
# #fig.savefig('CarinaRed.pdf', bbox_inches='tight')

# plt.figure(figsize=(20, 4))
# 
# plt.subplot(131)
# plt.scatter(df.HaRVb,df.Hasigmab, alpha=0.075, color='k', label='blue')
# plt.xlabel('centroid velocity [km/s]')
# plt.ylabel('$σ_{LOS}$ [km/s]')
# plt.legend()
# 
# plt.subplot(132)
# plt.scatter(df.HaRVr,df.Hasigmar, alpha=0.075, color='k', label='red')
# plt.xlabel('centroid velocity [km/s]')
# plt.ylabel('$σ_{LOS}$ [km/s]')
# plt.legend()
# 
# plt.show()
# 
# plt.rcParams["font.size"]="17"
# 
# #fig.savefig('CarinaLOSvsPOS.pdf', bbox_inches='tight')

# Combining Components

def combine_moments(f1, v1, s1, f2, v2, s2, return_skew=False):
    """Find combined flux, mean velocity, and sigma for two components 
    with fluxes `f1` and `f2`, velocities `v1` and `v2`, and sigmas `s1` and `s2`. 
    Returns tuple of the combined moments: `f`, `v`, `s`."""
    f = f1 + f2
    v = (v1*f1 + v2*f2)/f
    ss = (s1*s1*f1 + s2*s2*f2)/f
    ss += f1*f2*(v1 - v2)**2 / f**2
    s = np.sqrt(ss)
    if return_skew:
        p1 = f1/f
        p2 = f2/f
        skew = p1*p2*(v1 - v2)*((1 - 2*p1)*(v1 - v2)**2 + 3*(s1**2 - s2**2))
        skew /= (p1*(p2*(v1 - v2)**2 + s1**2 - s2**2) + s2**2)**1.5
#        vmode = np.where(f1 > f2, v1, v2)
#        mskew = (v - vmode)/s
        return f, v, s, skew
    else:
        return f, v, s


fHa, vHa, sHa, gHa = combine_moments(
    df["[SII]2Nr"],df["[NII]RVr"],df["[NII]sigmar"],
    df["[SII]2Nb"],df["[NII]RVb"],df["[NII]sigmab"],
    return_skew=True
)








dfHa = pd.DataFrame(
    {'log_F': np.log10(fHa), 
     'V_mean': vHa, 
     'sigma': sHa, 
     'skew': gHa,
     'R_B': df.Ha_rb_ratio,
     'dV': df.Ha_dV,
     'close': df.Ha_close,
     'RAdeg': df.RAdeg,
     'DEdeg': df.DEdeg,
    }
).dropna()


dfHa.describe()


# Maps

points_of_interest = {
    "eta Car": [161.26517, -59.684425],
    "Tr 14": [160.98911, -59.547698],
    "WR 25": [161.0433, -59.719735],
    "Finger": [161.13133, -59.664035],
}
def mark_points(ax):
    for label, c in points_of_interest.items():
        ax.plot(c[0], c[1], marker='+', markersize='12', color='k')


with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, c=df['[NII]Nb'], cmap='gray_r', vmin=0.0, vmax=4e5)
    fig.colorbar(scat, ax=ax)
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('H alpha blue layer brightness')


with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, c=df['[NII]Nr'], cmap='gray_r', vmin=0.0, vmax=4e5)
    fig.colorbar(scat, ax=ax)
    mark_points(ax)
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title('H alpha red layer brightness')


with sns.axes_style("darkgrid"):
    fig, [axr, axb] = plt.subplots(1, 2, figsize=(18, 8))
    scat = axr.scatter(df.RAdeg, df.DEdeg, 
                      s=40*(np.log10(df.HaNr/df.HaNb) + 1.3), 
                      c=df.HaRVr, cmap='RdBu_r',
                      vmin=-55, vmax=35, 
                     )
    
    axr.text(0.53, 0.2, '7 pc',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)
    
    axr.axhline(y=-59.65, xmin=0.655, xmax=0.91, linewidth=2, color = 'k')
    
    scat = axb.scatter(df.RAdeg, df.DEdeg, 
                      s=40*(np.log10(df.HaNb/df.HaNr) + 1.3), 
                      c=df.HaRVb, cmap='RdBu_r',
                      vmin=-55, vmax=35,
                     )
    
#    scat2 = ax.scatter(df.RAdeg, df.DEdeg, 
#                      s=50*(np.log10(df.HaNr) - 3), 
#                      c=df.HaRVr, cmap='RdBu_r',
#                      vmin=-55, vmax=35, marker='+',
#                     )
    fig.colorbar(scat, ax=[axr, axb])
    mark_points(axr)
    mark_points(axb)
    axr.invert_xaxis()
    axr.set_aspect(2.0)
    axb.invert_xaxis()
    axb.set_aspect(2.0)  
    axr.set_title('red layer velocity')
    axb.set_title('blue layer velocity')


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 12))
    scat = ax.scatter(dfHa.RAdeg, dfHa.DEdeg, s=8*(dfHa.sigma - 12), c=dfHa.V_mean-dfHa.V_mean.mean(), cmap='RdBu_r')
    mark_points(ax)
    fig.colorbar(scat, ax=ax).set_label("$V$")
    ax.invert_xaxis()
    ax.set_aspect(2)
    ax.set_title("H alpha mean velocity")
    
    ax.text(0.32, 0.2, '7 pc',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)
    ax.axhline(y=-59.83, xmin=0.14, xmax=0.395, linewidth=2, color = 'k')


# Structure Fucntion

# Dr. Will Blue

df2 = df[['RAdeg', 'DEdeg', 'HaRVb']].copy()
df2.rename(columns = {'RAdeg' : 'X', 'DEdeg' : 'Y', 'HaRVb' : 'RV'}, inplace = True)
#df2.describe()


# Red Dr. Will

df3 = df[['RAdeg', 'DEdeg', 'HaRVr']].copy()
df3.rename(columns = {'RAdeg' : 'X', 'DEdeg' : 'Y', 'HaRVr' : 'RV'}, inplace = True)
#df3.describe()


# Combined

dfHa





df4 = dfHa[['RAdeg', 'DEdeg', 'V_mean','log_F','sigma']].copy()
df4.rename(columns = {'RAdeg' : 'X', 'DEdeg' : 'Y', 'V_mean' : 'RV', 'log_F': 'I', 'sigma':'Sigma'}, inplace = True)
#df4


# Data to pc

(((df4.X[1]-df4.X[0])**2)-((df4.Y[1]-df4.Y[0])**2))**0.5


# Export archives

# List Form

data_export_list = {
        'name': name_export, 
        'pc' : pc,
        's0' : s0,
 #       'pix' : pix,
         name_export : np.array(df4),
      }
data_export_list


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


jsonfilename =name_export +"-l.json"
with open(datapath_res/jsonfilename, "w") as f:
    json.dump(data_export_list, fp=f, indent=3, cls=MyEncoder)


get_ipython().system('jupyter nbconvert --to script --no-prompt otv-FLA-Car-S.ipynb')


# 










print("--- %s seconds ---" % (time.time()-start_time))

