#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()


from pathlib import Path

from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_color_codes()
import json

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]="17"


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

data_file = 'FLA-M8'


name_export='FLA-M8-H'


# Input data of the region

dist = 1250 #parsecs
seeing = 0.9 #seeing arcsec FWHM


pc = dist*(2*np.pi) / (360 * 60 * 60)
s0 = (seeing*pc)/2.355 #seeing pc RMS

pc,s0


damiani_tab1_file = str(datapath_obs) + '\\' +data_file + ".fits"
tab = Table.read(damiani_tab1_file)
tab
df = tab.to_pandas()
df.describe()


m=df['sigHalpha'] < df['sigHalpha'].mean()+4*df['sigHalpha'].std()


df=df[m]


sns.pairplot(df[m],
             vars=["RVHalpha", "sigHalpha", "NormHalpha"],
             diag_kind='hist',
             plot_kws=dict(alpha=0.3, s=10, edgecolor='none', color="blue"),
             diag_kws=dict(bins=20, color="blue"),
            )


fig, ax = plt.subplots()
plt.scatter(np.log10(df.NormHalpha),df.sigHalpha, alpha=0.1, color='k', label='M8')

ax.set_xlabel('Log I')
ax.set_ylabel('$σ_{LOS}$ [km/s]')
plt.legend()

fig, ax = plt.subplots()
plt.scatter(np.log10(df.NormHalpha),df.RVHalpha, alpha=0.1, color='k', label='M8')

ax.set_xlabel('Log I')
ax.set_ylabel('centroid velocity [km/s]')
plt.legend()

fig, ax = plt.subplots()
plt.scatter(df.RVHalpha,df.sigHalpha, alpha=0.1, color='k', label='M8')

ax.set_ylabel('$σ_{LOS}$ [km/s]')
ax.set_xlabel('centroid velocity [km/s]')
plt.legend()

plt.rcParams["font.size"]="17"


df.describe()


df2 = df[['RAdeg', 'DEdeg', 'RVHalpha','NormHalpha','sigHalpha']].copy()
df2.rename(columns = {'RAdeg' : 'X', 'DEdeg' : 'Y', 'RVHalpha' : 'RV','NormHalpha':'I','sigHalpha':'Sig',}, inplace = True)
df2.describe()


data=df2


#fig, ax = plt.subplots()

sns.displot(data.RV-data.RV.mean(),bins=25)
plt.xlim(-10,10)


plt.text(0.7, 1.2,'n ='+str(data.RV.count()), ha='center', va='center', transform=ax.transAxes, color='k')
#plt.text(0.35, 0.78,'$μ$ ='+str(np.round(data.RV.mean(),2))+' km/s', ha='center', va='center', transform=ax.transAxes, color='k')
plt.text(0.7, 1.3,'$σ^{2}$ ='+str(np.round(data.RV.var(),2))+' km$^{2}$/s$^{2}$', ha='center', va='center', transform=ax.transAxes, color='k')

plt.title('Lagoon')

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]="15"

plt.xlabel('Radial velocity [km/s]')


types = ['HD 164536', '7 Sgr', 'Herschel 36', '9 Sgr', 'HD 164816', 'HD 164865', 'M8E-IR', 'HD 165052','HD 165246']
x_coords = [270.6609, 270.7129, 270.9180, 270.9685, 270.9869, 271.0634, 271.2244, 271.2940,271.5195]
y_coords = [-24.2554, -24.2825, -24.3785, -24.3607, -24.3126, -24.1834, -24.4448, -24.3986,-24.1955]


points_of_interest = {
    "HD 164536": [270.6609, -24.2554],
    "7 Sgr": [270.7129, -24.2825],
    "Herschel 36": [270.9180, -24.3785],
    "9 Sgr": [270.9685, -24.3607],
    "HD 164816": [270.9869, -24.3126],
    "HD 164865": [271.0634, -24.1834],
    "M8E-IR": [271.2244, -24.4448],
    "HD 165052": [271.2940, -24.3986],
    "HD 165246": [271.5195, -24.1955],
}
def mark_points(ax):
    for label, c in points_of_interest.items():
        ax.plot(c[0], c[1], marker='+', markersize='12', color='k')


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, 
                      s=0.0015*((df.NormHalpha)), 
                      c=df.RVHalpha,cmap='coolwarm' 
                     )
    fig.colorbar(scat, ax=[ax])
    #mark_points(ax)
    #ax.set_facecolor('k')
    #ax.axis('equal')
    ax.set_aspect('equal', 'datalim')
    fig.colorbar(scat, ax=ax).set_label("km/s")

    ax.invert_xaxis()

    ax.text(0.855, 0.1, '5 pc',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)
    
    plt.axhline(y=-24.65, xmin=0.725, xmax=0.905, linewidth=2, color = 'k')

    ax.set(xlabel='R.A.', ylabel='Dec')

    
for i,type in enumerate(types):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='+', color='yellow')
    plt.text(x, y, type, fontsize=14)


with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(figsize=(12, 6))
    scat = ax.scatter(df.RAdeg, df.DEdeg, s=100, c=np.log10(df.NormHalpha), cmap='inferno', vmin=3.5, vmax=5.5)
    fig.colorbar(scat, ax=ax).set_label("log10(F)")
    mark_points(ax)
    ax.set_title('H alpha brightness')
    ax.axis('equal')
    ax.axis([270.5, 271.7, -24.6, -24])
    ax.invert_xaxis()


data_export_list = {
        'name': name_export, 
        'pc' : pc,
        's0' : s0,
 #       'pix' : pix,
         name_export : np.array(data[m]),
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


print("--- %s seconds ---" % (time.time()-start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt otv-FLA-M8-H.ipynb')

