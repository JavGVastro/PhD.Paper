#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json


datapath_names = Path(open("path-name-list.txt", "r").read()).expanduser()


samples_H=pd.read_csv(str(datapath_names) + '//sample-names_H.csv',header=None)
samples_O=pd.read_csv(str(datapath_names) + '//sample-names_O.csv',header=None)
samples_N=pd.read_csv(str(datapath_names) + '//sample-names_N.csv',header=None)
samples_S=pd.read_csv(str(datapath_names) + '//sample-names_S.csv',header=None)


text_file_0 = open("path-results.txt", "r")
path_res = text_file_0.read()
datapath= Path(path_res).expanduser()


data_H= {}
Results_H = {}

for i in range(len(samples_H)):
    data_H[samples_H[0][i]] = json.load(open(str(datapath) + '/' + samples_H[0][i] + "-l.json"))
    Results_H[samples_H[0][i]]=pd.DataFrame(data_H[samples_H[0][i]][samples_H[0][i]])
    if Results_H[samples_H[0][i]].shape[1] == 6:
        Results_H[samples_H[0][i]].columns=['X','Y','RV','I','Sig','SigDisp']
    elif Results_H[samples_H[0][i]].shape[1] == 5:
        Results_H[samples_H[0][i]].columns=['X','Y','RV','I','Sig']
    elif Results_H[samples_H[0][i]].shape[1] == 4:
        Results_H[samples_H[0][i]].columns=['X','Y','RV','I']
    else:
        Results_H[samples_H[0][i]].columns=['X','Y','RV']


data_N= {}
Results_N = {}

for i in range(len(samples_N)):
    data_N[samples_N[0][i]] = json.load(open(str(datapath) + '/' + samples_N[0][i] + "-l.json"))
    Results_N[samples_N[0][i]]=pd.DataFrame(data_N[samples_N[0][i]][samples_N[0][i]])
    if Results_N[samples_N[0][i]].shape[1] == 6:
        Results_N[samples_N[0][i]].columns=['X','Y','RV','I','Sig','SigDisp']
    elif Results_N[samples_N[0][i]].shape[1] == 5:
        Results_N[samples_N[0][i]].columns=['X','Y','RV','I','Sig']
    elif Results_N[samples_N[0][i]].shape[1] == 4:
        Results_N[samples_N[0][i]].columns=['X','Y','RV','I']
    else:
        Results_N[samples_N[0][i]].columns=['X','Y','RV']


data_O= {}
Results_O = {}

for i in range(len(samples_O)):
    data_O[samples_O[0][i]] = json.load(open(str(datapath) + '/' + samples_O[0][i] + "-l.json"))
    Results_O[samples_O[0][i]]=pd.DataFrame(data_O[samples_O[0][i]][samples_O[0][i]])
    if Results_O[samples_O[0][i]].shape[1] == 6:
        Results_O[samples_O[0][i]].columns=['X','Y','RV','I','Sig','SigDisp']
    elif Results_O[samples_O[0][i]].shape[1] == 5:
        Results_O[samples_O[0][i]].columns=['X','Y','RV','I','Sig']
    elif Results_O[samples_O[0][i]].shape[1] == 4:
        Results_O[samples_O[0][i]].columns=['X','Y','RV','I']
    else:
        Results_O[samples_O[0][i]].columns=['X','Y','RV']


data_S= {}
Results_S = {}

for i in range(len(samples_S)):
    data_S[samples_S[0][i]] = json.load(open(str(datapath) + '/' + samples_S[0][i] + "-l.json"))
    Results_S[samples_S[0][i]]=pd.DataFrame(data_S[samples_S[0][i]][samples_S[0][i]])
    if Results_S[samples_S[0][i]].shape[1] == 6:
        Results_S[samples_S[0][i]].columns=['X','Y','RV','I','Sig','SigDisp']
    elif Results_S[samples_S[0][i]].shape[1] == 5:
        Results_S[samples_S[0][i]].columns=['X','Y','RV','I','Sig']
    elif Results_S[samples_S[0][i]].shape[1] == 4:
        Results_S[samples_S[0][i]].columns=['X','Y','RV','I']
    else:
        Results_S[samples_S[0][i]].columns=['X','Y','RV']


for i in range(len(samples_H)):
    RVhist = Results_H[samples_H[0][i]].RV - Results_H[samples_H[0][i]].RV.mean()
    Results_H[samples_H[0][i]]['RVhist'] = RVhist
    
for i in range(len(samples_N)):
    RVhist = Results_N[samples_N[0][i]].RV - Results_N[samples_N[0][i]].RV.mean()
    Results_N[samples_N[0][i]]['RVhist'] = RVhist
    
for i in range(len(samples_O)):
    RVhist = Results_O[samples_O[0][i]].RV - Results_O[samples_O[0][i]].RV.mean()
    Results_O[samples_O[0][i]]['RVhist'] = RVhist
    
for i in range(len(samples_S)):
    RVhist = Results_S[samples_S[0][i]].RV - Results_S[samples_S[0][i]].RV.mean()
    Results_S[samples_S[0][i]]['RVhist'] = RVhist


samples_H,samples_N,samples_O,samples_S


sns.set_context("talk", font_scale=1.2)


#mI=data.I>0

fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

name = 'Orion core'

h = 4
sns.histplot(data=Results_H[samples_H[0][h]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='b', stat='density', kde=True)
axes.text(.6, .80,'$\sigma_H^2$ = ' + str(np.round(Results_H[samples_H[0][h]].RV.var(),2))+ ' [km/s]$^2$',  color='b', transform=axes.transAxes)

n = 0
sns.histplot(data=Results_N[samples_N[0][n]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='g', stat='density', kde=True)
axes.text(.6, .70,'$\sigma_N^2$ = ' + str(np.round(Results_N[samples_N[0][n]].RV.var(),2))+ ' [km/s]$^2$',  color='g', transform=axes.transAxes)


o = 1
sns.histplot(data=Results_N[samples_N[0][o]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='r', stat='density', kde=True)
axes.text(.6, .60,'$\sigma_O^2$ = ' + str(np.round(Results_N[samples_N[0][o]].RV.var(),2))+ ' [km/s]$^2$',  color='r', transform=axes.transAxes)

s = 0
sns.histplot(data=Results_S[samples_S[0][s]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='orange', stat='density', kde=True)
axes.text(.6, .50,'$\sigma_S^2$ = ' + str(np.round(Results_S[samples_S[0][s]].RV.var(),2))+ ' [km/s]$^2$',  color='orange', transform=axes.transAxes)


axes.text(.6, .90,name,  color='k', transform=axes.transAxes)

plt.savefig('Imgs//hist_' + name + '.pdf', bbox_inches='tight')

#plt.xlim(-100,100)

#plt.legend()


#mI=data.I>0

fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

name = 'EON'

h = 7
sns.histplot(data=Results_H[samples_H[0][h]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='b', stat='density', kde=False)
axes.text(.6, .80,'$\sigma_H^2$ = ' + str(np.round(Results_H[samples_H[0][h]].RV.var(),2))+ ' [km/s]$^2$',  color='b', transform=axes.transAxes)

n = 2
sns.histplot(data=Results_N[samples_N[0][n]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='g', stat='density', kde=False)
axes.text(.6, .70,'$\sigma_N^2$ = ' + str(np.round(Results_N[samples_N[0][n]].RV.var(),2))+ ' [km/s]$^2$',  color='g', transform=axes.transAxes)


o = 3
sns.histplot(data=Results_N[samples_N[0][o]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='r', stat='density', kde=False)
axes.text(.6, .60,'$\sigma_O^2$ = ' + str(np.round(Results_N[samples_N[0][o]].RV.var(),2))+ ' [km/s]$^2$',  color='r', transform=axes.transAxes)

s = 2
sns.histplot(data=Results_S[samples_S[0][s]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='orange', stat='density', kde=False)
axes.text(.6, .50,'$\sigma_S^2$ = ' + str(np.round(Results_S[samples_S[0][s]].RV.var(),2))+ ' [km/s]$^2$',  color='orange', transform=axes.transAxes)


axes.text(.6, .90,name,  color='k', transform=axes.transAxes)

plt.savefig('Imgs//hist_' + name + '.pdf', bbox_inches='tight')

#plt.xlim(-100,100)

#plt.legend()


#mI=data.I>0

fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

name = 'Carina'

h = 9
sns.histplot(data=Results_H[samples_H[0][h]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='b', stat='density', kde=False)
axes.text(.6, .80,'$\sigma_H^2$ = ' + str(np.round(Results_H[samples_H[0][h]].RV.var(),2))+ ' [km/s]$^2$',  color='b', transform=axes.transAxes)

n = 4
sns.histplot(data=Results_N[samples_N[0][n]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='g', stat='density', kde=False)
axes.text(.6, .70,'$\sigma_N^2$ = ' + str(np.round(Results_N[samples_N[0][n]].RV.var(),2))+ ' [km/s]$^2$',  color='g', transform=axes.transAxes)


#o = 3
#sns.histplot(data=Results_N[samples_N[0][o]],
#                 x="RVhist", binwidth=1, element="step", fill=False, color='r', stat='density', kde=False)
#axes.text(.55, .60,'$\sigma_O^2$ = ' + str(np.round(Results_N[samples_N[0][o]].RV.var(),2))+ ' [km/s]$^2$',  color='r', transform=axes.transAxes)

s = 4
sns.histplot(data=Results_S[samples_S[0][s]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='orange', stat='density', kde=False)
axes.text(.6, .60,'$\sigma_S^2$ = ' + str(np.round(Results_S[samples_S[0][s]].RV.var(),2))+ ' [km/s]$^2$',  color='orange', transform=axes.transAxes)


axes.text(.6, .90,name,  color='k', transform=axes.transAxes)

plt.savefig('Imgs//hist_' + name + '.pdf', bbox_inches='tight')

#plt.xlim(-100,100)

#plt.legend()


#mI=data.I>0

fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

name = 'M8'

h = 8
sns.histplot(data=Results_H[samples_H[0][h]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='b', stat='density', kde=False)
axes.text(.6, .80,'$\sigma_H^2$ = ' + str(np.round(Results_H[samples_H[0][h]].RV.var(),2))+ ' [km/s]$^2$',  color='b', transform=axes.transAxes)

n = 3
sns.histplot(data=Results_N[samples_N[0][n]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='g', stat='density', kde=False)
axes.text(.6, .70,'$\sigma_N^2$ = ' + str(np.round(Results_N[samples_N[0][n]].RV.var(),2))+ ' [km/s]$^2$',  color='g', transform=axes.transAxes)


#o = 3
#sns.histplot(data=Results_N[samples_N[0][o]],
#                 x="RVhist", binwidth=1, element="step", fill=False, color='r', stat='density', kde=False)
#axes.text(.55, .60,'$\sigma_O^2$ = ' + str(np.round(Results_N[samples_N[0][o]].RV.var(),2))+ ' [km/s]$^2$',  color='r', transform=axes.transAxes)

s = 3
sns.histplot(data=Results_S[samples_S[0][s]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='orange', stat='density', kde=False)
axes.text(.6, .60,'$\sigma_S^2$ = ' + str(np.round(Results_S[samples_S[0][s]].RV.var(),2))+ ' [km/s]$^2$',  color='orange', transform=axes.transAxes)


axes.text(.6, .90,name,  color='k', transform=axes.transAxes)

plt.savefig('Imgs//hist_' + name + '.pdf', bbox_inches='tight')

#plt.xlim(-100,100)

#plt.legend()


#mI=data.I>0

fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

name = 'N346'

h = 6
sns.histplot(data=Results_H[samples_H[0][h]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='b', stat='density', kde=True)
axes.text(.6, .80,'$\sigma_H^2$ = ' + str(np.round(Results_H[samples_H[0][h]].RV.var(),2))+ ' [km/s]$^2$',  color='b', transform=axes.transAxes)

o = 2
sns.histplot(data=Results_N[samples_N[0][o]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='r', stat='density', kde=True)
axes.text(.6, .70,'$\sigma_O^2$ = ' + str(np.round(Results_N[samples_N[0][o]].RV.var(),2))+ ' [km/s]$^2$',  color='r', transform=axes.transAxes)

s = 1
sns.histplot(data=Results_S[samples_S[0][s]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='orange', stat='density', kde=True)
axes.text(.6, .60,'$\sigma_S^2$ = ' + str(np.round(Results_S[samples_S[0][s]].RV.var(),2))+ ' [km/s]$^2$',  color='orange', transform=axes.transAxes)


axes.text(.6, .90,name,  color='k', transform=axes.transAxes)

plt.savefig('Imgs//hist_' + name + '.pdf', bbox_inches='tight')

#plt.xlim(-100,100)

#plt.legend()


#mI=data.I>0

fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

name = 'Dor'

h = 5
sns.histplot(data=Results_H[samples_H[0][h]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='b', stat='density', kde=True)
axes.text(.6, .80,'$\sigma_H^2$ = ' + str(np.round(Results_H[samples_H[0][h]].RV.var(),2))+ ' [km/s]$^2$',  color='b', transform=axes.transAxes)



n = 1
sns.histplot(data=Results_N[samples_N[0][n]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='g', stat='density', kde=True)
axes.text(.6, .70,'$\sigma_N^2$ = ' + str(np.round(Results_N[samples_N[0][n]].RV.var(),2))+ ' [km/s]$^2$',  color='g', transform=axes.transAxes)



axes.text(.6, .90,name,  color='k', transform=axes.transAxes)

plt.savefig('Imgs//hist_' + name + '.pdf', bbox_inches='tight')

#plt.xlim(-100,100)

#plt.legend()


fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

sns.histplot(data=Results_H[samples_H[0][0]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='b', stat='density', kde=True)
axes.text(.55, .80,'$\sigma_H^2$ = ' + str(np.round(Results_H[samples_H[0][0]].RV.var(),2))+ ' [km/s]$^2$',  color='b', transform=axes.transAxes)



sns.histplot(data=Results_O[samples_O[0][0]],
                 x="RVhist", binwidth=1, element="step", fill=False, color='r', stat='density', kde=True)
axes.text(.55, .70,'$\sigma_O^2$ = ' + str(np.round(Results_O[samples_O[0][0]].RV.var(),2))+ ' [km/s]$^2$',  color='r', transform=axes.transAxes)

axes.text(.55, .90,'NGC 604',  color='k', transform=axes.transAxes)

plt.savefig('Imgs//hist_N604.pdf', bbox_inches='tight')

#plt.legend()


print("--- %s seconds ---" % (time.time()-start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt velocities-distributions_multiple_emission_lines.ipynb')

