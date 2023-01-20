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
datapath_res = Path(open("path-results.txt", "r").read()).expanduser()


samples_H=pd.read_csv(str(datapath_names) + '//sample-names_H.csv',header=None)
samples_O=pd.read_csv(str(datapath_names) + '//sample-names_O.csv',header=None)
samples_N=pd.read_csv(str(datapath_names) + '//sample-names_N.csv',header=None)
samples_S=pd.read_csv(str(datapath_names) + '//sample-names_S.csv',header=None)


data_H = {}
data_O = {}
data_N = {}
data_S = {}

for i in range(len(samples_H)):
        data_H[samples_H[0][i]] = json.load(open(str(datapath_res) + '/' + samples_H[0][i] + ".json"))
for i in range(len(samples_O)):
        data_O[samples_O[0][i]] = json.load(open(str(datapath_res) + '/' + samples_O[0][i] + ".json"))
for i in range(len(samples_N)):
        data_N[samples_N[0][i]] = json.load(open(str(datapath_res) + '/' + samples_N[0][i] + ".json"))
for i in range(len(samples_S)):
        data_S[samples_S[0][i]] = json.load(open(str(datapath_res) + '/' + samples_S[0][i] + ".json"))


sig2_H = [[0]*(1) for i in range(len(samples_H))]
sig2_O = [[0]*(1) for i in range(len(samples_O))]
sig2_N = [[0]*(1) for i in range(len(samples_N))]
sig2_S = [[0]*(1) for i in range(len(samples_S))]

sig2vm_H = [[0]*(1) for i in range(len(samples_H))]
sig2vm_O = [[0]*(1) for i in range(len(samples_O))]
sig2vm_N = [[0]*(1) for i in range(len(samples_N))]
sig2vm_S = [[0]*(1) for i in range(len(samples_S))]

r0_H = [[0]*(1) for i in range(len(samples_H))]
r0_O = [[0]*(1) for i in range(len(samples_O))]
r0_N = [[0]*(1) for i in range(len(samples_N))]
r0_S = [[0]*(1) for i in range(len(samples_S))]

m_H = [[0]*(1) for i in range(len(samples_H))]
m_O = [[0]*(1) for i in range(len(samples_O))]
m_N = [[0]*(1) for i in range(len(samples_N))]
m_S = [[0]*(1) for i in range(len(samples_S))]

box_size_H = [[0]*(1) for i in range(len(samples_H))]
box_size_O = [[0]*(1) for i in range(len(samples_O))]
box_size_N = [[0]*(1) for i in range(len(samples_N))]
box_size_S = [[0]*(1) for i in range(len(samples_S))]


data_H[samples_H[0][i]]['preres']['sig2']


for i in range(len(samples_H)):

        sig2_H[i] = data_H[samples_H[0][i]]['results_2sig']['sig2'][0]
        sig2vm_H[i] = data_H[samples_H[0][i]]['preres']['sig2']
        r0_H[i]    = data_H[samples_H[0][i]]['results_2sig']['r0'][0]
        m_H[i]    = data_H[samples_H[0][i]]['results_2sig']['m'][0]
        box_size_H[i] = data_H[samples_H[0][i]]['properties']['box_size']

results_H = pd.DataFrame(
        {
            "sig2" : sig2_H,
            "sig2_obs" : sig2vm_H,
            "r0" : r0_H,
            "m" : m_H,
            "L" : box_size_H,
            "L/r0" : np.array(box_size_H)/np.array(r0_H)
            }
        )

results_H.insert(loc=0, column='Region', value=samples_H)


for i in range(len(samples_O)):

        sig2_O[i] = data_O[samples_O[0][i]]['results_2sig']['sig2'][0]
        sig2vm_O[i] = data_O[samples_O[0][i]]['preres']['sig2']
        r0_O[i]    = data_O[samples_O[0][i]]['results_2sig']['r0'][0]
        m_O[i]    = data_O[samples_O[0][i]]['results_2sig']['m'][0]
        box_size_O[i] = data_O[samples_O[0][i]]['properties']['box_size']

results_O = pd.DataFrame(
        {
            "sig2" : sig2_O,
            "sig2_obs" : sig2vm_O,
            "r0" : r0_O,
            "m" : m_O,
            "L" : box_size_O,
            "L/r0" : np.array(box_size_O)/np.array(r0_O)
            }
        )

results_O.insert(loc=0, column='Region', value=samples_O)


for i in range(len(samples_N)):

        sig2_N[i] = data_N[samples_N[0][i]]['results_2sig']['sig2'][0]
        sig2vm_N[i] = data_N[samples_N[0][i]]['preres']['sig2']
        r0_N[i]    = data_N[samples_N[0][i]]['results_2sig']['r0'][0]
        m_N[i]    = data_N[samples_N[0][i]]['results_2sig']['m'][0]
        box_size_N[i] = data_N[samples_N[0][i]]['properties']['box_size']

results_N = pd.DataFrame(
        {
            "sig2" : sig2_N,
            "sig2_obs" : sig2vm_N,
            "r0" : r0_N,
            "m" : m_N,
            "L" : box_size_N,
            "L/r0" : np.array(box_size_N)/np.array(r0_N)
            
            }
        )

results_N.insert(loc=0, column='Region', value=samples_N)


for i in range(len(samples_S)):

        sig2_S[i] = data_S[samples_S[0][i]]['results_2sig']['sig2'][0]
        sig2vm_S[i] = data_S[samples_S[0][i]]['preres']['sig2']
        r0_S[i]    = data_S[samples_S[0][i]]['results_2sig']['r0'][0]
        m_S[i]    = data_S[samples_S[0][i]]['results_2sig']['m'][0]
        box_size_S[i] = data_S[samples_S[0][i]]['properties']['box_size']

results_S = pd.DataFrame(
        {
            "sig2" : sig2_S,
            "sig2_obs" : sig2vm_S,
            "r0" : r0_S,
            "m" : m_S,
            "L" : box_size_S,
            "L/r0" : np.array(box_size_S)/np.array(r0_S)
            }
        )

results_S.insert(loc=0, column='Region', value=samples_S)


results_H, results_O, results_N, results_S


frames = [results_H, results_O, results_N, results_S]
result = pd.concat(frames)
result = result.sort_values('Region')
result.round(2).to_latex('latex-files/multiple_lines.tex', escape=False, caption='Multiple emission lines',index=False,label ='tab:preresults')


result


sns.set_context("talk", font_scale=1.2)
fig, ax = plt.subplots(figsize = (8,8))

ax.scatter(data_H[samples_H[0][0]]['results_2sig']['sig2'][0],data_H[samples_H[0][0]]['results_2sig']['r0'][0], c = 'b', marker = 'o', label = 'N604')
ax.scatter(data_O[samples_O[0][0]]['results_2sig']['sig2'][0],data_O[samples_O[0][0]]['results_2sig']['r0'][0], c = 'r', marker = 'o')

ax.scatter(data_H[samples_H[0][4]]['results_2sig']['sig2'][0],data_H[samples_H[0][4]]['results_2sig']['r0'][0], c = 'b', marker = 's', label = 'Orion')
ax.scatter(data_O[samples_O[0][1]]['results_2sig']['sig2'][0],data_O[samples_O[0][1]]['results_2sig']['r0'][0], c = 'r', marker = 's')
ax.scatter(data_N[samples_N[0][0]]['results_2sig']['sig2'][0],data_N[samples_N[0][0]]['results_2sig']['r0'][0], c = 'g', marker = 's')
ax.scatter(data_S[samples_S[0][0]]['results_2sig']['sig2'][0],data_S[samples_S[0][0]]['results_2sig']['r0'][0], c = 'orange', marker = 's')

ax.scatter(data_H[samples_H[0][6]]['results_2sig']['sig2'][0],data_H[samples_H[0][6]]['results_2sig']['r0'][0], c = 'b', marker = 'X', label = 'N346')
ax.scatter(data_O[samples_O[0][2]]['results_2sig']['sig2'][0],data_O[samples_O[0][2]]['results_2sig']['r0'][0], c = 'r', marker = 'X')
ax.scatter(data_S[samples_S[0][1]]['results_2sig']['sig2'][0],data_S[samples_S[0][1]]['results_2sig']['r0'][0], c = 'orange', marker = 'X')

ax.scatter(data_H[samples_H[0][7]]['results_2sig']['sig2'][0],data_H[samples_H[0][7]]['results_2sig']['r0'][0], c = 'b', marker = '^', label = 'EON')
ax.scatter(data_O[samples_O[0][3]]['results_2sig']['sig2'][0],data_O[samples_O[0][3]]['results_2sig']['r0'][0], c = 'r', marker = '^')
ax.scatter(data_N[samples_N[0][2]]['results_2sig']['sig2'][0],data_N[samples_N[0][2]]['results_2sig']['r0'][0], c = 'g', marker = '^')
ax.scatter(data_S[samples_S[0][2]]['results_2sig']['sig2'][0],data_S[samples_S[0][2]]['results_2sig']['r0'][0], c = 'orange', marker = '^')

ax.scatter(data_H[samples_H[0][5]]['results_2sig']['sig2'][0],data_H[samples_H[0][5]]['results_2sig']['r0'][0], c = 'b', marker = 'v', label = '30Dor')
ax.scatter(data_N[samples_N[0][1]]['results_2sig']['sig2'][0],data_N[samples_N[0][1]]['results_2sig']['r0'][0], c = 'g', marker = 'v')

ax.scatter(data_H[samples_H[0][8]]['results_2sig']['sig2'][0],data_H[samples_H[0][8]]['results_2sig']['r0'][0], c = 'b', marker = 'P', label = 'M8')
ax.scatter(data_N[samples_N[0][3]]['results_2sig']['sig2'][0],data_N[samples_N[0][3]]['results_2sig']['r0'][0], c = 'g', marker = 'P')
ax.scatter(data_S[samples_S[0][3]]['results_2sig']['sig2'][0],data_S[samples_S[0][3]]['results_2sig']['r0'][0], c = 'orange', marker = 'P')

ax.scatter(data_H[samples_H[0][9]]['results_2sig']['sig2'][0],data_H[samples_H[0][9]]['results_2sig']['r0'][0], c = 'b', marker = '*', label = 'Carina')
ax.scatter(data_N[samples_N[0][4]]['results_2sig']['sig2'][0],data_N[samples_N[0][4]]['results_2sig']['r0'][0], c = 'g', marker = '*')
ax.scatter(data_S[samples_S[0][4]]['results_2sig']['sig2'][0],data_S[samples_S[0][4]]['results_2sig']['r0'][0], c = 'orange', marker = '*')

ax.set( xscale = 'log',  yscale = 'log',
#    xlim=[-1, 3], ylim=[-1.5, 1.5],
    xlabel=r"log $σ^2$", ylabel=r"log $r_{0}$ [pc]",
)

plt.legend()

plt.savefig('Imgs//params_multiple.pdf', bbox_inches='tight')


get_ipython().system('jupyter nbconvert --to script --no-prompt results-compiler-v2_emission_lines.ipynb')

