#!/usr/bin/env python
# coding: utf-8

import time
start_time=time.time()


from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import json


plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]="10"


datapath_names = Path(open("path-name-list.txt", "r").read()).expanduser()


# Previous structure functions

# 50
# 
# http://articles.adsabs.harvard.edu//full/1951ZA.....30...17V/0000045.000.html
# 
# http://articles.adsabs.harvard.edu//full/1958RvMP...30.1035M/0001037.000.html
# 
# 60
# 
# http://articles.adsabs.harvard.edu//full/1961MNRAS.122....1F/0000012.000.html
# 
# 70
# 
# https://ui.adsabs.harvard.edu/abs/1970A%26A.....8..486L/abstract
# 
# 80
# 
# http://articles.adsabs.harvard.edu//full/1985ApJ...288..142R/0000146.000.html 
# 
# https://ui.adsabs.harvard.edu/abs/1986ApJ...304..767O/abstract
# 
# http://articles.adsabs.harvard.edu//full/1987ApJ...317..686O/0000688.000.html
# 
# http://articles.adsabs.harvard.edu//full/1988ApJS...67...93C/0000122.000.html
# 
# 90
# 
# http://articles.adsabs.harvard.edu//full/1995ApJ...454..316M/0000324.000.html 
# 
# https://iopscience.iop.org/article/10.1086/304573/fulltext/33796.text.html#fg4
# 
# https://ui.adsabs.harvard.edu/abs/1999intu.conf..154J/abstract 
# 
# https://ui.adsabs.harvard.edu/abs/1999A%26A...346..947C/abstract
# 
# 00
# 
# 
# https://iopscience.iop.org/article/10.1088/0004-637X/700/2/1847#apj311588f19
# 
# 10
# 
# 
# https://academic.oup.com/mnras/article/413/2/721/1062900
# 
# https://academic.oup.com/mnras/article/463/3/2864/2646581
# 
# https://ui.adsabs.harvard.edu/abs/2019arXiv191203543M/abstract
# 
# 

# Galactic HII regions

Sample='galactic-regions'

samples=pd.read_csv(str(datapath_names) + '/' + Sample+'.csv',header=None)

DataNH=dict()
DataH=dict()


for i in range(len(samples)):
    DataNH[i]=samples[0][i]
    
for i in range(len(samples)):
    DataH[i]=pd.read_csv('data-previous-structure-functions//'+DataNH[i]+'.csv')    


samples


plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]="17"


fig, ax=plt.subplots(figsize=(8,8))

#plt.loglog(DataH[0].pc,DataH[0].S,marker='.',color='orangered',alpha=0.75, markersize=10, label=  DataNH[0])
#plt.loglog(DataH[1].pc,DataH[1].S,marker='.',color='orangered',alpha=0.75, markersize=10, label=  DataNH[1])
#plt.loglog(DataH[2].pc,DataH[2].S,marker='.',color='orange',alpha=0.75, markersize=10, label=  DataNH[2])
plt.loglog(DataH[3].pc,DataH[3].S,marker='.',color='red',alpha=0.75, markersize=7, label=  DataNH[3])
plt.loglog(DataH[4].pc,DataH[4].S,marker='.',color='maroon',alpha=0.75, markersize=7, label=  DataNH[4])
plt.loglog(DataH[5].pc,DataH[5].S,marker='o',color='blue',alpha=0.75, markersize=7, label=  DataNH[5])
plt.loglog(DataH[6].pc,DataH[6].S,marker='s',color='blue',alpha=0.75, markersize=7, label=  DataNH[6])
plt.loglog(DataH[7].pc,DataH[7].S,marker='^',color='blue',alpha=0.75, markersize=7, label=  DataNH[7])
plt.loglog(DataH[8].pc,DataH[8].S,marker='v',color='purple',alpha=0.75, markersize=7, label=  DataNH[8])
plt.loglog(DataH[9].pc,DataH[9].S,marker='X',color='purple',alpha=0.75, markersize=7, label=  DataNH[9])
plt.loglog(DataH[10].pc,DataH[10].S,marker='X',color='purple',alpha=0.75, markersize=7, label=  DataNH[10])
plt.loglog(DataH[17].pc,DataH[17].S,marker='.',color='darkorange',alpha=0.75, markersize=7, label=  DataNH[17])
#plt.loglog(DataH[18].pc,DataH[18].S,marker='o',color='red',alpha=0.75, markersize=10, label=  DataNH[18])
#plt.loglog(DataH[20].pc,DataH[20].S,marker='o',color='red',alpha=0.75, markersize=10, label=  DataNH[20])
plt.loglog(DataH[25].pc*0.006,DataH[25].S,marker='s',color='green',alpha=0.75, markersize=7, label=  DataNH[25])
plt.loglog(DataH[26].pc*0.363,DataH[26].S**2,marker='^',color='green',alpha=0.75, markersize=7, label=  DataNH[26])


    
ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')

plt.title('Regiones HII galácticas')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    


fig.savefig('plots/funciones-ghr.pdf', 
              bbox_inches='tight')
#


# GEHR

Sample='extragalactic-regions'

samples1=pd.read_csv(str(datapath_names) + '/' + Sample+'.csv',header=None)

DataNG=dict()
DataG=dict()


for i in range(len(samples1)):
    DataNG[i]=samples1[0][i]
    
for i in range(len(samples1)):
    DataG[i]=pd.read_csv('data-previous-structure-functions//'+DataNG[i]+'.csv')    


samples1


fig, ax=plt.subplots(figsize=(8,8))

n=7

plt.loglog(DataG[0].pc,DataG[0].S,marker='^',color='dimgray',alpha=0.75, markersize=n, label= DataNG[0])
plt.loglog(DataG[1].pc,DataG[1].S,marker='o',color='blue',alpha=0.75, markersize=n, label= DataNG[1])
plt.loglog(DataG[2].pc,DataG[2].S,marker='o',color='turquoise',alpha=0.75, markersize=n, label= DataNG[2])
plt.loglog(DataG[3].pc,DataG[3].S,marker='^',color='turquoise',alpha=0.75, markersize=n, label= DataNG[3])
plt.loglog(DataG[4].pc,DataG[4].S,marker='s',color='turquoise',alpha=0.75, markersize=n, label= DataNG[4])
plt.loglog(DataG[5].pc,DataG[5].S*5.92**2,marker='o',color='limegreen',alpha=0.75, markersize=5, label= DataNG[5])
plt.loglog(DataG[6].pc,DataG[6].S*5.92**2,marker='o',color='green',alpha=0.75, markersize=n, label= DataNG[6],linestyle='dotted')
plt.loglog(DataG[7].pc,DataG[7].S*8.12**2,marker='^',color='limegreen',alpha=0.75, markersize=5, label= DataNG[7])
plt.loglog(DataG[8].pc,DataG[8].S*8.12**2,marker='^',color='green',alpha=0.75, markersize=n, label= DataNG[8],linestyle='dotted')
plt.loglog(DataG[9].pc,DataG[9].S*4.85**2,marker='s',color='limegreen',alpha=0.75, markersize=5, label= DataNG[9])
plt.loglog(DataG[10].pc,DataG[10].S*4.85**2,marker='s',color='green',alpha=0.75, markersize=n, label= DataNG[10],linestyle='dotted')

plt.loglog(DataG[11].pc,DataG[11].S*18.2**2,marker='^',color='black',alpha=0.75, markersize=n, label= DataNG[11])
plt.loglog(DataG[12].pc,DataG[12].S*14.5**2,marker='s',color='black',alpha=0.75, markersize=n, label= DataNG[12])
plt.loglog(DataG[13].pc,DataG[13].S*11.6**2,marker='v',color='black',alpha=0.75, markersize=n, label= DataNG[13])

plt.loglog(DataG[14].pc,DataG[14].S*7.2**2,marker='o',color='slateblue',alpha=0.75, markersize=n, label= DataNG[14])

    
ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('Regiones HII extragalácticas')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
fig.savefig('plots/funciones-gehr.pdf', 
              bbox_inches='tight')


# Comparison with our results

datapath_names = Path(open("path-name-list.txt", "r").read()).expanduser()


samples=pd.read_csv(str(datapath_names) +'//sample-names.csv',header=None)
samples


Names=pd.read_csv(str(datapath_names) +'//formal-names.csv',header=None)
Names





results_path = Path(r"~/Documents/Aeon/GitHub/PhD.Paper/result-files").expanduser()


data = {}
Results = {}

for i in range(len(samples)):
    data[samples[0][i]] = json.load(open(str(results_path) + '/' + samples[0][i] + ".json"))


# - Orion

B_Orion = data[samples[0][5]]['B']
r_Orion = data[samples[0][5]]['r']
B_Orion_N = data[samples[0][6]]['B']
r_Orion_N = data[samples[0][6]]['r']
B_Orion_O = data[samples[0][7]]['B']
r_Orion_O = data[samples[0][7]]['r']
B_Orion_S = data[samples[0][8]]['B']
r_Orion_S = data[samples[0][8]]['r']


B_EON = data[samples[0][14]]['B']
r_EON = data[samples[0][14]]['r']
B_EON_N = data[samples[0][15]]['B']
r_EON_N = data[samples[0][15]]['r']
B_EON_O = data[samples[0][16]]['B']
r_EON_O = data[samples[0][16]]['r']
B_EON_S = data[samples[0][17]]['B']
r_EON_S = data[samples[0][17]]['r']


fig, ax=plt.subplots(figsize=(6,6))

plt.loglog(r_Orion,B_Orion,marker='o',color='red',alpha=0.75, markersize=5, label= 'Ha')
plt.loglog(r_Orion_O,B_Orion_O,marker='o',color='blue',alpha=0.75, markersize=5, label= 'O')
plt.loglog(r_Orion_N,B_Orion_N,marker='o',color='orange',alpha=0.75, markersize=5, label= 'N')
plt.loglog(r_Orion_S,B_Orion_S,marker='o',color='green',alpha=0.75, markersize=5, label= 'S')


plt.loglog(r_EON,B_EON,marker='o',color='red',alpha=0.75, markersize=5, )
plt.loglog(r_EON_O,B_EON_O,marker='o',color='blue',alpha=0.75, markersize=5,)
plt.loglog(r_EON_N,B_EON_N,marker='o',color='orange',alpha=0.75, markersize=5, )
plt.loglog(r_EON_S,B_EON_S,marker='o',color='green',alpha=0.75, markersize=5,)



ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('Orion')
plt.legend()
#fig.savefig('plots/comp-604.pdf', 
#              bbox_inches='tight')


# - NGC 604

B_604 = data[samples[0][0]]['B']
r_604 = data[samples[0][0]]['r']


B_604_O = data[samples[0][1]]['B']
r_604_O = data[samples[0][1]]['r']


fig, ax=plt.subplots(figsize=(6,6))

plt.loglog(r_604,B_604,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(DataG[1].pc,DataG[1].S,color='black',alpha=1,linestyle='dotted' , label= DataNG[1])
plt.loglog(DataG[14].pc,DataG[14].S*7.2**2,color='black',alpha=1,linestyle='dashed' , label= DataNG[14])

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('NGC 604')
plt.legend()
fig.savefig('plots/comp-604.pdf', 
              bbox_inches='tight')


fig, ax=plt.subplots(figsize=(6,6))

plt.loglog(r_604,B_604,marker='o',color='red',alpha=0.75, markersize=5, label= 'Ha')
plt.loglog(r_604_O,B_604_O,marker='o',color='blue',alpha=0.75, markersize=5, label= 'O')



ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('NGC 604')
plt.legend()
fig.savefig('plots/comp-604.pdf', 
              bbox_inches='tight')





# 30 Doradus

B_Dor = data[samples[0][9]]['B']
r_Dor = data[samples[0][9]]['r']


B_Dor_N = data[samples[0][10]]['B']
r_Dor_N = data[samples[0][10]]['r']


fig, ax=plt.subplots(figsize=(6,6))

plt.loglog(r_Dor,B_Dor,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(DataG[11].pc,DataG[11].S*18.2**2,color='black',alpha=0.75,linestyle='dashed' , label= DataNG[11])
plt.loglog(DataG[12].pc,DataG[12].S*14.5**2,color='black',alpha=0.75,linestyle='dashed' , label= DataNG[12])
plt.loglog(DataG[13].pc,DataG[13].S*11.6**2,color='black',alpha=0.75,linestyle='dashed' , label= DataNG[13])
plt.loglog(DataG[0].pc,DataG[0].S,color='dimgray',alpha=1,linestyle='dotted' , label= DataNG[0])

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('30 Doradus')
plt.legend()

fig.savefig('plots/comp-30Dor.pdf', 
              bbox_inches='tight')


fig, ax=plt.subplots(figsize=(6,6))

plt.loglog(r_Dor,B_Dor,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(r_Dor_N,B_Dor_N,marker='o',color='orange',alpha=0.75, markersize=5)


ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('30 Doradus')
plt.legend()

#fig.savefig('plots/comp-30Dor.pdf', 
#              bbox_inches='tight')


B_Dor = data[samples[0][9]]['B']
r_Dor = data[samples[0][9]]['r']


# NGC 595

B_595 =  data[samples[0][2]]['B']
r_595 =  data[samples[0][2]]['r']


fig, ax=plt.subplots(figsize=(6,6))


plt.loglog(r_595,B_595,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(DataG[2].pc,DataG[2].S,color='black',alpha=0.75,linestyle='dashed' , label= DataNG[2])
plt.loglog(DataG[5].pc,DataG[5].S*5.92**2,color='black',alpha=0.75,linestyle='dotted' , label= DataNG[5])
plt.loglog(DataG[6].pc,DataG[6].S*5.92**2,color='black',alpha=0.75,label= DataNG[6],linestyle='dashdot')

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('NGC 595')
plt.legend()
fig.savefig('plots/comp-595.pdf', 
              bbox_inches='tight')


# M8

#mask = SFresults[samples_results[0][7]]["SF"]["N pairs"] > 0
B_M8 = data[samples[0][18]]['B']
r_M8 =data[samples[0][18]]['r']
B_M8_N = data[samples[0][19]]['B']
r_M8_N =data[samples[0][19]]['r']
B_M8_S = data[samples[0][20]]['B']
r_M8_S =data[samples[0][20]]['r']


fig, ax=plt.subplots(figsize=(6,6))


plt.loglog(r_M8,B_M8,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(DataH[9].pc,DataH[9].S,marker='X',color='purple',alpha=0.75, markersize=7, label=  DataNH[9])
plt.loglog(DataH[10].pc,DataH[10].S,marker='X',color='purple',alpha=0.75, markersize=7, label=  DataNH[10])
plt.loglog(DataH[25].pc*0.006,DataH[25].S,marker='s',color='green',alpha=0.75, markersize=7, label=  DataNH[25])
plt.loglog(DataH[26].pc*0.363,DataH[26].S**2,marker='^',color='green',alpha=0.75, markersize=7, label=  DataNH[26])


ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('M8')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
fig.savefig('plots/comp-M8.pdf', 
              bbox_inches='tight')


fig, ax=plt.subplots(figsize=(6,6))


plt.loglog(r_M8,B_M8,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(r_M8_N,B_M8_N,marker='o',color='orange',alpha=0.75, markersize=5)
plt.loglog(r_M8_S,B_M8_S,marker='o',color='green',alpha=0.75, markersize=5)

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('M8')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
#fig.savefig('plots/comp-M8.pdf', 
#              bbox_inches='tight')


#mask = SFresults[samples_results[0][7]]["SF"]["N pairs"] > 0
B_Car = data[samples[0][21]]['B']
r_Car =data[samples[0][21]]['r']
B_Car_N = data[samples[0][22]]['B']
r_Car_N =data[samples[0][22]]['r']
B_Car_S = data[samples[0][23]]['B']
r_Car_S =data[samples[0][23]]['r']


fig, ax=plt.subplots(figsize=(6,6))


plt.loglog(r_Car,B_Car,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(r_Car_N,B_Car_N,marker='o',color='orange',alpha=0.75, markersize=5)
plt.loglog(r_Car_S,B_Car_S,marker='o',color='green',alpha=0.75, markersize=5)

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
plt.grid(which='minor')
plt.title('Carina)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
#fig.savefig('plots/comp-M8.pdf', 
#              bbox_inches='tight')


# 







get_ipython().system('jupyter nbconvert --to script --no-prompt structure-functions-catalogue.ipynb')


print("--- %s seconds ---" % (time.time()-start_time))

