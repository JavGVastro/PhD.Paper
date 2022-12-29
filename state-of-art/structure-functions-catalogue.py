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
import cmasher as cmr
sns.set_color_codes()


sns.set_context("talk")


#plt.rcParams["font.family"]="Times New Roman"
#plt.rcParams["font.size"]="10"


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





alpha = .55

fig, ax=plt.subplots(figsize=(8,8))

plt.loglog(DataH[26].pc*0.363,DataH[26].S**2,marker='o',color='darkorange',alpha=alpha, markersize=7, label='M8')
plt.loglog(DataH[3].pc,DataH[3].S,marker='o',color='red',alpha=alpha, markersize=7, label=  'Sh 142')
plt.loglog(DataH[4].pc,DataH[4].S,marker='o',color='maroon',alpha=alpha, markersize=7, label=  'M17')
plt.loglog(DataH[5].pc,DataH[5].S,marker='o',color='purple',alpha=alpha, markersize=7, label=  'N1499')
plt.loglog(DataH[6].pc,DataH[6].S,marker='s',color='purple',alpha=alpha, markersize=7, label=  'N7000')
plt.loglog(DataH[7].pc,DataH[7].S,marker='^',color='purple',alpha=alpha, markersize=7, label= 'S252')
plt.loglog(DataH[8].pc,DataH[8].S,marker='o',color='blue',alpha=alpha, markersize=7, label=  'N6414 [OIII]', linestyle=':')
plt.loglog(DataH[9].pc,DataH[9].S,marker='s',color='blue',alpha=alpha, markersize=7, label=  'N6523 A [OIII]', linestyle=':')
plt.loglog(DataH[10].pc,DataH[10].S,marker='^',color='blue',alpha=alpha, markersize=7, label=  'N6523 B [OIII]', linestyle=':')
plt.loglog(DataH[17].pc,DataH[17].S,marker='o',color='midnightblue',alpha=alpha, markersize=7, label=  'S170')
plt.loglog(DataH[25].pc*0.006,DataH[25].S,marker='o',color='green',alpha=alpha, markersize=7, label=  'M8 [OIII]', linestyle=':')

    
ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')

plt.title('Regiones HII galácticas')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    


fig.savefig('plots/funciones-ghr.pdf', 
              bbox_inches='tight')
#


#x = np.linspace(0, 2*np.pi, 64)
#n = 20
colors = plt.cm.hsv(np.linspace(0,1,29))

#colors = cmr.take_cmap_colors(
#    "cmr.nuclear",
#    50,
#    cmap_range=(0.25, 0.95),
#)

#samples=pd.read_csv('galactic-regions-noM42.csv',header=None)

#DataNH=dict()
#DataH=dict()

#for i in range(len(samples)):
#    DataNH[i]=samples[0][i]
    
#for i in range(len(samples)):
#    DataH[i]=pd.read_csv('data-previous-structure-functions//'+DataNH[i]+'.csv')    

#marker=itertools.cycle(('o','o','o','+','+','+','+','+','+','o','o','o','o','o','o',
#                        'x','x','o','x','o','o','o','o'))

#marker=itertools.cycle((r"$a$",r"$a$",r"$b$",r"$c$",r"$e$",r"$e$",r"$e$",r"$e$",r"$e$",r"$e$",r"$f$",r"$f$",r"$f$",r"$f$",r"$f$",r"$f$",
#                        r"$g$",r"$h$",r"$h$",r"$h$",r"$i$",r"$i$",r"$i$",r"$i$"))

#color=itertools.cycle(("magenta","magenta","red","purple","black","black","black","black","black","black","blue","blue","blue","blue","blue","blue",
#                       "darkred","green","green","green","orange","orange","orange","orange"))
     
fig, ax=plt.subplots(figsize=(8,8))

    
for i in range(len(samples)):
    plt.loglog(DataH[i].pc,DataH[i].S,marker='.', color=colors[i], alpha=0.55, markersize=10)
    
ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')

#samples


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

n=5

plt.loglog(DataG[0].pc,DataG[0].S,marker='o',color='darkorange',alpha=0.5, markersize=n, label= '30 Dor')
plt.loglog(DataG[1].pc,DataG[1].S,marker='o',color='red',alpha=0.5, markersize=n, label= 'N604')
#plt.loglog(DataG[2].pc,DataG[2].S,marker='o',color='maroon',alpha=0.5, markersize=n, label= 'N595')
#plt.loglog(DataG[3].pc,DataG[3].S,marker='o',color='maroon',alpha=0.5, markersize=n, label= 'N595 [OIII]',linestyle='dotted')
#plt.loglog(DataG[4].pc,DataG[4].S,marker='s',color='maroon',alpha=0.5, markersize=n, label= DataNG[4])
plt.loglog(DataG[5].pc,DataG[5].S*5.92**2,marker='o',color='purple',alpha=0.5, markersize=5, label= 'N595')
plt.loglog(DataG[6].pc,DataG[6].S*5.92**2,marker='x',color='purple',alpha=0.5, markersize=n, label= 'N595 + filtro')
plt.loglog(DataG[7].pc,DataG[7].S*8.12**2,marker='o',color='purple',alpha=0.5, markersize=5, label= 'N595 [OIII]',linestyle='dotted')
plt.loglog(DataG[8].pc,DataG[8].S*8.12**2,marker='x',color='purple',alpha=0.5, markersize=n, label= 'N595 [OIII] + filtro',linestyle='dotted')
plt.loglog(DataG[9].pc,DataG[9].S*4.85**2,marker='o',color='purple',alpha=0.5, markersize=5, label= 'N595 [SII]',linestyle='--')
plt.loglog(DataG[10].pc,DataG[10].S*4.85**2,marker='x',color='purple',alpha=0.5, markersize=n, label= 'N595 [SII] + filtro',linestyle='--')
plt.loglog(DataG[14].pc,DataG[14].S*7.2**2,marker='o',color='green',alpha=0.5, markersize=n, label= 'N604')
plt.loglog(DataG[11].pc,DataG[11].S*18.2**2,marker='o',color='blue',alpha=0.5, markersize=n, label= '30 Dor Peak')
plt.loglog(DataG[12].pc,DataG[12].S*14.5**2,marker='s',color='blue',alpha=0.5, markersize=n, label= '30 Dor Gaussian')
plt.loglog(DataG[13].pc,DataG[13].S*11.6**2,marker='^',color='blue',alpha=0.5, markersize=n, label= '30 Dor Singles')

    
ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
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


alpha = .4

fig, ax=plt.subplots(figsize=(8,8))

#plt.loglog(DataH[26].pc*0.363,DataH[26].S**2,marker='o',color='darkorange',alpha=alpha, markersize=7, label='M8')
plt.loglog(DataH[0].pc,DataH[0].S,marker='o',color='darkorange',alpha=alpha, markersize=5, label= 'von Horner 1')
plt.loglog(DataH[1].pc,DataH[1].S,marker='s',color='darkorange',alpha=alpha, markersize=5, label=  'von Horner 2')
plt.loglog(DataH[2].pc,DataH[2].S,marker='o',color='red',alpha=alpha, markersize=5, label=  'Munch O', linestyle=':')
plt.loglog(DataH[11].pc,DataH[11].S,marker='o',color='maroon',alpha=alpha, markersize=5, label=  'Cast 1a O')
plt.loglog(DataH[12].pc,DataH[12].S,marker='s',color='maroon',alpha=alpha, markersize=5, label=  'Cast 1b O')
plt.loglog(DataH[13].pc,DataH[13].S,marker='^',color='maroon',alpha=alpha, markersize=5, label=  'Cast 1c O')
plt.loglog(DataH[14].pc,DataH[14].S,marker='o',color='purple',alpha=alpha, markersize=5, label= 'Cast 2a O', linestyle=':')
plt.loglog(DataH[15].pc,DataH[15].S,marker='s',color='purple',alpha=alpha, markersize=5, label=  'Cast 2b O', linestyle=':')
plt.loglog(DataH[16].pc,DataH[16].S,marker='^',color='purple',alpha=alpha, markersize=5, label=  'Cast 2c O', linestyle=':')
plt.loglog(DataH[27].pc*0.002,DataH[27].S,marker='^',color='violet',alpha=alpha, markersize=5, label=  'Wen S', linestyle='-.')

plt.loglog(DataH[19].pc,DataH[19].S,marker='o',color='blue',alpha=alpha, markersize=5, label=  'Joncas')
plt.loglog(DataH[21].pc,DataH[21].S*9.37,marker='o',color='green',alpha=alpha, markersize=5, label=  'Arthur H')
plt.loglog(DataH[22].pc,DataH[22].S*5.62,marker='s',color='green',alpha=alpha, markersize=5, label=  'Arthur N', linestyle=':')
plt.loglog(DataH[23].pc,DataH[23].S*10.2,marker='^',color='green',alpha=alpha, markersize=5, label= 'Arthur O', linestyle='--')
plt.loglog(DataH[24].pc,DataH[24].S*5.27,marker='X',color='green',alpha=alpha, markersize=5, label=  'Arthur S', linestyle='-.')
plt.loglog(r_EON,B_EON,marker='o',color='k',alpha=0.35, markersize=5, label='Haenel H')
plt.loglog(r_EON_O,B_EON_O,marker='^',color='k',alpha=0.35, markersize=5,label='Haenel O')
plt.loglog(r_EON_N,B_EON_N,marker='s',color='k',alpha=0.35, markersize=5, label='Haenel N')
plt.loglog(r_EON_S,B_EON_S,marker='X',color='k',alpha=0.35, markersize=5,label='Haenel S')
    
ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')

plt.title('Orión (M42)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    


fig.savefig('plots/funciones-M42.pdf', 
              bbox_inches='tight')


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


fig, ax=plt.subplots(figsize=(7,7))

plt.loglog(r_Orion,B_Orion,marker='.',color='black',alpha=0.7, markersize=7, label= 'Orión núcleo', linestyle ='')
plt.loglog(r_EON,B_EON,marker='^',color='black',alpha=0.7, markersize=7,label= 'EON', linestyle ='' )

plt.loglog(DataH[21].pc,DataH[21].S*9.37,marker='o',color='blue',alpha=0.6, markersize=7, label=  'Arthur et al. (2016)')




ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
plt.title('Orión (M42)')
plt.legend()
fig.savefig('plots/comp-Orion.pdf',              bbox_inches='tight')


# - NGC 604

B_604 = data[samples[0][0]]['B']
r_604 = data[samples[0][0]]['r']


B_604_O = data[samples[0][1]]['B']
r_604_O = data[samples[0][1]]['r']


fig, ax=plt.subplots(figsize=(7,7))

plt.loglog(r_604,B_604,marker='.',color='black',alpha=0.75, markersize=8, linestyle ='')

plt.loglog(DataG[1].pc,DataG[1].S,color='blue',linestyle='-' , label= 'Medina-Tanco et al. (1997)',marker='o',alpha=0.5)
plt.loglog(DataG[14].pc,DataG[14].S*7.2**2,color='blue',linestyle='dashed' , label= 'Melnick et al. (2021)',marker='^',alpha=0.5)

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
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
#fig.savefig('plots/comp-604.pdf', 
#              bbox_inches='tight')





# 30 Doradus

B_Dor = data[samples[0][9]]['B']
r_Dor = data[samples[0][9]]['r']


B_Dor_N = data[samples[0][10]]['B']
r_Dor_N = data[samples[0][10]]['r']


fig, ax=plt.subplots(figsize=(7,7))

plt.loglog(r_Dor,B_Dor,marker='.',color='black',alpha=0.75, markersize=8, linestyle ='')

plt.loglog(DataG[0].pc,DataG[0].S,color='blue',alpha=0.5,linestyle='-' , label= 'Feast (1961)',marker='o')
plt.loglog(DataG[11].pc,DataG[11].S*18.2**2,color='blue',alpha=0.5,linestyle='dashed' , label= 'Melnick et al. (2021)',marker='^')
plt.loglog(DataG[12].pc,DataG[12].S*14.5**2,color='blue',alpha=0.5,linestyle='dotted' , label= 'Melnick et al. (2021)',marker='+')
plt.loglog(DataG[13].pc,DataG[13].S*11.6**2,color='blue',alpha=0.5,linestyle='-' , label= 'Melnick et al. (2021)',marker='x')

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
plt.title('30 Dor')
plt.legend()

fig.savefig('plots/comp-30Dor.pdf', 
              bbox_inches='tight')


fig, ax=plt.subplots(figsize=(6,6))

plt.loglog(r_Dor,B_Dor,marker='o',color='red',alpha=0.75, markersize=5)
plt.loglog(r_Dor_N,B_Dor_N,marker='o',color='orange',alpha=0.75, markersize=5)


ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
plt.title('30 Doradus')
plt.legend()

#fig.savefig('plots/comp-30Dor.pdf', 
#              bbox_inches='tight')


B_Dor = data[samples[0][9]]['B']
r_Dor = data[samples[0][9]]['r']


# NGC 595

B_595 =  data[samples[0][2]]['B']
r_595 =  data[samples[0][2]]['r']


fig, ax=plt.subplots(figsize=(7,7))


plt.loglog(r_595,B_595,marker='.',color='black',alpha=0.75, markersize=7, linestyle ='')


plt.loglog(DataG[2].pc,DataG[2].S,color='blue',alpha=0.6, label= 'Lagrois & Joncas (2009)', linestyle ='-.',marker='+',markersize=5)
plt.loglog(DataG[5].pc,DataG[5].S*5.92**2,color='blue',alpha=0.6, label= 'Lagrois & Joncas (2011)', linestyle ='-',marker='o',markersize=5)
plt.loglog(DataG[6].pc,DataG[6].S*5.92**2,color='blue',alpha=0.6,label= 'Lagrois & Joncas (2011)',linestyle='-',marker='^',markersize=5)

ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
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


fig, ax=plt.subplots(figsize=(8,8))


plt.loglog(r_M8,B_M8,marker='.',color='black',alpha=0.75, markersize=7, linestyle ='')

plt.loglog(DataH[26].pc*0.363,DataH[26].S**2,marker='.',color='blue',alpha=0.6, markersize=7, label=  'Louise & Monet (1970)', linestyle = ':')
plt.loglog(DataH[9].pc,DataH[9].S,marker='^',color='blue',alpha=0.6, markersize=5, label=  "O'Dell & Castañeda (1987)", linestyle = '-.')
plt.loglog(DataH[10].pc,DataH[10].S,marker='+',color='blue',alpha=0.6, markersize=5, label=  "O'Dell & Castañeda (1987)", linestyle = '-.')
plt.loglog(DataH[25].pc*0.006,DataH[25].S,marker='o',color='blue',alpha=0.6, markersize=5, label=  "Chakraborty (1999)", linestyle = '-')


ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
plt.title('Laguna (M8,NGC 6253)')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
plt.legend()    
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








fig, ax=plt.subplots(figsize=(8,8))

#M8
plt.loglog(DataH[26].pc*0.363,DataH[26].S**2,marker='^',color='blue',alpha=0.75, markersize=7, label=  DataNH[26], linestyle = '-.')
plt.loglog(DataH[9].pc,DataH[9].S,marker='^',color='blue',alpha=0.75, markersize=5, label=  DataNH[9], linestyle = ':')
#plt.loglog(DataH[10].pc,DataH[10].S,marker='^',color='blue',alpha=0.75, markersize=5, label=  DataNH[10], linestyle = ':')
plt.loglog(DataH[25].pc*0.006,DataH[25].S,marker='^',color='blue',alpha=0.75, markersize=5, label=  DataNH[25], linestyle = '-')
#595
#plt.loglog(DataG[2].pc,DataG[2].S,color='blue',alpha=0.35, label= DataNG[2], linestyle ='-',marker='o',markersize=5)
plt.loglog(DataG[5].pc,DataG[5].S*5.92**2,color='blue',alpha=0.35, label= DataNG[5], linestyle ='-',marker='o',markersize=5)
#plt.loglog(DataG[6].pc,DataG[6].S*5.92**2,color='blue',alpha=0.35,label= DataNG[6],linestyle='-',marker='o',markersize=5)
#604
#plt.loglog(DataG[1].pc,DataG[1].S,color='blue',linestyle='-.' , label= DataNG[1],marker='d',alpha=0.5)
plt.loglog(DataG[14].pc,DataG[14].S*7.2**2,color='blue',linestyle='-.' , label= DataNG[14],marker='+',alpha=0.5)
#Orion
plt.loglog(DataH[21].pc,DataH[21].S*9.37,marker='s',color='blue',alpha=0.75, markersize=7, label=  DataNH[21])
#30
plt.loglog(DataG[0].pc,DataG[0].S,color='blue',alpha=0.45,linestyle=':' , label= DataNG[0],marker='*')
#plt.loglog(DataG[11].pc,DataG[11].S*18.2**2,color='blue',alpha=0.45,linestyle='-' , label= DataNG[11],marker='*')
plt.loglog(DataG[12].pc,DataG[12].S*14.5**2,color='blue',alpha=0.45,linestyle='-' , label= DataNG[12],marker='*')
plt.loglog(DataG[13].pc,DataG[13].S*11.6**2,color='blue',alpha=0.45,linestyle='-.' , label= DataNG[13],marker='*')

plt.loglog(r_M8,B_M8,marker='^',color='black',alpha=0.65, markersize=7, linestyle ='', label= 'Lagoon')
plt.loglog(r_595,B_595,marker='o',color='black',alpha=0.65, markersize=7, linestyle ='', label= 'N595')
plt.loglog(r_604,B_604,marker='+',color='black',alpha=0.65, markersize=8, linestyle ='', label= 'N604')
plt.loglog(r_Orion,B_Orion,marker='s',color='black',alpha=0.65, markersize=7, label= 'Orion core', linestyle ='')
plt.loglog(r_EON,B_EON,marker='x',color='black',alpha=0.65, markersize=7,label= 'EON', linestyle ='' )
plt.loglog(r_Dor,B_Dor,marker='*',color='black',alpha=0.65, markersize=8, linestyle ='', label= '30 Dor')


ax.set(xlabel='separación [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')
#plt.title('Laguna (M8,NGC 6253)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
#plt.legend()    
fig.savefig('plots/comp-comp-same.pdf', 
              bbox_inches='tight')








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
plt.title('Carina')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
#fig.savefig('plots/comp-M8.pdf', 
#              bbox_inches='tight')


# 







get_ipython().system('jupyter nbconvert --to script --no-prompt structure-functions-catalogue.ipynb')


print("--- %s seconds ---" % (time.time()-start_time))

