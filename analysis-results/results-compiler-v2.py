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


from results import loadresults,loadresults2


x,y = loadresults('sample-names-corr','formal-names-corr','2sig')


a,b = loadresults2('sample-names-corr','formal-names-corr','LM')


c,d = loadresults2('sample-names-corr','formal-names-corr','MCMC')


b


d


#plt.rcParams["axes.edgecolor"] = "yellow"
#plt.rcParams["axes.linewidth"]  = 2

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(projection='3d')


ax.scatter(b['sig2'],b['r0'],b['m'],marker='o',s=150,color='b')
#ax.scatter(b['sig2']+b['sig2+'],b['r0']+b['r0+'],b['m']+b['m+'],marker='.',s=100,color='blue')
#ax.scatter(b['sig2']-b['sig2+'],b['r0']-b['r0+'],b['m']-b['m+'],marker='.',s=100,color='blue')

ax.scatter(d['sig2'],d['r0'],d['m'],marker='o',s=150,color='red')
#ax.scatter(d['sig2']+d['sig2+'],d['r0']+d['r0+'],d['m']+d['m+'],marker='.',s=100,color='red')
#ax.scatter(d['sig2']-d['sig2+'],d['r0']-d['r0+'],d['m']-d['m+'],marker='.',s=100,color='red')

ax.set_xlabel('sig2')
ax.set_ylabel('r0')
ax.set_zlabel('m')
#ax.set_xlim(0, 1)
#ax.set_ylim(0, 1)
#ax.set_zlim(0.7, 1.5)

#ax.view_init(elev=30, azim=-45)

plt.show()


fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(b['sig2'],b['r0'], c="b",s=40, alpha=0.6)
ax.errorbar(b['sig2'],b['r0'], xerr=b['sig2+'], yerr=b['r0+'], ls=" ", elinewidth=0.5, alpha=1.0, c="b")
ax.scatter(d['sig2'],d['r0'], c="r",s=40, alpha=0.6)
ax.errorbar(d['sig2'],d['r0'], xerr=d['sig2+'], yerr=d['r0+'], ls=" ", elinewidth=0.5, alpha=1.0, c="r")
#ax.scatter(y['sig2'],y['r0'], c="r",s=40, alpha=0.6)
#ax.errorbar(y['sig2'],y['r0'], xerr=y['sig2+'], yerr=y['r0+'], ls=" ", elinewidth=0.5, alpha=1.0, c="g")


fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(b['sig2'],b['m'], c="b")
ax.errorbar(b['sig2'],b['m'], xerr=b['sig2+'], yerr=b['m+'], ls=" ", elinewidth=0.75, alpha=1.0, c="b")
ax.scatter(d['sig2'],d['m'], c="r")
ax.errorbar(d['sig2'],d['m'], xerr=d['sig2+'], yerr=d['m+'], ls=" ", elinewidth=0.75, alpha=1.0, c="r")


get_ipython().system('jupyter nbconvert --to script --no-prompt results-compiler-v2.ipynb')

