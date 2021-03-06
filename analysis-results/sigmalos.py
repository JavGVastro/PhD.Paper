#!/usr/bin/env python
# coding: utf-8

from pathlib import Path


import time
start_time=time.time()
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import lmfit
import statsmodels.api as sm
import linmix
from scipy.stats import pearsonr
import pickle
import math
import itertools
import json
from logerr import logify


# * s_obs \
# 604: https://articles.adsabs.harvard.edu//full/1986ApJ...300..624R/0000628.000.html \
# 595/HV/HX: https://articles.adsabs.harvard.edu//full/1986A%26A...160..374H/0000377.000.html \
# Car: obs \
# Orion: https://ui.adsabs.harvard.edu/abs/2008RMxAA..44..181G/abstract \
# 30 Dor: https://www.aanda.org/articles/aa/full_html/2013/07/aa20474-12/aa20474-12.html \
# M8:
# 
# * s_inst \
# 604: https://articles.adsabs.harvard.edu/full/1988A%26A...198..283O \
# 595/HV/HX: https://articles.adsabs.harvard.edu//full/1986A%26A...160..374H/0000377.000.html \
# Carina: https://www.aanda.org/articles/aa/full_html/2016/07/aa28169-16/aa28169-16.html \
# Orion: https://ui.adsabs.harvard.edu/abs/2008RMxAA..44..181G/abstract \
# 30Dor: https://www.aanda.org/articles/aa/full_html/2013/07/aa20474-12/aa20474-12.html \
# M8: https://articles.adsabs.harvard.edu//full/1973ApJ...183..851B/0000860.000.html \
# 
# * s2_fs \
# H: https://ui.adsabs.harvard.edu/abs/2008RMxAA..44..181G/abstract 
# 
# * Te \
# 595/604: https://iopscience.iop.org/article/10.1088/0004-637X/700/1/654 \
# HX/HV: \
# https://articles.adsabs.harvard.edu//full/1986ApJ...300..624R/0000628.000.html \
# https://ui.adsabs.harvard.edu/abs/2001A%26A...367..388H/abstract \
# Dor: \
# https://www.aanda.org/articles/aa/full_html/2018/06/aa32084-17/aa32084-17.html \
# https://ui.adsabs.harvard.edu/abs/2010ApJS..191..160P/abstract \
# Carina: https://www.aanda.org/articles/aa/full_html/2016/07/aa28169-16/aa28169-16.html \
# M8: \
# https://articles.adsabs.harvard.edu//full/1973ApJ...183..851B/0000860.000.html \
# https://www.aanda.org/articles/aa/full_html/2017/08/aa30986-17/aa30986-17.html \
# https://articles.adsabs.harvard.edu//full/1973ApJ...184...93B/0000094.000.html \
# Orion: \
# https://ui.adsabs.harvard.edu/abs/2008RMxAA..44..181G/abstract 
# 
# * s_los_lit \
# 604: \
# (value) https://articles.adsabs.harvard.edu//full/1986A%26A...160..374H/0000377.000.html \
# https://articles.adsabs.harvard.edu//full/1996AJ....112.1636M/0001640.000.html \
# https://ui.adsabs.harvard.edu/abs/1995ApJ...444..200S/abstract \
# 595: \
# (value) https://iopscience.iop.org/article/10.1088/0004-637X/700/2/1847 \
# https://articles.adsabs.harvard.edu//full/1986A%26A...160..374H/0000377.000.html \
# HV/HX: \
# https://articles.adsabs.harvard.edu//full/1986A%26A...160..374H/0000377.000.html \
# 346: \
# https://iopscience.iop.org/article/10.1086/367959/fulltext/ \
# 30Dor: \
# (value) https://www.aanda.org/articles/aa/full_html/2013/07/aa20474-12/aa20474-12.html \
# https://articles.adsabs.harvard.edu//full/1999MNRAS.302..677M/0000682.000.html \
# M8:
# https://articles.adsabs.harvard.edu//full/1973ApJ...183..851B/0000860.000.html
# 

sigma_data = pd.read_table('sigmaH-data.csv', delimiter=',')
sigma_data['s2_th [km^2/s^2]']=(sigma_data['Te [K]']/10000)*82.5*(1/1.008)
sigma_data['s_los_calc [km/s]']=(sigma_data['s_obs [km/s]']**2-sigma_data['s_inst [km/s]']**2-sigma_data['s2_fs [km^2/s^2]']-sigma_data['s2_th [km^2/s^2]'])**0.5
sigma_data['~s_los_lit [km/s]'] = [6,11.7,0,22,10.1,10.5,10.3,16.83,17.8]
sigma_data['~s_los_lit_err [km/s]'] = [1,1.6,0,0,0,0.02,0.03,0.09,0.3]

sigma_data.round(2)


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


combine_moments(0.3966,-6.404371,6.44256,0.60335,3.454723,6.069641)


print("--- %s seconds ---" % (time.time() - start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt sigmalos.ipynb')

