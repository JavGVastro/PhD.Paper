#!/usr/bin/env python
# coding: utf-8

import time


start_time = time.time()


from pathlib import Path
import cmasher as cmr
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import lmfit
import json
import pandas as pd
import corner
import sys

import astropy.units as u

sys.path.insert(1, 'C:/Users/ZAINTEL2/Documents/Aeon/GitHub/PhD.Paper/py-modules')  
import bplot_mod
import bfunc


data_in = json.load(open("data_dens_cte.json"))


mask = np.array(data_in["sf"]["N pairs"]) > 0


B = np.array(data_in["sf"]["Unweighted B(r)"])[mask]
r = 10**np.array(data_in["sf"]["log10 r"])[mask]
box_size = 362
pc = 1
pix = 1
pc_per_arcsec = pc


model = lmfit.Model(bfunc.bfunc03s)
model.param_names


# Correlation length between 1/10 and 2 x box_size
model.set_param_hint("r0", value=0.1 * box_size, min=0.01 * box_size, max=2.0 * box_size)

# sig2 between 1/4 and 2 x max value of B(r)
model.set_param_hint("sig2", value=0.5 * B.max(), min=0.25 * B.max(), max=2.0 * B.max())

# m between 1/2 and 5/3
model.set_param_hint("m", value=1, min=0.5, max=1.5)

#Seeing RMS between 0.5 and 1.5 arcsec
#model.set_param_hint(
#    "s0", value=0.85 * pc_per_arcsec, min=0.5 * pc_per_arcsec, max=1.5 * pc_per_arcsec
#)

# Seeing pegged at ZERO
model.set_param_hint(
    "s0", value=0.0 * 1, vary=False,
)


# Noise cannot be much larger than smallest B(r)
model.set_param_hint("noise", value=0.5 * B.min(), min=0.0, max=3 * B.min())

# box_size is fixed
# model.set_param_hint("box_size", value=box_size, vary=False)


pd.DataFrame(model.param_hints)


relative_uncertainty = 0.08
weights = 1.0 / (relative_uncertainty * B)
large_scale = r > 0.5* box_size
weights[large_scale] /= 2.0
#weights[:1] /= 2.0


to_fit = r <= 0.5* box_size
#to_fit = ~large_scale
result = model.fit(B[to_fit], weights=weights[to_fit], r=r[to_fit])


result


fig, ax = plt.subplots(figsize=(12, 12))

# Plot the underlying model without instrumental effects
Bu = bfunc.bfunc00s(
    r, result.params["r0"].value, result.params["sig2"].value, result.params["m"].value
)
ax.plot(r, Bu, color="k", linestyle="dashed", label="underlying")

# Plot the fit results
result.plot_fit(ax=ax)

# Add in the points not included in fit
ax.plot(r[large_scale], B[large_scale], "o")

# Dotted lines for 2 x rms seeing and for box size
ax.axvline(2 * result.params["s0"].value, color="k", linestyle="dotted")
ax.axvline(box_size, color="k", linestyle="dotted")

# Dashed lines for best-fit r0 and sig2
ax.axvline(result.params["r0"].value, color="k", linestyle="dashed")
ax.axhline(result.params["sig2"].value, color="k", linestyle="dashed")

# Gray box to indicate the large scale values that are excluded from the fit
ax.axvspan(box_size / 2, r[-1], color="k", alpha=0.05, zorder=-1)

ax.set(
    xscale="log",
    yscale="log",
    xlabel="r [pc]",
    ylabel=r"B(r) [km$^{2}$/s$^{2}$]",
)
sns.despine()





# emcee

emcee_kws = dict(
    steps=10000, burn=500, thin=50, is_weighted=True, progress=False, workers=16
)
emcee_params = result.params.copy()
# emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))


result_emcee = model.fit(
    data=B[to_fit],
    r=r[to_fit],
    weights=weights[to_fit],
    params=emcee_params,
    method="emcee",
    nan_policy="omit",
    fit_kws=emcee_kws,
)


result_emcee


plt.plot(result_emcee.acceptance_fraction, "o")
plt.xlabel("walker")
plt.ylabel("acceptance fraction")


if hasattr(result_emcee, "acor"):
    print("Autocorrelation time for the parameters:")
    print("----------------------------------------")
    for i, p in enumerate(result_emcee.params):
        try:
            print(f"{p} = {result_emcee.acor[i]:.3f}")
        except IndexError:
            pass


name = 'Density cte'
data = 'k_3'


bplot_mod.corner_plot(
    result_emcee, result_emcee, name, data, data_ranges=[0.95, 0.99, 0.995, 0.995]
)
# data_ranges=[0.95, 0.99, 0.995, 0.995, 0.999]


bplot_mod.strucfunc_plot(
    result_emcee, result, r, B, to_fit, name, data, box_size, large_scale
)


# LM results

LM = {
    'sig2': [result.params['sig2'].value,result.params['sig2'].stderr],
    'r0': [result.params['r0'].value,result.params['r0'].stderr],
    'm' : [result.params['m'].value,result.params['m'].stderr],
    's0': [result.params['s0'].value,result.params['s0'].stderr],
    'noise' : [result.params['noise'].value,result.params['noise'].stderr]
}


LM


# MCMC results

MCMC = {
    'sig2': [result_emcee.params['sig2'].value,result_emcee.params['sig2'].stderr],
    'r0': [result_emcee.params['r0'].value,result_emcee.params['r0'].stderr],
    'm' : [result_emcee.params['m'].value,result_emcee.params['m'].stderr],
    's0': [result_emcee.params['s0'].value,result_emcee.params['s0'].stderr],
    'noise' : [result_emcee.params['noise'].value,result_emcee.params['noise'].stderr]
}


MCMC


# MCMC 2 sigma confidence interval

sig2s2 = np.percentile(result_emcee.flatchain['sig2'],[2.5, 97.5])
r0s2 = np.percentile(result_emcee.flatchain['r0'],[2.5, 97.5])
ms2 = np.percentile(result_emcee.flatchain['m'],[2.5, 97.5])
#s0s2 = np.percentile(result_emcee.flatchain['s0'],[2.5, 97.5])
b0s2 = np.percentile(result_emcee.flatchain['noise'],[2.5, 97.5])


sig2s2p = sig2s2[1]-result.params['sig2'].value
sig2s2m = result.params['sig2'].value-sig2s2[0]

r0s2p = r0s2[1]-result.params['r0'].value
r0s2m = result.params['r0'].value-r0s2[0]

ms2p = ms2[1]-result.params['m'].value
ms2m = result.params['m'].value-ms2[0]

#s0s2p = s0s2[1]-result.params['s0'].value
#s0s2m = result.params['s0'].value-s0s2[0]
s0s2p = 0
s0s2m = 0

b0s2p = b0s2[1]-result.params['noise'].value
b0s2m = result.params['noise'].value-b0s2[0]


# MCMC 1 sigma confidence interval

sig2s1 = np.percentile(result_emcee.flatchain['sig2'],[16, 85])
r0s1 = np.percentile(result_emcee.flatchain['r0'],[16, 85])
ms1 = np.percentile(result_emcee.flatchain['m'],[16, 85])
#s0s1 = np.percentile(result_emcee.flatchain['s0'],[16, 85])
s0s1 = 0
b0s1 = np.percentile(result_emcee.flatchain['noise'],[16, 85])


sig2s1p = sig2s1[1]-result.params['sig2'].value
sig2s1m = result.params['sig2'].value-sig2s1[0]

r0s1p = r0s1[1]-result.params['r0'].value
r0s1m = result.params['r0'].value-r0s1[0]

ms1p = ms1[1]-result.params['m'].value
ms1m = result.params['m'].value-ms1[0]

#s0s1p = s0s1[1]-result.params['s0'].value
#s0s1m = result.params['s0'].value-s0s1[0]
s0s1p = 0
s0s1m = 0

b0s1p = b0s1[1]-result.params['noise'].value
b0s1m = result.params['noise'].value-b0s1[0]


# LM + MCMC 2 sigma

results_2sig = {
    'sig2': [result.params['sig2'].value,sig2s2p,sig2s2m],
    'r0': [result.params['r0'].value,r0s2p,r0s2m],
    'm' : [result.params['m'].value,ms2p,ms2m],
    's0': [result.params['s0'].value,s0s2p,s0s2m],
    'noise' : [result.params['noise'].value,b0s2p,b0s2m] 
    
}


results_2sig


results_1sig = {
    'sig2': [result.params['sig2'].value,sig2s1p,sig2s1m],
    'r0': [result.params['r0'].value,r0s1p,r0s1m],
    'm' : [result.params['m'].value,ms1p,ms1m],
    's0': [result.params['s0'].value,s0s1p,s0s1m],
    'noise' : [result.params['noise'].value,b0s1p,b0s1m] 
    
}


results_1sig


# Previous SF results and obs

#observations ={
#    'sb':data_in['VF']['sb'],
#    'vv':data_in['VF']['vv'],
#    'ss':data_in['VF']['ss']   
#}


#properties = {
#    'pix' : data_in['pix'],
#    'pc' : data_in['pc'],
#    'box_size' : data_in['box_size']
#}


#fit_results = {
#    'name' : name_in,
#    'results_1sig' : results_1sig,
#    'results_2sig' : results_2sig,
#    'LM':LM,
#    'MCMC':MCMC,
#    'properties' : properties,
#    'B' : B,
#    'r' : r,
#     'preres' : data_in['results'],
#     'SFresults' : data_in['SF'],
#    'observations' : observations
   
#}


#class MyEncoder(json.JSONEncoder):
#    def default(self, obj):
#        if isinstance(obj, np.integer):
#            return int(obj)
#        elif isinstance(obj, np.floating):
#            return float(obj)
#        elif isinstance(obj, np.ndarray):
#            return obj.tolist()
#        else:
#            return super(MyEncoder, self).default(obj)


#jsonfilename =name_in +".json"
#with open(datapath_res/jsonfilename, "w") as f:
#    json.dump(fit_results, fp=f, indent=3, cls=MyEncoder)


print("--- %s seconds ---" % (time.time() - start_time))


get_ipython().system('jupyter nbconvert --to script --no-prompt ci-fake-3d-maps-constant-density_m1.ipynb')

