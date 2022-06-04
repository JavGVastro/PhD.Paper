# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import time

start_time = time.time()

# +
import cmasher as cmr
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import lmfit
import astropy.units as u
import pandas as pd
import corner
from scipy import interpolate

import pickle
import json

import bplot
import bfunc

# -

# Data load and region parameters

data = "CarC"

name = "Carina"

pickle_in = open("Results//SF" + data + ".pkl", "rb")
SFresults = pickle.load(pickle_in)

# +
# mask = SFresults['SF']["N pairs"] > 0
# -

B = np.array(SFresults["b2"])  # [mask]
r = np.array(SFresults["s"])  # [mask]
pc = SFresults["pc"]
# pix =  SFresults['pix']
box_size = SFresults["box_size"]
pc_per_arcsec = pc

# Merge first K + 1 points
K = 1
r[K] = np.mean(r[: K + 1])
B[K] = np.mean(B[: K + 1])
r = r[K:]
B = B[K:]

model = lmfit.Model(bfunc.bfunc03s)
model.param_names

# +
# Correlation length between 1/10 and 2 x box_size
model.set_param_hint("r0", value=0.1 * box_size, min=0.01 * box_size, max=2 * box_size)

# sig2 between 1/4 and 2 x max value of B(r)
model.set_param_hint("sig2", value=0.5 * B.max(), min=0.25 * B.max(), max=2 * B.max())

# m between 1/2 and 5/3
model.set_param_hint("m", value=1.0, min=0.5, max=2.0)

# Seeing RMS between 0.1 and 1.5 arcsec
model.set_param_hint(
    "s0", value=0.5 * pc_per_arcsec, min=0.1 * pc_per_arcsec, max=1.5 * pc_per_arcsec
)

# Noise cannot be much larger than smallest B(r)
model.set_param_hint("noise", value=0.5 * B.min(), min=0.0, max=3 * B.min())

# box_size is fixed
# model.set_param_hint("box_size", value=box_size, vary=False)
# -

pd.DataFrame(model.param_hints)

# +
#relative_uncertainty = 0.15
#weights = 1.0 / (relative_uncertainty * B)
#large_scale = r > 0.2 * box_size
#weights[large_scale] /= 1.25
#weights[:14] /= 1.5
# -

relative_uncertainty = 0.055
weights = 1.0 / (relative_uncertainty * B)
large_scale = r > 0.35 * box_size
weights[large_scale] /= 3.0
weights[:14] /= 3.0

to_fit = r <= 0.5 * box_size
#to_fit = ~large_scale
result = model.fit(B[to_fit], weights=weights[to_fit], r=r[to_fit])

result

# +
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
# -




# emcee

emcee_kws = dict(
    steps=50000, burn=500, thin=50, is_weighted=True, progress=False, workers=16
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

bplot.corner_plot(
    result_emcee, result, name, data, data_ranges=[0.95, 0.99, 0.995, 0.997, 0.999]
)
# data_ranges=[0.95, 0.99, 0.995, 0.995, 0.999]

bplot.STYLE["data label element"] = 4
bplot.STYLE["model label offset"] = (-60, 40)
bplot.STYLE["true model label offset"] = (30, -60)
bplot.strucfunc_plot(
    result_emcee, result, r, B, to_fit, name, data, box_size, large_scale
)

# +
#bplot.strucfunc_plot(
#    result_emcee, result_emcee, r, B, to_fit, name, data, box_size, large_scale
#)
# -



CIresults = {'result_emcee': result_emcee,
            'result' : result,
             'r' : r,
             'B' : B,
             'to_fit': to_fit,
             'name' : name,
             'data' : data,
             'box_size' : box_size,
             'large_scale' : large_scale
          }

f = open("Results//CI" + data + ".pkl", "wb")
pickle.dump(CIresults, f)
f.close()

# + tags=[]
print("--- %s seconds ---" % (time.time() - start_time))
# -


