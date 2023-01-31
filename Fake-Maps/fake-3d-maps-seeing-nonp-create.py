#!/usr/bin/env python
# coding: utf-8

# # Fake maps with fake seeing 3dfield
# 
# + Generate simulated emissivity and velocity cubes
# + Integrate them to get simulated intensity, velocity (and sigma) maps
# + Make them non periodic by dividing them in 4.
# + Apply fake seeing to the cubes
# + Export 2x2 moment1 maps: non-tapared, tapered and smoothed tapered
# 

import time
start_time=time.time()


from pathlib import Path
import numpy as np
import json
from astropy.io import fits
from astropy.utils.misc import JsonCustomEncoder
import astropy.units as u
import cmasher as cmr
from matplotlib import pyplot as plt
import turbustat.statistics as tss
import turbustat.simulator
#from turbustat.simulator import make_3dfield
from turb_utils import make_extended
import seaborn as sns
import sys
from turbustat.simulator import make_ppv
from turb_utils import make_extended, make_3dfield
from spectral_cube import SpectralCube  

from astropy.convolution import Gaussian2DKernel, convolve_fft

sys.path.append("../structure-functions")

import strucfunc

sns.set_color_codes()
sns.set_context("talk")





# ### Non-periodic boundaries
# 
# We can make a map that is twice as big and then analyze 1/4 of it. That way, it will not be periodic.  

widths = [1, 2, 4, 8, 16, 32]
r0 = 32.0
N = 256
m = 1.0


def split_square_in_4(arr):
    ny, nx = arr.shape
    assert nx == ny and nx % 2 == 0
    slices = slice(None, nx // 2), slice(nx // 2, None)
    corners = []
    for i, j in [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]:
        corners.append(arr[slices[i], slices[j]])
    return corners



# #### Effects of smoothing the non-periodic maps

density = np.ones((2*N,2*N,2*N))* u.cm**-3  
#density = make_3dfield(N, powerlaw=3.0 + m, amp=1.,randomseed=328764) * u.cm**-3  

n = 100
density*n
density += density.std()  
density[density.value < 0.] = 0. * u.cm**-3  


# non-tapered

v_nt = make_3dfield(
    2*N,
    ellip=0.5,
    theta=45,
    powerlaw=3.0 + m,
    randomseed=2021_10_08)* u.km / u.s  



cube_hdu_nt = make_ppv(v_nt, density, los_axis=0,
                    T=10000 * u.K, chan_width=0.5 * u.km / u.s,
                    v_min=-20 * u.km / u.s,
                    v_max=20 * u.km / u.s)  

cube_nt = SpectralCube.read(cube_hdu_nt)  

vms_nt = split_square_in_4(cube_nt.moment1().value)


# Tapared maps

velocity = make_3dfield(
    2*N,
    ellip=0.5,
    theta=45,
    correlation_length=r0,  
    powerlaw=3.0 + m,
    randomseed=2021_10_08)* u.km / u.s  

density = np.ones((2*N,2*N,2*N))* u.cm**-3  
#density = make_3dfield(N, powerlaw=3.0 + m, amp=1.,randomseed=328764) * u.cm**-3  

n = 100
density = density*n
density += density.std()  
density[density.value < 0.] = 0. * u.cm**-3  

cube_hdu = make_ppv(velocity, density, los_axis=0,
                    T=10000 * u.K, chan_width=0.5 * u.km / u.s,
                    v_min=-20 * u.km / u.s,
                    v_max=20 * u.km / u.s)  

cube = SpectralCube.read(cube_hdu)  

vms_t = split_square_in_4(cube.moment1().value)


widths = [1, 2, 4, 8, 16, 32]
cubes_smooth = {}
for width in widths:
    kernel = Gaussian2DKernel(x_stddev=width)
    cubes_smooth[width] = cube.spatial_smooth(kernel)
    


vmap_nps = {}

for width in widths:
    vmap_nps[width] = split_square_in_4(cubes_smooth[width].moment1().value)
    


v_maps_3ds = {}
v_maps_3ds[0] = vms_nt
v_maps_3ds[1] = vms_t
v_maps_3ds[2] = vmap_nps


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


jsonfilename = "v_maps_3ds.json"
with open(jsonfilename, "w") as f:
    json.dump(v_maps_3ds, fp=f, indent=3, cls=MyEncoder)


imshow_kwds = dict(origin="lower", cmap="RdBu_r")


ncols = len(widths) + 1
nrows = 4
fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(8, 5.1),
    sharex=True,
    sharey=True,
)
for j, vm in enumerate(vms_t):
    im = axes[j, 0].imshow(vm, **imshow_kwds)
axes[0, 0].set_title("original")
for i, width in enumerate(widths):
    for j, vm in enumerate(vmap_nps[width]):
        im = axes[j, i + 1].imshow(vm, **imshow_kwds)
    axes[0, i + 1].set_title(fr"$s_0 = {width}$")

for ax in axes.flat:
    ax.set(xticks=[], yticks=[])
sns.despine(left=True, bottom=True)
fig.tight_layout(h_pad=0.2, w_pad=0.2)
fig.savefig("fake-seeing-nonp-thumbnails-3d.pdf")


get_ipython().system('jupyter nbconvert --to script --no-prompt fake-maps-seeing-nonp-3dfields-create.ipynb')


print("--- %s seconds ---" % (time.time()-start_time))

