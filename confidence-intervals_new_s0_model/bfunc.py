"""
New improved structure function models 

Will Henney 2021-10-16: Hopefully, these are the final versions

bfunc00s is the basic model for B(r) based on an exponential autocorrelation function
bfunc03s includes the effect of seeing with rms width s0 plus noise
bfunc04s is the same with additional effect of finite box size

See new-model-strucfuncs.{py,ipynb} for more details
"""
import numpy as np

##Observational constraints
#---------------------------------------------------------------------
def ratio_empirical(r, s0, a = 0.75):
    return (1 + (2.6 * s0 / r) ** (2*a))

def bfac(x):
    return (1 + 1.25 * x )

def seeing_empirical(r, s0, r0, a = 0.75):
    return  (bfac( s0 / r0) * ratio_empirical(r, s0, a) )**-1

##Depecrated                
#def finite_box_effect(r0, L, scale=4.0):
#    return 1 - np.exp(-L / (scale * r0))

##Models
#---------------------------------------------------------------------
def bfunc00s(r, r0, sig2, m):
    "Simple 3-parameter structure function"
    C = np.exp(-np.log(2) * (r / r0) ** m)
    return 2.0 * sig2 * (1.0 - C)

def bfunc03s(r, r0, sig2, m, s0, noise):
    "Structure function with better seeing (scale `s0`) and noise"
    return seeing_empirical(r, s0, r0) * bfunc00s(r, r0, sig2, m) + noise

##Depecrated
#def bfunc04s(r, r0, sig2, m, s0, noise, box_size):
#    "Structure function with better seeing (scale `s0`) and noise, plus finite box effect"
#    boxeff = finite_box_effect(r0, box_size)
#    return (
#        # Note that the seeing is unaffected by boxeff
#        seeing_empirical(r, s0, r0) * bfunc00s(r, boxeff * r0, boxeff * sig2, m)
#        + noise
#    )
