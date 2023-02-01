from matplotlib import pyplot as plt
import seaborn as sns
from scipy import interpolate
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
from rebin_utils import downsample, oversample
from astropy.modeling import models, fitting
fitter = fitting.LevMarLSQFitter()

def sbfluct(s, min, kmax, name):
        """
        Brightness fluctuations in log space
        """
        Sfluct_label = r"\langle \delta S^2 \rangle^{1/2} / S_0"
        resamples = [2, 4, 8, 16, 32, 64]

        m = (s > min) & np.isfinite(s)
        w = np.ones_like(s)

        for n in resamples[: 2]:
            [s,], m, w = downsample(
                [s,],
                m,
                weights=w,
                mingood=1,
                )
        s /= np.mean(s[m])

        smin, smax = -3.1, 3.1

        H, edges, patches = plt.hist(
            np.log(s[m]),
            # weights=s[m],
            density=True,
            bins=40,
            range=[-3.1, 3.1],)

        # Calculate bin centers
        x = 0.5 * (edges[:-1] + edges[1:])
        # Fit Gaussian
        g = models.Gaussian1D(amplitude=H.max(), stddev=0.5)
        core = H > 0.01
        g = fitter(g, x, H)
        sigS = np.sqrt(np.exp(g.stddev.value ** 2) - 1)
        esigS = 0.5 * np.sqrt(np.exp(g.mean.value ** 2) - 1)
        xx = np.linspace(smin, smax, 200)

        plt.plot(xx, g(xx), "orange", lw=2)

        X = np.exp(xx)
        Xmean = np.average(X, weights=g(xx))
        Xvariance = np.average((X - Xmean) ** 2, weights=g(xx))
        eps_rms = np.sqrt(Xvariance) / Xmean

        #plt.set_ylim(-0.05, 1.45)
        biglabel = (
            name
            + "\n"
            + f"${Sfluct_label} = {sigS:.2f}$"
             + f", ${eps_rms:.2f}$"
            )
        #plt.text(
        #    0.10,
        #    0.10,
        #    biglabel,
         #  ha="right",
        #    transform=ax.transAxes,
          # va="top",
            #bbox=whitebox,
            #fontsize="small",
    #    )

        print(eps_rms)
