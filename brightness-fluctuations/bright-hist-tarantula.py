import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns
from astropy.io import fits
from astropy.modeling import models, fitting
from tetrabloks.rebin_utils import downsample, oversample
from dataclasses import dataclass

@dataclass

class Image:
  name: str
  fitsfile: str
  ihdu: int = 1





plotfile = 'bright-hist-tarantula.pdf'
images = [
  Image("30 Dor", "MUSE-Dor-H.fits"),
  Image("NGC 346", "MUSE-N346-H-sum.fits"),
]

DATADIR = Path.cwd().parent / "astronomical-observations"
sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_color_codes("deep")
whitebox = {'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'}
fitter = fitting.LevMarLSQFitter()
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(6, 3)
for image, ax in zip(images, axes.flat):
    s = fits.open(DATADIR / image.fitsfile)[image.ihdu].data.astype(float)
    m = (s > 0.0) & np.isfinite(s)
    w = np.ones_like(s)
    # for n in [2, 4, 8, 16, 32, 64]:
    #     [s,], m, w = downsample([s,], m, weights=w, mingood=1)
    s /= np.mean(s[m])

    #ax.hist(s, bins=100, range=[0.0, 5.0], weights=s)
    smin, smax = -3.1, 3.1
    H, edges, patches = ax.hist(np.log(s[m]), weights=s[m],
                                density=True, bins=25, range=[smin, smax])
    # Calculate bin centers
    x = 0.5*(edges[:-1] + edges[1:])
    # Fit Gaussian 
    g = models.Gaussian1D(amplitude=H.max(), stddev=0.5)
    core = H > 0.01*H.max()
    g = fitter(g, x[core], H[core])
    xx = np.linspace(smin, smax, 200)
    ax.plot(xx, g(xx), 'orange', lw=2)
    # Calculate equivalent fractional RMS width in linear brightness space
    X = np.exp(xx)
    Xmean = np.average(X, weights=g(xx))
    Xvariance = np.average((X - Xmean)**2, weights=g(xx))
    eps_rms = np.sqrt(Xvariance)/Xmean
    ax.set_xlim(smin, smax)
    ax.set_ylim(-0.05, 1.05)
    biglabel = (image.name
                + '\n' + r'$\sigma_{\ln(S/S_0)} '
                + '= {:.2f}$'.format(g.stddev.value))
    ax.text(-2.2, 0.8, biglabel, bbox=whitebox, fontsize='small')
axes[0].set_xlabel(r'Surface brightness: $\ln\, (S / S_0)$')
axes[0].set_ylabel('Probability density')
fig.tight_layout()
fig.savefig(plotfile)
print(plotfile, end="")
