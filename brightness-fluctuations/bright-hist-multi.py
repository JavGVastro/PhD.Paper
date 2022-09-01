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
    siglos: tuple[float, float]
    sig2pos: tuple[float, float]
    ihdu: int = 1
    fmin: float = 0.0
    kmax: int = 0


plotfile = "bright-hist-multi.pdf"
images = [
    Image(
        "Orion",
        "KPech-Orion-H-sum.fits",
        ihdu=0,
        kmax=4,
        siglos=(9.9, 1.2),
        sig2pos=(11.0, 1.0),
    ),
    Image(
        "NGC 346",
        "MUSE-N346-H-sum.fits",
        kmax=0,
        siglos=(9.6, 1.0),
        sig2pos=(33.0, 3.0),
    ),
    Image(
        "30 Dor",
        "MUSE-Dor-H.fits",
        kmax=4,
        siglos=(21.7, 2.2),
        sig2pos=(297.0, 30.0),
    ),
    Image(
        "NGC 604",
        "TAURUS-604-Ha-Flux.fits",
        ihdu=0,
        fmin=1e-8,
        kmax=2,
        siglos=(17.5, 0.3),
        sig2pos=(86.0, 15.0),
    ),
    Image(
        "NGC 595",
        "TAURUS-595-Ha-Flux.fits",
        ihdu=0,
        fmin=0.1,
        kmax=0,
        siglos=(16.5, 0.1),
        sig2pos=(53.0, 3.0),
    ),
    Image(
        "Hubble V",
        "TAURUS-HV-Ha-Flux.fits",
        ihdu=0,
        fmin=0.1,
        kmax=0,
        siglos=(9.8, 0.03),
        sig2pos=(10.0, 2.0),
    ),
    Image(
        "Hubble X",
        "TAURUS-HX-Ha-Flux.fits",
        ihdu=0,
        fmin=0.1,
        kmax=0,
        siglos=(10.0, 0.02),
        sig2pos=(15.0, 2.0),
    ),
]

DATADIR = Path.cwd().parent / "astronomical-observations"
sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_color_codes("deep")
whitebox = {"facecolor": "white", "alpha": 0.7, "edgecolor": "none"}
fitter = fitting.LevMarLSQFitter()
LAYOUT = """
abc
def
..g
"""
fig, axes = plt.subplot_mosaic(
    LAYOUT,
    sharex=True,
    sharey=True,
)
fig.set_size_inches(10, 8)
resamples = [2, 4, 8, 16, 32, 64]
sigS_vals = []
rat_vals = []
erat_vals = []
Sfluct_label = r"\langle \delta S^2 \rangle^{1/2} / S_0"
for image, ax in zip(images, axes.values()):
    s = fits.open(DATADIR / image.fitsfile)[image.ihdu].data.astype(float)
    m = (s > image.fmin) & np.isfinite(s)
    w = np.ones_like(s)

    for n in resamples[: image.kmax]:
        [s,], m, w = downsample(
            [
                s,
            ],
            m,
            weights=w,
            mingood=1,
        )
    s /= np.mean(s[m])

    # ax.hist(s, bins=100, range=[0.0, 5.0], weights=s)
    smin, smax = -3.1, 3.1
    H, edges, patches = ax.hist(
        np.log(s[m]),
        # weights=s[m],
        density=True,
        bins=40,
        range=[smin, smax],
    )
    # Calculate bin centers
    x = 0.5 * (edges[:-1] + edges[1:])
    # Fit Gaussian
    g = models.Gaussian1D(amplitude=H.max(), stddev=0.5)
    core = H > 0.01
    g = fitter(g, x, H)
    sigS = np.sqrt(np.exp(g.stddev.value ** 2) - 1)
    xx = np.linspace(smin, smax, 200)
    ax.plot(xx, g(xx), "orange", lw=2)
    # Calculate equivalent fractional RMS width in linear brightness space
    X = np.exp(xx)
    Xmean = np.average(X, weights=g(xx))
    Xvariance = np.average((X - Xmean) ** 2, weights=g(xx))
    eps_rms = np.sqrt(Xvariance) / Xmean
    ax.set_xlim(smin, smax)
    ax.set_ylim(-0.05, 1.45)
    biglabel = (
        image.name
        + "\n"
        + f"${Sfluct_label} = {sigS:.2f}$"
        # + f", ${eps_rms:.2f}$"
    )
    ax.text(smin + 0.1, 1.0, biglabel, bbox=whitebox, fontsize="small")

    # Value of the sigma ratio
    sigpos_los = np.sqrt(image.sig2pos[0]) / image.siglos[0]
    # Relative error in sig_pos
    epos = 0.5 * image.sig2pos[1] / image.sig2pos[0]
    # Relative error in sig_los
    elos = image.siglos[1] / image.siglos[0]
    # Absolute error in ratio
    erat = np.hypot(epos, elos) * sigpos_los

    sigS_vals.append(sigS)
    rat_vals.append(sigpos_los)
    erat_vals.append(erat)

axes["g"].set_xlabel(r"Surface brightness: $\ln\, (S / S_0)$")
axes["a"].set_ylabel("Probability density")

axx = fig.add_subplot(3, 5, (11, 13))
axx.errorbar(sigS_vals, rat_vals, yerr=erat_vals, fmt="none", color="r")
axx.scatter(sigS_vals, rat_vals, color="r")
axx.set_xlim(0.0, None)
axx.set_ylim(0.0, None)
axx.set_xlabel(f"Surface brightness fluctuations: ${Sfluct_label}$")
axx.set_ylabel(r"$\sigma_\mathrm{pos} / \sigma_\mathrm{los}$")

fig.tight_layout()
fig.savefig(plotfile)
print(plotfile, end="")
