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

CS_SQUARED = 11.0 ** 2
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
abcd
...e
...f
...g
"""
fig, axes = plt.subplot_mosaic(
    LAYOUT,
    sharex=True,
    sharey=True,
)
fig.set_size_inches(11, 9)
resamples = [2, 4, 8, 16, 32, 64]
sigS_vals = []
esigS_vals = []
sig2_vals = []
esig2_vals = []
rat_vals = []
erat_vals = []
Sfluct_label_long = r"\langle \delta S^2 \rangle^{1/2} / S_0"
Sfluct_label = r"\sigma_S"
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
    esigS = 0.5 * np.sqrt(np.exp(g.mean.value ** 2) - 1)
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
    ax.text(
        0.97,
        0.95,
        biglabel,
        ha="right",
        transform=ax.transAxes,
        va="top",
        bbox=whitebox,
        fontsize="small",
    )

    # Value of the sigma ratio
    sigpos_los = np.sqrt(image.sig2pos[0]) / image.siglos[0]
    # Relative error in sig_pos
    epos = 0.5 * image.sig2pos[1] / image.sig2pos[0]
    # Relative error in sig_los
    elos = image.siglos[1] / image.siglos[0]
    # Absolute error in ratio
    erat = np.hypot(epos, elos) * sigpos_los

    sigS_vals.append(sigS)
    esigS_vals.append(esigS)

    rat_vals.append(sigpos_los)
    erat_vals.append(erat)

    mach = np.sqrt(3 * image.sig2pos[0] / CS_SQUARED)
    sig2_vals.append(mach)
    esig2_vals.append(epos * mach)

axes["g"].set_xlabel("Surface brightness:\n" + r"$\ln (S / S_0)$")
axes["a"].set_ylabel("Probability\ndensity")

for label, ax in axes.items():
    ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", fontweight="bold")

axes["c"].text(
    0.5,
    0.4,
    "**",
    transform=axes["c"].transAxes,
    va="center",
    ha="center",
    fontweight="bold",
    fontsize="x-large",
)

smax = 1.39

axx = fig.add_subplot(4, 4, (13, 15))
alpha = np.logspace(0.0, 1.6, 500)
sig_S = np.abs(alpha - 1) / (1 + alpha)
sig_pos = 0.5 * np.abs(alpha - 1) / (1 + alpha)
sig_los = np.sqrt(alpha) / (1 + alpha)
extra_sig = 0.25
axx.plot(sig_S, sig_pos / sig_los, linestyle="solid", zorder=1)
axx.plot(
    sig_S,
    sig_pos / np.hypot(sig_los, extra_sig),
    color=axx.lines[-1].get_color(),
    linestyle="dashed",
    zorder=1,
)
axx.errorbar(
    sigS_vals, rat_vals, xerr=esigS_vals, yerr=erat_vals, fmt="none", color="b"
)
axx.scatter(sigS_vals, rat_vals, color="b")
axx.text(0.02, 0.97, "i", transform=axx.transAxes, va="top", fontweight="bold")
axx.text(
    1.2,
    0.6,
    "**",
    va="center",
    ha="center",
    fontweight="bold",
    fontsize="x-large",
    color="k",
)
axx.set_xlim(0.0, smax)
axx.set_ylim(0.0, 0.95)
axx.set_xlabel(f"RMS brightness fluctuation:\n${Sfluct_label} = {Sfluct_label_long}$")
axx.set_ylabel(r"$\sigma_\mathrm{pos} / \sigma_\mathrm{los}$")

axxx = fig.add_subplot(4, 4, (5, 11))
axxx.fill_between(
    [0, smax],
    [0, 1.5 * smax],
    [0, 4.5 * smax],
    color="k",
    linewidth=0,
    alpha=0.1,
)
axxx.errorbar(
    sigS_vals, sig2_vals, xerr=esigS_vals, yerr=esig2_vals, fmt="none", color="r"
)
axxx.scatter(sigS_vals, sig2_vals, color="r")
axxx.text(0.02, 0.97, "h", transform=axxx.transAxes, va="top", fontweight="bold")
axxx.text(
    1.2,
    2.5,
    "**",
    va="center",
    ha="center",
    fontweight="bold",
    fontsize="x-large",
    color="k",
)
axxx.set_xlim(0.0, smax)
axxx.set_ylim(0.0, 2.9)
axxx.set_ylabel(r"$\mathcal{M} \approx \sqrt{3} \sigma_\mathrm{pos} / c_\mathrm{s}$")
axxx.set_xticklabels([])
# fig.subplots_adjust(0.16, 0.2, 0.97, 0.98, wspace=0.1, hspace=0.1)
# fig.tight_layout(rect=(0.1, 0.2, 1.1, 1.2), pad=0.0)

fig.savefig(plotfile, bbox_inches="tight")
print(plotfile, end="")
