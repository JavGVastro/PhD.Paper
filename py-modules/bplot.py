"""
PLot models for the paper

Will Henney

corner_plot
spread_span
strucfunc_plot

See new-model-strucfuncs.{py,ipynb} for more details
"""
import cmasher as cmr
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import lmfit
import json
from pathlib import Path

# import astropy.units as u
import pandas as pd
import corner
import bfunc

FIGPATH = Path("Imgs")
STYLE = {
    "data label element": 3,
    "model label element": 1,
    "true model label element": 10,
    "data label offset": (40, -20),
    "model label offset": (20, -40),
    "true model label offset": (-30, 60),
}

sns.set_color_codes()
sns.set_context("talk")


def corner_plot(
    result_emcee,
    result_orig,
    source_name,
    suffix,
    data_ranges=None,
    also_reverse=False,
    **corner_kwds,
):
    """
    Make a corner plot of parameter correlations from model fit

    Optionally, make a second reflected plot if also_reverse=True
    """
    fancy_names = {
        "r0": "Corr. length:\n" + r"$r_0$ [parsec]",
        "sig2": "Vel. variance:\n" + r"$\sigma^2$ [km$^2$ s$^{-1}$]",
        "m": "Power-law\n" + r"slope: $m$",
        "s0": "RMS seeing:\n" + r"$s_0$ [parsec]",
        "noise": "Noise:\n" + r"[km$^2$ s$^{-1}$]",
    }
    # We need to remove the frozen parameters from the list before
    # passing it as the truths argument
    truths = [_.value for _ in result_orig.params.values() if _.vary]
    figfile_and_reverse = [[f"corner-emcee-{suffix}.pdf", False]]
    if also_reverse:
        # Optionally make a second plot that is mirrored about the diagonal
        figfile_and_reverse.append([f"corner-emcee-{suffix}-rev.pdf", True])
    for figfile, reverse in figfile_and_reverse[::-1]:
        fig = corner.corner(
            result_emcee.flatchain,
            labels=[fancy_names[_] for _ in result_emcee.var_names],
            truths=truths,
            labelpad=0.4,
            range=[0.995] * len(truths) if data_ranges is None else data_ranges,
            color="#b04060",
            hist_kwargs=dict(color="k"),
            data_kwargs=dict(color="k"),
            reverse=reverse,
            max_n_ticks=4,
            **corner_kwds,
        )
        sns.despine()
        # fig.set_size_inches(10, 10)
        fig.tight_layout(h_pad=0, w_pad=0)
        # fig.suptitle(source_name)
        fig.text(
            0.5,
            0.95,
            source_name,
            color="black",
            bbox=dict(facecolor="none", edgecolor="black", pad=5.0),
        )
        fig.savefig(FIGPATH / figfile)


def spread_span(
    ax,
    values,
    orient="h",
    pranges=[[2.5, 97.5], [16, 84]],
    alphas=[0.1, 0.3],
    colors=["m", "m"],
):
    span_kwds = dict(zorder=-1, linestyle="solid")
    """
    Show the spread of values as overlapping translucent boxes.
    A box is plotted for each percentile range in pranges
    Orientation (horizontal or vertical) is controlled by orient
    """
    for prange, alpha, color in zip(pranges, alphas, colors):
        if orient == "h":
            ax.axhspan(
                *np.percentile(values, prange), alpha=alpha, color=color, **span_kwds
            )
        else:
            ax.axvspan(
                *np.percentile(values, prange), alpha=alpha, color=color, **span_kwds
            )


def strucfunc_plot(
    result_emcee,
    result_orig,
    r,
    B,
    to_fit,
    source_name,
    suffix,
    box_size,
    large_scale,
    func=bfunc.bfunc03s,
    func00=bfunc.bfunc00s,
):

    whitebox = dict(facecolor="white", pad=5, edgecolor="0.5", linewidth=0.5)
    label_kwds = dict(ha="center", va="center", bbox=whitebox, zorder=100)
    label_kwds2 = dict(ha="left", va="center", bbox=whitebox, zorder=100)

    xmin, xmax = 5e-4 * box_size, 8 * box_size
    ymin, ymax = 5e-4 * B.max(), 8 * B.max()
    xarr = np.logspace(np.log10(xmin), np.log10(xmax))

    fig, ax = plt.subplots(figsize=(10, 10))

    # best = result_emcee.best_values
    # Use the original lev-marq fit for the "best" parameter values
    best = result_orig.best_values

    # Plot the data
    yerr = 1.0 / result_emcee.weights
    data_points = ax.errorbar(r[to_fit], B[to_fit], yerr=yerr, fmt="o", zorder=99)
    c_data = data_points[0].get_color()
    ax.annotate(
        "observed",
        (r[STYLE["data label element"]], B[STYLE["data label element"]]),
        xytext=STYLE["data label offset"],
        textcoords="offset points",
        color=c_data,
        arrowprops=dict(arrowstyle="->", color=c_data, shrinkB=8),
        **label_kwds2,
    )

    # Plot the full model including instrumental effects
    Ba = func(xarr, **best)
    line_apparent = ax.plot(xarr, Ba)
    c_apparent = line_apparent[0].get_color()
    ia = STYLE["model label element"]
    xa = 0.5 * (r[ia] + r[ia + 1])
    ya = func(xa, **best)
    ax.annotate(
        "model",
        (xa, ya),
        xytext=STYLE["model label offset"],
        textcoords="offset points",
        color=c_apparent,
        arrowprops=dict(arrowstyle="->", color=c_apparent, shrinkB=8),
        **label_kwds2,
    )
    # ax.text(xmax / 1.5, Ba[-1], "model apparent", color=c_apparent, **label_kwds2)

    # Plot the underlying model without instrumental effects
    Bu = func00(xarr, best["r0"], best["sig2"], best["m"])
    line_true = ax.plot(xarr, Bu, linestyle="dashed")
    c_true = line_true[0].get_color()
    ax.annotate(
        "model true",
        (
            xarr[STYLE["true model label element"]],
            Bu[STYLE["true model label element"]],
        ),
        xytext=STYLE["true model label offset"],
        textcoords="offset points",
        color=c_true,
        arrowprops=dict(arrowstyle="->", color=c_true, shrinkB=8),
        **label_kwds,
    )
    # ax.text(xmax / 1.5, Bu[-1], "model true", color=c_true, **{**label_kwds2, "zorder": 99})

    # Plot the fit results
    # result_emcee.plot_fit(ax=ax)

    # Plot a random sample of the emcee chain
    inds = np.random.randint(len(result_emcee.flatchain), size=100)
    for ind in inds:
        sample = result_emcee.flatchain.iloc[ind]
        try:
            s0 = sample.s0
        except AttributeError:
            s0 = best["s0"]
        try:
            m = sample.m
        except AttributeError:
            m = best["m"]
        Bsamp = func(
            xarr,
            sample.r0,
            sample.sig2,
            m,
            s0,
            sample.noise,
            # best["box_size"]
        )
        ax.plot(xarr, Bsamp, alpha=0.05, color="orange")
        Busamp = func00(xarr, sample.r0, sample.sig2, m)
        ax.plot(xarr, Busamp, alpha=0.05, color="g")

    # Dotted lines for 2 x rms seeing and for box size
    try:
        spread_span(ax, result_emcee.flatchain.s0, orient="v")
        ax.axvline(best["s0"], color="k", linestyle="dotted")
        ax.annotate(
            r"$s_0$",
            (best["s0"], 1.5 * ymin),
            xytext=(-40, 0),
            textcoords="offset points",
            color="k",
            arrowprops=dict(arrowstyle="->", color="k", shrinkB=2),
            **label_kwds,
        )
    except AttributeError:
        # Case where s0 parameter is frozen
        pass

    ax.axvline(box_size, color="k", linestyle="dotted")
    ax.text(box_size, 1.5 * ymin, r"$L$", **label_kwds)

    # Dashed lines for best-fit r0 and sig2
    ax.axvline(best["r0"], color="k", linestyle="dashed")
    spread_span(ax, result_emcee.flatchain.r0, orient="v")
    # ax.text(best["r0"], 1.5 * ymin, r"$r_0$", **label_kwds)
    ax.annotate(
        r"$r_0$",
        (best["r0"], 1.5 * ymin),
        xytext=(40, 20),
        textcoords="offset points",
        color="k",
        arrowprops=dict(arrowstyle="->", color="k", shrinkB=2),
        **label_kwds,
    )

    ax.axhline(best["sig2"], color="k", linestyle="dashed")
    spread_span(ax, result_emcee.flatchain.sig2, orient="h")
    ax.text(1.5 * xmin, best["sig2"], r"$\sigma^2$", **label_kwds)

    ax.axhline(best["noise"], color="k", linestyle="dotted")
    spread_span(ax, result_emcee.flatchain.noise, orient="h")
    if best["noise"] > 0.5 * best["sig2"]:
        # Avoid collision of labels
        ytext = -40
    else:
        ytext = 40
    ax.annotate(
        r"$B_\mathrm{noise}$",
        (1.5 * xmin, best["noise"]),
        xytext=(20, ytext),
        textcoords="offset points",
        color="k",
        arrowprops=dict(arrowstyle="->", color="k", shrinkB=2),
        **label_kwds,
    )

    if np.any(~to_fit):
        # Add in the points not included in fit
        ax.plot(
            r[~to_fit],
            B[~to_fit],
            "o",
            color=c_data,
            mew=3,
            fillstyle="none",
            zorder=99,
        )
        # Translucent overlay box to indicate the large scale values that are excluded from the fit
        ax.axvspan(box_size / 2, ymax * 3, color="w", alpha=0.5, zorder=50)

    ax.text(
        np.sqrt(xmin * xmax), ymax / 1.5, source_name, fontsize="large", **label_kwds
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Separation: $r$ [pc]",
        ylabel=r"Structure function: $B(r)$ [km$^{2}$/s$^{2}$]",
        xlim=[xmin, xmax],
        ylim=[ymin, ymax],
    )
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGPATH / f"sf-emcee-{suffix}.pdf")
