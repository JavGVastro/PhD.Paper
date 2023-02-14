from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import math
import itertools
import json

def loadresults(a,b,c):
    """
    Load .json load_results
    """
    #Path names
    datapath_names = Path(open("path-name-list.txt", "r").read()).expanduser()
    samples=pd.read_csv(str(datapath_names) + '//' + a + '.csv',header=None)
    Names=pd.read_csv(str(datapath_names) + '//' + b + '.csv',header=None)

    #load Results
    datapath_res = Path(open("path-results.txt", "r").read()).expanduser()

    data = {}
    Results = {}

    for i in range(len(samples)):
        data[samples[0][i]] = json.load(open(str(datapath_res) + '/' + samples[0][i] + ".json"))

    #sigma
    sig = [[0]*(1) for i in range(len(samples))]
    siger = [[0]*(1) for i in range(len(samples))]

    #velocity dispersion with 2-sig intervals
    sig2 = [[0]*(1) for i in range(len(samples))]
    #sig2er = [[0]*(1) for i in range(len(samples))]
    sig2s2 = [[0]*(1) for i in range(len(samples))]
    sig2s2p = [[0]*(1) for i in range(len(samples))]
    sig2s2m = [[0]*(1) for i in range(len(samples))]

    #correlation length with 2-sig intervals
    r0 = [[0]*(1) for i in range(len(samples))]
    #r0er = [[0]*(1) for i in range(len(samples))]
    r0s2 = [[0]*(1) for i in range(len(samples))]
    r0s2p = [[0]*(1) for i in range(len(samples))]
    r0s2m = [[0]*(1) for i in range(len(samples))]

    #power-law
    m = [[0]*(1) for i in range(len(samples))]
    #mer = [[0]*(1) for i in range(len(samples))]
    ms2 = [[0]*(1) for i in range(len(samples))]
    ms2p = [[0]*(1) for i in range(len(samples))]
    ms2m = [[0]*(1) for i in range(len(samples))]

    #noise with 2-sig intervals
    bn = [[0]*(1) for i in range(len(samples))]
    #ner = [[0]*(1) for i in range(len(samples))]
    bns2 = [[0]*(1) for i in range(len(samples))]
    bns2p = [[0]*(1) for i in range(len(samples))]
    bns2m = [[0]*(1) for i in range(len(samples))]

    #seeing with 2-sig intervals
    s0 = [[0]*(1) for i in range(len(samples))]
    #s0er = [[0]*(1) for i in range(len(samples))]
    s0s2 = [[0]*(1) for i in range(len(samples))]
    s0s2p = [[0]*(1) for i in range(len(samples))]
    s0s2m = [[0]*(1) for i in range(len(samples))]

    pc = [[0]*(1) for i in range(len(samples))]
    box_size = [[0]*(1) for i in range(len(samples))]

    results_='results_' + c

    for i in range(len(samples)):

        sig2[i] = data[samples[0][i]][results_]['sig2'][0]
        sig2s2p[i] = data[samples[0][i]][results_]['sig2'][1]
        sig2s2m[i] = data[samples[0][i]][results_]['sig2'][2]

        r0[i]    = data[samples[0][i]][results_]['r0'][0]
        r0s2p[i] = data[samples[0][i]][results_]['r0'][1]
        r0s2m[i] = data[samples[0][i]][results_]['r0'][2]

        m[i]    = data[samples[0][i]][results_]['m'][0]
        ms2p[i] = data[samples[0][i]][results_]['m'][1]
        ms2m[i] = data[samples[0][i]][results_]['m'][2]

        bn[i]    = data[samples[0][i]][results_]['noise'][0]
        bns2p[i] = data[samples[0][i]][results_]['noise'][1]
        bns2m[i] = data[samples[0][i]][results_]['noise'][2]

        s0[i]    = data[samples[0][i]][results_]['s0'][0]
        s0s2p[i] = data[samples[0][i]][results_]['s0'][1]
        s0s2m[i] = data[samples[0][i]][results_]['s0'][2]

        box_size[i] = data[samples[0][i]]['properties']['box_size']
        pc[i] = data[samples[0][i]]['properties']['pc']

    s0f = pd.DataFrame(
        {
        "s0 [RMS]":s0,
        "s0+[RMS]": s0s2p,
        "s0-[RMS]": s0s2m,
        "s0 [FWHM]": np.array(s0)*2.35/np.array(pc),
        "s0- [FWHM]": np.array(s0s2m)*2.35/np.array(pc),
        "s0+ [FWHM]": np.array(s0s2p)*2.35/np.array(pc),
        "bn ":bn,
        "bn+": bns2p,
        "bn- ": bns2m,
        }
    )

    s0f.insert(loc=0, column='Region', value=Names)
    #s0f.round(4)

    s1f = pd.DataFrame(
        {
            "sig2":sig2,
            "sig2+": sig2s2p,
            "sig2-": sig2s2m,
            "r0":r0,
            "r0+": r0s2p,
            "r0-": r0s2m,
            "m":m,
            "m+": ms2p,
            "m-": ms2m,
            }
        )

    s1f.insert(loc=0, column='Region', value=Names)
    #s1f.round(4)

    #data = pd.DataFrame(
    #        {
    #        "sig2 [km/s]": sig2,
    #        "sig2er": sig2s2p,
    #        "sig [km/s]": np.array(sig2)**0.5,
    #        "siger": (np.array(sig2s2p)/np.array(sig2))*np.array(sig2)**0.5,
    #        "m": m,
    #        "mer": ms2p,
    #        "r0 [pc]": r0,
    #        "r0er": r0s2p,
    #        },
    #)

    #data.insert(loc=0, column='Region', value=Names)

    return s0f, s1f

def loadresults2(a,b,c):
    """
    Load .json load_results
    """
    #Path names
    datapath_names = Path(open("path-name-list.txt", "r").read()).expanduser()
    samples=pd.read_csv(str(datapath_names) + '//' + a + '.csv',header=None)
    Names=pd.read_csv(str(datapath_names) + '//' + b + '.csv',header=None)

    #load Results
    datapath_res = Path(open("path-results.txt", "r").read()).expanduser()

    data = {}
    Results = {}

    for i in range(len(samples)):
        data[samples[0][i]] = json.load(open(str(datapath_res) + '/' + samples[0][i] + ".json"))

    #sigma
    sig = [[0]*(1) for i in range(len(samples))]
    siger = [[0]*(1) for i in range(len(samples))]

    #velocity dispersion with 2-sig intervals
    sig2 = [[0]*(1) for i in range(len(samples))]
    #sig2er = [[0]*(1) for i in range(len(samples))]
    sig2s2 = [[0]*(1) for i in range(len(samples))]
    sig2s2p = [[0]*(1) for i in range(len(samples))]
    sig2s2m = [[0]*(1) for i in range(len(samples))]

    #correlation length with 2-sig intervals
    r0 = [[0]*(1) for i in range(len(samples))]
    #r0er = [[0]*(1) for i in range(len(samples))]
    r0s2 = [[0]*(1) for i in range(len(samples))]
    r0s2p = [[0]*(1) for i in range(len(samples))]
    r0s2m = [[0]*(1) for i in range(len(samples))]

    #power-law
    m = [[0]*(1) for i in range(len(samples))]
    #mer = [[0]*(1) for i in range(len(samples))]
    ms2 = [[0]*(1) for i in range(len(samples))]
    ms2p = [[0]*(1) for i in range(len(samples))]
    ms2m = [[0]*(1) for i in range(len(samples))]

    #noise with 2-sig intervals
    bn = [[0]*(1) for i in range(len(samples))]
    #ner = [[0]*(1) for i in range(len(samples))]
    bns2 = [[0]*(1) for i in range(len(samples))]
    bns2p = [[0]*(1) for i in range(len(samples))]
    bns2m = [[0]*(1) for i in range(len(samples))]

    #seeing with 2-sig intervals
    s0 = [[0]*(1) for i in range(len(samples))]
    #s0er = [[0]*(1) for i in range(len(samples))]
    s0s2 = [[0]*(1) for i in range(len(samples))]
    s0s2p = [[0]*(1) for i in range(len(samples))]
    s0s2m = [[0]*(1) for i in range(len(samples))]

    pc = [[0]*(1) for i in range(len(samples))]
    box_size = [[0]*(1) for i in range(len(samples))]

    results_= c

    for i in range(len(samples)):

        sig2[i] = data[samples[0][i]][results_]['sig2'][0]
        sig2s2p[i] = data[samples[0][i]][results_]['sig2'][1]
        #sig2s2m[i] = data[samples[0][i]][results_]['sig2'][2]

        r0[i]    = data[samples[0][i]][results_]['r0'][0]
        r0s2p[i] = data[samples[0][i]][results_]['r0'][1]
        #r0s2m[i] = data[samples[0][i]][results_]['r0'][2]

        m[i]    = data[samples[0][i]][results_]['m'][0]
        ms2p[i] = data[samples[0][i]][results_]['m'][1]
        #ms2m[i] = data[samples[0][i]][results_]['m'][2]

        bn[i]    = data[samples[0][i]][results_]['noise'][0]
        bns2p[i] = data[samples[0][i]][results_]['noise'][1]
        #bns2m[i] = data[samples[0][i]][results_]['noise'][2]

        s0[i]    = data[samples[0][i]][results_]['s0'][0]
        s0s2p[i] = data[samples[0][i]][results_]['s0'][1]
        #s0s2m[i] = data[samples[0][i]][results_]['s0'][2]

        box_size[i] = data[samples[0][i]]['properties']['box_size']
        pc[i] = data[samples[0][i]]['properties']['pc']

    s0f = pd.DataFrame(
        {
        "s0 [RMS]":s0,
        "s0+[RMS]": s0s2p,
        #"s0-[RMS]": s0s2m,
        "s0 [FWHM]": np.array(s0)*2.35/np.array(pc),
        #"s0- [FWHM]": np.array(s0s2m)*2.35/np.array(pc),
        "s0+ [FWHM]": np.array(s0s2p)*2.35/np.array(pc),
        "bn ":bn,
        "bn+": bns2p,
        #"bn- ": bns2m,
        }
    )

    s0f.insert(loc=0, column='Region', value=Names)
    #s0f.round(4)

    s1f = pd.DataFrame(
        {
            "sig2":sig2,
            "sig2+": sig2s2p,
            #"sig2-": sig2s2m,
            "r0":r0,
            "r0+": r0s2p,
            #"r0-": r0s2m,
            "m":m,
            "m+": ms2p,
            #"m-": ms2m,
            }
        )

    s1f.insert(loc=0, column='Region', value=Names)
    #s1f.round(4)

    #data = pd.DataFrame(
    #        {
    #        "sig2 [km/s]": sig2,
    #        "sig2er": sig2s2p,
    #        "sig [km/s]": np.array(sig2)**0.5,
    #        "siger": (np.array(sig2s2p)/np.array(sig2))*np.array(sig2)**0.5,
    #        "m": m,
    #        "mer": ms2p,
    #        "r0 [pc]": r0,
    #        "r0er": r0s2p,
    #        },
    #)

    #data.insert(loc=0, column='Region', value=Names)

    return s0f, s1f
