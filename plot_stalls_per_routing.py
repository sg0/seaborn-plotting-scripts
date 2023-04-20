import functools
import operator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import matplotlib.ticker as mtick
import pathlib
import math
import mmap
import re
import os
import itertools
from pathlib import Path

def main():
    dirpath = '../Interconnect_Data/'
    paths = ('routing_data_lulesh_dragonfly_and_Jellyfish/', 'routing_data_sweep3d_dragonfly_and_Jellyfish/', 'routing_data_Graph500_BFS_Dragonfly_and_Jellyfish/', 'routing_data_minivite_dragonfly_and_Jellyfish/', 'routing_data_tric_dragonfly_and_Jellyfish/')
    key = "Number of packets wait time equal"
    topos = ('rrg_153_24_87_', 'dfly_17_9_')
    rt_algs = ('am', 'min', 'um', 'par', 'ugal', 'val')
    
    l_jam   = []
    l_jmin  = []
    l_jum   = []
    l_dmin  = []
    l_dpar  = []
    l_dval  = []
    l_dugal = []
 
    for path in paths:
        collect(key, dirpath+path, topos, rt_algs, l_jam, l_jmin, l_jum, l_dmin, l_dpar, l_dval, l_dugal)

    #df = pd.DataFrame(columns=["Jellyfish (AM)", "Jellyfish (MIN)", "Jellyfish (UM)", "Dragonfly (MIN)", "Dragonfly (PAR)", "Dragonfly (VAL)", "Dragonfly (UGAL)"])
    df = pd.DataFrame(columns=["Jellyfish (AM)", "Jellyfish (UM)", "Dragonfly (MIN)", "Dragonfly (PAR)", "Dragonfly (VAL)", "Dragonfly (UGAL)"])
    df['Jellyfish (AM)']   = pd.Series(l_jam) 
    #df['Jellyfish (MIN)']  = pd.Series(l_jmin) 
    df['Jellyfish (UM)']   = pd.Series(l_jum) 
    df['Dragonfly (MIN)']  = pd.Series(l_dmin) 
    df['Dragonfly (PAR)']  = pd.Series(l_dpar) 
    df['Dragonfly (VAL)']  = pd.Series(l_dval) 
    df['Dragonfly (UGAL)'] = pd.Series(l_dugal) 

    #df = df.fillna(0)

    sns.set_theme()
    sns.set(font_scale=2)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax = sns.boxplot(data=df)
    ax.set(xlabel="", ylabel="Queue lengths")
    #ax.set_yscale("log")
    plt.xticks(rotation=30, fontsize=18, ha='right')
    #plt.xticks(fontsize=16)
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    print("routing-stalls-networks.pdf saving...")
    fig.savefig('routing-stalls-networks.pdf', bbox_inches='tight')

def collect(key, path, topos, rt_algs, l_jam, l_jmin, l_jum, l_dmin, l_dpar, l_dval, l_dugal):
    main_dir = Path(path)
    lg = []
    for topo in topos:
        for rt in rt_algs:
            pat = "*" + topo + rt + ".out"
            l = list(main_dir.glob(pat))
            if l != []:
                lg.append(l)
    for f in lg:
        for k in f:
            if re.search('rrg_153_24_87_', str(k), re.IGNORECASE) and re.search('am', str(k), re.IGNORECASE):
                l_jam.append(calc_queue_length(k, key))
            if re.search('rrg_153_24_87_', str(k), re.IGNORECASE) and re.search('min', str(k), re.IGNORECASE):
                l_jmin.append(calc_queue_length(k, key))
            if re.search('rrg_153_24_87_', str(k), re.IGNORECASE) and re.search('um', str(k), re.IGNORECASE):
                l_jum.append(calc_queue_length(k, key))
            if re.search('dfly_17_9_', str(k), re.IGNORECASE) and re.search('min', str(k), re.IGNORECASE):
                l_dmin.append(calc_queue_length(k, key))
            if re.search('dfly_17_9_', str(k), re.IGNORECASE) and re.search('par', str(k), re.IGNORECASE):
                l_dpar.append(calc_queue_length(k, key))
            if re.search('dfly_17_9_', str(k), re.IGNORECASE) and re.search('val', str(k), re.IGNORECASE):
                l_dval.append(calc_queue_length(k, key))
            if re.search('dfly_17_9_', str(k), re.IGNORECASE) and re.search('ugal', str(k), re.IGNORECASE):
                l_dugal.append(calc_queue_length(k, key))

def calc_queue_length(f, key):
    with open(f, 'r') as f:
        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) # size 0 means whole file
        c = 0
        while True:
            line = m.readline()
            text = line.strip()
            if str(text).find(key) != -1:
                if qstalls(text) is not None:
                    c = c + 1
            if not line: 
                break
        m.close()
        return c

def qstalls(l):
    kval = l.split(b"=")
    if int(kval[1]) == 0:
        return None
    return int(kval[1])

if (__name__ == "__main__"):
    sys.exit(main())
