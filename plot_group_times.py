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
from collections import defaultdict

def main():
    dirpath = '../Interconnect_Data/'
    paths = ('routing_data_lulesh_dragonfly_and_Jellyfish/', 'routing_data_lulesh_fat_tree_and_Jellyfish/', 'routing_data_sweep3d_dragonfly_and_Jellyfish/', 'routing_data_sweep3d_fat_tree_and_Jellyfish/', 'routing_data_Graph500_BFS_Dragonfly_and_Jellyfish/',  'routing_data_Graph500_BFS_fat_tree_and_Jellyfish/', 'routing_data_minivite_dragonfly_and_Jellyfish/', 'routing_data_minivite_fat_tree_and_Jellyfish/', 'routing_data_tric_dragonfly_and_Jellyfish/', 'routing_data_tric_fat_tree_and_Jellyfish/')
    keys = ('Inactive', 'Compute', 'Estimated total runtime of', 'Modularity, #Iterations')
    #inputs = ('orkut_', 'uk-2002_', 'stokes_', 'graph500-scale16-ef16_', 'graph500-scale17-ef16_', 'graph500-scale18-ef16_', 'graph500-scale19-ef16_', 'graph500-scale25-ef16_', 'graph500-scale24-ef16_', 'graph500-scale23-ef16_', 'sx-superuser_', 'soc-Epinions1_', 'com-Amazon_', 'lulesh_', 'sweep3d_')
    topos = ('rrg_153_24_87_', 'dfly_17_9_', 'rrg_120_38_446_', 'fat_tree_3_48_2')
    rt_algs = ('am', 'min', 'um', 'par', 'ugal', 'val')
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    for path in paths:
        lg = []
        main_dir = Path(dirpath+path)
        for topo in topos:
            if topo == "fat_tree_3_48_2":
                key = "*" + topo + ".out"
                l = list(main_dir.glob(key))
                if l != []:
                    lg.extend(l)
            else:
                for rt in rt_algs:
                    key = "*" + topo + rt + ".out"
                    l = list(main_dir.glob(key))
                    if l != []:
                        lg.extend(l)
        df = fetch_data(lg, keys)
        plt.figure()
        kind = "box"
        if re.search('lulesh', path, re.IGNORECASE) or re.search('sweep3d', path, re.IGNORECASE):
            kind = "bar"
        g = sns.catplot(data=df, kind=kind, height=6, aspect=1.5)
        title = title_builder(path)
        g.set(xlabel="Networks")
        g.set(ylabel="MPI time(s)")
        #g.set_title(title,fontsize=20)
        g.fig.subplots_adjust(top=1.5)
        g.fig.suptitle(title,fontsize=25)
        #g.set_yscale("log")
        plt.xticks(rotation=30, fontsize=20, ha='right')
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
        figname = os.path.basename(os.path.normpath(path)) + "_mpi_times.pdf"
        print(figname, "saving...")
        plt.savefig(figname, bbox_inches='tight')

def fetch_data(f, keys):
    d_net = {}
    for h in f:
        filename = title_extract(h)
        title = xtics_builder(filename)
        if title not in d_net:
            d_net[title] = []
        d_net[title].append(calc_mpi_time(h, keys))
    #cols = list(d_net.keys())
    df = pd.concat({k: pd.Series(v) for k, v in d_net.items()}, axis=1)
    return df
    #df = pd.DataFrame(columns = ["Network", "Time"])
    #df['Network'] = pd.Series(l_net) 
    #df['Time'] = pd.Series(l_time) 
    #g = sns.lineplot(x='Network', y='Time', data=concat_df, style='dataset', hue='dataset', markers=True, markersize=10)
    #g = sns.lineplot(x='Network', y='Time', data=concat_df, markers=True, markersize=10)
    #g = sns.violinplot(x='Network', y='Time', data=mdf)
    #g = sns.boxplot(x='Network', y='Time', data=df)
    #g = sns.boxplot(data=df)

def calc_mpi_time(f, keys):
    with open(f, 'r') as f:
        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) # size 0 means whole file
        i = 0.0
        c = 0.0
        e = 0.0
        t = 1.0
        while True:
            line = m.readline()
            text = line.strip()
            if bytes(keys[0], encoding='utf8') in text:
                i = split_tokenize(str(text), keys[0]) # inactive
            if bytes(keys[1], encoding='utf8') in text:
                c = split_tokenize(str(text), keys[1]) # compute
            if bytes(keys[2], encoding='utf8') in text:
                e = split_tokenize(str(text), keys[2]) # estimate
            if bytes(keys[3], encoding='utf8') in text:
                t = split_tokenize(str(text), keys[3], 1) # iterations
            if not line: 
                break
        m.close()
        #formula: (Total Run Time * #process) - (Inactive Time + Compute Time)
        return (e * float(ranks(str(f))) - (i + c)) / t

def title_builder(f):
    s = ''
    # application
    if re.search('lulesh', f, re.IGNORECASE):
        s = 'LULESH ('
    if re.search('sweep3d', f, re.IGNORECASE):
        s = 'SWEEP3D ('
    if re.search('minivite', f, re.IGNORECASE):
        s = 'Graph Clustering ('
    if re.search('tric', f, re.IGNORECASE):
        s = 'Graph Triangle Counting ('
    if re.search('BFS', f, re.IGNORECASE):
        s = 'Graph BFS ('
    # network
    if re.search('fat_tree', f, re.IGNORECASE):
        s += 'Fat tree vs. Jellyfish)' 
    if re.search('dragonfly', f, re.IGNORECASE):
        s += 'Dragonfly vs. Jellyfish)' 
    return s

def ranks(f):
    if re.search('lulesh', f, re.IGNORECASE):
        return 1000
    return 1024

def find_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def title_extract(f):
    t = os.path.basename(f)
    t = t.rsplit(".", 1)[0]
    if re.search('rrg', t, re.IGNORECASE):
        return t[re.search('rrg', t, re.IGNORECASE).start():].upper()
    if re.search('dfly', t, re.IGNORECASE):
        return t[re.search('dfly', t, re.IGNORECASE).start():].upper()
    if re.search('fat_tree', t, re.IGNORECASE):
        return t[re.search('fat_tree', t, re.IGNORECASE).start():].upper()
    return t

def xtics_builder(f):
    s = ''
    # network
    if re.search('fat_tree', str(f), re.IGNORECASE):
        s += 'Fat Tree' 
        return s
    if re.search('dfly', str(f), re.IGNORECASE):
        s += 'Dragonfly ('
    if re.search('rrg', str(f), re.IGNORECASE):
        s += 'Jellyfish ('
    # routing
    s += f.rsplit('_', 1)[1] + ')'
    return s

def factors(number):
    x = int(math.sqrt(number))
    while(number % x):
        x += 1
    return [int(number/x), x]

def flatten(L):
    islist = map(lambda x: isinstance(x, list), L)
    if (all(islist)):
        L1 = functools.reduce(operator.concat, L) 
        return flatten(L1)
    else:
        return L

def split_tokenize(ls, key, idx = 0):
    if ls.find(key) != -1:
        key = ls
        return float(re.findall("\d+\.\d+|[\d]+", key)[idx])
    return None

def get_num_str(s):
    l = [int(n) for n in s.split() if n.isdigit()]
    return l[-1] if l else None

if (__name__ == "__main__"):
    sys.exit(main())
