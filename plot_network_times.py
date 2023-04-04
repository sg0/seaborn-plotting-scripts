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
    paths = ('routing_data_lulesh_dragonfly_and_Jellyfish/', 'routing_data_lulesh_fat_tree_and_Jellyfish/', 'routing_data_sweep3d_dragonfly_and_Jellyfish/', 'routing_data_sweep3d_fat_tree_and_Jellyfish/', 'routing_data_Graph500_BFS_fat_tree_and_Jellyfish/', 'routing_data_minivite_fat_tree_and_Jellyfish/', 'routing_data_Graph500_BFS_Dragonfly_and_Jellyfish/', 'routing_data_tric_dragonfly_and_Jellyfish/', 'routing_data_tric_fat_tree_and_Jellyfish/', 'routing_data_minivite_dragonfly_and_Jellyfish/')
    keys = ('Inactive', 'Compute', 'Estimated total runtime of')
    #inputs = ('orkut_', 'uk-2002_', 'stokes_', 'graph500-scale16-ef16_', 'graph500-scale17-ef16_', 'graph500-scale18-ef16_', 'graph500-scale19-ef16_', 'graph500-scale25-ef16_', 'graph500-scale24-ef16_', 'graph500-scale23-ef16_', 'sx-superuser_', 'soc-Epinions1_', 'com-Amazon_', 'lulesh_', 'sweep3d_')
    topos = ('rrg_153_24_87_', 'dfly_17_9_', 'rrg_120_38_446_', 'fat_tree_3_48_2')
    rt_algs = ('am', 'min', 'um', 'par', 'ugal', 'val')

    for path in paths:
        plot(keys, dirpath+path, topos, rt_algs)

def plot(keys, path, topos, rt_algs):
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    main_dir = Path(path)
    lg = []
    for topo in topos:
        if topo == "fat_tree_3_48_2":
            key = "*" + topo + ".out"
            l = list(main_dir.glob(key))
            if l != []:
                lg.append(l)
        else:
            for rt in rt_algs:
                key = "*" + topo + rt + ".out"
                l = list(main_dir.glob(key))
                if l != []:
                    lg.append(l)

    tot = len(lg)
    nrows,ncols = factors(tot) 
    concat_df = pd.DataFrame(columns = ["Network", "Time"])
    
    for i in range(nrows):
        for j in range(ncols):
            f = lg[i*ncols+j]
            title = title_extract(f[0])
            ll = [] 
            for k in range(len(f)):
                ll.append([title, calc_mpi_time(f[k], keys)])
                df = pd.DataFrame(ll, columns = ["Network", "Time"])
                concat_df = pd.concat([concat_df, df])
            
    #g = sns.lineplot(x='Network', y='Time', data=concat_df, style='dataset', hue='dataset', markers=True, markersize=10)
    #g = sns.lineplot(x='Network', y='Time', data=concat_df, markers=True, markersize=10)
    g = sns.barplot(x='Network', y='Time', data=concat_df)
    g.set(xlabel="Networks")
    g.set(ylabel="MPI time(s)")
    title = title_builder(path)
    g.set_title(title,fontsize=20)
    g.set_yscale("log")

    plt.xticks(rotation=30, fontsize=16, ha='right')
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    figname = os.path.basename(os.path.normpath(path)) + "_mpi_times.pdf"
    print(figname, "saving...")
    plt.savefig(figname, bbox_inches='tight')

def calc_mpi_time(f, keys):
    with open(f, 'r') as f:
        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) # size 0 means whole file
        i = 0.0
        c = 0.0
        e = 0.0
        while True:
            line = m.readline()
            text = line.strip()
            if bytes(keys[0], encoding='utf8') in text:
                i = split_tokenize(str(text), keys[0]) # inactive
            if bytes(keys[1], encoding='utf8') in text:
                c = split_tokenize(str(text), keys[1]) # compute
            if bytes(keys[2], encoding='utf8') in text:
                e = split_tokenize(str(text), keys[2]) # estimate
            if not line: 
                break
        m.close()
        #formula: (Total Run Time * #process) - (Inactive Time + Compute Time)
        return (e * float(ranks(str(f))) - (i + c))

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
    t = t.split('_', 1)[1].replace('.', '').upper()
    prefix = 'ADJ_'
    if t.startswith(prefix):
        return t[len(prefix):]
    return t

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

def split_tokenize(ls, key):
    if ls.find(key) != -1:
        key = ls
        return float(re.findall("\d+\.\d+|[\d]+", key)[0])
    return None

def get_num_str(s):
    l = [int(n) for n in s.split() if n.isdigit()]
    return l[-1] if l else None

if (__name__ == "__main__"):
    sys.exit(main())
