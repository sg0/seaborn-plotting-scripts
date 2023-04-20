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
from adjustText import adjust_text

def main():
    dirpath = '../Interconnect_Data/'
    paths = ('routing_data_lulesh_dragonfly_and_Jellyfish/', 'routing_data_lulesh_fat_tree_and_Jellyfish/', 'routing_data_sweep3d_dragonfly_and_Jellyfish/', 'routing_data_sweep3d_fat_tree_and_Jellyfish/', 'routing_data_Graph500_BFS_Dragonfly_and_Jellyfish/',  'routing_data_Graph500_BFS_fat_tree_and_Jellyfish/', 'routing_data_minivite_dragonfly_and_Jellyfish/', 'routing_data_minivite_fat_tree_and_Jellyfish/', 'routing_data_tric_dragonfly_and_Jellyfish/', 'routing_data_tric_fat_tree_and_Jellyfish/')
    keys = ('Number of packets traversing', 'Number of packets wait time equal')
    #inputs = ('orkut_', 'uk-2002_', 'stokes_', 'kron_g500-logn16_', 'graph500-scale16-ef16_', 'graph500-scale17-ef16_', 'graph500-scale18-ef16_', 'graph500-scale19-ef16_', 'graph500-scale25-ef16_', 'graph500-scale24-ef16_', 'graph500-scale23-ef16_', 'sx-superuser_', 'soc-Epinions1_', 'com-Amazon_', 'lulesh_', 'sweep3d_')
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
    #nrows,ncols = factors(tot) 
    nrows = 1
    ncols = len(lg)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5,nrows*4), squeeze=False)
    
    for i in range(nrows):
        for j in range(ncols):
            f = lg[i*ncols+j]
            filename = title_extract(f[0])
            title = title_builder(filename)
            concat_df = pd.DataFrame()
            merged_df = pd.DataFrame()
            for k in range(len(f)):
                stalls_df, hops_df = transform(f[k], keys)
                stalls_hops = pd.concat([stalls_df.assign(dataset='stalls'), hops_df.assign(dataset='hops')])
                concat_df = pd.concat([concat_df, stalls_hops])
            #g = sns.scatterplot(x='Costs', y='#Packets', data=concat_df, style='dataset', hue='dataset', s=75, ax=axes[i,j])
            g = sns.lineplot(x='Costs', y='#Packets', data=concat_df, style='dataset', hue='dataset', markers=True, markersize=10, ax=axes[i,j])
            if g is not None:
                #g.legend_.set_title('') 
                g.legend(fontsize=20) 
                g.set(xlabel=None)
                g.set(ylabel=None)
                g.set_title(title,fontsize=20)
                g.set_yscale("log")
            ts = []
            #https://stackoverflow.com/questions/9074996/how-to-annotate-point-on-a-scatter-automatically-placed-arrow
            for p in range(0,3):
                ts.append(plt.text(axes[p], axes[p + j], 'stalls='+str(concat_df['#Packets'][p])))
                #ts.append(plt.text(concat_df.Costs[p], concat_df['#Packets'][p], 'stalls='+str(concat_df['#Packets'][p])))
            adjust_text(ts, x=concat_df.Costs, y=concat_df['#Packets'], force_points=0.1, arrowprops=dict(arrowstyle='->', color='red'))


    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    figname = os.path.basename(os.path.normpath(path)) + ".pdf"
    print(figname, "saving...")
    fig.savefig(figname, bbox_inches='tight')

def transform(f, keys):
    with open(f, 'r') as f:
        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) # size 0 means whole file
        l_hops = []
        l_stal = []
        while True:
            line = m.readline()
            x = split_tokenize(line.strip(), bytes(keys[0], encoding='utf8')) # hops
            if x is None:
                y = split_tokenize(line.strip(), bytes(keys[1], encoding='utf8')) # stalls
                if y is not None:
                    l_stal.append(y)      
            else:
                l_hops.append(x) 

            if not line: 
                break
        m.close()
        #hops_df = pd.DataFrame(l_hops, columns = ['#Hops' , '#Packets'])
        #stalls_df = pd.DataFrame(l_stal, columns = ['#Stalls' , '#Packets'])
        hops_df = pd.DataFrame(l_hops, columns = ['Costs' , '#Packets'])
        stalls_df = pd.DataFrame(l_stal, columns = ['Costs' , '#Packets'])
        return stalls_df, hops_df

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

def title_builder(f):
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

def split_tokenize(ls, key):
    if ls.find(key) != -1:
        key = ls
        kval = key.split(b"=")
        if int(kval[1]) == 0:
            return None
        tnum = get_num_str(str(kval[0]))
        return [tnum, int(kval[1])]
    else:
        return None

def get_num_str(s):
    l = [int(n) for n in s.split() if n.isdigit()]
    return l[-1] if l else None

if (__name__ == "__main__"):
    sys.exit(main())
