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

def main():

    path = './data/qdrep_report_gpu_'
    devices = ['tonga_', 'a100_']
    options = ['mp0_', 'mp1_']
    version = []
    prefix = []

    v = 0
    if (len(sys.argv) > 1):
        v = int(sys.argv[1])
   
    # full
    #version = ['Baseline', 'TFDataGen', 'TFDataOptMGPU', 'TFDataOptMGPUAcc', 'PyTorch']
    #prefix = ['ng1_nc16_e20_b65536_tb0_', 'ng2_nc16_e20_b65536_tb0_', 'ng4_nc16_e20_b65536_tb0_', 'ng6_nc16_e20_b65536_tb0_', 'ng8_nc16_e20_b65536_tb0_']
    
    if v > 0: # multi-GPU
        version = ['TFDataOptMGPU', 'TFDataOptMGPUAcc', 'PyTorch']
        prefix = ['ng2_nc16_e20_b65536_tb0_', 'ng4_nc16_e20_b65536_tb0_', 'ng6_nc16_e20_b65536_tb0_', 'ng8_nc16_e20_b65536_tb0_']
    else:
        version = ['Baseline', 'TFDataGen', 'PyTorch']
        prefix = ['ng1_nc16_e20_b65536_tb0_']

    main_cuda(path, devices, options, version, prefix)

    main_tf(path, devices, options, version, prefix)

def main_cuda(path, devices, options, version, prefix):
    path_cuda = [[[[ (path + dev + pref + opt + ver + '_cudaapisum.csv') 
        for ver in version ] 
        for opt in options ]
        for pref in prefix ]
        for dev in devices ]
    path_cuda = flattenL(path_cuda)
    path_list = [item for item in path_cuda if pathlib.Path(item).exists() is True]
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    tot = len(path_list)
    nrows,ncols = factors(tot) 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5,nrows*4))

    for i in range(nrows):
        for j in range(ncols):
            f = path_list[i*ncols+j]
            title = titlegen(f)
            df = pd.read_csv(f)
            cdf = cuda_df_manip(df) 
            g = sns.barplot(data=cdf, ax=axes[i,j], y='Time(%)', x='Name')
            #g = sns.heatmap(data=cdf, ax=axes[i,j], y='Time(%)', x='Name', annot=True, cbar=True, cmap='RdBu_r')
            g.set(xlabel=None)
            g.set(ylabel=None)
            g.set_title(title,fontsize=20)
            g.set_xticklabels(g.get_xticklabels(), fontsize=20)#,rotation=45)
            g.set_yscale("log")
            #g.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0f}'))
            g.yaxis.set_minor_formatter(mtick.NullFormatter())
            g.set_yticks([20, 40, 80])
            fmt = '%.0f%%' 
            yticks = mtick.FormatStrFormatter(fmt)
            g.yaxis.set_major_formatter(yticks)
            #g.yaxis.set_major_locator(mtick.MultipleLocator(20))

    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    #fig.savefig('pdmd-cuda.pdf', bbox_inches='tight')
    fig.savefig('pdmd-cuda.png', bbox_inches='tight')
    #plt.show()

def main_tf(path, devices, options, version, prefix):
    path_cuda = [[[[ (path + dev + pref + opt + ver + '_gpukernsum.csv') 
        for ver in version ] 
        for opt in options ]
        for pref in prefix ]
        for dev in devices ]
    path_cuda = flattenL(path_cuda)
    path_list = [item for item in path_cuda if pathlib.Path(item).exists() is True]
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    tot = len(path_list)
    nrows,ncols = factors(tot) 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5,nrows*4))

    for i in range(nrows):
        for j in range(ncols):
            f = path_list[i*ncols+j]
            title = titlegen(f)
            df = pd.read_csv(f)
            cdf = tf_df_manip(df) 
            g = sns.barplot(data=cdf, ax=axes[i,j], y='Time(%)', x='Name')
            #g = sns.heatmap(data=cdf, ax=axes[i,j], y='Time(%)', x='Name', annot=True, cbar=True, cmap='RdBu_r')
            g.set(xlabel=None)
            g.set(ylabel=None)
            g.set_title(title,fontsize=20)
            g.set_xticklabels(g.get_xticklabels(), fontsize=20)#, rotation=45)
            g.set_yscale("log")
            #g.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0f}'))
            g.yaxis.set_minor_formatter(mtick.NullFormatter())
            g.set_yticks([10, 50, 100])
            fmt = '%.0f%%' 
            yticks = mtick.FormatStrFormatter(fmt)
            g.yaxis.set_major_formatter(yticks)
            #g.yaxis.set_major_locator(mtick.MultipleLocator(20))

    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    #fig.savefig('pdmd-tf.pdf', bbox_inches='tight')
    fig.savefig('pdmd-tf.png', bbox_inches='tight')
    #plt.show()

def cuda_df_manip(df):
    cdf = df[['Time(%)', 'Name']]
    cdf.loc[cdf['Name'].str.contains(r'Memcpy', case=False), 'Name'] = "xfer"
    cdf.loc[cdf['Name'].str.contains(r'Alloc|Memset|Free|Mem', case=False), 'Name'] = "mem"
    cdf.loc[cdf['Name'].str.contains(r'Event', case=False), 'Name'] = "event"
    cdf.loc[cdf['Name'].str.contains(r'Stream', case=False), 'Name'] = "stream"
    cdf.loc[cdf['Name'].str.contains(r'Module', case=False), 'Name'] = "mod"
    cdf.loc[cdf['Name'].str.contains(r'Launch', case=False), 'Name'] = "exec"
    cdf.loc[cdf['Name'].str.contains(r'Init|Ctx|Device', case=False), 'Name'] = "dev"
    cdf = cdf[cdf['Time(%)'] >= 1] 
    cdf2 = cdf.groupby(['Name'],as_index=False).agg({'Time(%)': 'sum'}).sort_values('Time(%)', ascending=False)
    return cdf2

def tf_df_manip(df):
    cdf = df[['Time(%)', 'Name']]
    cdf.loc[cdf['Name'].str.contains(r'fp16', case=False), 'Name'] = "fp16"
    cdf.loc[cdf['Name'].str.contains(r'gemm|scal_kernel|cutlass', case=False), 'Name'] = "gemm"
    cdf.loc[cdf['Name'].str.contains(r'redzone|Cleanup', case=False), 'Name'] = "mem"
    cdf.loc[cdf['Name'].str.contains(r'fusion', case=False), 'Name'] = "fusion"
    cdf.loc[cdf['Name'].str.contains(r'broadcast|scatter', case=False), 'Name'] = "bcast"
    cdf.loc[cdf['Name'].str.contains(r'reduce', case=False), 'Name'] = "reduce"
    cdf.loc[cdf['Name'].str.contains(r'Eigen', case=False), 'Name'] = "eigen"
    cdf.loc[cdf['Name'].str.contains(r'elementwise', case=False), 'Name'] = "elem"
    cdf.loc[cdf['Name'].str.contains(r'convert|cat|chunk|split|kernel|sqrt|rng|comparison|add|Launch|Abs|KernelLaunch|slice', case=False), 'Name'] = "other"
    cdf = cdf[cdf['Time(%)'] >= 1] 
    cdf2 = cdf.groupby(['Name'],as_index=False).agg({'Time(%)': 'sum'}).sort_values('Time(%)', ascending=False)
    return cdf2

def factors(number):
    x = int(math.sqrt(number))
    while(number % x):
        x += 1
    return [int(number/x), x]

def titlegen(f):
    title = f.rsplit('_',1)[0].rsplit('_',1)[1]
    if (title == 'TFDataGen'):
        title = 'TF.data'
    if (title == 'Baseline'):
        title = 'TF.numpy'
    if (title == 'TFDataOptMGPU'):
        title = 'TF.dist'
    if (title == 'TFDataOptMGPUAcc'):
        title = 'TF.dist[upd]'

    st = '_ng'
    en = '_nc'
    x = f[f.find(st)+len(st):f.rfind(en)]
    if (f.find('tonga') != -1):
        title += ',' + 'V100(' + x + ')'
    if (f.find('a100') != -1):
        title += ',' + 'A100(' + x + ')'
    if (f.find('mp1') != -1):
        title += ',' + 'MP'
    return title

def flattenL(L):
    islist = map(lambda x: isinstance(x, list), L)
    if (all(islist)):
        L1 = functools.reduce(operator.concat, L) 
        return flattenL(L1)
    else:
        return L

if (__name__ == "__main__"):
    sys.exit(main())
