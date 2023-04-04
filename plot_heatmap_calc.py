import functools
import operator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import matplotlib.ticker as mtick

def main():

    search_key = 'Reducer:'
    path = './const_random_'
    prefix = ('30000', '50')
    methods = ('NearestNeighbors', 'SVM', 'DecisionTree', 'RandomForest', 'AdaBoost', 'NeuralNet', 'NaiveBayes', 'Mean')

    score_diff(path, search_key, prefix, methods)

def score_diff(path, search_key, prefix, methods):
    path_inp = path + '_'.join(prefix) + '.txt'
    print('Reading file: ', path_inp)

    lidx = []
    lname = []
    with open(path_inp) as inpfile:
        for num, line in enumerate(inpfile, 1):
            if search_key in line:
                lname.append(line.rstrip())
                lidx.append(num)

    df_tp     = pd.DataFrame()
    df_fp     = pd.DataFrame()
    df_tn     = pd.DataFrame()
    ll_tp_avg = []
    ll_fp_avg = []
    ll_tn_avg = []
    
    for idx, x in enumerate(lidx):
        df = pd.read_csv(path_inp, skiprows=x+1, nrows=len(methods)-1, header=None, delimiter=r"\s+")
        #print(df)
        idx_name = lname[idx].split(':')[1]
        df_tp.insert(idx, idx_name, [round(100*(abs(x-y)/(x+y)/2),2) for x, y in zip(df[1], df[4])])
        df_fp.insert(idx, idx_name, [round(100*(abs(x-y)/(x+y)/2),2) for x, y in zip(df[2], df[5])])
        df_tn.insert(idx, idx_name, [round(100*(abs(x-y)/(x+y)/2),2) for x, y in zip(df[3], df[6])])
        
        ll_tp_avg.append(round(df_tp.loc[:, idx_name].mean(),2))
        ll_fp_avg.append(round(df_fp.loc[:, idx_name].mean(),2))
        ll_tn_avg.append(round(df_tn.loc[:, idx_name].mean(),2))

    #print(df_tp)
    #print(df_fp)
    #print(df_tn)
    s_tp = pd.Series(ll_tp_avg, index=df_tp.columns)
    df_tp = df_tp.append(s_tp, ignore_index=True)

    s_fp = pd.Series(ll_fp_avg, index=df_fp.columns)
    df_fp = df_fp.append(s_fp, ignore_index=True)

    s_tn = pd.Series(ll_tn_avg, index=df_tn.columns)
    df_tn = df_tn.append(s_tn, ignore_index=True)

       
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    xticks = list(df_tp.columns)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,5))

    ax = sns.heatmap(data=df_tp, ax=axes[0], xticklabels=xticks, yticklabels=methods, annot=True, fmt="g", cmap='RdBu_r')
    ax.set_xticklabels(xticks, rotation = 20, ha="right")
    ax.set_title('Accuracy %-differences')
    
    ax = sns.heatmap(data=df_fp, ax=axes[1], xticklabels=xticks, yticklabels=[], annot=True, fmt="g", cmap='RdBu_r')
    ax.set_xticklabels(xticks, rotation = 20, ha="right")
    ax.set_title('Precision %-differences')

    ax = sns.heatmap(data=df_tn, ax=axes[2], xticklabels=xticks, yticklabels=[], annot=True, fmt="g", cmap='RdBu_r')
    ax.set_xticklabels(xticks, rotation = 20, ha="right")
    ax.set_title('Recall %-differences')


    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    fig.savefig(path_inp+'.pdf', bbox_inches='tight')
    print('Saved plotfile: ', path_inp+'.pdf')

if (__name__ == "__main__"):
    sys.exit(main())
