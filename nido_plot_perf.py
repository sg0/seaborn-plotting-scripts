import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import sys

def get_loops(lines):
    loops = []
    for i in range(len(lines)):
        if "Final Q" in lines[i]:
            target = lines[i-1]
            loops.append(int(target.split(" ")[0]))
    return loops

def get_total_loops(lines):
    for i in range(len(lines)):
        if "Total Loops:" in lines[i]:
            target = lines[i]
            return int(target.split(" ")[2])

def get_phases(lines):
    phases = []
    for i in range(len(lines)):
        if "PHASE" in lines[i]:
            target = lines[i]
            x=re.findall('[0-9]+',target)[0]
            phases.append(int(x))

    return phases

def get_total_phases(lines):
    return get_phases(lines)[-1]

def get_modularities(lines):
    q0 = 0
    for i in range(len(lines)):
        if "PHASE #0:" in lines[i]:
            target = lines[i]
            i=i+3
            q0=float(lines[i].split(" ")[1])
            break
    q0 = [q0]
    qs = []
    for i in range(len(lines)):
        if "Final Q" in lines[i]:
            qs.append(float(lines[i].split(" ")[2]))
    qs = q0+qs
    return qs

def get_modularity_trace(lines, phase):
    q=[]
    for i in range(len(lines)):
        if "PHASE" in lines[i]:
            target = lines[i]
            x=int(re.findall('[0-9]+',target)[0])
            if(x==phase):
                i=i+3
                for j in range(i, len(lines)):
                    words = lines[i].split(" ")
                    if(words[0]!="Final"):
                        qs.append(float(words[1]))
                    else:
                        return qs

def get_time_per_loops(lines):
    t = 0
    for i in range(len(lines)):
        if "Time per loop" in lines[i]:
            t = float(lines[i].split(" ")[3])
    return t

def get_total_time(lines):
    total_time = 0
    for i in range(len(lines)):
        if "Total time elapse:" in lines[i]:
            total_time = float(lines[i].split(" ")[3])
    return total_time

def get_time_per_iterations(lines):
    tot = get_total_time(lines)
    loops = get_total_loops(lines)
    return tot/loops


def get_results(dir_name):
    files=glob.glob(dir_name+"/log*.txt")
    print(files)
    #n_baches = []
    key_vals = dict([])
    for f in files:
        n=re.findall('[0-9]+',f)[-1]
        n=int(n)
        with open(f, "r") as f:
            lines = f.readlines()
            t = get_time_per_iterations(lines)
            key_vals[n]=t
            print(n)
    key_vals = dict(sorted(key_vals.items(), key=lambda item: item[0]))
    n_batches = []
    time = []
    for key, val in key_vals.items():
        n_batches.append(key)
        time.append(val)
    return n_batches, time
 
markers=["C0o-", "C1o-", "C2o-", "C3o-", "C4o-", "C5o-", "C6o-", "C7o-", "C8o-", "C9o-"]
gpus=[1,2,4,8]
graphs=["com-orkut", "com-friendster", "mycielski20", "twitter7"]
graphs_name = ["com-orkut", "com-friendster", "mycielskian20", "twitter7"]
nrows = 1
ncols = 4
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(32,6))
handles=[]
labels=[]
for i, axi in enumerate(ax.flat):
    # i runs from 0 to (nrows*ncols-1)
    # axi is equivalent with ax[rowid][colid]
    # get indices of row/column
    b=[]
    t=[]
    for k, g in enumerate(gpus):
        dir_name=graphs[i]+"/"+str(g).zfill(2)+"gpu"
        x,y = get_results(dir_name)
        b.append(x)
        t.append(y)
    b=np.transpose(b)
    t=np.transpose(t)
    axi.set_xlabel('#GPU', fontsize=32)
    axi.set_ylabel('Time per Iteration', fontsize=32)
    axi.xaxis.set_tick_params(labelsize=24)
    axi.yaxis.set_tick_params(labelsize=24)

    for j in range(len(t)):
        if(b[j][0] == 1):
            axi.plot(gpus,t[j], markers[j], markersize=12, linewidth=5,label=str(b[j][0])+" batch")
        else:
            axi.plot(gpus,t[j], markers[j], markersize=12, linewidth=5, label=str(b[j][0])+" batches")    
    # write row/col indices as axes' title for identification
    axi.set_title(graphs_name[i], fontsize=32)
    if(i==0):
        handles, labels = axi.get_legend_handles_labels()
    #    axi.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4, fontsize=26)
fig.legend(handles, labels, loc='lower left', ncol=4, fontsize=26)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
#plt.legend()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4, fontsize=26)
#plt.figlegend( lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
plt.tight_layout()
plt.savefig("time_per_iteration_hard_1.svg")
plt.savefig("time_per_iteration_hard_1.pdf")
plt.show()
'''
#ax1.xticks(fontsize=28)
#ax1.yticks(fontsize=28)
ax1.set_xlabel('Phases', fontsize=32)
ax1.set_ylabel('Modularity', fontsize=32)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_tick_params(labelsize=24)

files=glob.glob("./log*.txt")
#n_baches = []
key_vals = dict([])
for f in files:
    n=re.findall('[0-9]+',f)[0]
    n=int(n)
    print(n)
    with open(f, "r") as f:
        lines = f.readlines()
        qs = get_modularities(lines)
        key_vals[n]=qs

key_vals = dict(sorted(key_vals.items(), key=lambda item: item[0]))

i = 0
for key, val in key_vals.items():
    phases=range(0,len(val))
    titles=[]
    if(key==1):
        titles=str(key)+" batch"
    else:
        titles=str(key)+" batches"

    ax1.plot(phases, val, markers[i], markersize=10, linewidth=4, label=titles)
    i=i+1
ax1.legend(fontsize=24)
#fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
#left, bottom, width, height = [0.55, 0.55, 0.3, 0.225]
#ax2 = fig.add_axes([left, bottom, width, height])

#plt.plot(range(10), color='red')
#i=0
#ax2.set_xlim([0.,7])
#ax2.set_ylim([0.6,0.7])
#ax2.xaxis.set_tick_params(labelsize=12)
#ax2.yaxis.set_tick_params(labelsize=12)

#for key, val in key_vals.items():
#    phases=range(0,len(val))

#    ax2.plot(phases, val[0:], markers[i], markersize=6, linewidth=2.4)
#    i=i+1
#ax2.plot(, color='green')
#plt.savefig("com-orkut_01gpu.svg")
#plt.savefig("com-orkut_01gpu.pdf")
#plt.show()
'''
