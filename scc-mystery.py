import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

if len(sys.argv) == 1:
    sys.exit("Expecting: python file.py path/to/tot-times-csv-file path/to/2-6times-csv-file path/to/3-6times-csv-file path/to/mod-csv-file")
tdata = np.genfromtxt(sys.argv[1], delimiter=',')
pdata = np.genfromtxt(sys.argv[2], delimiter=',')
ppdata = np.genfromtxt(sys.argv[3], delimiter=',')
mdata = np.genfromtxt(sys.argv[4], delimiter=',')

tdf = pd.DataFrame(data=tdata, columns=["Task1", "Task2", "Task3", "Task4", "Task5", "Task6"])
tdf26 = pd.DataFrame(data=pdata, columns=["Task2", "Task3", "Task4", "Task5", "Task6"])
tdf36 = pd.DataFrame(data=ppdata, columns=["Task3", "Task4", "Task5", "Task6"])
mdf = pd.DataFrame(data=mdata, columns=["Task1", "Task2", "Task3", "Task4", "Task5", "Task6"])

xlabels = ["FAU", "UIUC", "MIT", "Peking", "Tsinghua", "UCSD", "Texas A&M", "ShanghaiTech", "TACC", "SUSTech", "GATech", "Wake Forest", "Warsaw", "Shanghai Jiao Tong", "Clemson", "Northeastern", "NCState", "ETH Zurich", "NTU", "Singapore"]
sns.set_theme()
sns.set(font_scale=2)

# times
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax = sns.violinplot(data=tdf, xticklabels=xlabels, scale='width', cut=0)
ax.set_title("Reported time (in seconds)", fontsize=20, fontweight='bold')
fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
fig.savefig('scc-mystery-times.pdf', bbox_inches='tight')
plt.show()

# task 2-6 times
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax = sns.violinplot(data=tdf26, xticklabels=xlabels, scale='width', cut=0)
ax.set_title("Reported time (in seconds)", fontsize=20, fontweight='bold')
fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
fig.savefig('scc-mystery-times-part26.pdf', bbox_inches='tight')
plt.show()

# task 3-6 times
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax = sns.violinplot(data=tdf36, xticklabels=xlabels, scale='width', cut=0)
ax.set_title("Reported time (in seconds)", fontsize=20, fontweight='bold')
fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
fig.savefig('scc-mystery-times-part36.pdf', bbox_inches='tight')
plt.show()

# modularities
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax = sns.violinplot(data=mdf, xticklabels=xlabels, scale='width', cut=0)
ax.set_title("Modularities (0-1)", fontsize=20, fontweight='bold')
fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
fig.savefig('scc-mystery-mods.pdf', bbox_inches='tight')
plt.show()
