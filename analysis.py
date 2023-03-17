import LIA_3
import matplotlib.pyplot as plt
import My_Plots
import pickle as pkl
import pathlib
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

experiment_save = pathlib.Path(r"C:\Users\josia\Dropbox (GaTech)\MNBL\Bead Diagnostics\experiments\ex. 75) serum repeats\ex. 75 LIA part 4")


experiment_folders = [r"C:\Users\josia\Dropbox (GaTech)\MNBL\Bead Diagnostics\experiments\ex. 75) serum repeats\ex. 75 LIA part 4",
                      ]

file_names = ['experiment.pkl',
              ]

# load in experiments
experiments = []
for i, experiment_folder in enumerate(experiment_folders):
    exp  = pathlib.Path(experiment_folder)
    file = file_names[i]
    with open(exp / file, 'rb') as f:
        experiment = pkl.load(f)
    experiments.append(experiment)

# load in frequency of recorded peaks
experiment = experiments[0]
demods_freqs = {}
runs = list(experiment.runs.keys())
run = experiment.runs[runs[0]] # should rename this to trial
for key, val in run.demods.items():
    demods_freqs[key] = int(val.frequency)

# concatenate all experiment dataframes together into one
frames = []
for i, experiment in enumerate(experiments):
    frames.append(experiment.dataframe_filtered)
frame = pd.concat(frames)

# get relevant data (z and phi at 4th demodulator)
attributes = ['z_waveform_height', 'phi_waveform_height']
indices = list(itertools.product(demods_freqs.keys(), attributes))
indices = list(itertools.product(['4'], attributes))
indices_run = indices + [('run', 'run')]

# map trial names to molarity / serum type
def name_to_molarity(row): # this is setting x200's as x20's... quite naturally
    name = row[('run', 'run')]
    if re.search(r'C.', name):
        mol = 'Covid+'
    if re.search(r'H.', name):
        mol = 'Healthy'
    if re.search(r'PE', name):
        mol = 'No Metal'
    if re.search(r'HRP', name):
        mol = 'High Metal'
    return mol

frame['mol'] = frame.apply(name_to_molarity, axis=1)

# create training frame
control_mol = 'No Metal'
metal_mol = 'High Metal'
train_frame = frame[frame['mol'].isin([control_mol, metal_mol])]
# set classes
def name_to_class(row):
    name = row[('run', 'run')]
    if re.search(r'PE', name):
        class_num = 0
    if re.search(r'HRP', name):
        class_num = 1
    return class_num
train_frame['class'] = train_frame.apply(name_to_class, axis=1)

# split into X and y
X = np.array(train_frame[indices])
y = np.array(train_frame['class'])

# create scaler and classifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

################################################################################
#
# # apply classifier to all data
X = np.array(frame[indices]) # isolate X data
X = scaler.transform(X) # scale
frame['transformed'] = clf.transform(X) # transform and save
frame['predicted'] = clf.predict(X)

means = frame.groupby(['mol', ('run', 'run')]).mean()[['transformed', 'predicted']]
means = means.reset_index()

mean_means = means.groupby('mol').mean()
#
# # now let's plot
# # i want error bars
# def get_error_bounds(x):
#     mean = np.mean(x)
#     std = np.std(x)
#     return (mean - 1*std, mean + 1*std)
#
# fig, ax = plt.subplots()
# sns.stripplot(ax=ax,
#               data=means,
#               y='transformed',
#               x='mol')
# sns.pointplot(ax=ax,
#               data=means,
#               y='transformed',
#               x='mol',
#               estimator=np.mean,
#               errorbar=get_error_bounds,
#               markers='x',
#               color='black',
#               join=False,
#               n_boot=3)
# ax.set(title='LDA Means', xlabel='Conc. (molarity)', ylabel='"transformed"')
# fig.tight_layout()
# filename = experiment_save / 'transformed_titration.png'
# fig.savefig(fname = filename, format='png', dpi=100)
#
# fig, ax = plt.subplots()
# sns.stripplot(ax=ax,
#               data=means,
#               y='predicted',
#               x='mol',)
# sns.pointplot(ax=ax,
#               data=means,
#               y='predicted',
#               x='mol',
#               estimator=np.mean,
#               errorbar=get_error_bounds,
#               markers='x',
#               color='black',
#               join=False,
#               n_boot=3)
# ax.set(title='LDA Means', xlabel='Conc. (molarity)', ylabel='"predicted %"')
# fig.tight_layout()
# filename = experiment_save / 'predicted_titration.png'
# fig.savefig(fname = filename, format='png', dpi=100)