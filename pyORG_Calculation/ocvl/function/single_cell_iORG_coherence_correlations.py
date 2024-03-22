# Creating a file to calculate correlation coefficient between individual cell signals acquired under different
# imaging conditions


# Importing all the things...
import os
from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import Tk, filedialog, ttk, HORIZONTAL

import numpy as np
import pandas as pd
import matplotlib
import re
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from ocvl.function.analysis.cell_profile_extraction import extract_profiles, norm_profiles, standardize_profiles
from ocvl.function.analysis.iORG_profile_analyses import signal_power_iORG
from ocvl.function.utility.generic import PipeStages
from ocvl.function.utility.meao import MEAODataset
from ocvl.function.utility.resources import save_video
from ocvl.function.utility.temporal_signal_utils import reconstruct_profiles
from datetime import datetime, date, time, timezone

# User selects dataset from file
root = Tk()
fName_1 = filedialog.askopenfilename(title="Select the Stimulus_cell_power_iORG csv for the first condition .", parent=root)
print('selected path: ' + fName_1)

if not fName_1:
    quit()


fName_2 = filedialog.askopenfilename(title="Select the Stimulus_cell_power_iORG csv for the second condition .", parent=root)
print('selected path: ' + fName_2)

if not fName_2:
    quit()

# Reading in datasets as dataframe
cell_pwr_iORG_1 = pd.read_csv(fName_1)
cell_pwr_iORG_2 = pd.read_csv(fName_2)

# calculating correlation coefficient
testCorrW = cell_pwr_iORG_1.corrwith(cell_pwr_iORG_2, axis=0, drop=False, method='pearson')

# plotting the signals from the highest and lowest correlation coeffs
print('Max Pearson correlation: %.5f' % testCorrW.max())
print('Median Pearson correlation: %.5f' % testCorrW.median())
print('Min Pearson correlation: %.5f' % testCorrW.min())

# calculating median manually because something is weird with finding that index
sort_testCorrW = testCorrW.sort_values(axis=0)
length_testCorrW = testCorrW.size
calc_med = round(length_testCorrW/2)
sort_testCorrW.iloc[calc_med]
med_loc = testCorrW.index.get_loc(sort_testCorrW[sort_testCorrW == sort_testCorrW.iloc[calc_med]].index[0])

min_loc = int(testCorrW.idxmin())
max_loc = int(testCorrW.idxmax())

plt.figure(1)
plt.plot(cell_pwr_iORG_1.iloc[min_loc])
plt.plot(cell_pwr_iORG_2.iloc[min_loc])
plt.title('Min Correlation')

plt.figure(2)
plt.plot(cell_pwr_iORG_1.iloc[max_loc])
plt.plot(cell_pwr_iORG_2.iloc[max_loc])
plt.title('Max Correlation')

plt.show()

print(' ')
cell_pwr_iORG_2 = pd.read_csv(fName_2)
print('fiheoiusfhoighoge')