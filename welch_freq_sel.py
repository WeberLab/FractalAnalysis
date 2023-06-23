#!/usr/bin/env python3

import multiprocessing
from joblib import Parallel, delayed
import os 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import math
from scipy.signal import welch as welch
from sklearn.linear_model import LinearRegression
import nibabel as nib
import sys
from numpy import var
from math import sqrt
from numpy import mean

##########################################################################################
# this script conducts whole-brain fractal analysis using Welch's method within the provided 
# frequency range

# inputs: 
# 	1. Bold file (whole brain .nii.gz file)
#   2. Minumim frequency of the frequency range of interest (e.g. 0.01)
#   3. Maximum frequency of the frequency range of interest (e.g. 0.3)
#   4. Output folder


# outputs: 
#   1. PSD plot showing the full frequency and selected frequency (.png image)
#   2. Whole-brain 3D heat map of H values (.nii.gz file)


##########################################################################################

##########################################################################################
# system arguments
##########################################################################################

bold = sys.argv[1]
min_freq =sys.argv[2]
max_freq = sys.argv[3]
outfol = sys.argv[4]


##########################################################################################
# load bold image
##########################################################################################

slice_img = nib.load(bold, mmap=False)
n1_header = slice_img.header
TR = n1_header.get_zooms()
TR = np.asarray(TR)[3]
slice_array = slice_img.get_fdata()
[N1, N2, slice_num, N3] = slice_array.shape
row = np.arange(N1)
column = np.arange(N2)

##########################################################################################
# get full frequency vector
##########################################################################################

samp_slice = slice_array[:,:,slice_num-1,:]
vox = (samp_slice[N1-1,N2-1])
nperseg = math.floor(len(vox)/8)
noverlap = math.floor(nperseg/2)
w = welch(vox, fs = 1/TR, nperseg = nperseg, noverlap = noverlap)
np.seterr(divide = 'ignore')
full_freq=w[0]

##########################################################################################
# calculate power for full range of frequencies
##########################################################################################

def PSD_voxel(i,j):
    '''calculates the power spectrum of one voxel using Welchs method'''
    global TR 
    global full_freq
    voxel = (slice_sq[i,j])
    if np.mean(voxel) == 0:
        return np.zeros(len(full_freq))
    else:
        nperseg = math.floor(len(voxel)/8)
        noverlap = math.floor(nperseg/2)
        w = welch(voxel, fs = 1/TR, nperseg = nperseg, noverlap = noverlap)
        np.seterr(divide = 'ignore')
        power=w[1]
        return power

num_cores = multiprocessing.cpu_count()
tot = np.zeros(len(full_freq))
count = 0
for x in range(slice_num):
    slice_sq = slice_array[:,:,x,:]
    out = Parallel(n_jobs=num_cores)(delayed(PSD_voxel)(i,j) for i in row for j in column)
    out = np.array(out, dtype=object) 
    for x in np.arange(len(out)):
        tot = tot + out[x]
        if np.mean(out[x]) == 0:
            pass
        else:
            count = count+1
            tot = tot + out[x]

##########################################################################################
# calculate average power spectra in full frequency range
##########################################################################################

avg=tot/count
avg = avg.astype(float)

##########################################################################################
# plot full frequency PSD
##########################################################################################

x_full = np.log10(full_freq)[1:-1].reshape((-1, 1))
y_full = np.log10(avg)[1:-1]
fig, ax = plt.subplots()
fig.canvas.draw()
ax.plot(x_full,y_full)

##########################################################################################
# narrow full frequency range to selected range 
##########################################################################################

min_freq = float(min_freq)
max_freq = float(max_freq)
min_ind = np.where(full_freq<min_freq)[0][-1] + 1
max_ind = np.where(full_freq>max_freq)[0][0]
sel_freq=full_freq[min_ind:max_ind+1]
sel_avg=avg[min_ind:max_ind+1]

print("You selected a frequency range between ", min_freq, " and ", max_freq)
print("The closest I can do is:")
print("Min Frequency = ", full_freq[min_ind], " and Max Frequency = ", full_freq[max_ind])

##########################################################################################
# plot selected frequency PSD
##########################################################################################

x_sel = np.log10(sel_freq)[:].reshape((-1, 1))
y_sel = np.log10(sel_avg)[:]

model = LinearRegression().fit(x_sel, y_sel)
x_new=x_sel
y_new=model.predict(x_sel)
plt.plot(x_new, y_new)

##########################################################################################
# name and save PSD png output
##########################################################################################
sepvar = '.'
base = os.path.basename(bold)
base = base.split(sepvar,1)[0]
png_name = outfol + "/" + base + "_PSD_" + str(round(full_freq[min_ind],3)) + "_to_" + str(round(full_freq[max_ind],3))+ ".png"
plt.savefig(png_name)


##########################################################################################
# calculate H for selected frequency 
##########################################################################################

def welch_voxel(i,j):
    global TR 
    voxel = (slice_sq[i,j])
    if np.mean(voxel) == 0:
        return None
    else:
        nperseg = math.floor(len(voxel)/8)
        noverlap = math.floor(nperseg/2)
        w = welch(voxel, fs = 1/TR, nperseg = nperseg, noverlap = noverlap)
        np.seterr(divide = 'ignore')
        freq=w[0][min_ind:max_ind+1]
        power=w[1][min_ind:max_ind+1]
        x = np.log10(freq)[:].reshape((-1, 1))
        y = np.log10(power)[:]
        model = LinearRegression().fit(x, y)
        negbeta = model.coef_
        beta = negbeta*-1
        H = (beta + 1)/2
#        if H < 0:
#            H = 0
#        elif H > 1:
#            H = 1
        return H

num_cores = multiprocessing.cpu_count()
base_welch = np.zeros((N1,N2))
for x in range(slice_num):
    slice_sq = slice_array[:,:,x,:]
    output_welch = Parallel(n_jobs=num_cores)(delayed(welch_voxel)(i,j) for i in row for j in column)
    output_welch = np.array(output_welch, dtype=object) #convert output into numpy array
    output_welch = output_welch.astype(np.float64) #convert type object list elements to type float
    Hurst_matrix_welch = output_welch.reshape(N1,N2)
    base_welch = np.dstack((base_welch,Hurst_matrix_welch))


##########################################################################################
# output heat map nifti file for whole brain H 
##########################################################################################

base_welch = base_welch[:,:,1:slice_num+1]

base_welch[np.isnan(base_welch)] = 0
ni_img = nib.nifti1.Nifti1Pair(base_welch, None, n1_header)

hurst_name = outfol + "/" + base + "_WelchHurst_" + str(round(full_freq[min_ind],3)) + "_to_" + str(round(full_freq[max_ind],3)) + ".nii.gz"
nib.save(ni_img, hurst_name)

