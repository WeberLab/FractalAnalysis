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
#   4. Grey matter (gm) mask of the bold file (.nii.gz file)
#   5. White matter (wm) mask of the bold file (.nii.gz file)


# outputs: 
#   1. PSD plot showing the full frequency and selected frequency (.png image)
#   2. Whole-brain 3D heat map of H values (.nii.gz file)
#   3. Print statements of the mean of the gm and wm H values 
#   4. Print statement of the Cohen's D effect size of H between gm and wm 


##########################################################################################

##########################################################################################
# system arguments
##########################################################################################

bold = sys.argv[1]
min_freq =sys.argv[2]
max_freq = sys.argv[3]
gm = sys.argv[4]
wm = sys.argv[5]


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
sel_freq=full_freq[min_ind:max_ind]
sel_avg=avg[min_ind:max_ind]

##########################################################################################
# plot selected frequency PSD
##########################################################################################

x_sel = np.log10(sel_freq)[1:-1].reshape((-1, 1))
y_sel = np.log10(sel_avg)[1:-1]
ax.plot(x_sel,y_sel)

##########################################################################################
# name and save PSD png output
##########################################################################################
base = os.path.basename(bold)
png_name = "PSD" + ".png"
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
        freq=w[0][min_ind:max_ind]
        power=w[1][min_ind:max_ind]
        x = np.log10(freq)[1:].reshape((-1, 1))
        y = np.log10(power)[1:]
        model = LinearRegression().fit(x, y)
        negbeta = model.coef_
        beta = negbeta*-1
        H = (beta + 1)/2
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

#name = bold.rstrip("nii.gz") 
base_welch = base_welch[:,:,1:slice_num+1]
ni_img = nib.nifti1.Nifti1Pair(base_welch, None, n1_header)
nib.save(ni_img, f'Hurst.nii.gz')

##########################################################################################
# load hurst file
##########################################################################################

slice_img = nib.load('Hurst.nii.gz', mmap=False)
hurst = slice_img.get_fdata()
[N1, N2, N3] = hurst.shape
row = np.arange(N1)
column = np.arange(N2)
slices = np.arange(N3)

##########################################################################################
# load gm mask
##########################################################################################

slice_img = nib.load(gm, mmap=False)
gm = slice_img.get_fdata()
[N1, N2, N3] = gm.shape
row = np.arange(N1)
column = np.arange(N2)
slices = np.arange(N3)

##########################################################################################
# get all H values in gm 
##########################################################################################

list_gm = []
for i in slices: 
    for j in column: 
        for k in row: 
            if gm[k,j,i] == 0: 
                pass
            else:
                list_gm.append([hurst[k,j,i]])

##########################################################################################
# calculate gm H mean and std
##########################################################################################

arr = np.array(list_gm)
gm_mean = arr.mean()
gm_std = arr.std()
      

##########################################################################################
# load wm mask
##########################################################################################

slice_img = nib.load(wm, mmap=False)
wm = slice_img.get_fdata()
[N1, N2, N3] = wm.shape
row = np.arange(N1)
column = np.arange(N2)
slices = np.arange(N3)

##########################################################################################
# get all H values in wm 
##########################################################################################

list_wm = []
for i in slices: 
    for j in column: 
        for k in row: 
            if wm[k,j,i] == 0: 
                pass
            else:
                list_wm.append([hurst[k,j,i]])

##########################################################################################
# calculate gm H mean and std
##########################################################################################

arr = np.array(list_wm)
wm_mean = arr.mean()
wm_std = arr.std()                

##########################################################################################
# calculate gm and wm effect size
##########################################################################################

n1, n2 = len(list_gm), len(list_wm)
s1, s2 = var(list_gm, ddof=1), var(list_wm, ddof=1)
s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
u1, u2 = mean(list_gm), mean(list_wm)
cohens_d = (u1 - u2) / s

##########################################################################################
# print statements
##########################################################################################

 
print('GM Mean:', gm_mean)  
print('WM Mean:', wm_mean)      
print('Cohens D:', cohens_d) 
