# FractalAnalysis
The scripts in this repository can be used to visualize the power spectral densities of fMRI BOLD signals and to calculate the Hurst exponent using Welch's method. 

## Files Summary

The following are the scrips written by Olivia Campbell at BCCHR from May 2020 - May 2022 and details on how to run them. 

## Files

### 1. PSD.ipynb
This Jupyter notebook can be used to select the frequency over which power law scaling exists. It calculates the power spectra for every voxel of the fMRI BOLD image and averages them. It also allows you to change the frequency range that you want to visualize. It displays the average power spectral density (PSD) for the full frequency range and then the average PSD for the selected frequncy range. On the PSDs, the slope, or Beta, is displayed. The y-axis is log(power) and the x-axis is log(freq). There is a secondary x-axis above in freq in order to aid in frequency selection. 

### 2. PSD_Welch.py
After selected the frequency range over which power law scaling holds, this script can be used to calculate the Hurst exponent. It takes five inputs: the fMRI file, minimum frequency, maximum frequency, a grey matter mask, and white matter mask. It outputs a .png image of the PSD showing the full frequency range and selected frequency range, a 3D .nii.gz file of the Hurst exponent calculated for every voxel, and the effect size between H in the grey matter and the white matter. 
