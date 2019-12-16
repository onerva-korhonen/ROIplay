#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:23:26 2019

@author: onerva

A bunch of functions for handling voxel and ROI masks and time series and for all sort of playing with ROIs.
"""
import numpy as np
import nibabel as nib

def pickROITs(dataPath, ROIMaskPath, returnVoxels=False, grayMaskPath=None):
    """
    Starting from full 4D fMRI data matrix, picks the time series of ROIs based
    on a mask. The ROI time series is defined as the average of the voxel time
    series of the ROI.
    
    Parameters:
    -----------
    dataPath: str, path to which the data has been saved in .nii form
    ROIMaskPath: str, path to which the ROI mask has been saved in .nii form. In the
                 mask, the value of each voxel should show to which ROI the voxel belongs
                 (voxels outside of the brain should have value 0).
    returnVoxels: boolean; if True, time series of each voxel are returned instead
                  of ROI time series
    grayMaskPath: str, path to which the gray matter mask has been saved in .nii form. If
                  grayMaskPath is given, the ROI mask will be multiplied by the gray matter mask
                  before extracting ROI time series.
                  
    Returns:
    --------
    ROITs: a 2D (nROIs x t) matrix of ROI time series
    ROIMaps: list of ROISize x 3 np.arrays, coordinates (in voxels) of voxels belonging to each ROI
    """
    data = readNii(dataPath)
    ROIMask = readNii(ROIMaskPath)
    if not grayMaskPath == None:
        grayMask = readNii(grayMaskPath)
        ROIMask = ROIMask*grayMask
    if returnVoxels:
        x,y,z = np.where(ROIMask>0)
        ROIMap = np.transpose(np.stack((x,y,z)))
        ROITs = data[(x,y,z)]
        ROIMaps = [ROIMap]
    else:
        nROIs = len(np.unique(ROIMask)) - 1
        nT = data.shape[-1]
        ROITs = np.zeros((nROIs,nT))
        ROIMaps = []
        for i in range(1,nROIs+1):
            x,y,z = np.where(ROIMask==i)
            ROIMap = np.transpose(np.stack((x,y,z)))
            ROIMaps.append(ROIMap)
            ROIVoxelTs = data[(x,y,z)]
            ROITs[i-1,:] = np.mean(ROIVoxelTs,axis=0)
    return ROITs, ROIMaps
            
def readNii(path):
    """
    Reads the given NIFTI file using nibabel.
    
    Parameters:
    -----------
    path: str, path to the file to be read
    
    Returns:
    --------
    data: np.array, the data of the NIFTI file
    """
    img = nib.load(path)
    data = img.get_fdata()
    return data
    
