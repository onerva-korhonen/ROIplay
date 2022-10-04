#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:23:26 2019

@author: onerva

A bunch of functions for handling voxel and ROI masks and time series and for all sort of playing with ROIs.
"""
import numpy as np
import nibabel as nib

def pickROITs(dataPath, ROIMaskPath, returnVoxels=False, greyMaskPath=None):
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
    greyMaskPath: str, path to which the grey matter mask has been saved in .nii form. If
                  greyMaskPath is given, the ROI mask will be multiplied by the grey matter mask
                  before extracting ROI time series.
                  
    Returns:
    --------
    ROITs: a 2D (nROIs x t) matrix of ROI time series
    ROIMaps: list of ROISize x 3 np.arrays, coordinates (in voxels) of voxels belonging to each ROI
    """
    data = readNii(dataPath)
    ROIMask = readNii(ROIMaskPath)
    if not greyMaskPath == None:
        greyMask = readNii(greyMaskPath)
        ROIMask = ROIMask*greyMask
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

def pickVoxelTs(dataPath, greyMaskPath, saveTs=False, tsSavePath=''):
    """
    Starting from full 4D fMRI data matrix, picks time series of voxels belonging
    to the grey matter, forms a 2D numpy array of the time series, and (if required)
    saves them as .npy
    
    Parameters:
    -----------
    dataPath: str, path to which the data has been saved in .nii form
    greyMaskPath: str, path to which the grey matter mask has been saved in .nii form.
    saveTs: bln, if True, voxel time series are saved as .npy
    tsSavePath: str, path to which to save the voxel time series
    
    Returns:
    --------
    voxelTs: np.array, a 2D (voxels x time) array of voxel time series
    """
    data = readNii(dataPath)
    greyMask = readNii(greyMaskPath)
    x,y,z = np.where(greyMask>0)
    voxelTs = data[(x,y,z)]
    if saveTs:
        np.save(tsSavePath,voxelTs)
    return voxelTs

def atlas2map(ROIMaskPath, greyMaskPath, saveMap=False, mapSavePath=''):
    """
    Based on the 4D ROI atlas matrix in NIFTI format, builds a 2D numpy array 
    containing a voxel -> ROI map
    
    Parameters:
    -----------
    ROIMaskPath: str, path to which the ROI mask has been saved in .nii form. In the
                 mask, the value of each voxel should show to which ROI the voxel belongs
                 (voxels outside of the brain should have value 0).
    greyMaskPath: str, path to which the grey matter mask has been saved in .nii form.
    saveMap: bln, if True, the voxel -> ROI map is saved as .npy
    mapSavePath: str, path to which to save the voxel -> ROI map
    
    Returns:
    --------
    ROIMap: np.array, a voxel -> ROI map. Each row corresponds to a voxel - ROI pair:
            first column contains the index of the ROI and second column the index
            of the voxel belonging to the ROI. Values of the second column are unique,
            while values of the first column repeat as a ROI contains multiple voxels.
    """
    ROIMask = readNii(ROIMaskPath)
    greyMask = readNii(greyMaskPath)
    x,y,z = np.where(greyMask>0)
    ROIIndices = ROIMask[(x,y,z)]
    ROIMap = np.zeros((len(x),2))
    c = 0
    for i in range(1,np.amax(ROIIndices)+1):
        ind = np.where(ROIIndices == i)
        ROIMap[c:c+len(ind),0] = np.ones(len(ind)*i)
        ROIMap[c:c+len(ind),1] = ind
        c += len(ind)
    if saveMap:
        np.save(mapSavePath,ROIMap)
    return ROIMap
            
    
    

# Data IO:
            
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
    
