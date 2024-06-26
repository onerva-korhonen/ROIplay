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

def calculateSpatialConsistency(voxelTsPath, voxel2ROIMapPath, fTransform=False, saveConsistency=False, savePath=''):
    """
    Calculates the spatial consistency (mean Pearson correlation between voxel time series, see
    Korhonen et al. 2017, Net Neurosci) of ROIs.
    
    Parameters:
    -----------
    voxelTsPath: str, path to the voxel time series file created by pickVoxelTs
    voxel2ROIMapPath: str, path to the voxel2ROIMap file created by atlas2map
    fTransform: bln, if True, the Pearson correlations are f-transformed before averaging
    saveConsistency: bln, if True, consistencies are saved as a .npy file
    savePath: str, path to which to save the consistencies
    
    Returns:
    --------
    consistencies: np.array, consistencies of ROIs in the order of ROIs in voxel2ROIMap
    """
    voxelTs = np.load(voxelTsPath)
    voxel2ROIMap = np.load(voxel2ROIMapPath)
    ROIIndices = np.unique(voxel2ROIMap[:,0])
    ROIIndices = ROIIndices[(ROIIndices >= 0)]
    spatialConsistencies = np.zeros(len(ROIIndices))
    for i, ROIIndex in enumerate(ROIIndices):
        voxelIndicesInROI = voxel2ROIMap[:,1][np.where(voxel2ROIMap[:,0]==ROIIndex)]
        voxelIndicesInROI = voxelIndicesInROI.astype('int')
        if len(voxelIndicesInROI) == 1:
            spatialConsistency = 1.
        else:
            voxelTsInROI = voxelTs[voxelIndicesInROI]
            correlations = np.corrcoef(voxelTsInROI) # NOTE: replacing this with calculatePearsonR doesn't speed up the calculation
            correlations = correlations[np.tril_indices(voxelTsInROI.shape[0],k=-1)] # keeping only the lower triangle, diagonal is discarded
            if fTransform:
                correlations = np.arctanh(correlations)
                spatialConsistency = np.tanh(np.mean(correlations))
            else:
                spatialConsistency = np.mean(correlations)
        spatialConsistencies[i] = spatialConsistency
    if saveConsistency:
        np.save(savePath,spatialConsistencies)
    return spatialConsistencies

# Mask operations

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
    assert ROIMask.shape == greyMask.shape,'ROI mask has different shape than the grey mask, check space and resolution'
    x,y,z = np.where(greyMask>0)
    ROIIndices = ROIMask[(x,y,z)]
    ROIMap = np.zeros((len(x),2))
    counter = 0
    for i in range(0,int(np.amax(ROIIndices)+1)):
        ind = np.where(ROIIndices == i)[0]
        if i == 0: # voxels that don't belong to any ROI; ROI index set to -1
            ROIMap[counter:counter+len(ind),0] = np.ones(len(ind))*-1
        else:
            ROIMap[counter:counter+len(ind),0] = np.ones(len(ind))*i
        ROIMap[counter:counter+len(ind),1] = ind
        counter += len(ind)
    if saveMap:
        np.save(mapSavePath,ROIMap)
    return ROIMap

def makeGroupMask(indMaskPaths, saveGroupMask=False, groupMaskSavePath=''):
    """
    Combines individual mask arrays into a group mask that has non-zero values
    only for voxels that have a non-zero value in all individual masks.
    
    Parameters:
    -----------
    indMaskPaths: list of str, paths of the individual masks
    saveGroupMask: bln, if True, the mask is saved as .nii using the affine of
                   the first individual mask in indMaskPaths
    groupMaskDataPath: str, path to which to save the group mask
    
    Returns:
    -------
    groupMask: np.array, the group mask
    """
    for i, indMaskPath in enumerate(indMaskPaths):
        indMask = readNii(indMaskPath)
        if i == 0:
            groupMask = indMask.copy()
        else:
            groupMask = groupMask*indMask
    if saveGroupMask:
        template = nib.load(indMaskPaths[0])
        affine = template.affine # using the affine of the first individual mask, should apply for all individual masks
        header = nib.Nifti1Header() # creating an empty header; this will automatically adapt to the data
        groupMaskImage = nib.Nifti1Image(groupMask,affine,header)
        nib.save(groupMaskImage,groupMaskSavePath)
    return groupMask

def getROICentroids(ROIMaskPath, applyGreyMask=False, greyMaskPath='', saveCentroids=False, centroidSavePath='', calculateDistances=False, saveDistances=False, distanceSavePath=''):
    """
    Calculates the centroid coordinates of each ROI as the mean of coordinates
    of voxels belonging to the ROI. Note that depending on the shape of the ROI,
    the centroid may fall outside of the ROI. Contains also possibility to 
    calculate the distance matrix between ROI centroids.
    
    Parameters:
    -----------
    ROIMaskPath: str, path to which the ROI mask has been saved in .nii form. In the
                 mask, the value of each voxel should show to which ROI the voxel belongs
                 (voxels outside of the brain should have value 0).
    applyGreyMask: bln, if True, the ROI mask is multiplied by a subject or group-specific grey
              matter mask before calculating the centroid coordinates
    greyMaskPath: str, path to which the grey matter mask has been saved in .nii form.
    saveCentroids: bln, if True, the centroids are saved as .npy
    centroidSavePath: str, path to which to save the centroid array
    calculateDistances: bln, if True, a distance matrix between ROI centroids is calculated
    saveDistances: bln, if True, the distance matrix is saved as .npy
    distanceSavePath: str, path to which to save the distance matrix
    
    Returns:
    --------
    centroids: np.array, N_ROIs x 3 array of centroid coordinates
    distances: np.array, N_ROIs x N_ROIs distance matrix (only returned if calculateDistances == True)
    """
    ROIMask = readNii(ROIMaskPath)
    if applyGreyMask:
        greyMask = readNii(greyMaskPath)
        assert ROIMask.shape == greyMask.shape,'ROI mask has different shape than the grey mask, check space and resolution'
        ROIMask = ROIMask*greyMask
    ROIIndices = np.unique(ROIMask)
    ROIIndices = np.delete(ROIIndices, np.where(ROIIndices==np.amin(ROIIndices))) # Removing the smallest unique value of ROIMask, this is typically 0 and marks voxels that don't belong to any ROI
    centroids = np.zeros((len(ROIIndices), 3))
    for i, ROIIndex in enumerate(ROIIndices):
        ROIVoxels = np.where(ROIMask==ROIIndex)
        centroid = np.mean(ROIVoxels, axis=1)
        centroids[i,:] = centroid
    if saveCentroids:
        np.save(centroidSavePath, centroids)
    if calculateDistances:
        distances = np.zeros((len(ROIIndices),len(ROIIndices)))
        for i in np.arange(len(ROIIndices)):
            for j in np.arange(i+1,len(ROIIndices)):
                d = np.sqrt((centroids[i,0]-centroids[j,0])**2 + (centroids[i,1]-centroids[j,1])**2 + (centroids[i,2]-centroids[j,2])**2)
                distances[i,j] = d
                distances[j,i] = d
        if saveDistances:
            np.save(distanceSavePath, distances)
        return centroids, distances
    else:
        return centroids

def calculateVoxelDistances(ROIMaskPath, applyGreyMask=False, greyMaskPath='', saveDistances=False, distanceSavePath=''):
    """
    Calculates the distance matrix between all voxels in a ROI mask. Note that depending on the resolution, the voxel-voxel
    distance matrix may be notably large.

    Parameters:
    -----------
    ROIMaskPath: str, path to which the ROI mask has been saved in .nii form. In the
                 mask, the value of each voxel should show to which ROI the voxel belongs
                 (voxels outside of the brain should have value 0).
    applyGreyMask: bln, if True, the ROI mask is multiplied by a subject or group-specific grey
              matter mask before calculating the centroid coordinates
    greyMaskPath: str, path to which the grey matter mask has been saved in .nii form.
    saveDistances: bln, if True, the distance matrix is saved as .npy
    distanceSavePath: str, path to which to save the distance matrix
    
    Returns:
    --------
    distances: np.array, N_ROIs x N_ROIs distance matrix
    """
    ROIMask = readNii(ROIMaskPath)
    if applyGreyMask:
        greyMask = readNii(greyMaskPath)
        assert ROIMask.shape == greyMask.shape,'ROI mask has different shape than the grey mask, check space and resolution'
        ROIMask = ROIMask*greyMask
    ROIIndices = np.unique(ROIMask)
    ROIVoxels = np.transpose(np.array(np.where(ROIMask > np.amin(ROIIndices))))
    nVoxels = ROIVoxels.shape[0]
    distances = np.zeros((nVoxels,nVoxels))
    for i in np.arange(nVoxels):
        for j in np.arange(i+1,nVoxels):
            d = np.sqrt((ROIVoxels[i,0]-ROIVoxels[j,0])**2 + (ROIVoxels[i,1]-ROIVoxels[j,1])**2 + (ROIVoxels[i,2]-ROIVoxels[j,2])**2)
            distances[i,j] = d
            distances[j,i] = d
    if saveDistances:
        np.save(distanceSavePath, distances)
    return distances

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

def writeNii(data, template_path, save_path):
    """
    Writes the given data into a NIFTI file using nibabel.
    
    Parameters:
    -----------
    data : np.array 
        voxel values to be written as NIFTI, 3D or 4D
    template : str
        path to a template NIFTI, i.e. any NIFTI image in the same space as the data to be saved;
        the affine of the template NIFTI will be used for saving the data
    save_path : str
        path to which save the data
    """
    template = nib.load(template_path)
    affine = template.affine
    header = nib.Nifti1Header()
    data_image = nib.Nifti1Image(data, affine, header)
    nib.save(data_image, save_path)
