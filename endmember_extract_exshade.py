# -*- coding: utf-8 -*-
"""
shadow-aware endmember extraction.
"""

# import packages
import numpy as np
from sklearn.decomposition import PCA # PCA
from skimage import feature # for canny detector
import cv2 # image processing
from numpy import random # generate random numbers
from VCA import vca
import math
import matplotlib.pyplot as plt

def endmember_extract_exshade(pixels, nrow, ncol, nband, n_group, n_endmem, verbose=False):
    """Extract endmembers while excluding shadow pixels.
    
    Parameters
    ----------
    pixels : float64, nband x npixel
        input image pixels.
    nrow : int
        row size.
    ncol : int
        column size.
    n_group : int
        number of groups for endmeber extraction. 
    n_endmem : int, 1x1
        number of endmembers per group for endmember extraction. 
    verbose : bool
        plot endmember locations in the target image. The default is false.
    Returns
    -------
    E_total : float64, nband x n_endmem
        extracted endmembers.
    """
    # define parameters
    RGB_bands = [60,40,30]
    
    ######################################
    #%% extract edge pixels
    ######################################
    # run PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(np.transpose(pixels))
    
    # detect edges
    edges1 = feature.canny(components[:,0].reshape((nrow, ncol)),sigma=3)
    edges2 = feature.canny(components[:,1].reshape((nrow, ncol)),sigma=3)
    edges = np.bitwise_or(edges1,edges2)
    
    # dilation
    kernel = np.ones((2, 2), np.uint8)    
    edges = cv2.dilate(edges.astype(float), kernel, iterations=1).astype(int)
    
    ######################################
    #%% select candidate sunlit pixels
    ######################################    
    pix_sunlit = np.bitwise_and(np.mean(pixels, axis=0) > 0.085, np.max(pixels, axis=0)<0.7)
    # dilation
    kernel = np.ones((2, 2), np.uint8)    
    pix_sunlit = cv2.erode(pix_sunlit.reshape((nrow,ncol)).astype(float), kernel, iterations=1).astype(int)  
    
    ######################################
    #%% exclude edge pixels from the candidate pixels
    ######################################      
    pix_select = np.bitwise_and(pix_sunlit, (edges != 1))
    
    pix_select = pix_select.flatten()
    pix_select = np.where(pix_select == 1)[0]
    pix_candidate = pix_select
    # shuffle candidate pixels
    random.shuffle(pix_candidate)
    

    ######################################
    #%% interations to select endmembers. Select #n_endmem in #n_group
    ######################################       
    pixels_per_group = int(pix_candidate.shape[0]/n_group)
    E_total = np.empty((nband,1)) 
    count = 1
    
    while count <= n_group:
        select_currentgroup = np.arange(pixels_per_group*(count-1),pixels_per_group*count)
        #%%% run vca
        U, indicies, _ = vca(pixels[:,pix_candidate[select_currentgroup]],n_endmem)
        
        #%%% plot locations of endmembers
        if verbose:
            # compute X Y position of endmembers in the image       
            temp = pix_candidate[select_currentgroup]
            indicies_inwholeimg = temp[indicies]        
            floor = np.vectorize(math.floor)
            loc_y = floor(indicies_inwholeimg/ncol)
            loc_x = indicies_inwholeimg - loc_y * ncol
            if count == 1:
                print('The location is: '+str(loc_x[0]) + "," + str(loc_y[0]))
            RGB = np.transpose(pixels[RGB_bands,:]).reshape((nrow,ncol,3),order='F')*3
            
            
            if count == 1:
                E_total = U
                # new a figure 
                fig_ncol = 5 # five columns in fig plot
                fig, ax = plt.subplots(math.ceil(n_group/fig_ncol),fig_ncol)
            else:
                E_total = np.append(E_total, U, axis=1)
    
            # plot the locations of selected pixels
            cnt = count -1 
            ax[math.floor(cnt/fig_ncol), cnt-math.floor(cnt/fig_ncol)*fig_ncol].imshow(RGB)
            ax[math.floor(cnt/fig_ncol), cnt-math.floor(cnt/fig_ncol)*fig_ncol].scatter(loc_y, loc_x, marker='+', color='r') 
            plt.show()
        
        count = count + 1
        
    return E_total


def endmember_merge(E_total):
    """Merge similar spectra given endmember library.
    
    Parameters
    ----------
    E_total : nband x n_endmem
        input endmembers.

    Returns
    -------
    E : float64, nband x n_endmem
        output endmembers.

    """
    nband, n_endmem = E_total.shape
    
    # compute distance matrix 
    distance = np.zeros((n_endmem,n_endmem))
    
    for i in np.arange(0,n_endmem):
        for j in np.arange(0,n_endmem):
            distance[i,j] = np.sum((E_total[:,i] - E_total[:,j]) ** 2)/nband


    flag = np.ones((n_endmem,1))
    count = 1

# compute a smaller subset of E, whose spectra has significant distance compared to all other spectra
    for i in np.arange(0,n_endmem):
        temp = np.arange(0,n_endmem)
        loc = (distance[i,temp] < 0.005)
        
        if sum(loc) != 0:
            flag[temp[loc]] = 0
            if count == 1:
                E = np.mean(E_total[:,temp[loc]], axis=1, keepdims=True)
            else:
                E = np.append(E, np.mean(E_total[:,temp[loc]], axis=1, keepdims=True) , axis=1)
            count = count + 1
        
    return E