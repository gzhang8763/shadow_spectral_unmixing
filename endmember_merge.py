# -*- coding: utf-8 -*-
"""
Merge similar spectra given endmember library
"""
# import packages
import numpy as np

def endmember_merge(E_total):
    """
    merge similar spectra given endmember library

    Parameters
    ----------
    E_total : nband x n_endmem
        input.

    Returns
    -------
    E : float64, nband x n_endmem
        output.

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