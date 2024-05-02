# -*- coding: utf-8 -*-
"""
Compute difference oprerator 
"""
import os
import scipy.io 
import numpy as np
from scipy.sparse import csr_matrix

# file_directory = os.path.join(os.getcwd(),'data')


def weighted_difference_operator(x, n_lin, n_col, **kwargs):
    """Weighted difference operator

    Parameters
    ----------
    x : double, bands x pixels
        image.
    n_lin : int, 1 x 1
        image rows.
    n_col : int, 1 x 1
        image columns.
    R : array of object 4 x 1, in which n_lin x n_col
        corelation matrix in up, down, left, right directions.

    Returns
    -------
    W: array of object 4 x 1, in which n_lin*n_col x n_lin*n_col
    Weighted difference operator in up, down, left, right directions.
    For example: AW computes value difference in four directions for each pixel
        weighted by R

    """
    N = n_lin * n_col
    
    ones_matrix = np.ones((n_lin,n_col))
    R_default = np.stack((ones_matrix,ones_matrix,ones_matrix,ones_matrix)).reshape(4,1,n_lin,n_col)
    
    R = kwargs.get('R',R_default)

    """ set W as sparse matrices: up, down, left, right """
    
    # UP
    R_up = R[0,0].reshape(N,-1,order='F').flatten()
    # fill in locations where values = 0
    i_idx1 = n_lin * np.arange(0,n_col)
    j_idx1 = n_lin * np.arange(0,n_col)
    value1 = np.zeros(len(i_idx1),)
    
    value1_R = R_up[i_idx1]
    
    # fill in locations where values = 1
    i_idx2 = np.arange(0,N) 
    k = np.arange(1,n_col+1)
    i_idx2_remove = k * n_lin - 1
    i_idx2 = np.setdiff1d(i_idx2, i_idx2_remove)
    j_idx2 = i_idx2 + 1 
    value2 = np.ones(i_idx2.shape[0],)
    
    value2_R = R_up[i_idx2+1]
    
    # fill in locations where values = -1
    i_idx3 = np.arange(0,N) 
    k = np.arange(1,n_col+1)
    i_idx3_remove = n_lin * (k-1)
    i_idx3 = np.setdiff1d(i_idx3, i_idx3_remove)
    j_idx3 = i_idx3 
    value3 = np.ones(i_idx3.shape[0],) * (-1)
    
    value3_R = R_up[i_idx3] 
    
    # join 0, 1, -1
    i_idx = np.concatenate((i_idx1,i_idx2,i_idx3),axis=0)
    j_idx = np.concatenate((j_idx1,j_idx2,j_idx3),axis=0)
    values = np.concatenate((value1_R*value1,value2_R*value2,value3_R*value3),axis=0)
    
    W_up = csr_matrix((values, (i_idx, j_idx)), shape=(N, N))
    
    
    # DOWN
    R_down = R[1,0].reshape(N,-1,order='F').flatten()
    
    # fill in locations where values = 0
    k = np.arange(1,n_col+1)
    i_idx1 = n_lin* k - 1
    j_idx1 = n_lin* k - 1
    value1 = np.zeros(len(i_idx1),)
    value1_R = R_down[i_idx1]
    
    # fill in locations where values = 1
    i_idx2 = np.arange(1,N)
    k = np.arange(1,n_col)
    i_idx2_remove = k * n_lin
    i_idx2 = np.setdiff1d(i_idx2, i_idx2_remove)
    j_idx2 = i_idx2 - 1 
    value2 = np.ones(len(i_idx2),)
    
    value2_R = R_down[i_idx2-1]
    
    # fill in locations where values = -1
    i_idx3 = np.arange(0,N)
    k = np.arange(1,n_col+1)
    i_idx3_remove = n_lin * k - 1
    i_idx3 =  np.setdiff1d(i_idx3, i_idx3_remove)
    j_idx3 = i_idx3 
    value3 = np.ones(len(i_idx3),) * (-1)
    
    value3_R = R_down[i_idx3]
    
    # join 0, 1, -1
    i_idx = np.concatenate((i_idx1,i_idx2,i_idx3),axis=0)
    j_idx = np.concatenate((j_idx1,j_idx2,j_idx3),axis=0)
    values = np.concatenate((value1_R*value1,value2_R*value2,value3_R*value3),axis=0)
    
    W_down = csr_matrix((values, (i_idx, j_idx)), shape=(N, N))    
    
    # LEFT
    R_left = R[2,0].reshape(N,-1,order='F').flatten()
    
    # fill in locations where values = 0
    k = np.arange(0,n_lin)
    i_idx1 = k 
    j_idx1 = k
    value1 = np.zeros(len(i_idx1),)
    value1_R = R_left[i_idx1]
    
    # fill in locations where values = 1
    i_idx2 = np.arange(0,n_lin*(n_col-1))
    j_idx2 = i_idx2 + n_lin 
    value2 = np.ones(len(i_idx2),)
    value2_R = R_left[i_idx2 + n_lin]
    
    # fill in locations where values = -1
    i_idx3 = np.arange(n_lin, N)
    j_idx3 = i_idx3 
    value3= np.ones(len(i_idx3),) * (-1)
    value3_R = R_left[i_idx3]
    
    # join 0, 1, -1
    i_idx = np.concatenate((i_idx1,i_idx2,i_idx3),axis=0)
    j_idx = np.concatenate((j_idx1,j_idx2,j_idx3),axis=0)
    values = np.concatenate((value1_R*value1,value2_R*value2,value3_R*value3),axis=0)
    
    W_left = csr_matrix((values, (i_idx, j_idx)), shape=(N, N))
    
    
    # RIGHT
    R_right = R[3,0].reshape(N,-1,order='F').flatten()
    
    # fill in locations where values = 0
    i_idx1 = np.arange(n_lin*(n_col-1),n_lin*n_col)
    j_idx1 = i_idx1 
    value1 = np.zeros(len(i_idx1),)
    value1_R = R_right[i_idx1]
    
    # fill in locations where values = 1
    i_idx2 = np.arange(n_lin,N)
    j_idx2 = i_idx2 - n_lin 
    value2 = np.ones(len(i_idx2),)
    value2_R = R_right[i_idx2-n_lin]
    
    # fill in locations where values = -1
    i_idx3 = np.arange(0,n_lin*(n_col-1))
    j_idx3 = i_idx3 
    value3 = np.ones(len(i_idx3),) * (-1)
    value3_R = R_right[i_idx3]
    
    # join 0, 1, -1
    i_idx = np.concatenate((i_idx1,i_idx2,i_idx3),axis=0)
    j_idx = np.concatenate((j_idx1,j_idx2,j_idx3),axis=0)
    values = np.concatenate((value1_R*value1,value2_R*value2,value3_R*value3),axis=0)
    
    W_right = csr_matrix((values, (i_idx, j_idx)), shape=(N, N))    
    
    W = np.stack((W_up,W_down,W_left,W_right))
    
#    mat_data = scipy.io.loadmat(file_directory+'\\test_data.mat')
#    W = mat_data['W'][0,:]
#    for m in np.arange(0, 4):
#        W[m] = W[m].tocsr()
#    W = [W_u, W_d, W_l, W_r]
    return W

'''
def difference_operator(x,n_lin,n_col):
    """Non-weighted difference operator

    Parameters
    ----------
    x : double, bands x pixels
        image.
    n_lin : int, 1 x 1
        image rows.
    n_col : int, 1 x 1
        image columns.
    R : array of object 4 x 1, in which n_lin x n_col
        corelation matrix in up, down, left, right directions.

    Returns
    -------
    W: array of object 4 x 1, in which n_lin*n_col x n_lin*n_col
    Non-weighted difference operator in up, down, left, right directions.
    For example: AW computes value difference in four directions for each pixel

    """    
    mat_data = scipy.io.loadmat(file_directory+'\\test_data.mat')
    W = mat_data['W2'][0,:]
    for m in np.arange(0, 4):
        W[m] = W[m].tocsr()
        
    return W
'''
