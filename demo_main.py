# -*- coding: utf-8 -*-
from __future__ import print_function
import os

# functions
# from file.py import function(a,b) -> to import funcitons from another py file

## import packages for sptools
# sptools: https://pysptools.sourceforge.io/index.html
import pysptools.util as util # load envi file
import pysptools.eea as eea # endmember extraction
import pysptools.abundance_maps.amaps as amaps # abundance estimation

# other packages 
import numpy as np
import scipy.io # load mat files in python
from scipy.optimize import minimize as fmin # nonlinear optimization 
import matplotlib.pyplot as plt # to plot
from PIL import Image

# functions
from endmember_extract_exshade import endmember_extract_exshade, endmember_merge
from S3AM import S3AM
from sunsal import sunsal
from sp_models import model_LMM, model_S3AM, model_S3AM_deshade



#############################################################
#%%           load and reshape data 
#############################################################

file_directory = os.path.join(os.getcwd(),'data')

# load data
mat_data = scipy.io.loadmat(file_directory+'\\test_image.mat')

image = mat_data['Image'].astype(np.double) # input image: n_row * n_col * n_band
image_clean = mat_data['Image_clean'].astype(np.double) # clean image without noise and shadow
E = mat_data['E'] # n_band x n_endmembers
wavelength_norm = mat_data['wavelength_norm'] # n_band x 1
wavelength = mat_data['wavelength'] # n_band x 1
selected_band = mat_data['selected_band'] # 1 x n_band
k = mat_data['k'].reshape(3,) # precomputed shadow parameters. recomputation required for different dataset
SVF = mat_data['SVF_test_normalize'] # SkyViewFactor nomalized, n_row x n_col
Q_prior = mat_data['Q'] # 1 x n_pixels, where  n_pixels = n_row * n_col
DSM = mat_data['DSM_test_normalize'] # n_row x n_col
nn = mat_data['nn'] # averaged spectrum in local neighorhood (window size=3), n_band x n_pixels
R = mat_data['R'] # weighted spatial relation matrices, 4 x nrow x ncol. 4: up,down,left,right directions. 
# Pixel in each direction computes the spectral and height similarities between the pixel and its (up,down,left,or right) neighbors
#R_eq = mat_data['R_eq'] # spatial relatinon matrices, 4 x nrow x ncol

# data shape
(n_row,n_col,n_band) = image.shape
print('The size of the image is:' + str(n_row) +'x'+ str(n_col)+'x'+ str(n_band))
n_endmem = E.shape[1]  
Q_prior = Q_prior.reshape(1, n_row*n_col)  
# reshape data
x = image.reshape((n_row*n_col,n_band), order='F')
x = np.transpose(x)
SVF = SVF.reshape((1, n_row*n_col))
    
#############################################################
#%%          endmember extraction
#############################################################

# shadow-excluded endmember extraction - enable to extract endmembers 
'''
n_group = 10
n_endmem = 10
# VCA endmember extraction, from n_group groups, each group extracts n_endmem 
E_total = endmember_extract_exshade(x, n_row, n_col, n_band, n_group, n_endmem, verbose=True)
# refine endmembers, remove highly similar spectra
E = endmember_merge(E_total)
'''

fig, ax = plt.subplots()
ax.plot(wavelength, E)
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('reflectance')
ax.set_title('endmember library')

#############################################################
#%%          Run FCLS 
#############################################################

U_fcls,_,_,_ = sunsal(E,x,AL_iters=100,lambda_0=0,positivity=True,addone=True,tol=1e-4,x0 = None,verbose=False) # solve abundances
x_fcls = model_LMM(U_fcls, E) # re-compute simulated pixels

#############################################################
#%%     Run FCLS with extended zero-reflectance endmember
#############################################################

E_extend = np.append(E,np.zeros((n_band,1)),axis=1)
U_fcls_zero,_,_,_ = sunsal(E_extend,x,AL_iters=100,lambda_0=0,positivity=True,addone=True,tol=1e-4,x0 = None,verbose=False) # solve abundances
x_fcls_zero = model_LMM(U_fcls_zero, E_extend) # re-compute simulated pixels


#############################################################
#%%     Run shadow-aware ADMM
#############################################################

# parameter setting
lambda_tv = 0.5 # regularization
n_iteration = 90
mu = 0.001
toll = 5e-4

[U_s3am,Q_s3am,K_s3am,_,_,_] = \
    S3AM(E, x, DSM, SVF, R, k, nn, wavelength_norm, n_row, n_col, lambda_tv = lambda_tv, n_iters = n_iteration, mu = mu, Q0 = Q_prior, toll = toll)  
    
x_s3am = model_S3AM(U_s3am, E, Q_s3am, K_s3am, SVF, k, nn, wavelength_norm)


#############################################################
#%%     Compare abundances
#############################################################

#%%% group endmembers
endmember_groups = np.array([[0],[1],[2],[3],[4],[5]], dtype=object)
#endmember_groups = np.array([[0, 2, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19], [1, 3, 4, 5, 8, 10, 11], [6]], dtype=object)
n_group = endmember_groups.shape[0]

if not os.path.exists('abundances_result'):
   os.makedirs('abundances_result')
   
directory_save = os.path.join(os.getcwd(),"abundances_result")


# make plots
fig, ax = plt.subplots(3,6)


for group in np.arange(0,n_group):
    selected_endmember = endmember_groups[group][0]
    
    # plot FCLS
    im = ax[0,group].imshow(U_fcls[selected_endmember,:].reshape(-1,1).sum(axis=1).reshape(n_row, n_col, order='F'), vmin=0.1, vmax=0.9)
    ax[0,group].set_title("FCLS: g "+ str(group))
    
    # FCLS zeros
    ax[1,group].imshow(U_fcls_zero[selected_endmember,:].reshape(-1,1).sum(axis=1).reshape(n_row, n_col, order='F'), vmin=0.1, vmax=0.9)
    ax[1,group].set_title("FCLS zero: g "+ str(group))
    
    #S3AM
    ax[2,group].imshow(U_s3am[selected_endmember,:].reshape(-1,1).sum(axis=1).reshape(n_row, n_col, order='F'), vmin=0.1, vmax=0.9)
    ax[2,group].set_title("S3AM: g "+ str(group))
    
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # remove ticks  
fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.5) # add colorbar

# plot input image
fig, ax = plt.subplots(1,3)
ax[0].imshow(image[:,:,[58,38,10]]*2)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # remove ticks  
ax[0].set_title('input: shadow and noisy image')

# plot output image
image_output = model_S3AM_deshade(U_s3am, E, Q_s3am, K_s3am, nn).transpose().reshape(n_row,n_col,n_band, order='F')

ax[1].imshow(image_output[:,:,[58,38,10]]*2)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # remove ticks  
ax[1].set_title('output: shadow and noise-removed image')

ax[2].imshow(image_clean[:,:,[58,38,10]]*2)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # remove ticks  
ax[2].set_title('original clean image')

# save abundances and images
for group in np.arange(0,n_group):
    # save FCLS
    extent = ax[0,group].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(directory_save + '\\FCLS_group' + str(group) + '.png', bbox_inches=extent)
    
    # save FCLS zeros
    extent = ax[1,group].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(directory_save+'\\FCLS_zeros_group'+ str(group) +'.png', bbox_inches=extent)
    
    # save S3AM
    extent = ax[2,group].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(directory_save+'\\S3AM_group' + str(group) + '.png', bbox_inches=extent)    
    
