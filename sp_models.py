# -*- coding: utf-8 -*-
"""
Spectral mixing models

E: endmembers [bands x endmembers]
U: abundances [endmembers x pixels]
"""
import numpy as np

def model_LMM(U, E): # LMM
    """Model pixels, given abundances, endmembers and hyperparameters.
    
    Parameters [double]
    ----------
    U : abundances. endmembers x pixels.
    E : endmembers. bands x endmembers
    
    Returns [double]
    -------
    modeled pixels. bands x pixels    
    """
    return np.dot(E, U)

def model_S3AM(U, E, Q, K, F, k, nn, wavelength_norm): # simplified ESMLM
    """Model pixels, given abundances, endmembers and hyperparameters.
    
    Parameters [double]
    ----------
    U : abundances. endmembers x pixels.
    E : endmembers. bands x endmembers
    Q : shadow fractions. 1 x pixels
    K : double reflection stength. 1 x pixels
    F : sky view factor. 1 x pixels
    k : atmosphere parameters. 3 x 1
    nn : averaged spectra for each pixel in the firsr order neighborhood. bands x pixels
    wavelength_norm : wavelengths normalized. bands x 1

    Returns [double]
    -------
    x_hat : modeled pixels. bands x pixels
    """
    # sizes
    n_band = wavelength_norm.shape[0]
    N = Q.shape[1]
    x_hat = np.zeros((n_band, N))
    
    sky = k[0] * wavelength_norm ** (-k[1]) + k[2]
    
    for pixel in np.arange(0, N):
        F_term = F[:,pixel] * sky / (1 + F[:,pixel] * sky) # n_band x 1
        E_prim =  E * ((1 - Q[:,pixel]) + (Q[:,pixel] * F_term + (K[:,pixel] * nn[:,pixel]).reshape(-1,1))) # n_band x n_endmem
        x_hat[:,pixel] = np.dot(E_prim, U[:,pixel])
        
    return x_hat

def model_S3AM_deshade(U, E, Q, K, nn):
    """

    Parameters [double]
    ----------
    U : abundances. endmembers x pixels.
    E : endmembers. bands x endmembers
    Q : shadow fractions. 1 x pixels
    K : double reflection stength. 1 x pixels
    nn : averaged spectra for each pixel in the firsr order neighborhood. bands x pixels

    Returns [double]
    -------
    x_restore : shadow restored pixels. bands x pixels

    """
    y = np.dot(E, U)
    x_restore = (1-Q) * y + Q * y + K * nn * y
    
    return x_restore