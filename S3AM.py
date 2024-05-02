# -*- coding: utf-8 -*-
"""
A nonlinear regression method with weighted-TV regularization based on ADMM,
 applied to shadow-aware nonlinear spectral unmixing

Reference:
Zhang, Guichen, Paul Scheunders, and Daniele Cerra.
"Shadow-aware nonlinear spectral unmixing with spatial regularization."
IEEE Transactions on Geoscience and Remote Sensing (2023). DOI: 10.1109/TGRS.2023.3289570


----- This method solves the optimization problem -----
    min{X,Q,K}  (1/2) ||(~E)*X-Y||^2_F  + lambda_tv  ||XW1||_{1,1} + lambda_tv ||KW2||_{1,1}
    s.t.
    1) Positivity X(:,i) >= 0, for i=1,...,N
    2) Sum-To-One sum( X(:,i)) = 1, for i=1,...,N
    3) Q, K >=0 and <= 1 for i=1,...,N

    (~E) is computed by endmemebrs E, unknown parameters Q and K

    W is a sparse matrix to compute difference between a target pixel and
    its neighnoring pixels in the four canonical directions, weighted by
    R. R is weight matricies

    W2 is a sparse matrix to compute difference between a target pixel and
    its neighnoring pixels in the four canonical directions, not weighted
    
    lambda_tv is the regularization parameter
    
"""
# import packages
import sys
import numpy as np
from scipy.linalg import inv # inverse
from scipy.linalg import lstsq # MATLAB ldivide
from scipy.linalg import norm # Frobenius norm by default
from scipy.sparse import eye # sparse eye function
from scipy.sparse.linalg import lsqr # solves x: Ax = b in sparse condition
from scipy.sparse.linalg import spsolve 
from basic_func import soft # soft function
from difference_operator import weighted_difference_operator


# "pip install sparse-dot-mkl" for efficient sparse matrix multiplication
# matrix multiplication | solution ATA | A\b method to compute inverse
from sparse_dot_mkl import dot_product_mkl as sparse_multiply, gram_matrix_mkl as sparse_ata #sparse_qr_solve_mkl as sparse_inv




def S3AM(M, Y, DSM, F, R, k, nn, wavelength_norm, n_lin, n_col, **kwargs):
    """S3AM function for shadow-aware constraint spectral unmixing.
    
    Parameters
    ----------
    M : double
        endmembers, bands x endmembers.
    Y : double
        input image, bands x pixels.
    DSM : double
        DSM, nrow x ncol.
    F : double
        sky view factor (SVF), 1 x pixels.
    R : object contains 4 x 1 double matrices
        correlation matrix in a local neighborhood, 4 x row x col
    k : double
        constant value, atmosphere parameters, 3 x 0
    nn : double
        averaged for each pixel in the first order neighborhood, bands x pixels.
    wavelength_norm : double
        normalized wavelengths, bands x 1
    n_lin : int
        rows of input image, 1
    n_col : int
        columns of input image, 1.
    **kwargs : other parameters
        
        lambda_tv: TV regularizer
        niters: maximum number of iterations
        mu: lagrange weight
        A0: initial abundance solution
        Q0: initlal Q solution
        toll: stop criteria, a small value
    Returns
    -------
    A : double
        abundances, endmembers x pixels.
    Q : double
        shadow fraction, 1 x pixels.
    K : double
        double reflection parameter, 1 x pixels.
    res_primal : double
        primal residual, 8 x 1.
    res_dual : double
        dual residual, 8 x 1.
    res_total : double
        total residuals for each iteration, n_iter x 1
    """
    ##########################################
    #%% preprare parameters and data sizes
    ##########################################
    # read optional parameters
    parameters = {"lambda_tv": 0.1, "n_iters": 100, "mu": 0.001, "A0": 0, "Q0": 0, "toll": 0.01}
    
    for key, para in parameters.items():
        if key not in kwargs.keys():
            kwargs[key] = parameters[key]
    
    N = n_lin * n_col
    lambda_tv = kwargs["lambda_tv"]
    n_iters = kwargs["n_iters"]
    mu = kwargs["mu"]
    A0 = kwargs["A0"]
    Q = kwargs["Q0"]
    toll = kwargs["toll"]
    
    K = np.zeros((1, N)) # N: number of pixels

    # compute sizes
    lm, n = M.shape # n_band, n_endmembers
    
    # dataset size
    l, N = Y.shape # n_band, n_pixels
    
    if (lm != l):
        sys.exit("mixing matrix M and data set Y are inconsistent.")
    

    # compute sky factor
    sky = k[0] * wavelength_norm ** (-k[1]) + k[2]
    
    ##########################################
    #%% compute (weighted) difference operator W and W2
    ##########################################
    
    W = weighted_difference_operator(Y,n_lin,n_col,R=R) # weighted
    W2 = weighted_difference_operator(Y,n_lin,n_col) # not weighted
    
    ##########################################
    #%% initialize unknown variables
    ##########################################
    
    # number regularizers
    reg_l1 = 1 # tv regu
    reg_pos = 1 # positivity
    reg_add = 1 # add to one
    n_reg = reg_l1 + reg_pos + reg_add # total
    
    invers = inv(np.dot(M.T, M) + n_reg * np.eye(n)) 
    n_reg = n_reg + 1   
    
    # if A0 is not given
    if A0 == 0:
        A = np.dot(np.dot(invers, M.T), Y)
    
    # initialize G 1 - 4 and U 1 - 4
    G2 = [sparse_multiply(A, W[0]), sparse_multiply(A, W[1]), sparse_multiply(A, W[2]), sparse_multiply(A, W[3])]
    G = [A.copy(), G2.copy(), A.copy(), A.copy()]
    U2 = [np.zeros(A.shape), np.zeros(A.shape), np.zeros(A.shape), np.zeros(A.shape)]
    U_p1 = [np.zeros(A.shape), U2.copy(), np.zeros(A.shape), np.zeros(A.shape)]
    
    # initialize H 1 - 4 and U 5 - 8
    H3 = [sparse_multiply(K, W2[0]), sparse_multiply(K, W2[1]), sparse_multiply(K, W2[2]), sparse_multiply(K, W2[3])]
    H = [Q.copy(), K.copy(), H3.copy(), K.copy()]
    
    U7 = [np.zeros(K.shape), np.zeros(K.shape), np.zeros(K.shape), np.zeros(K.shape)]
    U_p2 = [np.zeros(Q.shape), np.zeros(K.shape), U7.copy(), np.zeros(K.shape)]
    
    U_p1.extend(U_p2)
    U = U_p1.copy()
    

    ##########################################
    #%%  iterations : main body
    ##########################################    

    i = 1
    res = float('INF') 
    res_total = np.empty((1,1))    
    
    while (i <= n_iters) and (np.sqrt(np.sum(res)) > np.sqrt(N * M.shape[1]) * toll):
        # back up G
        G_bk = G.copy()
        H_bk = H.copy()
        
        ##########################################
        #%%% update A and G 1-4, fixing Q and K and H 1-4 
        ##########################################
        
        # updata A
        for pixel in np.arange(0, N):
            F_term = F[:,pixel] * sky / (1 + F[:,pixel] * sky) # n_band x 1
            M_prim =  M * ((1 - Q[:,pixel]) + (Q[:,pixel] * F_term + (K[:,pixel] * nn[:,pixel]).reshape(-1,1))) # n_band x n_endmem
            Xi = np.dot(M_prim.T, Y[:,pixel].reshape(-1,1)) + \
                mu * (G[0][:,pixel].reshape(-1,1) + U[0][:,pixel].reshape(-1,1) + \
                      G[2][:,pixel].reshape(-1,1) + U[2][:,pixel].reshape(-1,1) + \
                          G[3][:,pixel].reshape(-1,1) + U[3][:,pixel].reshape(-1,1))
            A[:,pixel] = lstsq((np.dot(M_prim.T, M_prim) + 3 * mu * np.eye(n)) , Xi)[0].reshape(-1,) # equivalent to A\b in matlab

        # update G1 2
        fi_W = sum([sparse_multiply((G[1][m] + U[1][m]), W[m].T) for m in np.arange(0,4)])

        # sparse matrix computation ATA : transpose=False, AAT: transpose=True 
        temp = sum([sparse_ata(W[m], transpose=True, cast=True) for m in np.arange(0,4)]) # W{0}*W{0}' + W{2}*W{2}' + W{3}*W{3}' +  W{4}*W{4}'
        
        lsqr_A = (eye(N) + temp).T # WWT = temp
        lsqr_b = (A - U[0] + fi_W).T
        G[0] = spsolve(lsqr_A, lsqr_b).T

        for m in np.arange(0,4):
            G[1][m] = soft(sparse_multiply(G[0], W[m]) - U[1][m], lambda_tv / mu)
    
        
        # update G3 4
        G[2] = np.maximum(A - U[2], 0)
        G[3] = A - U[3] + (1 - np.sum(A - U[3], axis = 0, keepdims= True)) / n 
        
        
        ##########################################
        #%%% update Q and K, H1-4, fixing A
        ##########################################
        
        # update Q and K
        Y_update = np.dot(M, A)
        
        F_term = F * sky / (1 + F * sky)
        
        c1 = np.sum((F_term - 1)**2 * Y_update**2, axis=0, keepdims=True) + mu
        c2 = np.sum(nn * (F_term - 1) * Y_update * Y_update, axis = 0, keepdims=True)
        c3 = np.sum((F_term - 1) * Y_update * Y,axis=0, keepdims=True) - \
            np.sum((F_term - 1) * Y_update**2, axis=0, keepdims=True) + \
                mu * (H[0] + U[4])
        
        d1 = np.sum((F_term - 1) * nn * Y_update**2, axis=0, keepdims=True)
        d2 = np.sum(nn**2 * Y_update**2, axis=0, keepdims=True) + 2 * mu
        d3 = np.sum(nn * Y_update * Y, axis=0, keepdims=True) - \
            np.sum(nn * Y_update**2, axis=0, keepdims=True) + mu * (H[1] + U[5] + H[3] + U[7])      
        
        Q = (c3 * d2 - d3 * c2) / (c1 * d2 - c2 * d1)
        K = (c3 * d1 - d3 * c1) / (c2 * d1 - c1 * d2)
        
        # update H 1234
        
        # H1
        H[0] = np.minimum(np.maximum(Q - U[4], 0), 1)
        
        # H2 H3
        fi_W = sum([sparse_multiply((H[2][m] + U[6][m]), W2[m].T) for m in np.arange(0,4)]) 
        
        # sparse matrix computation ATA : transpose=False, AAT: transpose=True 
        temp = sum(sparse_ata(W2[m], transpose=True, cast=True) for m in np.arange(0,4)) # W2{0}*W2{0}' + W2{2}*W2{2}' + W2{3}*W2{3}' +  W2{4}*W2{4}'        
        
        lsqr_A = (eye(N) + temp).T # WWT2 = temp
        lsqr_b = (K - U[5] + fi_W).T
        
        H[1] = lsqr(lsqr_A,lsqr_b)[0].reshape(-1,1).T
        
        for m in np.arange(0,4):
            H[2][m] = soft(sparse_multiply(H[1], W2[m]) - U[6][m], lambda_tv / mu)
        
        # H4
        H[3] = np.minimum(np.maximum(K - U[7], 0.1), 1)
        
        
        ##########################################
        #%%% update U 1 - 8
        ##########################################
        
        for j in np.arange(0, n_reg *2):
            if j == 0 or j == 2 or j == 3:
                U[j] = U[j] - A + G[j]
            elif j == 1:
                for m in np.arange(0, 4):
                    U[j][m] = U[j][m] - sparse_multiply(G[j-1], W[m]) + G[j][m]
            elif j == 4:
                U[j] = U[j] - Q + H[j-4]
            elif j == 5 or j == 7:
                U[j] = U[j] - K + H[j-4]
            elif j == 6:
                for m in np.arange(0, 4):
                    U[j][m] = U[j][m] - sparse_multiply(H[j-5], W2[m]) + H[j-4][m]


        ##########################################
        #%%% update primal and dual variables
        ##########################################
        
        ##### primal residuals: res_primal
        res = np.zeros((n_reg*2,))
    
        for j in np.arange(0, n_reg*2):
            if j == 0 or j == 2 or j == 3:
                res[j] = norm(A - G[j]) ** 2
            elif j == 1:
                for m in np.arange(0,4):
                    if m == 0:
                        res[j] = ((sparse_multiply(G[j-1], W[m]) - G[j][m]) ** 2).sum() # 2d sum for matrix
                    else:
                        res[j] = res[j] + ((sparse_multiply(G[j-1], W[m]) - G[j][m]) ** 2).sum()
            elif j == 4:
                res[j] = norm(Q - H[j-4]) ** 2
            elif j == 6:
                for m in np.arange(0,4):
                    if m == 0:
                        res[j] = ((sparse_multiply(H[j-5], W2[m]) - H[j-4][m]) ** 2).sum() # 2d sum for matrix
                    else:
                        res[j] = res[j] + ((sparse_multiply(H[j-5], W2[m]) - H[j-4][m]) ** 2).sum() 
            elif j == 5 or j == 7:
                res[j] = norm(K - H[j-4]) ** 2
            
        res_primal = np.sqrt(np.sum(res))

    
        ##### dual residual: res_dual
        res_dual_p1 = 0
        res_dual_p2 = 0
        for m in np.array([0, 2, 3]):
            res_dual_p1 = res_dual_p1 + norm(G[m] - G_bk[m])**2
        for m in np.array([0, 1, 3]):
            res_dual_p1 = res_dual_p1 + norm(H[m] - H_bk[m]) **2
            
        for m in np.arange(0, 4):
            res_dual_p2 = res_dual_p2 + norm(G[1][m] - G_bk[1][m])**2 + norm(H[2][m] - H_bk[2][m]) **2                                              
                      
        res_dual = np.sqrt(res_dual_p1 + res_dual_p2)
    
        # total residual
        if i == 1:
            res_total = (res_primal / np.sqrt(N * M.shape[1]))
        else:
            res_total = np.append(res_total,res_primal / np.sqrt(N * M.shape[1]))
    
        ##########################################
        #%%% update mu, based on the scaling factor between primal and dual residuals
        ##########################################
        
        scaling_factor = 1.5
    
        if i == 1:
            print(['ADMM scaling value is:',str(scaling_factor),'\n'])
        elif res_primal > 10 * mu * res_dual and np.mod(i,14) == 1 and i != 1:
            mu = scaling_factor * mu
            for j in np.arange(0, n_reg * 2):
                if j == 1 or j == 6:
                    U[j] = [U[j][m] / scaling_factor for m in np.arange(0, 4)]
                else:
                    U[j] = U[j] / scaling_factor
        elif  mu * res_dual > 10 * res_primal and np.mod(i,14) == 1 and i != 1:
            mu = mu / scaling_factor
            for j in np.arange(0, n_reg * 2):
                if j == 1 or j == 6:
                    U[j] = [U[j][m] * scaling_factor for m in np.arange(0, 4)]
                else:
                    U[j] = U[j] * scaling_factor        
                
        i = i + 1        
        print('The total primal residual at iteration ' + str(i) + ' : ' + str(res_primal / np.sqrt(N * M.shape[1])),'\n')
    
    return A, Q, K, res_primal, res_dual, res_total
    

























    