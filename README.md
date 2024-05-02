# shadow_spectral_unmixing
## A nonlinear regression method with weighted-TV regularization based on ADMM,  applied to shadow-aware nonlinear spectral unmixing problem.

If you find this code helpful to your work, please consider citing this work:

Zhang, Guichen, Paul Scheunders, and Daniele Cerra. "Shadow-aware nonlinear spectral unmixing with spatial regularization." IEEE Transactions on Geoscience and Remote Sensing (2023). DOI: 10.1109/TGRS.2023.3289570

The function S3AM solves the optimization problem

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
    
