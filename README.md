# shadow_spectral_unmixing
## A spectral unmixing method robust to shadow and noise, based on constrained nonliner optimization via ADMM.

If you find this code helpful to your work, please consider citing this work:

Zhang, Guichen, Paul Scheunders, and Daniele Cerra. "Shadow-aware nonlinear spectral unmixing with spatial regularization." IEEE Transactions on Geoscience and Remote Sensing (2023). DOI: 10.1109/TGRS.2023.3289570

Given input image Y and endmember library E, the optimization algorithm S3AM solves X (abundances), Q (shadow fraction), and K (nonlinearity). 

    min{X,Q,K}  (1/2) ||(~E)*X-Y||^2_F  + lambda_tv  ||XW1||_{1,1} + lambda_tv ||KW2||_{1,1}
    s.t.
    1) Positivity X(:,i) >= 0, for i=1,...,N
    2) Sum-To-One sum( X(:,i)) = 1, for i=1,...,N
    3) Q, K >=0 and <= 1 for i=1,...,N

    (~E) is computed by endmemebrs E, shadow fraction Q, and nonlinearity K.

    W1 is a sparse matrix to compute difference between a target pixel and its neighnoring pixels in the four canonical directions, weighted by R. R is weight matricies

    W2 is a sparse matrix to compute difference between a target pixel and its neighnoring pixels in the four canonical directions, not weighted.
    
    lambda_tv is the regularization parameter.
    
