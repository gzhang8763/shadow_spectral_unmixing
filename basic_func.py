# -*- coding: utf-8 -*-
"""
Basic functions in nonlinear optimization - ADMM
"""
# import packages
import numpy as np


# soft threshold: threshod values in x using t
def soft(x, t):
    t = t + np.spacing(1) # np.spacing(1) is a very small value, similar to eps in matlab
    y = np.maximum(np.abs(x) - t, 0)
    y = y / (y + t) * x
    
    return y
    