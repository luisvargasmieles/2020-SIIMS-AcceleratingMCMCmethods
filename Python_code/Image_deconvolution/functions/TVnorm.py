"""
Total-variation norm
Implemented from eq. (1) of:
Chambolle, A. An Algorithm for Total Variation Minimization and Applications.
Journal of Mathematical Imaging and Vision 20, 89–97 (2004). 
https://doi.org/10.1023/B:JMIV.0000011325.36760.1e
input:
    x: image
output:
    tvNorm: the discrete total-variation norm of the image
"""

import numpy as np

def TVnorm(x):
    [rows,columns]=[np.size(x,0),np.size(x,1)]
    tvNorm = 0
    for i in range(0,rows):
        for j in range(0,columns):
            if (j != columns-1) and (i != rows-1):
                tvNorm = tvNorm + np.sqrt((x[i+1,j]-x[i,j])**2 +(x[i,j+1]-x[i,j])**2)
            if (j == columns-1) and (i != rows-1):
                tvNorm = tvNorm + abs(x[i+1,j]-x[i,j])
            if (j != columns-1) and (i == rows-1):
                tvNorm = tvNorm + abs(x[i,j+1]-x[i,j])
    return tvNorm