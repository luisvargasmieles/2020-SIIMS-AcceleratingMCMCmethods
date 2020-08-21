# -*- coding: utf-8 -*-
"""
Functions for the MRI experiment

@author: Luis Vargas, School of Mathematics, University of Edinburgh
"""

import numpy as np

# fast fourier transforms (orthonormal)
"""res = fft2c(x), orthonormal forward 2D FFT, (c) Michael Lustig 2005"""
fft2c = lambda x: 1/np.sqrt(x.size) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
"""res = ifft2c(x), orthonormal centered 2D ifft, (c) Michael Lustig 2005"""
ifft2c = lambda x: np.sqrt(x.size) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))

# mask operator
def LineMask(angles,dim):
    """
    Create the subsampling mask operator for the MRI imaging experiment
    inputs:
    angles: number of angles to consider in the mask
    dim: the dimention of the mask, dim x dim
    output:
    the mask operator    
    """
    thc = np.linspace(0,np.pi -np.pi/angles,angles)
    M = np.zeros((dim,dim))
    for i in range(0,angles):
        if((thc[i]<= np.pi/4) or (thc[i]>3*np.pi/4)):
            yr = np.round(np.tan(thc[i])*np.arange(-dim/2 +1,dim/2))+ dim/2
            for j in range(0,dim-1):
                M[int(yr[j]),j+1] = 1
        else:
            xc = np.round((1/np.tan(thc[i]))*np.arange(-dim/2 +1,dim/2))+ dim/2
            for j in range(0,dim-1):
                M[j+1,int(xc[j])] = 1
    return np.fft.ifftshift(M)

# AF operator
def masked_FFT(x,mask):
    """
    Computes a 2D partial FFT transform,
    only for those frequencies for which 
    mask is not zero. The result is returned
    in an array of size (k,1) where k is
    the number of non-zeros in mask.
    The transpose of this operator is 
    available in the function masked_FFT_t

    Copyright, 2006, Mario Figueiredo.

    """
    Rf = fft2c(x)
    return Rf[abs(mask)>0]

#AF' operator
def masked_FFT_t(x,mask):
    """
    This is the transpose operator of the
    partial FFT transform, implemented in 
    masked_FFT.  See that function for details.
    """
    gg = np.zeros(mask.shape,dtype=complex)
    gg[abs(mask)>0]=x
    return ifft2c(gg)

