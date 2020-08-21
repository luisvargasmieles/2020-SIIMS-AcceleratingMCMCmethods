"""
f = chambolle_prox_TV(g, lambda, maxiter)
Proximal  point operator for the TV regularizer 

Uses the Chambolle's projection  algorithm:

A. chambolle_TV, "An Algorithm for Total Variation Minimization and
Applications", J. Math. Imaging Vis., vol. 20, pp. 89-97, 2004.

Optimization problem:  

    arg min = (1/2) || y - x ||_2^2 + lambda TV(x)
        x

=========== Required inputs ====================

'g'       : noisy image (size X: ny * nx)

'lambda'  : regularization  parameter according

'maxiter' :maximum number of iterations
"""
import numpy as np

def DivergenceIm(p1,p2):
    """The divergence function that the prox_TV function needs"""
    z = p2[:,1:-1] - p2[:,0:-2]
    v = np.c_[p2[:,0],z,-p2[:,-1]]
    
    z = p1[1:-1, :] - p1[0:-2,:]
    u = np.c_[p1[0,:],z.T,-p1[-1,:]]
    u = u.T
    return v + u

def GradientIm(u):
    """The Gradient of the image that prox_TV function needs"""
    z = u[1:, :] - u[0:-1,:]
    dux = np.c_[z.T,np.zeros(np.size(z,1))]
    dux = dux.T
    
    z = u[:,1:] - u[:,0:-1]
    duy = np.c_[z,np.zeros((np.size(z,0),1))]
    return  dux,duy

def chambolle_prox_TV(g,apprParam,MaxIter):
    """
    total variation proximal operator
    inputs:
    g: image
    apprParam: the approximation parameter of the proximal algorithm (\lambda)
    MaxIter: number of iterations of the optimisation algorithm
    output:
    the total-variation Prox operator of the image 'g'
    """
    # initialize
    px = np.zeros((np.size(g,0),np.size(g,1)))
    py = np.zeros((np.size(g,0),np.size(g,1)))
    cont = 1       
    k    = 0
    tau = 0.249

    while cont: 
        k = k+1
        # compute Divergence of (px, py)
        divp = DivergenceIm(px,py) 
        u = divp - g/apprParam
        # compute gradient
        upx,upy = GradientIm(u)
        tmp = np.sqrt(np.multiply(upx,upx) + np.multiply(upy,upy))  
        px = np.divide(px + tau * upx,1 + tau * tmp)
        py = np.divide(py + tau * upy,1 + tau * tmp)
        cont = (k<MaxIter)

    return g - apprParam * DivergenceIm(px,py)

def TVnorm(x):
    """Compute the discrete Total variation norm"""
    upx,upy = GradientIm(x)
    return np.sum(np.sqrt(np.power(upx,2) + np.power(upy,2)))