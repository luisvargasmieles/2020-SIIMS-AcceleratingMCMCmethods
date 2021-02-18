"""
MYULA SAMPLING METHOD                        

This function samples the distribution \pi(x) = exp(-U(x)) thanks to a 
proximal MCMC algorithm called MYULA (see "Efficient Bayesian Computation
by Proximal Markov Chain Monte Carlo: When Langevin Meets Moreau", Alain
Durmus, Eric Moulines and Marcelo Pereyra, SIAM Journal
on Imaging Sciences, 2018).

    INPUTS:
        X: current MCMC iterate (2D-array)
        Lipschitz_U: user-defined lipschitz constant of the model
        gradU: function that computes the gradient of the potential U
        
    OUTPUT:
        XkMYULA: new value for X (2D-array).

@author: Luis Vargas Mieles
"""
import numpy as np

def MYULA(X,Lipschitz_U,gradU):
    # size of the sample
    N = len(X)
    
    # variable to save new sample
    XkMYULA = np.zeros((N,N))
    
    # MYULA step-size
    dtMYULA = 1/Lipschitz_U # step-size

    # Sampling the variable X (MYULA)
    Q=np.sqrt(2*dtMYULA)*np.random.randn(N,N) # diffusion term

    # MYULA sample
    XkMYULA = X - dtMYULA*gradU(X) + Q

    return XkMYULA # new sample produced by the MYULA algorithm
