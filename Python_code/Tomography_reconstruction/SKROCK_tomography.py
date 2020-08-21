"""
TOMOGRAPHIC IMAGE RECONSTRUCTION
We implement the SK-ROCK algorithm described in: "Accelerating
Proximal Markov Chain Monte Carlo by Using an Explicit Stabilized
Method", Marcelo Pereyra, Luis Vargas Mieles, and Konstantinos C.
Zygalakis, SIAM Journal on Imaging Sciences, Vol. 13, No. 2, 2020
Permalink: https://doi.org/10.1137/19M1283719 

@author: Luis Vargas Mieles
"""

# initialize the random number generator to make the results repeatable
# initialize the generator using a seed of 10
import random
random.seed(10)

import numpy as np
from skimage import data
from skimage.transform import resize
from functions.chambolle_prox_TV import chambolle_prox_TV, TVnorm
from functions.SKROCK import SKROCK
from functions.plot_results import plot_results
from functions.MRIfunctions import fft2c, ifft2c, LineMask, masked_FFT, masked_FFT_t
from tqdm import tqdm
import time

#%% Setup experiment
N=128 # image dimension
x = np.load('src/phantom.npy')
# x = resize(x, (N,N),preserve_range=True,anti_aliasing=False) # 128x128 dim

angles = 22
mask_temp = LineMask(angles,N)
mask = np.fft.fftshift(mask_temp)
del mask_temp
# A operator
A = lambda x: masked_FFT(x,mask)
# A transpose operator
AT = lambda x: np.real(masked_FFT_t(x,mask))
# A^T * A operator
ATA = lambda x: np.real(ifft2c(np.multiply(mask,fft2c(x))))

# generate the blurred and noisy observation 'y'
y = A(x)
sigma = 1e-2
sigma2 = sigma**2
y = y + sigma*(np.random.randn(y.size) + 1j*np.random.randn(y.size))

# Algorithm parameters
lambda_prox = 0.2*sigma2 # regularization parameter
alpha = 1e2 # hyperparameter of the prior

# Lipschitz Constants
Lf = 1/sigma2 # Lipschitz constant of the likelihood
Lg = 1/lambda_prox # Lipshcitz constant of the prior
Lfg = Lf + Lg # Lipschitz constant of the model

# Gradients, proximal and \log\pi trace generator function
proxG = lambda x: chambolle_prox_TV(x,alpha*lambda_prox,25)
ATy = AT(y)
gradF = lambda x: (ATA(x) - ATy)/sigma2 # gradient of the likelihood
gradG = lambda x: (x -proxG(x))/lambda_prox # gradient of the prior
gradU = lambda x: gradF(x) + gradG(x) # gradient of the model
logPi = lambda x: -(np.linalg.norm(y-A(x))**2)/(2*sigma2) -alpha*TVnorm(x)

# SK-ROCK PARAMETERS
# number of internal stages 's'
nStagesROCK = 10
# fraction of the maximum step-size allowed in SK-ROCK (0,1]
percDeltat = 0.8

nSamplesBurnIn = int(5e2) # number of samples to produce in the burn-in stage
nSamples = int(1e3) # number of samples to produce in the sampling stage
XkSKROCK = AT(y) # Initial condition
logPiTrace=np.zeros(nSamplesBurnIn+nSamples)
logPiTrace[0]=logPi(XkSKROCK)
# to save the mean of the samples from burn-in stage
meanSamples_fromBurnIn = XkSKROCK
# to save the evolution of the MSE from burn-in stage
mse_fromBurnIn=np.zeros(nSamplesBurnIn+nSamples)
mse_fromBurnIn[0]=np.square(np.subtract(meanSamples_fromBurnIn,x)).mean()
# to save the mean of the samples in the sampling stage
meanSamples = np.zeros([N,N])
#to save the evolution of the MSE in the sampling stage
mse=np.zeros(nSamples)
#-------------------------------------------------------------------------

print(' ')
print('BEGINNING OF THE SAMPLING')

#-------------------------------------------------------------------------

print('Burn-in stage...')
progressBar = tqdm(total=nSamplesBurnIn-1)
start_exec = time.time()
for i in range(1,nSamplesBurnIn):
    # produce a sample using SK-ROCK
    XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU)
    # save \log \pi trace of the new sample
    logPiTrace[i]=logPi(XkSKROCK)
    # mean
    meanSamples_fromBurnIn = (i/(i+1))*meanSamples_fromBurnIn \
        + (1/(i+1))*(XkSKROCK)
    # mse
    mse_fromBurnIn[i] = np.square(np.subtract(meanSamples_fromBurnIn,x)).mean()
    # update iteration progress bar
    progressBar.update(1)

print('\nEnd of burn-in stage')
progressBar.close()
#-------------------------------------------------------------------------
print('Sampling stage...') 
progressBar = tqdm(total=nSamples)  
for i in range(nSamples):
    # produce a sample using SK-ROCK
    XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU)
    # save \log \pi trace of the new sample
    logPiTrace[i+nSamplesBurnIn]=logPi(XkSKROCK)
    # mean from burn-in stage
    meanSamples_fromBurnIn = \
       ((i+nSamplesBurnIn)/(i+nSamplesBurnIn+1))*meanSamples_fromBurnIn \
       + (1/(i+nSamplesBurnIn+1))*XkSKROCK
    # mse from burn-in stabe
    mse_fromBurnIn[i+nSamplesBurnIn] = np.square(np.subtract(meanSamples_fromBurnIn,x)).mean()
    # mean from sampling stage
    meanSamples = (i/(i+1))*meanSamples + (1/(i+1))*XkSKROCK
    # mse from sampling stage
    mse[i] = np.square(np.subtract(meanSamples,x)).mean()
    # update iteration progress bar
    progressBar.update(1)

end_exec = time.time()
progressBar.close()
print('END OF THE SK-ROCK SAMPLING')
print('Execution time of the SK-ROCK sampling: '+str(end_exec-start_exec)+' sec')

# %% Plot of the results
plot_results(x,nStagesROCK,meanSamples,logPiTrace,mse,mask,sigma)