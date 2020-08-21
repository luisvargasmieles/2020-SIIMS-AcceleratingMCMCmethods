import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython import get_ipython

def plot_results(X,nStagesSKROCK,meanSamples,logPiTrace,mseValues,mask,sigma):

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    'exec(%matplotlib inline)'

    # 1. PLOT ORIGINAL, OBSERVATIONS AND ESTIMATES

    # Plot the original image
    plt.figure(figsize=(16,8))
    plt.subplot(231)
    plt.gray()
    plt.imshow(X)
    plt.title("Original image")

    # Plot the noisy observation - amplitude fourier coefficients
    tomObs = np.multiply(mask,np.real(np.log(np.fft.fft2(X + \
        sigma*(np.random.randn(np.size(X,0),np.size(X,0)))))))
    tomObs[tomObs==0]=np.amin(tomObs)
    plt.subplot(232)
    plt.gray()
    plt.imshow(tomObs)
    plt.title("Tomographic Observation $y$ (amp. Fourier coeff. - log-scale)")
    plt.colorbar()

    # Plot the MMSE of x
    plt.subplot(233)
    plt.gray()
    plt.imshow(meanSamples)
    plt.title("MMSE estimate of $x$")
    
    # Plot the \log\pi trace of the samples
    plt.subplot(234)
    plt.plot(np.arange(1,nStagesSKROCK*len(logPiTrace),nStagesSKROCK),logPiTrace)
    # plt.xscale('log')
    plt.xlabel('number of gradient evaluations')
    plt.ylabel('$\log\pi(X_n)$')
    plt.title('$\log\pi$ trace of $X_n$')
    
    # % Plot the evolution of the MSE in stationarity
    plt.subplot(235)
    plt.plot(np.arange(1,nStagesSKROCK*len(mseValues),nStagesSKROCK),mseValues)
    plt.xlabel('number of gradient evaluations')
    plt.ylabel('MSE')
    plt.title('Evolution of MSE in stationarity')

    # to show all the plots
    plt.show()