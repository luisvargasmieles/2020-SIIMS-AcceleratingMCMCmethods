import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython import get_ipython

# Function to plot the autocorrelation of the slowest component
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

def plot_results(Y,X,nStagesSKROCK,meanSamples,logPiTrace,mseValues,\
                     slowComponent):

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

    # Plot the noisy observation
    plt.subplot(232)
    plt.gray()
    plt.imshow(Y)
    plt.title("Blurred and noisy observation")

    # Plot the MMSE of x
    plt.subplot(233)
    plt.gray()
    plt.imshow(meanSamples)
    plt.title("MMSE estimate of $x$")
    
    # Plot the \log\pi trace of the samples
    plt.subplot(234)
    plt.plot(np.arange(1,nStagesSKROCK*len(logPiTrace),nStagesSKROCK),logPiTrace)
    plt.xscale('log')
    plt.xlabel('number of gradient evaluations')
    plt.ylabel('$\log\pi(X_n)$')
    plt.title('$\log\pi$ trace of $X_n$')
    
    # % Plot the evolution of the MSE in stationarity
    plt.subplot(235)
    plt.plot(np.arange(1,nStagesSKROCK*len(mseValues),nStagesSKROCK),mseValues)
    plt.xlabel('number of gradient evaluations')
    plt.ylabel('MSE')
    plt.title('Evolution of MSE in stationarity')
    
    # % Plot the autocorrelation function of the slowest component
    plt.subplot(236)
    lag  = int(2e2)
    autocorSKROCK = autocorr(slowComponent)
    autocorSKROCK = autocorSKROCK / autocorSKROCK.max()
    xVal = np.arange(0,lag+1,np.round(lag/20))
    xVal = xVal.astype(int)
    plt.stem(xVal,autocorSKROCK[xVal],markerfmt='^',use_line_collection=True)
    plt.xlabel('lag')
    plt.ylabel('ACF')
    plt.xlim(0,lag)
    plt.title('ACF')

    # to show all the plots
    plt.show()