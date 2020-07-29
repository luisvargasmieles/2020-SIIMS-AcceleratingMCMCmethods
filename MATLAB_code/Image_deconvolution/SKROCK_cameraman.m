%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%          IMAGE DEBLURRING EXPERIMENT - CAMERAMAN TEST IMAGE            %
%     We implement the SK-ROCK algorithm described in: "Accelerating     %
%    Proximal Markov Chain Monte Carlo by Using an Explicit Stabilized   %
%    Method", Marcelo Pereyra, Luis Vargas Mieles, and Konstantinos C.   %
%    Zygalakis, SIAM Journal on Imaging Sciences, Vol. 13, No. 2, 2020   %
%                Permalink: https://doi.org/10.1137/19M1283719           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

addpath([pwd,'/functions']);

%%% initialize the random number generator to make the results repeatable
rng('default');
%%% initialize the generator using a seed of 1
rng(1);

%% Setup experiment
x=imread('cameraman.tif'); % Cameraman image to be used for the experiment
x = double(x);
[nRows, nColumns] = size(x); % size of the image

%%%% function handle for uniform blur operator (acts on the image
%%%% coefficients)
dim_blur = 5; %%% dimension of the blur operator
h = fspecial('average',dim_blur);
H = zeros(nRows,nColumns);
H(1:dim_blur,1:dim_blur) = h;
clear h dim_blur;

%%% operators A and A'
H_FFT = fft2(H);
HC_FFT = conj(H_FFT);

A = @(x) real(ifft2(H_FFT.*fft2(x))); % A operator
AT = @(x) real(ifft2(HC_FFT.*fft2((x)))); % A transpose operator
ATA = @(x) real(ifft2((HC_FFT.*H_FFT).*fft2((x)))); % AtA operator

% generate 'y'
y = A(x);
BSNRdb = 42; % we will use this noise level
sigma = sqrt(var(y(:)) / 10^(BSNRdb/10));
sigma2 = sigma^2;
y = y + sigma*randn(nRows, nColumns); % we generate the observation 'y'

%%% Algorithm parameters
lambda = sigma2; %%% regularization parameter
alpha = 0.01/sigma2; %%% hyperparameter of the prior

% Lipschitz Constants
Lf = 1/sigma2; %%% Lipschitz constant of the likelihood
Lg = 1/lambda; %%% Lipshcitz constant of the prior
Lfg = Lf + Lg; %%% Lipschitz constant of the model

% Gradients, proximal and \log\pi trace generator function
proxG = @(x) chambolle_prox_TV_stop(x, 'lambda',alpha*lambda,'maxiter',25);
ATy = AT(y);
gradF = @(x) (ATA(x) - ATy)/sigma2; %%% gradient of the likelihood
gradG = @(x) (x -proxG(x))/lambda; %%% gradient of the prior
gradU = @(x) gradF(x) + gradG(x); %%% gradient of the model
logPi = @(x) -(norm(y-A(x),'fro')^2)/(2*sigma2) -alpha*TVnorm(x);

% SK-ROCK PARAMETERS
%%% number of internal stages 's'
nStagesROCK = 15;
%%% fraction of the maximum step-size allowed in SK-ROCK (0,1]
percDeltat = 0.8;

nSamplesBurnIn = 6e2; % number of samples to produce in the burn-in stage
nSamples = 1e3; % number of samples to produce in the sampling stage
XkSKROCK = y; % Initial condition
logPiTrace=zeros(1,nSamplesBurnIn+nSamples);
logPiTrace(1)=logPi(XkSKROCK);
fastFComponent=zeros(1,nSamples); % to save the fastest comp.
slowFComponent=zeros(1,nSamples); % to save the slowest comp.
%%% to save the mean of the samples from burn-in stage
meanSamples_fromBurnIn = XkSKROCK;
%%% to save the evolution of the MSE from burn-in stage
mse_fromBurnIn=zeros(1,nSamplesBurnIn+nSamples);
mse_fromBurnIn(1)=immse(meanSamples_fromBurnIn,x);
%%% to save the mean of the samples in the sampling stage
meanSamples = zeros(nRows,nColumns);
%%% to save the evolution of the MSE in the sampling stage
mse=zeros(1,nSamples);

%-------------------------------------------------------------------------

disp(' ');
disp('BEGINNING OF THE SAMPLING');

%-------------------------------------------------------------------------

progressBar = waitbar(0,'Sampling in progress...');

%-------------------------------------------------------------------------
disp('Burn-in stage...');
tic;
for i=2:nSamplesBurnIn
    %%% produce a sample using SK-ROCK
    XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU);
    % save \log \pi trace of the new sample
    logPiTrace(i)=logPi(XkSKROCK);
    %%% mean
    meanSamples_fromBurnIn = ((i-1)/i)*meanSamples_fromBurnIn ...
        + (1/i)*(XkSKROCK);
    %%% mse
    mse_fromBurnIn(i) = immse(meanSamples_fromBurnIn,x);
    %%% update iteration progress bar
    waitbar(i/(nSamplesBurnIn+nSamples));
end
disp('End of burn-in stage');

disp('Sampling stage...');    
for i=1:nSamples
    %%% produce a sample using SK-ROCK
    XkSKROCK=SKROCK(XkSKROCK,Lfg,nStagesROCK,percDeltat,gradU);
    % save \log \pi trace of the new sample
    logPiTrace(i+nSamplesBurnIn)=logPi(XkSKROCK);
    %%% mean from burn-in stage
    meanSamples_fromBurnIn = ...
       ((i+nSamplesBurnIn-1)/(i+nSamplesBurnIn))*meanSamples_fromBurnIn ...
       + (1/(i+nSamplesBurnIn))*XkSKROCK;
    %%% mse from burn-in stabe
    mse_fromBurnIn(i+nSamplesBurnIn) = immse(meanSamples_fromBurnIn,x);
    %%% mean from sampling stage
    meanSamples = ((i-1)/i)*meanSamples + (1/i)*XkSKROCK;
    %%% mse from sampling stage
    mse(i) = immse(meanSamples,x);
    %%% to save the slowest and fastest component
    fftSample=fft2(XkSKROCK);
    slowFComponent(i)=fftSample(11,51);
    fastFComponent(i)=fftSample(5,3);
    %%% update iteration progress bar
    waitbar((i+nSamplesBurnIn)/(nSamplesBurnIn+nSamples));
end

%-------------------------------------------------------------------------

t_end = toc;
close(progressBar);
disp('END OF THE SK-ROCK SAMPLING');
disp(['Execution time of the SK-ROCK sampling: ' num2str(t_end) ' sec']);

%-------------------------------------------------------------------------%
% Display MSE associated to the MMSE estimator of x
disp(['MSE (x): ' num2str(mse(end-1))]);
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% Plot the results                                                        
plot_results(y,x,nStagesROCK,meanSamples,logPiTrace,mse,slowFComponent);     
%-------------------------------------------------------------------------%