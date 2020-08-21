%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                   TOMOGRAPHIC IMAGE RECONSTRUCTION                     %
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
N = 128;
x = phantom(N);

% Define forward operator A and adjoint AT for MRI observation model
% (Fourier transform + mask)
angles = 22;
[mask_temp,~,~,~] = LineMask(angles,N);
mask = fftshift(mask_temp);
A = @(x)  masked_FFT(x,mask);
AT = @(x) real(masked_FFT_t(x,mask));
ATA = @(x) real(ifft2c(mask.*fft2c(x)));

% generate 'y'
Ax = A(x);
sigma = 1e-2; %%% noise level
sigma2 = sigma^2;
y = Ax + sigma*(randn(size(Ax)) +1i*randn(size(Ax)));

%%% Algorithm parameters
lambda = 0.2*sigma2; %%% regularization parameter
alpha = 100; %%% hyperparameter of the prior

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
nStagesROCK = 10;
%%% fraction of the maximum step-size allowed in SK-ROCK (0,1]
percDeltat = 0.8;

nSamplesBurnIn = 5e2; % number of samples to produce in the burn-in stage
nSamples = 1e3; % number of samples to produce in the sampling stage
XkSKROCK = AT(y); % Initial condition
logPiTrace=zeros(1,nSamplesBurnIn+nSamples);
logPiTrace(1)=logPi(XkSKROCK);
%%% to save the mean of the samples from burn-in stage
meanSamples_fromBurnIn = zeros(N);
%%% to save the evolution of the MSE from burn-in stage
mse_fromBurnIn=zeros(1,nSamplesBurnIn+nSamples);
%%% to save the mean of the samples in the sampling stage
meanSamples = zeros(N);
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
plot_RESULT(x,sigma,mask,nStagesROCK,meanSamples,logPiTrace,mse);     
%-------------------------------------------------------------------------%