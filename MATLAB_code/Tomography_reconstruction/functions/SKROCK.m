%-------------------------------------------------------------------------%
%                          SK-ROCK SAMPLING METHOD                        %
%-------------------------------------------------------------------------%

function X_new = SKROCK(X,Lipschitz_U,nStages,dt_perc,gradU)

%-------------------------------------------------------------------------%
% This function samples the distribution \pi(x) = exp(-U(x)) thanks to a 
% proximal MCMC algorithm called SK-ROCK (see "Accelerating Proximal Markov
% Chain Monte Carlo by Using an Explicit Stabilized Method", Marcelo
% Pereyra, Luis Vargas Mieles, and Konstantinos C. Zygalakis, SIAM Journal
% on Imaging Sciences, 2020).

    % INPUTS:
        % X: current MCMC iterate (2D-array)
        % Lipschitz_U: user-defined lipschitz constant of the model
        % nStages: the number of internal stages of the SK-ROCK iterations
        % dt_perc: the fraction of the max. stepsize to be used
        % gradU: function that computes the gradient of the potential U
        
    % OUTPUT:
        % X_new: new value for X (2D-array).
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% PRE-PROCESSING
imgDim = size(X);

%%% SK-ROCK parameters
%%% First kind Chebyshev function
T_s = @(s,x) cosh(s*acosh(x));

%%% First derivative Chebyshev polynomial first kind
T_prime_s = @(s,x) s*sinh(s*acosh(x))/sqrt(x^2 -1);

%%% computing SK-ROCK stepsize given a number of stages
%%% and parameters needed in the algorithm
eta=0.05;
denNStag=(2-(4/3)*eta);
rhoSKROCK = ((nStages - 0.5)^2)*denNStag -1.5; % stiffness ratio
dtSKROCK = dt_perc*rhoSKROCK/Lipschitz_U; %%% step-size
w0=1 + eta/(nStages^2); %%% parameter \omega_0
w1=T_s(nStages,w0)/T_prime_s(nStages,w0); %%% parameter \omega_1
mu1 = w1/w0; %%% parameter \mu_1
nu1=nStages*w1/2; %%% parameter \nu_1
kappa1=nStages*(w1/w0); %%% parameter \kappa_1

%%% Sampling the variable X (SKROCK)
Q=sqrt(2*dtSKROCK)*randn(imgDim); %%% diffusion term
%%% SKROCK
%%% SKROCK first internal iteration (s=1)
Xts= X - mu1*dtSKROCK*gradU(X + nu1*Q) +kappa1*Q;
XtsMinus2 = X;
for js = 2:nStages % s=2,...,nStages SK-ROCK internal iterations
    XprevSMinus2 = Xts;
    mu=2*w1*T_s(js-1,w0)/T_s(js,w0); %%% parameter \mu_js
    nu=2*w0*T_s(js-1,w0)/T_s(js,w0); %%% parameter \nu_js
    kappa=1-nu; %%% %%% parameter \kappa_js
    Xts=-mu*dtSKROCK*gradU(Xts) + nu*Xts + kappa*XtsMinus2;
    XtsMinus2=XprevSMinus2;
end
X_new=Xts; %%% new sample produced by the SK-ROCK algorithm
end