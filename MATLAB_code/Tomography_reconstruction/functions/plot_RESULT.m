function plot_RESULT(X,sigma,mask,nStagesSKROCK,meanSamples,logPiTrace,...
    mseValues)

close all;

% Dock the generated figures
set(0,'DefaultFigureWindowStyle','docked');

% Initial configurations for the plots
set(0, 'DefaultAxesFontSize',20);
set(0, 'DefaultLineLineWidth', 3);
set(0, 'defaultTextInterpreter', 'latex');

% 1. PLOT ORIGINAL, OBSERVATIONS AND ESTIMATES

    % Plot the original image
    figure(1);
    imagesc(X);
    title('Original image');
    axis equal; axis off;colormap('gray');

    % Plot the noisy and incomplete observation
    figure(2);
    tomographyIC = mask.*real(log(fft2(X + sigma*(randn(size(X))))));
    tomographyIC(tomographyIC==0)=min(min(tomographyIC));
    imagesc(tomographyIC); colormap(gray); colorbar;
    caxis([min(min(tomographyIC)) max(max(tomographyIC))])
    axis equal; axis off; colormap('gray');
    title('Tomographic observation (Amp. Fourier coeff. - log scale)');
 
    % Plot the MMSE of x
    figure(3);
    imagesc(meanSamples);
    axis equal; axis off;colormap('gray');
    title('MMSE estimate of x');
    
    % Plot the \log\pi trace of the samples
    figure(4);
    semilogx(1:nStagesSKROCK:nStagesSKROCK*length(logPiTrace),logPiTrace);
    xlabel('number of gradient evaluations');
    ylabel('$\log\pi(X_n)$','interpreter','latex');
    title('$\log\pi$ trace of $X_n$','Interpreter','latex');
    
    % Plot the evolution of the MSE in stationarity
    figure(5);
    semilogx(1:nStagesSKROCK:nStagesSKROCK*length(mseValues),mseValues);
    xlabel('number of gradient evaluations');
    ylabel('MSE','interpreter','latex');
    title('Evolution of MSE in stationarity');
end