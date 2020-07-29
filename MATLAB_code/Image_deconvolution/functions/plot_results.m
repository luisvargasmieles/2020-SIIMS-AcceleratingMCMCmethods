function plot_results(Y,X,nStagesSKROCK,meanSamples,logPiTrace,mseValues,...
                     slowComponent)

close all

% Dock the generated figures
set(0,'DefaultFigureWindowStyle','docked');

% Initial configurations for the plots
set(0, 'DefaultAxesFontSize',20);
set(0, 'DefaultLineLineWidth', 3);
set(0, 'defaultTextInterpreter', 'latex');

% 1. PLOT ORIGINAL, OBSERVATIONS AND ESTIMATES

    % Plot the original image
    figure(1);
    imagesc(X,[0 255]);
    title('Original image');
    axis equal; axis off;colormap('gray');

    % Plot the noisy observation
    figure(2);
    imagesc(Y, [0 255]); hold on
    axis equal; axis off;colormap('gray');
    title('Blurred and noisy observation');
 
    % Plot the MMSE of x
    figure(3);
    imagesc(meanSamples,[0 255]);
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
    title('Evolution of MSE in stationarity','Interpreter','latex');
    
    % Plot the autocorrelation function of the slowest component
    figure(6);
    lag  = 20; 
    [autocorSKROCK,lags] = autocorr(slowComponent, lag);
    stem(lags,autocorSKROCK,'filled','^','LineWidth',2,'MarkerSize',10);
    legend({'SK-ROCK $(s=10)$'},'interpreter','latex');
    xlabel('lag','interpreter','latex');
    ylabel('ACF','interpreter','latex');
    xlim([0 lag]);
    title('ACF','Interpreter','latex');

end