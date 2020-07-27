function [f,px,py] = chambolle_prox_TV_stop(g, varargin)
%
% [f,px,py] = chambolle_prox_TV(g, varargin)
% Proximal  point operator for the TV regularizer 
%
% Uses the Chambolle's projection  algorithm:
%
%  A. chambolle_TV, "An Algorithm for Total Variation Minimization and
%  Applications", J. Math. Imaging Vis., vol. 20, pp. 89-97, 2004.
%
%
%%
% Optimization problem:  
%
%    arg min = (1/2) || y - x ||_2^2 + lambda TV(x)
%        x
%
%
%%  =========== Required inputs ====================
%
% 'g'       : noisy image (size X: ny * nx)
%
%%  =========== Optional inputs ====================
%  
% 'lambda'  : regularization  parameter according
%
% 'maxiter' :maximum number of iterations
%  
% 'tol'     : tol for the stopping criterion
%
% 'tau'     : algorithm parameter
%
% 'dualvars' : dual variables: used to start the algorithm closer
%              to the solution. 
%              Input format: [px, py] where px amd py have the same size 
%              of g
%            
%  
% =========== Outputs ====================
%
% f      : denoised image
%
% px,py  : dual variables 
% ===================================================
%
% Adapted by: Jose Bioucas-Dias, June 2009, (email: bioucas@lx.it.pt)
%
% from 
%   
%   Chambolle_Exact_TV(g, varargin)
%  
%    written by  Dr.Wen Youwei, email: wenyouwei@graduate.hku.hk
%
%
% Last revision: July  2009

%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 1
     error('Wrong number of required parameters');
end

%--------------------------------------------------------------
% Initialization
%--------------------------------------------------------------

px = zeros(size(g)); 
py = zeros(size(g));
cont = 1;       
k    = 0;  

%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------

tau = 0.249;
tol = 1e-3;  
lambda = 1;      
MaxIter = 10;
verbose = 0;


%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
for i = 1:2:(length(varargin)-1)
    switch upper(varargin{i})
        case 'LAMBDA'
            lambda   = varargin{i+1};
        case 'VERBOSE'
            verbose   = varargin{i+1};
        case 'TOL'
            tol      = varargin{i+1};
        case 'MAXITER'
            MaxIter  = varargin{i+1};
        case 'TAU'      
            tau      = varargin{i+1}; 
        case 'DUALVARS' 
             [M, N] = size(g);
             [Maux, Naux] = size(varargin{i+1});
             if M ~= Maux || Naux ~= 2*N
                     error('Wrong size of the dual variables');
             end
             px = varargin{i+1};
             py = px(:,M+1:end);
             px = px(:,1:M);
    end
end



%--------------------------------------------------------------
% Main body
%--------------------------------------------------------------
%%%%%%%%%%%%%%
% CHanged by MTF & JMB 
%
%%%%%%%%%%%%%%
while cont 
    k = k+1;
    % compute Divergence of (px, py)
    divp = DivergenceIm(px,py); 
    u = divp - g/lambda;
    % compute gradient
    [upx,upy] = GradientIm(u);
    tmp = sqrt(upx.^2 + upy.^2);  
    err = sum((-upx(:)+tmp(:).*px(:)).^2 + (-upy(:)+tmp(:).*py(:)).^2)^0.5;
    px = (px + tau * upx)./(1 + tau * tmp);
    py = (py + tau * upy)./(1 + tau * tmp);
    cont = ((k<MaxIter) & (err>tol));
%     if k == 1
%         cont = 1;
%         old_px = px;
%         old_py = py;
%     else
%         dx = px - old_px;
%         dy = py - old_py;
%         err = norm(sqrt(dx.^2 + dy.^2),inf);
%         cont = ((k<MaxIter) & (err>tol));
%         old_px = px;
%         old_py = py;
%     end
end
if verbose
   fprintf(1,' \n\n k TV = %g,    \n\n', k)
   fprintf(1,' \n\n err TV = %g,    \n\n', err)
end
f = g - lambda * DivergenceIm(px,py);
    

function divp = DivergenceIm(p1,p2)
z = p2(:,2:end-1) - p2(:,1:end-2);
v = [p2(:,1) z -p2(:,end)];

z = p1(2:end-1, :) - p1(1:end-2,:);
u = [p1(1,:); z;  -p1(end,:)];

divp = v + u;

function [dux, duy] = GradientIm(u)
z = u(2:end, :) - u(1:end-1,:);
dux = [z;  zeros(1,size(z,2))];

z = u(:,2:end) - u(:,1:end-1);
duy = [z zeros(size(z,1),1)];

