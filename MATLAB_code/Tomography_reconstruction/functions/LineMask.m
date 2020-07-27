function [M,Mh,mi,mhi] = LineMask(angles,dim)


thc = linspace(0, pi-pi/angles, angles);
%thc = linspace(pi/(2*L), pi-pi/(2*L), L);

M = zeros(dim);

% full mask
for ll = 1:angles

	if ((thc(ll) <= pi/4) || (thc(ll) > 3*pi/4))
		yr = round(tan(thc(ll))*(-dim/2+1:dim/2-1))+dim/2+1;
    	for nn = 1:dim-1
      	M(yr(nn),nn+1) = 1;
      end
  else 
		xc = round(cot(thc(ll))*(-dim/2+1:dim/2-1))+dim/2+1;
		for nn = 1:dim-1
			M(nn+1,xc(nn)) = 1;
		end
	end

end


% upper half plane mask (not including origin)
Mh = M;
Mh(dim/2+2:dim,:) = 0;
Mh(dim/2+1,dim/2+1:dim) = 0;


M = ifftshift(M);
mi = find(M);
Mh = ifftshift(Mh);
mhi = find(Mh);
end