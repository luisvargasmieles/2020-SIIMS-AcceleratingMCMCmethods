function y = cshift(x,L)

N = length(x);
if (size(x) ~= N)
    error('\nInput x must be one dimensional.\n');
end

y = zeros(size(x));

if (L == 0)
    y = x;
    return;
end

if (L>0)
    y(L+1:end) = x(1:N-L);
    y(1:L) = x(N-L+1:N);
else
    L = -L;
    y(1:N-L) = x(L+1:N);
    y(N-L+1:N) = x(1:L);
end
