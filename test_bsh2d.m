clear all;
close all;

L = 5.0;
npts = 42;
alpha = 2;
coeff = 1.0;
mu = -0.7;

x = linspace(-L/2,L/2,npts);
% x = linspace(-L/2,L/2,npts+1);
% x = x(1:npts);
delx = x(2)-x(1);
[X,Y] = meshgrid(x,x);

f = zeros(npts,npts);
f = exp(-alpha*(X.^2 + Y.^2));
tol = 1e-10;
% maxR = round(sqrt(-(log(tol)-log(coeff))/alpha)/L)
% for iR = -maxR:maxR
%     XR = X+iR*L;
%     for jR = -maxR:maxR
%         YR = Y+jR*L;
%         f = f + exp(-alpha*(XR.^2 + YR.^2));
%     end
% end

% setup k-grid
sidesz = (npts-1)/2;
kx = 2*pi*(-sidesz:sidesz)'./delx./npts;
[KX,KY] = meshgrid(kx,kx);
K2 = KX.^2 + KY.^2;

Gf = fftshift(fft2(f));
Gf = ifftshift(Gf./(K2 + mu^2));
Gf = ifft2(Gf);

