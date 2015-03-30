clear all;
close all;

L = 5.0;
npts = 22;
alpha = 0.5;
coeff = 1.0;

x = linspace(-L/2,L/2,npts+1);
x = x(1:npts);
[X,Y,Z] = meshgrid(x,x,x);

f = zeros(npts,npts,npts);
tol = 1e-10;
maxR = round(sqrt(-(log(tol)-log(coeff))/alpha)/L)
for iR = -maxR:maxR
    XR = X+iR*L;
    for jR = -maxR:maxR
        YR = Y+jR*L;
        for kR = -maxR:maxR
            ZR = Y+kR*L;
            f = f + exp(-alpha*(XR.^2 + YR.^2 + ZR.^2));
        end
    end
end

% setup k-grid
sidesz = (npts-1)/2;
kx = 2*pi*(-sidesz:sidesz)'./delx./npts;
[KX,KY,KZ] = meshgrid(kx,kx,kx);
K2 = KX.^2 + KY.^2 + KZ.^2;

Gf = fftshift(fftn(f))./(K2 + mu^2);
Gf = ifftn(ifftshift(Gf));

