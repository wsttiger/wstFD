#include <cmath>
//#include "lanczos.h"
#include "wstTensor.h"

#define L 1.0
#define NPTS 50 
#define alpha 2.5
#define PI 3.141592653589793238

vector<double> linspace(double start, double end, int npts)
{
   double delx = (end-start)/(npts-1);
   vector<double> t(npts);
   double s1 = start;
   for (int i = 0; i < npts; i++)
   {
     t[i] = s1+i*delx;
   }
   return t;
}

inline
double v1(double x)
{
  return -alpha*(cos(2.0*PI*x/L) + 1.0);
}

inline
double v1(double x, double y)
{
  return -alpha*(cos(2.0*PI*x/L)*cos(2.0*PI*y/L) + 1.0);
}

inline
double v1(double x, double y, double z)
{
  return -alpha*(cos(2.0*PI*x/L)*cos(2.0*PI*y/L)*cos(2.0*PI*z/L) + 1.0);
}

inline
double v2(double x)
{
  return cos(2.0*PI*x/L);
}

inline
double v2(double x, double y)
{
  return cos(2.0*PI*x/L)*cos(2.0*PI*y/L);
}

inline
double v2(double x, double y, double z)
{
  return cos(2.0*PI*x/L)*cos(2.0*PI*y/L)*cos(2.0*PI*z/L);
}

inline
double v2r(double x)
{
  return -(4.0*PI*PI/L/L)*cos(2.0*PI*x/L);
}

inline
double v2r(double x, double y)
{
  return -(8.0*PI*PI/L/L)*cos(2.0*PI*x/L)*cos(2.0*PI*y/L);
}

inline
double v2r(double x, double y, double z)
{
  return -(12.0*PI*PI/L/L)*cos(2.0*PI*x/L)*cos(2.0*PI*y/L)*cos(2.0*PI*z/L);
}

void test1_3d()
{
  vector<double> x = linspace(-L/2, L/2, NPTS);
  vector<double> y = linspace(-L/2, L/2, NPTS);
  vector<double> z = linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);
  double hz = std::abs(z[1]-z[0]);

  //for (unsigned int i = 0; i < x.size(); i++)
  //  printf("%15.8f\n", x[i]);

  // Create the 5-point laplacian stencil
  int offsets5p[5] = {-2, -1, 0, 1, 2};
  double coeffs5p[5] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/16.0};
  vector<int> xoffset5p(15,0); 
  vector<int> yoffset5p(15,0); 
  vector<int> zoffset5p(15,0);
  vector<double> vcoeffs5p(15,0.0);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    xoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i];
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    yoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i];
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    zoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i];
  }

  wstTensor V;
  V.create(v2, x, y, z, NPTS, NPTS, NPTS, true, true, true);  
  wstTensor rho;
  rho.create(v2r, x, y, z, NPTS, NPTS, NPTS, true, true, true);  

  //V.print();
  //V.print(); printf("\n\n"); rho.print(); printf("\n\n");

  print(V, rho);

  wstKernel3D kernel;
  kernel.create(xoffset5p, yoffset5p, zoffset5p, vcoeffs5p);
  printf("kernel created\n");
  wstTensor rho2 = kernel.apply(V);
  printf("kernel applied\n");
  wstTensor errorT = rho-rho2;

  double error = (rho-rho2).norm2();

  printf("Error is: %15.8e\n", error);
}

void test1_2d()
{
  vector<double> x = linspace(-L/2, L/2, NPTS);
  vector<double> y = linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);

  //for (unsigned int i = 0; i < x.size(); i++)
  //  printf("%15.8f\n", x[i]);

  // Create the 3-point laplacian stencil
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(6,0); 
  vector<int> yoffset3p(6,0); 
  vector<double> vcoeffs3p(6,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hx/hx;
  }
  p += 3;
  for (int i = 0; i < 5; i++) {
    yoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hy/hy;
  }
  for (int i = 0; i < 6; i++)
    printf("%d    %d    %10.5f\n", xoffset3p[i], yoffset3p[i], vcoeffs3p[i]);
  
  wstTensor V;
  V.create(v2, x, y, NPTS, NPTS, true, true);  
  wstTensor rho;
  rho.create(v2r, x, y, NPTS, NPTS, true, true);  

  //V.print(); printf("\n\n"); rho.print(); printf("\n\n");

  wstKernel2D kernel;
  kernel.create(xoffset3p, yoffset3p, vcoeffs3p);
  wstTensor rho2 = kernel.apply(V);
  wstTensor errorT = rho-rho2;

  double error = (rho-rho2).norm2();

  print(rho, rho2, errorT);

  printf("Error is: %15.8e\n", error);
}

void test1_1d()

{
  vector<double> x = linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);

  //for (unsigned int i = 0; i < x.size(); i++)
  //  printf("%15.8f\n", x[i]);

  // Create the 3-point laplacian stencil
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(3,0); 
  vector<double> vcoeffs3p(3,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hx/hx;
  }
  for (int i = 0; i < 3; i++)
    printf("%d    %10.5f\n", xoffset3p[i], vcoeffs3p[i]);
  
  wstTensor V;
  V.create(v2, x, NPTS, true);
  printf("V created\n");
  wstTensor rho;
  rho.create(v2r, x, NPTS, true);  
  printf("rho created\n");

  //V.print(); printf("\n\n"); rho.print(); printf("\n\n");

  wstKernel1D kernel;
  kernel.create(xoffset3p, vcoeffs3p);
  printf("1D kernel created\n");
  wstTensor rho2 = kernel.apply(V);
  printf("applied 1D kernel\n");
  wstTensor errorT = rho-rho2;

  double error = (rho-rho2).norm2();

  print(rho, rho2, errorT);

  printf("Error is: %15.8e\n", error);
}

int main(int argc, char** argv)
{
  test1_3d(); 
  return 0;
}
