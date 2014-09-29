#include <cmath>
//#include "lanczos.h"
#include "wstTensor.h"
#include "wstKernel.h"

#define L 1.0
#define NPTS 100 
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

  wstTensor V;
  V.create(v2, x, y, z, NPTS, NPTS, NPTS, true, true, true);  
  wstTensor rho;
  rho.create(v2r, x, y, z, NPTS, NPTS, NPTS, true, true, true);  

  wstKernel3D kernel = create_laplacian_7p_3d(hx, hy, hz);
  wstTensor rho2 = kernel.apply(V);
  wstTensor errorT = rho-rho2;

  //print(rho, rho2, errorT);

  double error = (rho-rho2).norm2();

  printf("Error is: %15.8e\n", error);
}

void test1_2d()
{
  vector<double> x = linspace(-L/2, L/2, NPTS);
  vector<double> y = linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);

  wstTensor V;
  V.create(v2, x, y, NPTS, NPTS, true, true);  
  wstTensor rho;
  rho.create(v2r, x, y, NPTS, NPTS, true, true);  

  wstKernel2D kernel = create_laplacian_7p_2d(hx, hy);
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

  
  wstTensor V;
  V.create(v2, x, NPTS, true);
  wstTensor rho;
  rho.create(v2r, x, NPTS, true);  

  wstKernel1D kernel = create_laplacian_7p_1d(hx);
  wstTensor rho2 = kernel.apply(V);
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
