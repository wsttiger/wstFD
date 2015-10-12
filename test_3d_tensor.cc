#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <functional>
#include "lanczos.h"
#include "wstTensor.h"
#include "wstKernel.h"
#include "wstUtils.h"

#define PI 3.141592653589793238

inline
double v1(double alpha, double L, double x, double y, double z)
{
  return -alpha*(cos(2.0*PI*x/L)*cos(2.0*PI*y/L)*cos(2.0*PI*z/L) + 1.0);
}

inline
double v2(double L, double x, double y, double z)
{
  return cos(2.0*PI*x/L)*cos(2.0*PI*y/L)*cos(2.0*PI*z/L);
}

inline
double v2r(double L, double x, double y, double z)
{
  return -(12.0*PI*PI/L/L)*cos(2.0*PI*x/L)*cos(2.0*PI*y/L)*cos(2.0*PI*z/L);
}

inline
double gaussian(double coeff, double expnt, double x, double y, double z) {
  return coeff*std::exp(-expnt*(x*x + y*y + z*z));
}

inline
double pgaussian(double coeff, double expnt, double L, double x, double y, double z) {
  double tol = 1e-10;
  double rmax = std::sqrt((std::log(coeff)-std::log(tol))/expnt);
  int maxR = round(rmax/L);
  double s = 0.0;
  for (int iR = -maxR; iR <= maxR; iR++) {
    double xR2 = (x+iR*L)*(x+iR*L);
    for (int jR = -maxR; jR <= maxR; jR++) {
      double yR2 = (y+jR*L)*(y+jR*L);
      for (int kR = -maxR; kR <= maxR; kR++) {
        double zR2 = (z+kR*L)*(z+kR*L);
        s += coeff*std::exp(-expnt*(xR2+yR2+zR2)); 
      }
    }
  }
  return s;
}

bool test_7_pts_lap_3d()
{
  const double L = 5.0;
  const int NPTS = 350;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);
  double hz = std::abs(y[1]-y[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);

  wstKernel3D<double> kernel = create_laplacian_7p_3d(hx, hy, hz);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;
  double error = norm2(rho-rho2)*L/NPTS;
  printf("err: %15.5e\n", error);
  return (error < 1.e-8);
}

//bool test_7_pts_lap_3d_complex()
//{
//  const double L = 5.0;
//  const int NPTS = 150;
//
//  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
//  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
//  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);
//  double hx = std::abs(x[1]-x[0]);
//  double hy = std::abs(y[1]-y[0]);
//  double hz = std::abs(y[1]-y[0]);
//
//  wstTensorT< std::complex<double> > V;
//  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
//  wstTensorT< std::complex<double> > rho;
//  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
//
//  wstKernel3D< std::complex<double> > kernel = create_laplacian_7p_3d(hx, hy, hz);
//  wstTensorT< std::complex<double> > rho2 = kernel.apply(V);
//  wstTensorT< std::complex<double> > errorT = rho-rho2;
//  double error = (rho-rho2).norm2()*L/NPTS;
//  return (error < 1.e-8);
//}

bool test_5_pts_lap_3d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);
  double hz = std::abs(y[1]-y[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);

  wstKernel3D<double> kernel = create_laplacian_5p_3d(hx, hy, hz);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;
  double error = norm2(rho-rho2)*L/NPTS;
  printf("err: %15.5e\n", error);
  return (error < 1.e-5);
}

bool test_3_pts_lap_3d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);
  double hz = std::abs(z[1]-z[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);

  wstKernel3D<double> kernel = create_laplacian_3p_3d(hx, hy, hz);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;
  double error = norm2(rho-rho2)*L/NPTS;
  printf("err: %15.5e\n", error);
  return (error < 1.e-1);
}

bool test_fftshift()
{
  bool passed = true;
  {
    wstTensorT<double> f;
    f.create(8,true); 
    for (int i = 0; i < 8; i++) f(i) = i+1;
    wstTensorT<double> f2;
    f2.create(8,true); 
    for (int i = 0; i < 4; i++) f2(i) = i+5;
    for (int i = 4; i < 8; i++) f2(i) = i-3;
    fftshift(f);
    passed = passed && std::abs(norm2(f2-f)) < 1e-16;
  }

  {
    wstTensorT<double> f;
    f.create(4,4); 
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        f(i,j) = i*4+j+1;
      }
    }
    wstTensorT<double> f2;
    f2.create(4,4); 
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        f2(i,j) = i*4+j+11;
      }
    }
    for (int i = 2; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        f2(i,j) = (i-2)*4+j+3;
      }
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 2; j < 4; j++) {
        f2(i,j) = (i+2)*4+j-1;
      }
    }
    for (int i = 2; i < 4; i++) {
      for (int j = 2; j < 4; j++) {
        f2(i,j) = (i-2)*4+j-1;
      }
    }
    fftshift(f);
    passed = passed && std::abs(norm2(f2-f)) < 1e-16;
  }

  {
    wstTensorT<double> f;
    f.create(2,4,4); 
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
          f(i,j,k) = i*16+j*4+k+1;
        }
      }
    }
    print3d(f);
    fftshift(f);
    print3d(f);
  }
  return passed;
}

void test_3d_fft() {
  const double L = 5.0;
  const int NPTS = 22;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);

  wstTensorT<double> G;
  G.create(std::bind(gaussian, 1.0, 0.5, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS);
  wstTensorT<std::complex<double> > FG = fft(G);
  fftshift(FG);
  print(real(FG));
  //print(G);
}

void test_bsh() {
  const double L = 5.0;
  const int NPTS = 22;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS, true);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS, true);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS, true);

  wstTensorT<double> G;
  G.create(std::bind(pgaussian, 1.0, 0.5, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS);
  wstTensorT<std::complex<double> > FG = fft(G);
  //fftshift(FG);
  //print(real(FG));
  print(G);
}

int main(int argc, char** argv)
{
//  test_bsh();
//  //test_3d_fft();
//  assert(false);

  bool testResult = false;
  testResult = test_3_pts_lap_3d();
  if (testResult)
    printf("test_3_pts_lap_3d -- PASSED\n");
  else
    printf("test_3_pts_lap_3d -- FAILED\n");
  testResult = test_5_pts_lap_3d();
  if (testResult)
    printf("test_5_pts_lap_3d -- PASSED\n");
  else
    printf("test_5_pts_lap_3d -- FAILED\n");
  testResult = test_7_pts_lap_3d();
  if (testResult)
    printf("test_7_pts_lap_3d -- PASSED\n");
  else
    printf("test_7_pts_lap_3d -- FAILED\n");
  testResult = test_fftshift();
  if (testResult)
    printf("test_fftshift -- PASSED\n");
  else
    printf("test_fftshift -- FAILED\n");

  return 0;
}
