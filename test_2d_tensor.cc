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
double v1(double alpha, double L, double x, double y)
{
  return -alpha*(cos(2.0*PI*x/L)*cos(2.0*PI*y/L) + 1.0);
}

inline
double v2(double L, double x, double y)
{
  return cos(2.0*PI*x/L)*cos(2.0*PI*y/L);
}

inline
double v2r(double L, double x, double y)
{
  return -(8.0*PI*PI/L/L)*cos(2.0*PI*x/L)*cos(2.0*PI*y/L);
}

bool test_7_pts_lap_2d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2), x, y, NPTS, NPTS, true, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2), x, y, NPTS, NPTS, true, true);

  wstKernel2D<double> kernel = create_laplacian_7p_2d(hx, hy);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;
  double error = (rho-rho2).norm2()*L/NPTS;
  return (error < 1.e-10);
}

bool test_5_pts_lap_2d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2), x, y, NPTS, NPTS, true, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2), x, y, NPTS, NPTS, true, true);

  wstKernel2D<double> kernel = create_laplacian_5p_2d(hx, hy);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;
  double error = (rho-rho2).norm2()*L/NPTS;
  return (error < 1.e-6);
}

bool test_3_pts_lap_2d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2), x, y, NPTS, NPTS, true, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2), x, y, NPTS, NPTS, true, true);

  wstKernel2D<double> kernel = create_laplacian_3p_2d(hx, hy);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;
  double error = (rho-rho2).norm2()*L/NPTS;
  return (error < 1.e-2);
}


int main(int argc, char** argv)
{
  bool testResult = false;
  testResult = test_3_pts_lap_2d();
  if (testResult)
    printf("test_3_pts_lap_2d -- PASSED\n");
  else
    printf("test_3_pts_lap_2d -- FAILED\n");
  testResult = test_5_pts_lap_2d();
  if (testResult)
    printf("test_5_pts_lap_2d -- PASSED\n");
  else
    printf("test_5_pts_lap_2d -- FAILED\n");
  testResult = test_7_pts_lap_2d();
  if (testResult)
    printf("test_7_pts_lap_2d -- PASSED\n");
  else
    printf("test_7_pts_lap_2d -- FAILED\n");

  return 0;
}
