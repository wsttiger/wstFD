#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <functional>
#include "lanczos.h"
#include "wstTensor.h"
#include "wstKernel.h"
#include "wstModel.h"
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

bool test_7_pts_lap_3d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);
  double hz = std::abs(y[1]-y[0]);

  wstTensor V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
  wstTensor rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);

  wstKernel3D kernel = create_laplacian_7p_3d(hx, hy, hz);
  wstTensor rho2 = kernel.apply(V);
  wstTensor errorT = rho-rho2;
  double error = (rho-rho2).norm2()*L/NPTS;
  return (error < 1.e-8);
}

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

  wstTensor V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
  wstTensor rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);

  wstKernel3D kernel = create_laplacian_5p_3d(hx, hy, hz);
  wstTensor rho2 = kernel.apply(V);
  wstTensor errorT = rho-rho2;
  double error = (rho-rho2).norm2()*L/NPTS;
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

  wstTensor V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
  wstTensor rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);

  wstKernel3D kernel = create_laplacian_3p_3d(hx, hy, hz);
  wstTensor rho2 = kernel.apply(V);
  wstTensor errorT = rho-rho2;
  double error = (rho-rho2).norm2()*L/NPTS;
  return (error < 1.e-1);
}


int main(int argc, char** argv)
{
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

  return 0;
}
