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

bool test_7_pts_jacobi_3d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);
  double hz = std::abs(y[1]-y[0]);

  //real_tensor V = random_function_double(NPTS, NPTS, NPTS, true, true, true);
  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, NPTS, NPTS, NPTS, true, true, true);

  wstKernel3D<double> jkernel = create_laplacian_jacobi_7p_3d(rho, hx, hy, hz);
  wstKernel3D<double> kernel = create_laplacian_7p_3d(hx, hy, hz);
  for (int iter = 0; iter < 20; iter++) {
    V = jkernel.apply(V);
    wstTensorT<double> rho2 = kernel.apply(V);
    wstTensorT<double> errorT = rho-rho2;
    double error = norm2(rho-rho2)*L/NPTS;
    printf("error: %15.8e\n", error);
  }
  return (true);
}

int main(int argc, char** argv)
{
  bool testResult = test_7_pts_jacobi_3d();
  if (testResult)
    printf("test_7_pts_jacobi_3d -- PASSED\n");
  else
    printf("test_7_pts_jacobi_3d -- FAILED\n");

  return 0;
}
