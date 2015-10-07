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
double v1(double alpha, double L, double x)
{
  return -alpha*(cos(2.0*PI*x/L) + 1.0);
}

inline
double v2(double L, double x)
{
  return cos(2.0*PI*x/L);
}

inline
double v2r(double L, double x)
{
  return -(4.0*PI*PI/L/L)*cos(2.0*PI*x/L);
}

bool test_7_pts_lap_1d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1), x, NPTS, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1), x, NPTS, true);  

  wstKernel1D<double> kernel = create_laplacian_7p_1d(hx);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;
  double error = norm2(rho-rho2)*L/NPTS;
  return (error < 1.e-10);
}

bool test_5_pts_lap_1d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1), x, NPTS, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1), x, NPTS, true);  

  wstKernel1D<double> kernel = create_laplacian_5p_1d(hx);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;

  double error = norm2(rho-rho2)*L/NPTS;
  return (error < 1.e-7);
}

bool test_3_pts_lap_1d()
{
  const double L = 5.0;
  const int NPTS = 150;

  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);

  wstTensorT<double> V;
  V.create(std::bind(v2, L, std::placeholders::_1), x, NPTS, true);
  wstTensorT<double> rho;
  rho.create(std::bind(v2r, L, std::placeholders::_1), x, NPTS, true);  

  wstKernel1D<double> kernel = create_laplacian_3p_1d(hx);
  wstTensorT<double> rho2 = kernel.apply(V);
  wstTensorT<double> errorT = rho-rho2;

  double error = norm2(rho-rho2)*L/NPTS;
  return (error < 1.e-4);
}

//bool lanczos_test1_1d()
//{
//  const double L = 5.0;
//  const NPTS = 50;
//  const alpha = 2.5;
//
//  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
//  double hx = std::abs(x[1]-x[0]);
//  
//  wstTensorT<double> V;
//  V.create(v1, x, NPTS, true);  
//  wstKernel1D kernel = create_laplacian_7p_1d(hx);
//
//  const auto tstart = std::chrono::system_clock::now();
//  //wstModel model(kernel, V);
//  //Lanczos<wstTensorT<double>,wstModel> lanczos(&model, 100);
//  //lanczos.run();
//  printf("creating lanzcos\n");
//  wstLanczos1D lanczos(V, hx, 40);
//  printf("created lanzcos\n");
//  lanczos.run();
//  printf("finished lanzcos\n");
//  const auto tstop = std::chrono::system_clock::now();
//  const std::chrono::duration<double> time_elapsed = tstop - tstart;
//
//  std::cout << "Elapsed time:  " << time_elapsed.count() << std::endl;
//}

int main(int argc, char** argv)
{
  bool testResult = false;
  testResult = test_3_pts_lap_1d();
  if (testResult)
    printf("test_3_pts_lap_1d -- PASSED\n");
  else
    printf("test_3_pts_lap_1d -- FAILED\n");
  testResult = test_5_pts_lap_1d();
  if (testResult)
    printf("test_5_pts_lap_1d -- PASSED\n");
  else
    printf("test_5_pts_lap_1d -- FAILED\n");
  testResult = test_7_pts_lap_1d();
  if (testResult)
    printf("test_7_pts_lap_1d -- PASSED\n");
  else
    printf("test_7_pts_lap_1d -- FAILED\n");

  return 0;
}
