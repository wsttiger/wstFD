#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <functional>
#include "lanczos.h"
#include "wstMatrix.h"
#include "wstUtils.h"

#define MARRAYINT = {0,1,2,3,4,5}
#define MARRAY1 = {0.0,1.0,2.0,3.0,4.0,5.0}
#define MARRAY2 = {-7.0,3.2,-21.7,0.8,1.1,-2.6}
#define MARRAYP12 = {-7.0,4.2,-19.7,3.8,5.1,2.4}
#define MARRAYM12 = {7.0,-2.2,23.7,2.2,2.9,7.6}
#define MARRAYMM12 = {-80.4,-105.9,-8.2,-8.9}
#define MARRAYSQ = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0}
#define MARRAYSQCOL1 = {1.0,5.0,9.0,13.0}
#define MARRAYSQCOL2 = {2.0,6.0,10.0,14.0}
#define MARRAYSQCOL3 = {2.0,3.0,6.0,7.0,10.0,11.0,14.0,15.0}
#define MARRAY1SLICE1 = {1.0,2.0,4.0,5.0}
#define MARRAY1SLICE2 = {0.0,1.0,3.0,4.0}

bool test_matrix_create()
{
  bool passed = true; 
  {
    std::vector<int> v MARRAYINT;
    wstMatrixT<int> r = from_vector(2,3,v);
    passed = passed && (r.nrows() == 2);
    passed = passed && (r.ncols() == 3);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        passed == passed && ((i*3+j) == r(i,j));
      }
    }
  }

  {
    wstMatrixT<double> r = zeros<double>(4,5);
    passed = passed && (r.nrows() == 4);
    passed = passed && (r.ncols() == 5);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 5; j++) {
        passed = passed && ( std::abs(r(i,j)) < 1e-16);
      }
    }
  }

  {
    double value = -3.442211;
    wstMatrixT<double> r = constant<double>(6,4, value);
    passed = passed && (r.nrows() == 6);
    passed = passed && (r.ncols() == 4);
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 4; j++) {
        passed = passed && ( std::abs(r(i,j)-value) < 1e-16);
      }
    }
  }

  return passed;
}

bool test_matrix_math()
{
  bool passed = true;
  {
    std::vector<double> av MARRAY1;
    wstMatrixT<double> A = from_vector(2,3,av);
    std::vector<double> bv MARRAY2;
    wstMatrixT<double> B = from_vector(2,3,bv);
    std::vector<double> cpv MARRAYP12;
    wstMatrixT<double> Cp = from_vector(2,3,cpv);
    std::vector<double> cmv MARRAYM12;
    wstMatrixT<double> Cm = from_vector(2,3,cmv);

    wstMatrixT<double> Cp2 = A + B;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        passed = passed && ( std::abs(Cp(i,j)-Cp2(i,j)) < 1e-16);
      }
    }
    wstMatrixT<double> Cm2 = A - B;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        passed = passed && ( std::abs(Cm(i,j)-Cm2(i,j)) < 1e-16);
      }
    }
  }
  return passed;
}

bool test_matrix_cols()
{
  bool passed = true;
  {
    std::vector<double> av MARRAYSQ;
    wstMatrixT<double> A = from_vector(4,4,av);
    std::vector<double> ac MARRAYSQCOL1;
    wstMatrixT<double> Ac = from_vector(4,1,ac);
    passed = passed && norm2(Ac - A.col(0)) < 1e-16;
  }
  {
    std::vector<double> av MARRAYSQ;
    wstMatrixT<double> A = from_vector(4,4,av);
    std::vector<double> ac MARRAYSQCOL2;
    wstMatrixT<double> Ac = from_vector(4,1,ac);
    passed = passed && norm2(Ac - A.col(1)) < 1e-16;
  }
  {
    std::vector<double> av MARRAYSQ;
    wstMatrixT<double> A = from_vector(4,4,av);
    std::vector<double> ac0 MARRAYSQCOL1;
    wstMatrixT<double> Ac0 = from_vector(4,1,ac0);
    std::vector<double> ac1 MARRAYSQCOL2;
    wstMatrixT<double> Ac1 = from_vector(4,1,ac1);
    std::vector<wstMatrixT<double> > Acols = A.cols(wstSlice(0,1));
    passed = passed && norm2(Ac0 - Acols[0]) < 1e-16;
    passed = passed && norm2(Ac1 - Acols[1]) < 1e-16;
  }
  return passed;
}

bool test_matrix_slice()
{
  bool passed = true;
  {
    std::vector<double> av MARRAY1;
    wstMatrixT<double> A = from_vector(2,3,av);
    std::vector<double> avs MARRAY1SLICE1;
    wstMatrixT<double> As = from_vector(2,2,avs);
    passed = passed && norm2(As - A(wstSlice(0,1,1,2))) < 1e-16;
  }
  {
    std::vector<double> av MARRAY1;
    wstMatrixT<double> A = from_vector(2,3,av);
    std::vector<double> avs MARRAY1SLICE2;
    wstMatrixT<double> As = from_vector(2,2,avs);
    passed = passed && norm2(As - A(wstSlice(0,1,0,1))) < 1e-16;
  }
  return passed;
}

bool test_matrix_conv()
{
  bool passed = true;
  {
    std::vector<double> ar MARRAYSQ;
    wstMatrixT<double> Ar = from_vector(4,4,ar);
    std::vector<std::complex<double> > ac MARRAYSQ;
    wstMatrixT<std::complex<double> > Ac = from_vector(4,4,ac);
    passed = passed && norm2(Ac - (wstMatrixT<std::complex<double> >)(Ar)) < 1e-16;
  }
  return passed;
}

bool test_matrix_diag()
{
  bool passed = true;
  {
    std::vector<double> av MARRAYSQ;
    wstMatrixT<double> A = from_vector(4,4,av);
    A = 0.5*(A + transpose(A));
    std::pair<wstMatrixT<double>, wstMatrixT<double> > result = diag(A);
    wstMatrixT<double> eigs = result.first;
    passed = passed && (std::abs(eigs(0) + 3.34698994937580307507) < 1e-12);
    passed = passed && (std::abs(eigs(1)) < 1e-12);
    passed = passed && (std::abs(eigs(2)) < 1e-12);
    passed = passed && (std::abs(eigs(3) - 37.3469899493758106246) < 1e-12);
  }
  {
    std::vector<std::complex<double> > av MARRAYSQ;
    wstMatrixT<std::complex<double> > A = from_vector(4,4,av);
    A = 0.5*(A + transpose(A));
    std::pair<wstMatrixT<double>, wstMatrixT<std::complex<double> > > result = diag(A);
    wstMatrixT<double> eigs = result.first;
    passed = passed && (std::abs(eigs(0) + 3.34698994937580307507) < 1e-12);
    passed = passed && (std::abs(eigs(1)) < 1e-12);
    passed = passed && (std::abs(eigs(2)) < 1e-12);
    passed = passed && (std::abs(eigs(3) - 37.3469899493758106246) < 1e-12);
  }
  {
    std::vector<double> av MARRAYSQ;
    wstMatrixT<double> Ar = from_vector(4,4,av);
    wstMatrixT<double> Ai = from_vector(4,4,av);
    Ai.scale(0.1);
    wstMatrixT<std::complex<double> > A = make_complex(Ar,Ai);
    A = 0.5*(A + ctranspose(A));
    std::pair<wstMatrixT<double>, wstMatrixT<std::complex<double> > > result = diag(A);
    wstMatrixT<double> eigs = result.first;
    passed = passed && (std::abs(eigs(0) + 3.3580450927882576905858514) < 1e-12);
    passed = passed && (std::abs(eigs(1)) < 1e-12);
    passed = passed && (std::abs(eigs(2)) < 1e-12);
    passed = passed && (std::abs(eigs(3) - 37.358045092788259466942691) < 1e-12);

    // WSTHORNTON
    wstMatrixT<std::complex<double> > evecs = result.second;
    print(A);
    print(eigs);
    print(evecs.col(0));
    print(evecs.col(3));
  }

  return passed;
  
}

int main(int argc, char** argv)
{
  bool testResult = false;
  testResult = test_matrix_create();
  if (testResult)
    printf("test_matrix_create -- PASSED\n");
  else
    printf("test_matrix_create -- FAILED\n");
  testResult = test_matrix_math();
  if (testResult)
    printf("test_matrix_math -- PASSED\n");
  else
    printf("test_matrix_math -- FAILED\n");
  testResult = test_matrix_cols();
  if (testResult)
    printf("test_matrix_cols -- PASSED\n");
  else
    printf("test_matrix_cols -- FAILED\n");
  testResult = test_matrix_conv();
  if (testResult)
    printf("test_matrix_conv -- PASSED\n");
  else
    printf("test_matrix_conv -- FAILED\n");
  testResult = test_matrix_slice();
  if (testResult)
    printf("test_matrix_slice -- PASSED\n");
  else
    printf("test_matrix_slice -- FAILED\n");
  testResult = test_matrix_diag();
  if (testResult)
    printf("test_matrix_diag -- PASSED\n");
  else
    printf("test_matrix_diag -- FAILED\n");
  return 0;
}
