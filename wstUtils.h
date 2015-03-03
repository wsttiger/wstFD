#ifndef WSTUTILS_H_
#define WSTUTILS_H_

#include <cmath>
#include <vector>

extern "C" void dsyev_(char *jobz, char *uplo, int *n, double *a,
           int *lda, double *w, double *work, int *lwork,
           int *info);

using std::vector;

class wstUtils {
public:

  static long fact(int n) {
    long prod = 1;
    for (long i = 1; i <= n; i++) {
      prod *= i;
    }
    return prod;
  }

  static int nchoosek(int n, int k) {
    long rval = 1;
    for (long i = k+1; i <= n; i++) {
      rval *= i;
    }
    return (rval/fact(n-k));
  }

  static vector<double> daxpy(const double& a, const vector<double>& x,
                       const double& b, const vector<double>& y) {
    vector<double> rv(x.size(),0.0);
    for (unsigned int i = 0; i < rv.size(); i++) {
      rv[i] = a*x[i]+b*y[i];
    }
    return rv;
  }

  static void daxpy_inplace(const double& a, vector<double>& x,
                     const double& b, const vector<double>& y) {
    for (unsigned int i = 0; i < x.size(); i++) {
      x[i] = a*x[i]+b*y[i];
    }
  }

  static double dotvec(const vector<double>& v1, const vector<double>& v2) {
    double rval = 0.0;
    for (unsigned int iv = 0; iv < v1.size(); iv++) {
      rval += v1[iv]*v2[iv];
    }
    return rval;
  }
 
  static vector<double> linspace(double start, double end, int npts, bool periodic = true) {
     int npts2 = (periodic) ? npts+1 : npts;
     double delx = (end-start)/(npts2-1);
     vector<double> t(npts);
     double s1 = start;
     for (int i = 0; i < npts; i++) {
       t[i] = s1+i*delx;
     }
     return t;
  }

  static double norm2(const vector<double>& v) {
    double rnorm = 0.0;
    for (unsigned int iv = 0; iv < v.size(); iv++) {
      rnorm += v[iv]*v[iv];
    }
    return std::sqrt(rnorm);
  }
  
  static vector<double> matrix_vector_mult(vector<double> mat, vector<double> v) {
    unsigned int szmat = mat.size();
    unsigned int sz2 = v.size();
    unsigned int sz1 = szmat / sz1;
    vector<double> rv(sz2,0.0);
    for (unsigned int i = 0; i < sz2; i++) {
      for (unsigned int j = 0; j < sz1; j++) {
        rv[i] += mat[i*sz1+j];
      }
    }
    return rv;
  }
  
  static void normalize(vector<double>& v) {
    double rnorm = 0.0;
    for (unsigned int iv = 0; iv < v.size(); iv++) {
      rnorm += v[iv]*v[iv];
    }
    rnorm = std::sqrt(rnorm);
    for (unsigned int iv = 0; iv < v.size(); iv++) {
      v[iv] /= rnorm;
    }
  }
  
  static void print_vector(const vector<double>& v) {
    for (unsigned int i = 0; i < v.size(); i++) {
      printf("  %15.8f\n",v[i]);
    }
  }
  
  static void print_matrix(const vector<double>& mat, int m, int n) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%12.6f",mat[i*n+j]);
      }
      printf("\n");
    }
  }

  static vector<double> get_col_from_matrix(const vector<double> mat, int col, int m, int n) {
    vector<double> rv(m,0.0);
    for (int i = 0; i < m; i++) {
      rv[i] = mat[i*n+col];
    }
    return rv;
  }

  static int periodic_index(int idx, int size) {
    if (idx >= size) 
      return periodic_index(idx-size, size);
    else if (idx < 0) 
      return periodic_index(idx+size, size);
    else
      return idx;
  }

  static void diag_matrix(const vector<double>& mat, int n,
                   vector<double>& e, vector <double>& evec) {
    char jobz = 'V';
    char uplo = 'U';
    int info;
    double* a = new double[n*n];
    int lda = n;
    int lwork = 3*n-1;
    double *work = new double[lwork];
    double *et = new double[n];
    for (int i = 0; i < n*n; i++) a[i] = mat[i];
  
    dsyev_(&jobz, &uplo, &n, a, &lda, et, work, &lwork, &info);
  
    if (info != 0) {
      printf("[[Error:]] lapack::dsyev failed --- info = %d\n\n", info);
    }
    else {
      for (int i = 0; i < n; i++) {
        e[i] = et[i];
      }
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          evec[j*n+i] = a[i*n+j];
        }
      }
    }
    delete a;
    delete work;
    delete et;
  }

  static vector<double> random_vector(const int& nsize) {
    vector<double> v(nsize,0.0);
    for (int i = 0; i < nsize; i++) {
      int i1 = rand();
      double t1 = (i1 % 100000)/100000.0;
      v[i] = t1;
    }
    normalize(v);
    return v;
  }

  static bool is_equals(const vector<double>& a, const vector<double>& b, double tol = 1e-10) {
    for (unsigned int i = 0; i < a.size(); i++) {
      if (std::abs(a[i]-b[i]) > tol) return false;
    }
    return true;
  }

};

#endif
