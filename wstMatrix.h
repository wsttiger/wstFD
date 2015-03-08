#ifndef WSTMATRIX_H_
#define WSTMATRIX_H_

#include <vector>
#include <cassert>
#include <memory>
#include <complex>
//#include "wstUtils.h"

using std::vector;
using std::pair;

extern "C" void dsyev_(char *jobz, char *uplo, int *n, double *a,
           int *lda, double *w, double *work, int *lwork,
           int *info);

extern "C" void zheev_(char *jobz, char *uplo, int *n, std::complex<double> *a,
           int *lda, double *w, std::complex<double> *work, int *lwork,
           double *rwork, int *info);

struct wstMatrixSlice {
  wstMatrixSlice(int i0, int i1, int j0, int j1) 
   : i0(i0), i1(i1), j0(j0), j1(j1), ndim(2) {}
  
  wstMatrixSlice(int i0, int i1) 
   : i0(i0), i1(i1), j0(0), j1(0), ndim(1) {}

  int n0() const {
    return i1-i0+1;
  }

  int n1() const {
    return j1-j0+1;
  }

  int i0, i1, j0, j1, ndim;
};

template <typename T>
class wstMatrixT {
private:
  // the dimensions
  unsigned int _dim0, _dim1;
  // the data pointer
  std::shared_ptr<T> _sp;
  T* _p;
  // is the matrix currently allocated?
  bool _allocated;

  friend wstMatrixT copy(const wstMatrixT& t, bool empty = false) {
    wstMatrixT r;
    if (t._allocated) {
      r._dim0 = t._dim0;
      r._dim1 = t._dim1;


      int sz = r.size();
      r._p = new T[sz];
      r._sp = std::shared_ptr<T>(r._p, [](T *p) {delete[] p;});
      if (empty) {
        for (int i = 0; i < sz; i++) {
          r._p[i] = 0.0;
        }
      }
      else {
        for (int i = 0; i < sz; i++) {
          r._p[i] = t._p[i];
        }
      }
      r._allocated = t._allocated;
    }
    return r;
  }

public:
  wstMatrixT() 
    :  _dim0(0), _dim1(0), _p(0),_allocated(false) {}
  
  virtual ~wstMatrixT() {
    _allocated = false;
    _dim0 = 0;
    _dim1 = 0;
  }
  
  wstMatrixT& operator=(const wstMatrixT& t) {
    if (this != &t) {
      _dim0 = t._dim0;
      _dim1 = t._dim1;
      _p = t._p;
      _sp = t._sp;
      _allocated = t._allocated;
    }
    return *this;
  }

  wstMatrixT(const wstMatrixT& t) {
    (*this) = t;
  }

  void create(int d0, int d1) {
    // dims
    _dim0 = d0; _dim1 = d1;
    // allocation
    _p = new T[d0*d1];
    _sp = std::shared_ptr<T>(_p, [](T *p) {delete[] p;});
    _allocated = true;
  }

  void create(std::function<T (double, double)> f, const vector<double>& x, const vector<double>& y, 
              int d0, int d1) {
    create(d0, d1);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        _p[i*d1+j] = f(x[i], y[j]);
  }

  int size() const {
    return _dim0*_dim1;
  }

  int nrows() const {
    return _dim0; 
  }

  int ncols() const {
    return _dim1; 
  }

  // set all elements to zero
  void empty() {
    int sz = this->size();
    for (int i = 0; i < sz; i++) {
      _p[i] = T(0);
    }
  }

  // set all elements to value
  void value(T val) {
    int sz = this->size();
    for (int i = 0; i < sz; i++) {
      _p[i] = val;
    }
  }

//  void fillrandom() {
//    int sz = this->size();
//    for (int i = 0; i < sz; i++) {
//      int i1 = rand();
//      double t1 = (i1 % 100000000)/100000000.0;
//      _p[i] = t1;
//    }
//  }

//  void print() const {
//    int sz = this->size();
//    for (int i = 0; i < sz; i++)
//      printf("%15.8e\n", _p[i]);
//  }

  // pointer at a base position
  T* ptr() {
    return _p;
  }

  // pointer at a base position
  const T* ptr() const {
    return _p;
  }

  // pointer at a given (i,j) index
  T* ptr(int i0, int i1) {
    return &_p[i0*_dim1+i1];
  }

  // pointer at a given (i,j) index
  const T* ptr(int i0, int i1) const {
    return &_p[i0*_dim1+i0];
  }

  // index operator
  T& operator()(int i) {
    return _p[i];
  }

  // index operator
  T& operator()(int i) const {
    return _p[i];
  }

  // index operator
  T& operator()(int i0, int i1) {
    return _p[i0*_dim1+i1];
  }

  // index operator
  T& operator()(int i0, int i1) const {
    return _p[i0*_dim1+i1];
  }

  // perform inplace a*(*this) + b*t
  void gaxpy(const T& a, const wstMatrixT& t, const T& b) {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    for (int i = 0; i < sz1; i++) _p[i] = a*_p[i]+b*t._p[i];
  }

  // perform out-of-place a*(*this) + b*t
  wstMatrixT gaxpy_oop(const T& a, const wstMatrixT& t, const T& b) const {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    wstMatrixT r = copy(*this, true);
    for (int i = 0; i < sz1; i++) r._p[i] = a*_p[i]+b*t._p[i];
    return r;
  }

  // addition
  wstMatrixT operator+(const wstMatrixT& t) const {
    return gaxpy_oop(1.0, t, 1.0);
  }

  // subtraction
  wstMatrixT operator-(const wstMatrixT& t) const {
    return gaxpy_oop(1.0, t, -1.0);
  }

  // inplace scaling by a constant 
  void scale(T a) {
    int sz = this->size();
    for (int i = 0; i < sz; i++) _p[i] *= a;
  }

  // inner like a vector 
  T inner(const wstMatrixT& t) const {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    T rval = 0.0;
    for (int i = 0; i < sz1; i++) rval += _p[i]*t._p[i]; 
    return rval;
  }

  // two norm of the vector
  T norm2() const {
    int sz = this->size();
    T rval = 0.0;
    for (int i = 0; i < sz; i++) rval += _p[i]*_p[i]; 
    return std::sqrt(rval);
  }

  // normalize the values according to the two-norm of the vector
  void normalize() {
    T s = this->norm2();
    this->scale(1./s); 
  }

  // simple slicing
  wstMatrixT<T> operator()(const wstMatrixSlice& sl) {
    wstMatrixT<T> R;
    R.create(sl.n0(),sl.n1());
    T* p = ptr(sl.i0, sl.j0);
    for (int i = 0; i < sl.n0(); i++) {
      for (int j = 0; j < sl.n1(); j++) {
        R._p[i*R._dim1+j] = p[i*_dim1+j];
      }
    }
    return R;
  }

  // conversion from complex to real
  operator wstMatrixT<std::complex<T> > () const {
    wstMatrixT<std::complex<T> > Ac;
    Ac.create(nrows(), ncols());
    int sz = size();
    for (int i = 0; i < sz; i++) Ac(i) = std::complex<T>(_p[i],T(0));
    return Ac;
  }

  // extract a column vector from matrix
  wstMatrixT<T> col(int j) {
    wstMatrixT<T> R;
    int nr = nrows();
    R.create(nr,1);
    for (int i = 0; i < nr; i++) {
      R(i) = _p[i*_dim0+j];
    }
    return R;
  }

  // extract a contiguous group of column vectors from matrix
  std::vector<wstMatrixT<T> > cols(const wstMatrixSlice& sl) {
    assert(sl.ndim == 1);
    int nc = sl.n0();
    std::vector<wstMatrixT<T> > R(nc);
    for (int j = 0; j < nc; j++) {
      R[j] = col(j);
    }
    return R;
  }
};

template <typename Q>
wstMatrixT<Q> gaxpy(const Q& a, const wstMatrixT<Q>& T1, const Q& b, const wstMatrixT<Q>& T2) {
  return T1.gaxpy_oop(a, T2, b);
}

template <typename Q>
double inner(const wstMatrixT<Q>& t1, const wstMatrixT<Q>& t2) {
  return t1.inner(t2);
}

double norm2(const wstMatrixT<double>& t) {
  return t.norm2();
}

double norm2(const wstMatrixT<std::complex<double> >& t) {
  return std::abs(t.norm2());
}

void print(const wstMatrixT<double>& A) {
  int nr = A.nrows();
  int nc = A.ncols();
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nc; j++) {
      printf("%15.8e  ", A(i,j));
      //printf("%20.25e  ", A(i,j));
    }
    printf("\n");
  }
  printf("\n");
}

void print(const wstMatrixT<std::complex<double> >& A) {
  int nr = A.nrows();
  int nc = A.ncols();
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nc; j++) {
      printf("(%15.8e, %15.8e)  ", real(A(i,j)), imag(A(i,j)));
    }
    printf("\n");
  }
  printf("\n");
}

template <typename Q>
wstMatrixT<Q> operator*(const Q& s, const wstMatrixT<Q>& A) {
  wstMatrixT<Q> r = copy(A);
  r.scale(s);
  return r;
}

template <typename Q>
wstMatrixT<std::complex<Q> > operator*(const Q& s, const wstMatrixT<std::complex<Q> >& A) {
  wstMatrixT<std::complex<Q> > r = copy(A);
  r.scale(std::complex<Q>(s));
  return r;
}

// create zeros matrix
template <typename Q>
wstMatrixT<Q> zeros(int d0, int d1) {
  wstMatrixT<Q> r;
  r.create(d0, d1);
  r.empty();  
  return r;
}

// create an constant matrix
template <typename Q>
wstMatrixT<Q> constant(int d0, int d1, double val) {
  wstMatrixT<Q> r;
  r.create(d0, d1);
  r.value(val);  
  return r;
}

// create an constant matrix
template <typename Q>
wstMatrixT<Q> ones(int d0, int d1) {
  return constant(d0, d1, T(1.0));
}

template <typename Q>
wstMatrixT<Q> from_vector(int d0, int d1, const std::vector<Q>& v) {
  wstMatrixT<Q> r;
  r.create(d0, d1);
  for (int i = 0; i < d0; i++)
    for (int j = 0; j < d1; j++)
      r(i,j) = v[i*d1+j];
  return r;
}

template <typename Q> 
wstMatrixT<Q> real(const wstMatrixT<std::complex<Q> > Ac) {
  wstMatrixT<Q> Ar;
  Ar.create(Ac.nrows(), Ac.ncols()); 
  int sz = Ar.size();
  for (int i = 0; i < sz; i++) Ar(i) = std::real(Ac(i));
  return Ar;
}

template <typename Q> 
wstMatrixT<Q> imag(const wstMatrixT<std::complex<Q> > Ac) {
  wstMatrixT<Q> Ar;
  Ar.create(Ac.nrows(), Ac.ncols()); 
  int sz = Ar.size();
  for (int i = 0; i < sz; i++) Ar(i) = std::imag(Ac(i));
  return Ar;
}

template <typename Q> 
wstMatrixT<Q> abs(const wstMatrixT<std::complex<Q> > Ac) {
  wstMatrixT<Q> Ar;
  Ar.create(Ac.nrows(), Ac.ncols()); 
  int sz = Ar.size();
  for (int i = 0; i < sz; i++) Ar(i) = std::abs(Ac(i));
  return Ar;
}

template <typename Q>
wstMatrixT<Q> transpose(const wstMatrixT<Q> A) {
  int nr = A.nrows();
  int nc = A.ncols();
  wstMatrixT<Q> r = zeros<Q>(nc, nr);
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nc; j++) {
      r(j, i) = A(i, j);
    }
  }
  return r;
}

wstMatrixT<double> ctranspose(const wstMatrixT<double> A) {
  return transpose(A);
}

wstMatrixT<std::complex<double> > ctranspose(const wstMatrixT<std::complex<double> > A) {
  int nr = A.nrows();
  int nc = A.ncols();
  wstMatrixT<std::complex<double> > r = zeros<std::complex<double> >(nc, nr);
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nc; j++) {
      r(j, i) = std::conj(A(i, j));
    }
  }
  return r;
}

template<typename Q>
wstMatrixT<std::complex<Q> > make_complex(const wstMatrixT<Q>& rp, const wstMatrixT<Q>& ip) {
  int nr1 = rp.nrows();
  int nc1 = rp.ncols();
  int nr2 = ip.nrows();
  int nc2 = ip.ncols();
  assert(nr1 == nr2);
  assert(nc1 == nc2);
  wstMatrixT<std::complex<Q> > r = zeros<std::complex<Q> >(nr1, nc2);
  for (int i = 0; i < nr1; i++) {
    for (int j = 0; j < nc1; j++) {
      r(i,j) = std::complex<Q>(rp(i,j),ip(i,j)); 
    }
  }
  return r;
}

// For a symmetric real matrix
std::pair< wstMatrixT<double>, wstMatrixT<double> > diag(const wstMatrixT<double>& mat) {
  assert(mat.nrows() == mat.ncols());
  int n = mat.nrows();
  char jobz = 'V';
  char uplo = 'U';
  int info;
  int lda = n;
  int lwork = 3*n-1;
  double *work = new double[lwork];
  wstMatrixT<double> ev = copy(mat, false);
  wstMatrixT<double> e = zeros<double>(n,1);
  double* evptr = ev.ptr();
  double* eptr = e.ptr();

  dsyev_(&jobz, &uplo, &n, evptr, &lda, eptr, work, &lwork, &info);

  if (info != 0) {
    printf("[[Error:]] lapack::dsyev failed --- info = %d\n\n", info);
  }
  delete work;

  return std::pair< wstMatrixT<double>, wstMatrixT<double> >(transpose(e), transpose(ev));
}

// For a hermitian complex matrix
std::pair< wstMatrixT<double>, wstMatrixT<std::complex<double> > > diag(const wstMatrixT<std::complex<double> >& mat) {
  assert(mat.nrows() == mat.ncols());
  int n = mat.nrows();
  char jobz = 'V';
  char uplo = 'U';
  int info;
  int lda = n;
  int lwork = 3*n-1;
  std::complex<double>* work = new std::complex<double>[lwork];
  double* rwork = new double[3*n-2];
  wstMatrixT<std::complex<double> > ev = copy(mat, false);
  wstMatrixT<double> e = zeros<double>(n,1);
  std::complex<double>* evptr = ev.ptr();
  double* eptr = e.ptr();

  //for (int i = 0; i < n*n; i++) printf("%15.8f\n", evptr[i]);

  zheev_(&jobz, &uplo, &n, evptr, &lda, eptr, work, &lwork, rwork, &info);

  if (info != 0) {
    printf("[[Error:]] lapack::dsyev failed --- info = %d\n\n", info);
  }
  delete work;
  delete rwork;

  return std::pair< wstMatrixT<double>, wstMatrixT<std::complex<double> > >(transpose(e), transpose(ev));
}

#endif
