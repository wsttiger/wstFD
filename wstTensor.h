#ifndef WSTTENSOR_H_
#define WSTTENSOR_H_

#include <fftw3.h>
#include <vector>
#include <cassert>
#include <memory>
#include "wstUtils.h"

#include "wstMatrix.h"

using std::vector;
using std::pair;

template <typename T>
class wstTensorT {
private:
  // number of dimensions
  int _ndim;
  // array holding the dimensions
  std::vector<int> _dims;
  // are the dimensions periodic?
  std::vector<bool> _bc;
  // the data pointer
  std::shared_ptr<T> _sp;
  T* _p;
  // is the tensor currently allocated?
  bool _allocated;

  friend wstTensorT copy(const wstTensorT& t, bool empty = false) {
    wstTensorT r;
    if (t._allocated) {
      r._ndim = t._ndim; 
      r._dims = t._dims;
      r._bc = t._bc;
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

  friend wstTensorT copy_and_fill(const wstTensorT& t, wstMatrixT<T> A ) {
    assert(t.size() == A.size());
    wstTensorT r;
    if (t._allocated) {
      r._ndim = t._ndim; 
      r._dims = t._dims;
      r._bc = t._bc;
      int sz = r.size();
      r._p = new double[sz];
      r._sp = std::shared_ptr<double>(r._p, [](double *p) {delete[] p;});
      for (int i = 0; i < sz; i++) r._p[i] = A(i);
      r._allocated = t._allocated;
    }
    return r;
  }

public:
  wstTensorT() 
    : _ndim(0), _dims(std::vector<int>(0)), _bc(std::vector<bool>(false)), _p(0),_allocated(false) {}
  
  virtual ~wstTensorT() {
    _allocated = false;
    _ndim = 0;
  }
  
  wstTensorT& operator=(const wstTensorT& t) {
    if (this != &t) {
      _ndim = t._ndim;
      _dims = t._dims;
      _bc = t._bc;
      _p = t._p;
      _sp = t._sp;
      _allocated = t._allocated;
    }
    return *this;
  }

  wstTensorT(const wstTensorT& t) {
    (*this) = t;
  }

  template <typename Q>
  void create(const wstTensorT<Q>& t) {
    _ndim = t.ndim();
    _dims = std::vector<int>(_ndim,0);
    for (int d = 0; d < _ndim; d++) _dims[d] = t.dim(d);
    _bc = std::vector<bool>(_ndim,false);
    for (int d = 0; d < _ndim; d++) _bc[d] = t.bc(d);
    int sz = size();
    _p = new T[sz];
    _sp = std::shared_ptr<T>(_p, [](T *p) {delete[] p;});
    _allocated = true;
  }

  void create(int d0, bool periodic = false) {
    // dims
    _ndim = 1;
    _dims = vector<int>(1,d0);
    // boundary conditions
    _bc = std::vector<bool>(1,periodic);
    // allocation
    _p = new T[d0];
    _sp = std::shared_ptr<T>(_p, [](T *p) {delete[] p;});
    _allocated = true;
  }

  void create(std::function<T (double)> f, const vector<double>& x, int d0, bool periodic = false) {
    create(d0, periodic);
    for (int i = 0; i < d0; i++) _p[i] = f(x[i]);
  }

  void create(const wstMatrixT<T>& A, int d0, bool periodic = false) {
    assert(A.size() == d0);
    create(d0, periodic);
    for (int i = 0; i < d0; i++) _p[i] = A(i);
  }

  void create(int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
    // dims
    _ndim = 2;
    _dims = std::vector<int>(_ndim,0);
    _dims[0] = d0; _dims[1] = d1;
    // boundary conditions
    _bc = std::vector<bool>(_ndim,false);
    _bc[0] = periodic0; _bc[1] = periodic1;
    // allocation
    _p = new T[d0*d1];
    _sp = std::shared_ptr<T>(_p, [](T *p) {delete[] p;});
    _allocated = true;
  }

  void create(std::function<T (double, double)> f, const vector<double>& x, const vector<double>& y, 
              int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
    create(d0, d1, periodic0, periodic1);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        _p[i*d1+j] = f(x[i], y[j]);
  }

  void create(const wstMatrixT<T>& A, int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
    assert(A.size() == d0*d1);
    create(d0, d1, periodic0, periodic1);
    for (int i = 0; i < d0; i++) _p[i] = A(i);
  }

  void create(int d0, int d1, int d2, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
    // dims
    _ndim = 3;
    _dims = std::vector<int>(_ndim,0);
    _dims[0] = d0; _dims[1] = d1; _dims[2] = d2;
    // boundary conditions
    _bc = std::vector<bool>(_ndim,false);
    _bc[0] = periodic0; _bc[1] = periodic1; _bc[2] = periodic2;
    // allocation
    _p = new T[d0*d1*d2];
    _sp = std::shared_ptr<T>(_p, [](T *p) {delete[] p;});
    _allocated = true;
  }

  void create(std::function<T (double, double, double)> f, 
              const vector<double>& x, const vector<double>& y, const vector<double>& z, 
              int d0, int d1, int d2, 
              bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
    create(d0, d1, d2, periodic0, periodic1, periodic2);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        for (int k = 0; k < d2; k++)
          _p[i*d1*d2+j*d2+k] = f(x[i], y[j], z[k]);
  }

  void create(const wstMatrixT<T>& A, int d0, int d1, int d2, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
    assert(A.size() == d0*d1*d2);
    create(d0, d1, d2, periodic0, periodic1, periodic2);
    for (int i = 0; i < d0; i++) _p[i] = A(i);
  }

  int size() const {
    int sz = 1;
    for (int i = 0; i < _ndim; i++) sz *= _dims[i];
    return sz;
  }

  void empty() {
    int sz = this->size();
    for (int i = 0; i < sz; i++) {
      _p[i] = T(0);
    }
  }

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

  int ndim() const {return _ndim;}

  int dim(int i0) const {
    assert(i0 < _ndim);
    return _dims[i0];
  }

  bool bc(int i0) const {
    assert(i0 < _ndim);
    return _bc[i0];
  }
//  void print() const {
//    int sz = this->size();
//    for (int i = 0; i < sz; i++)
//      printf("%15.8e\n", _p[i]);
//  }

  T* ptr() {
    return _p;
  }

  const T* ptr() const {
    return _p;
  }

  T& operator[](int i0) {
    //assert(i0 >= 0 && i0 < _nsize);
    return _p[i0];
  }

  const T& operator[](int i0) const {
    //assert(i0 >= 0 && i0 < _nsize);
    return _p[i0];
  }

  T& operator()(int i0) {
    assert(_ndim == 1);
    int idx0 = (_bc[0]) ? wstUtils::periodic_index(i0, _dims[0]) : i0;
    return _p[idx0];
  }

  const T& operator()(int i0) const {
    assert(_ndim == 1);
    int idx0 = (_bc[0]) ? wstUtils::periodic_index(i0, _dims[0]) : i0;
    return _p[idx0];
  }

  T& operator()(int i0, int i1) {
    assert(_ndim == 2);
    int idx0 = (_bc[0]) ? wstUtils::periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? wstUtils::periodic_index(i1, _dims[1]) : i1;
    return _p[idx0*_dims[1]+idx1];
  }

  T& operator()(int i0, int i1) const {
    assert(_ndim == 2);
    int idx0 = (_bc[0]) ? wstUtils::periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? wstUtils::periodic_index(i1, _dims[1]) : i1;
    return _p[idx0*_dims[1]+idx1];
  }

  T& operator()(int i0, int i1, int i2) {
    assert(_ndim == 3);
    int idx0 = (_bc[0]) ? wstUtils::periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? wstUtils::periodic_index(i1, _dims[1]) : i1;
    int idx2 = (_bc[2]) ? wstUtils::periodic_index(i2, _dims[2]) : i2;
    return _p[idx0*_dims[1]*_dims[2]+idx1*_dims[2]+idx2];
  }

  T& operator()(int i0, int i1, int i2) const {
    assert(_ndim == 3);
    int idx0 = (_bc[0]) ? wstUtils::periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? wstUtils::periodic_index(i1, _dims[1]) : i1;
    int idx2 = (_bc[2]) ? wstUtils::periodic_index(i2, _dims[2]) : i2;
    return _p[idx0*_dims[1]*_dims[2]+idx1*_dims[2]+idx2];
  }

  // perform inplace a*(*this) + b*t
  void gaxpy(const T& a, const wstTensorT& t, const T& b) {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    for (int i = 0; i < sz1; i++) _p[i] = a*_p[i]+b*t._p[i];
  }

  // perform out-of-place a*(*this) + b*t
  wstTensorT gaxpy_oop(const T& a, const wstTensorT& t, const T& b) const {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    wstTensorT r = copy(*this, true);
    for (int i = 0; i < sz1; i++) r._p[i] = a*_p[i]+b*t._p[i];
    return r;
  }

  wstTensorT operator+(const wstTensorT& t) const {
    return gaxpy_oop(1.0, t, 1.0);
  }

  wstTensorT operator-(const wstTensorT& t) const {
    return gaxpy_oop(1.0, t, -1.0);
  }
 
  void scale(T a) {
    int sz = this->size();
    for (int i = 0; i < sz; i++) _p[i] *= a;
  }
 
//  T inner(const wstTensorT& t) const {
//    int sz1 = this->size();
//    int sz2 = t.size();
//    assert(sz1 == sz2);
//    T rval = 0.0;
//    for (int i = 0; i < sz1; i++) rval += _p[i]*t._p[i]; 
//    return rval;
//  }

//  T norm2() const {
//    int sz = this->size();
//    T rval = 0.0;
//    for (int i = 0; i < sz; i++) rval += _p[i]*_p[i]; 
//    return std::sqrt(rval);
//  }

//  void normalize() {
//    T s = this->norm2();
//    this->scale(1./s); 
//  }

  // conversion from complex to real
  operator wstTensorT<std::complex<T> > () const {
    wstTensorT<std::complex<T> > r;
    r.create(*this);
    int sz = size();
    for (int i = 0; i < sz; i++) r[i] = _p[i];
    return r;
  }

};

// typedefs
typedef wstTensorT<double> double_tensor;
typedef wstTensorT<std::complex<double> > complex_tensor;

template <typename Q>
wstTensorT<Q> operator*(const Q& s, const wstTensorT<Q>& A) {
  wstTensorT<Q> r = copy(A);
  r.scale(s);
  return r;
}

template <typename Q>
wstTensorT<Q> gaxpy(const Q& a, const wstTensorT<Q>& T1, const Q& b, const wstTensorT<Q>& T2) {
  return T1.gaxpy_oop(a, T2, b);
}

std::complex<double> inner(const complex_tensor& t1, const complex_tensor& t2) {
  std::complex<double> r = 0.0;
  assert(t1.size() == t2.size());
  int sz = t1.size();
  for (int i = 0; i < sz; i++) {
    r += std::conj(t1[i])*t2[i];
  }
  return r;
}

std::complex<double> inner(const double_tensor& t1, const complex_tensor& t2) {
  std::complex<double> r = 0.0;
  assert(t1.size() == t2.size());
  int sz = t1.size();
  for (int i = 0; i < sz; i++) {
    r += t1[i]*t2[i];
  }
  return r;
}

std::complex<double> inner(const complex_tensor& t1, const double_tensor& t2) {
  std::complex<double> r = 0.0;
  assert(t1.size() == t2.size());
  int sz = t1.size();
  for (int i = 0; i < sz; i++) {
    r += std::conj(t1[i])*t2[i];
  }
  return r;
}

double inner(const double_tensor& t1, const double_tensor& t2) {
  double r = 0.0;
  assert(t1.size() == t2.size());
  int sz = t1.size();
  for (int i = 0; i < sz; i++) {
    r += t1[i]*t2[i];
  }
  return r;
}

template <typename Q>
double norm2(const wstTensorT<Q>& t) {
  return std::abs(std::sqrt(inner(t, t)));
}

template <typename Q>
void normalize(wstTensorT<Q>& t) {
  double s = norm2(t);
  t.scale(1./s);
}

// create an empty 1-D function
template <typename Q>
wstTensorT<Q> empty_function(int d0, bool periodic = false) {
  wstTensorT<Q> r;
  r.create(d0, periodic);
  r.empty();  
  return r;
}

// create an empty 2-D function
template <typename Q>
wstTensorT<Q> empty_function(int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
  wstTensorT<Q> r;
  r.create(d0, d1, periodic0, periodic1);
  r.empty();  
  return r;
}

// create an empty 3-D function
template <typename Q>
wstTensorT<Q> empty_function(int d0, int d1, int d2, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
  wstTensorT<Q> r;
  r.create(d0, d1, d2, periodic0, periodic1, periodic2);
  r.empty();  
  return r;
}

// create an constant 1-D function
template <typename Q>
wstTensorT<Q> constant_function(int d0, double val, bool periodic = false) {
  wstTensorT<Q> r;
  r.create(d0, periodic);
  r.value(val);  
  return r;
}

// create an constant 2-D function
template <typename Q>
wstTensorT<Q> constant_function(int d0, int d1, double val, bool periodic0 = false, bool periodic1 = false) {
  wstTensorT<Q> r;
  r.create(d0, d1, periodic0, periodic1);
  r.value(val);  
  return r;
}

// create an constant 3-D function
template <typename Q>
wstTensorT<Q> constant_function(int d0, int d1, int d2, double val, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
  wstTensorT<Q> r;
  r.create(d0, d1, d2, periodic0, periodic1, periodic2);
  r.value(val);  
  return r;
}

template <typename Q> 
wstMatrixT<Q> outer(const std::vector<wstTensorT<Q> >& v1, const std::vector<wstTensorT<Q> >& v2) {
 
  // assume that the tensors are all the same dimensions
  int nsize = v1[0].size();
  // and the v1 and v2 have the same length
  int nvecs = v1.size();
  int nvecs2 = v2.size();
  assert(nvecs == nvecs2);

  wstMatrixT<Q> S = zeros<Q>(nsize,nsize);
  for (int i = 0; i < nsize; i++) {
    for (int j = 0; j < nsize; j++) {
      Q val = Q(0);
      for (int ivec = 0; ivec < nvecs; ivec++) {
        wstTensorT<Q> v1t = v1[ivec];
        wstTensorT<Q> v2t = v2[ivec];
        val += v1t[i]*v2t[j];
      }
      S(i,j) = val;
    }
  }
  return S;
}

// create a random 1-D function (obviously cannot be periodic)
double_tensor random_function_double(int d0, bool periodic0 = false) {
  double_tensor r;
  r.create(d0, periodic0);
  int sz = d0;
  double* p = r.ptr();
  for (int i = 0; i < sz; i++) {
    int i1 = rand();
    double t1 = (i1 % 100000000)/100000000.0;
    p[i] = t1;
  }
  return r;
}

// create a random 2-D function (obviously cannot be periodic)
double_tensor random_function_double(int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
  double_tensor r;
  r.create(d0, periodic0, periodic1);
  int sz = d0*d1;
  double* p = r.ptr();
  for (int i = 0; i < sz; i++) {
    int i1 = rand();
    double t1 = (i1 % 100000000)/100000000.0;
    p[i] = t1;
  }
  return r;
}

// create a random 3-D function (obviously cannot be periodic)
double_tensor random_function_double(int d0, int d1, int d2, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
  double_tensor r;
  r.create(d0, d1, d2, periodic0, periodic1, periodic2);
  int sz = d0*d1*d2;
  double* p = r.ptr();
  for (int i = 0; i < sz; i++) {
    int i1 = rand();
    double t1 = (i1 % 100000000)/100000000.0;
    p[i] = t1;
  }
  return r;
}

template <typename Q>
wstMatrixT<Q> matrix_inner(const std::vector<wstTensorT<Q> >& v1, const std::vector<wstTensorT<Q> >& v2) {
  int nsize = v1.size();
  wstMatrixT<Q> R = zeros<Q>(nsize, nsize);
  for (int i = 0; i < nsize; i++) {
    for (int j = 0; j < nsize; j++) {
      R(i,j) = inner(v1[i], v2[j]);
    }
  }
  return R;
}

template <typename Q>
void normalize(std::vector<wstTensorT<Q> >& vs) {
  int nsize = vs.size();
  for (int i = 0; i < nsize; i++) {
    wstTensorT<Q> f = vs[i];
    normalize(f); 
  }
}

template <typename Q>
std::vector<wstTensorT<Q> > transform(const std::vector<wstTensorT<Q> >& orbs, const std::vector<wstMatrixT<Q> >& vs) {
  int nvecs= vs.size();
  int norbs = orbs.size();
  std::vector<wstTensorT<Q> > orbs2;
  for (int i = 0; i < nvecs; i++) {
    wstMatrixT<Q> v = vs[i];
    int nsize = vs.size();
    assert(v.nrows() == 1 || v.ncols() == 1);
    wstTensorT<Q> f = copy(orbs[0], true);
    for (int j = 0; j < nsize; j++) {
      f.gaxpy(1.0, orbs[j], v(j));
    }
    orbs2.push_back(f);
  }
  return orbs2;
}

// for even
template <typename Q>
void fftshift(wstTensorT<Q>& t) {
  // Can only do even numbers right now
  for (int d = 0; d < t.ndim(); d++)
    assert(t.dim(d)%2 == 0);
  // 0 <--> 1
  if (t.ndim() == 1) {
    int m2 = t.dim(0)/2;
    for (int i = 0; i < m2; i++) {
      Q t01   = t(i);
      t(i)    = t(i+m2);
      t(i+m2) = t01;
    }
  }
  // 00 <--> 11
  // 01 <--> 10
  else if (t.ndim() == 2) {
    int m2 = t.dim(0)/2;
    int n2 = t.dim(1)/2;
    for (int i = 0; i < m2; i++) {
      for (int j = 0; j < n2; j++) {
        Q t03          = t(i,j);
        t(i,j)         = t(i+m2,j+n2);
        t(i+m2,j+n2)   = t03;
        Q t12          = t(i,j+n2);
        t(i,j+n2)      = t(i+m2,j);
        t(i+m2,j)      = t12;
      }
    }
  }
  // 000 <--> 111
  // 001 <--> 110
  // 010 <--> 101 
  // 100 <--> 011
  else if (t.ndim() == 3) {
    int m2 = t.dim(0)/2;
    int n2 = t.dim(1)/2;
    int p2 = t.dim(2)/2;
    for (int i = 0; i < m2; i++) {
      for (int j = 0; j < n2; j++) {
        for (int k = 0; k < p2; k++) {
          Q t07               = t(i,j,k);
          t(i,j,k)            = t(i+m2,j+n2,k+p2);
          t(i+m2,j+n2,k+p2)   = t07;
          Q t16               = t(i,j,k+p2);
          t(i,j,k+p2)         = t(i+m2,j+n2,k);
          t(i+m2,j+n2,k)      = t16;
          Q t25               = t(i,j+n2,k);
          t(i,j+n2,k)         = t(i+m2,j,k+p2);
          t(i+m2,j,k+p2)      = t25;
          Q t43               = t(i+m2,j,k);
          t(i+m2,j,k)         = t(i,j+n2,k+p2);
          t(i,j+n2,k+p2)      = t43;
        }
      }
    }
  }
  else {
    assert(false);
  }
}

void print(const double_tensor& t1) {
  int sz1 = t1.size();
  printf("# dims:  %d\n", t1.ndim());
  for (unsigned int i = 0; i < t1.ndim(); i++) {
      printf("%d\n", t1.dim(i)); 
  }
  for (int i = 0; i < sz1; i++)
    printf("%15.10f   \n", t1[i]);
}

void print2d(const double_tensor& t1) {
  assert(t1.ndim() == 2);
  printf("Dims (%d, %d)\n", t1.dim(0), t1.dim(1));
  for (unsigned int i = 0; i < t1.dim(0); i++) {
    for (unsigned int j = 0; j < t1.dim(1); j++) {
      printf("%15.10f  ", t1(i,j));
    }
    printf("\n");
  }
}

void print3d(const double_tensor& t1) {
  assert(t1.ndim() == 3);
  printf("Dims (%d, %d, %d)\n", t1.dim(0), t1.dim(1), t1.dim(2));
  for (unsigned int i = 0; i < t1.dim(0); i++) {
    printf("(%d,:,:)\n",i);
    for (unsigned int j = 0; j < t1.dim(1); j++) {
      for (unsigned int k = 0; k < t1.dim(2); k++) {
        printf("%15.10f  ", t1(i,j,k));
      }
      printf("\n");
    }
    printf("\n");
  }
}
void print(const double_tensor& t1, const double_tensor& t2) {
  int sz1 = t1.size(); 
  printf("# dims:  %d\n", t1.ndim());
  for (unsigned int i = 0; i < t1.ndim(); i++) {
      printf("%d     %d\n", t1.dim(i), t2.dim(i)); 
  }
  for (int i = 0; i < sz1; i++)
    printf("%15.10f     %15.10f\n", t1[i], t2[i]);
}

void print(const double_tensor& t1, const double_tensor& t2, const double_tensor& t3) {
  int sz1 = t1.size(); 
  printf("# dims:  %d\n", t1.ndim());
  for (unsigned int i = 0; i < t1.ndim(); i++) {
      printf("%d     %d     %d\n", t1.dim(i), t2.dim(i), t3.dim(i)); 
  }
  for (int i = 0; i < sz1; i++)
    printf("%15.10f     %15.10f     %15.10f\n", t1[i], t2[i], t3[i]);
}

void print(const double_tensor& t1, const double_tensor& t2, const double_tensor& t3, const double_tensor& t4) {
  int sz1 = t1.size(); 
  printf("# dims:  %d\n", t1.ndim());
  for (unsigned int i = 0; i < t1.ndim(); i++) {
      printf("%d     %d     %d     %d\n", t1.dim(i), t2.dim(i), t3.dim(i), t4.dim(i)); 
  }
  for (int i = 0; i < sz1; i++)
    printf("%15.10f     %15.10f     %15.10f     %15.10f\n", t1[i], t2[i], t3[i], t4[i]);
}

template <typename Q>
wstTensorT<Q> real(const wstTensorT<std::complex<Q> >&t) {
  wstTensorT<Q> R;
  R.create(t);
  int sz = R.size();
  for (int i = 0; i < sz; i++) R[i] = real(t[i]);
  return R;
}

template <typename Q>
wstTensorT<Q> imag(const wstTensorT<std::complex<Q> >&t) {
  wstTensorT<Q> R;
  R.create(t);
  int sz = R.size();
  for (int i = 0; i < sz; i++) R[i] = imag(t[i]);
  return R;
}

template <typename Q>
wstTensorT<Q> abs(const wstTensorT<std::complex<Q> >&t) {
  wstTensorT<Q> R;
  R.create(t);
  int sz = R.size();
  for (int i = 0; i < sz; i++) R[i] = std::abs(t[i]);
  return R;
}

// I realize that this code is totally sub-optimal
complex_tensor fft(const complex_tensor& t) {
  complex_tensor R = copy(t,false);
  std::complex<double>* ptr = new std::complex<double>[t.size()];
  fftw_plan plan;
  if (t.ndim() == 1) {
    plan = fftw_plan_dft_1d(t.dim(0), reinterpret_cast<fftw_complex*>(ptr), 
        reinterpret_cast<fftw_complex*>(ptr), FFTW_FORWARD, FFTW_ESTIMATE);
  }
  else if (t.ndim() == 2) {
    plan = fftw_plan_dft_2d(t.dim(0), t.dim(1), reinterpret_cast<fftw_complex*>(ptr), 
        reinterpret_cast<fftw_complex*>(ptr), FFTW_FORWARD, FFTW_ESTIMATE);
  }
  else if (t.ndim() == 3) {
    plan = fftw_plan_dft_3d(t.dim(0), t.dim(1), t.dim(2), reinterpret_cast<fftw_complex*>(ptr), 
        reinterpret_cast<fftw_complex*>(ptr), FFTW_FORWARD, FFTW_ESTIMATE);
  }
  else {
    assert(false);
  }
  // do dreadful copy (for now) 
  for (int i = 0; i < t.size(); i++) ptr[i] = R[i];
  fftw_execute(plan); 
  // do another dreadful copy (for now) 
  for (int i = 0; i < t.size(); i++) R[i] = ptr[i];
  fftw_destroy_plan(plan);
  return R;
}

// More sub-optimal code
complex_tensor ifft(const complex_tensor& t) {
  complex_tensor R = copy(t,false);
  std::complex<double>* ptr = new std::complex<double>[t.size()];
  fftw_plan plan;
  if (t.ndim() == 1) {
    plan = fftw_plan_dft_1d(t.dim(0), reinterpret_cast<fftw_complex*>(ptr), 
        reinterpret_cast<fftw_complex*>(ptr), FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else if (t.ndim() == 2) {
    plan = fftw_plan_dft_2d(t.dim(0), t.dim(1), reinterpret_cast<fftw_complex*>(ptr), 
        reinterpret_cast<fftw_complex*>(ptr), FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else if (t.ndim() == 3) {
    plan = fftw_plan_dft_3d(t.dim(0), t.dim(1), t.dim(2), reinterpret_cast<fftw_complex*>(ptr), 
        reinterpret_cast<fftw_complex*>(ptr), FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else {
    assert(false);
  }
  // do dreadful copy (for now) 
  for (int i = 0; i < t.size(); i++) ptr[i] = R[i];
  fftw_execute(plan); 
  // do another dreadful copy (for now) 
  for (int i = 0; i < t.size(); i++) R[i] = ptr[i];
  fftw_destroy_plan(plan);
  return R;
}

#endif
