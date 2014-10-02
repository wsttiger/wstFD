#ifndef WSTTENSOR_H_
#define WSTTENSOR_H_

#include <vector>
#include <cassert>

using std::vector;
using std::pair;

class wstTensor {
private:
  // number of dimensions
  int _ndim;
  // array holding the dimensions
  int* _dims;
  // are the dimensions periodic?
  bool* _bc;
  // the data pointer
  std::shared_ptr<double> _sp;
  double* _p;
  // is the tensor currently allocated?
  bool _allocated;

  friend void print(const wstTensor& t1) {
    int sz1 = t1.size();
    for (int i = 0; i < sz1; i++)
      printf("%15.10f   \n", t1._p[i]);
  }

  friend void print(const wstTensor& t1, const wstTensor& t2) {
    int sz1 = t1.size(); int sz2 = t2.size();
    for (int i = 0; i < sz1; i++)
      printf("%15.10f     %15.10f\n", t1._p[i], t2._p[i]);
  }

  friend void print(const wstTensor& t1, const wstTensor& t2, const wstTensor& t3) {
    int sz1 = t1.size(); int sz2 = t2.size();
    for (int i = 0; i < sz1; i++)
      printf("%15.10f     %15.10f     %15.10f\n", t1._p[i], t2._p[i], t3._p[i]);
  }

  friend void print(const wstTensor& t1, const wstTensor& t2, const wstTensor& t3, const wstTensor& t4) {
    int sz1 = t1.size(); int sz2 = t2.size();
    for (int i = 0; i < sz1; i++)
      printf("%15.10f     %15.10f     %15.10f     %15.10f\n", t1._p[i], t2._p[i], t3._p[i], t4._p[i]);
  }

  friend wstTensor copy(const wstTensor& t, bool empty = false) {
    wstTensor r;
    if (t._allocated) {
      r._ndim = t._ndim; 
      r._dims = new int[r._ndim];
      r._bc = new bool[r._ndim];
      for (int i = 0; i < r._ndim; i++) {
        r._dims[i] = t._dims[i];
        r._bc[i] = t._bc[i];
      }
      int sz = r.size();
      r._p = new double[sz];
      r._sp = std::shared_ptr<double>(r._p, [](double *p) {delete[] p;});
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
  wstTensor() 
    : _ndim(0), _dims(0), _bc(0), _p(0),_allocated(false) {}
  
  virtual ~wstTensor() {
    printf("(begin) wstTensor destructor called for %ld\n", this);
    if (_allocated) {
      printf("wstTensor destructor [_dims] called for %ld\n", _dims);
      delete [] _dims;
      printf("wstTensor destructor [_bc] called for %ld\n", _bc);
      delete [] _bc;
    }
    printf("(end) wstTensor destructor called for %ld\n", this);
    _allocated = false;
    _ndim = 0;
  }
  
  wstTensor& operator=(const wstTensor& t) {
    if (this != &t) {
      _ndim = t._ndim;
      _dims = new int[_ndim];
      _bc = new bool[_ndim];
      for (int i = 0; i < _ndim; i++) {
        _dims[i] = t._dims[i];
        _bc[i] = t._bc[i];
      }
      _p = t._p;
      _sp = t._sp;
      _allocated = t._allocated;
    }
    return *this;
  }

  wstTensor(const wstTensor& t) {
    (*this) = t;
  }

  void create(int d0, bool periodic = false) {
    // dims
    _ndim = 1;
    _dims = new int[_ndim];
    _dims[0] = d0;
    // boundary conditions
    _bc = new bool[_ndim];
    _bc[0] = periodic;
    // allocation
    _p = new double[d0];
    _sp = std::shared_ptr<double>(_p, [](double *p) {delete[] p;});
    _allocated = true;
  }

  void create(double (*f)(double), const vector<double>& x, int d0, bool periodic = false) {
    create(d0, periodic);
    for (int i = 0; i < d0; i++) _p[i] = f(x[i]);
  }

  void create(int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
    // dims
    _ndim = 2;
    _dims = new int[_ndim];
    _dims[0] = d0; _dims[1] = d1;
    // boundary conditions
    _bc = new bool[_ndim];
    _bc[0] = periodic0; _bc[1] = periodic1;
    // allocation
    _p = new double[d0*d1];
    _sp = std::shared_ptr<double>(_p, [](double *p) {delete[] p;});
    _allocated = true;
  }

  void create(double (*f)(double, double), const vector<double>& x, const vector<double>& y, 
              int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
    create(d0, d1, periodic0, periodic1);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        _p[i*d1+j] = f(x[i], y[j]);
  }

  void create(int d0, int d1, int d2, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
    // dims
    _ndim = 3;
    _dims = new int[_ndim];
    _dims[0] = d0; _dims[1] = d1; _dims[2] = d2;
    // boundary conditions
    _bc = new bool[_ndim];
    _bc[0] = periodic0; _bc[1] = periodic1; _bc[2] = periodic2;
    // allocation
    _p = new double[d0*d1*d2];
    _sp = std::shared_ptr<double>(_p, [](double *p) {delete[] p;});
    _allocated = true;
  }

  void create(double (*f)(double, double, double), 
              const vector<double>& x, const vector<double>& y, const vector<double>& z, 
              int d0, int d1, int d2, 
              bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
    create(d0, d1, d2, periodic0, periodic1, periodic2);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        for (int k = 0; k < d2; k++)
          _p[i*d1*d2+j*d2+k] = f(x[i], y[j], z[k]);
  }

  int size() const {
    int sz = 1;
    for (int i = 0; i < _ndim; i++) sz *= _dims[i];
    return sz;
  }

  void empty() {
    int sz = this->size();
    for (int i = 0; i < sz; i++) {
      _p[i] = 0.0;
    }
  }

  void fillrandom() {
    int sz = this->size();
    for (int i = 0; i < sz; i++) {
      int i1 = rand();
      double t1 = (i1 % 100000000)/100000000.0;
      _p[i] = t1;
    }
  }

  int periodic_index(int idx, int size) const {
    if (idx >= size) 
      return periodic_index(idx-size+1, size);
    else if (idx < 0) 
      return periodic_index(idx+size-1, size);
    else
      return idx;
  }

  int ndim() const {return _ndim;}

  int dim(int i0) const {
    assert(i0 < _ndim);
    return _dims[i0];
  }

  void print() const {
    int sz = this->size();
    for (int i = 0; i < sz; i++)
      printf("%15.8e\n", _p[i]);
  }

  double* ptr() {
    return _p;
  }

  const double* ptr() const {
    return _p;
  }

  double& operator()(int i0) {
    assert(_ndim == 1);
    int idx0 = (_bc[0]) ? periodic_index(i0, _dims[0]) : i0;
    return _p[idx0];
  }

  const double& operator()(int i0) const {
    assert(_ndim == 1);
    int idx0 = (_bc[0]) ? periodic_index(i0, _dims[0]) : i0;
    return _p[idx0];
  }

  double& operator()(int i0, int i1) {
    assert(_ndim == 2);
    int idx0 = (_bc[0]) ? periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? periodic_index(i1, _dims[1]) : i1;
    return _p[idx0*_dims[1]+idx1];
  }

  double& operator()(int i0, int i1) const {
    assert(_ndim == 2);
    int idx0 = (_bc[0]) ? periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? periodic_index(i1, _dims[1]) : i1;
    return _p[idx0*_dims[1]+idx1];
  }

  double& operator()(int i0, int i1, int i2) {
    assert(_ndim == 3);
    int idx0 = (_bc[0]) ? periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? periodic_index(i1, _dims[1]) : i1;
    int idx2 = (_bc[2]) ? periodic_index(i2, _dims[2]) : i2;
    return _p[idx0*_dims[1]*_dims[2]+idx1*_dims[2]+idx2];
  }

  double& operator()(int i0, int i1, int i2) const {
    assert(_ndim == 3);
    int idx0 = (_bc[0]) ? periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? periodic_index(i1, _dims[1]) : i1;
    int idx2 = (_bc[2]) ? periodic_index(i2, _dims[2]) : i2;
    return _p[idx0*_dims[1]*_dims[2]+idx1*_dims[2]+idx2];
  }

  // perform inplace a*(*this) + b*t
  void gaxpy(const double& a, const wstTensor& t, const double& b) {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    for (int i = 0; i < sz1; i++) _p[i] = a*_p[i]+b*t._p[i];
  }

  // perform out-of-place a*(*this) + b*t
  wstTensor gaxpy_oop(const double& a, const wstTensor& t, const double& b) const {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    wstTensor r = copy(*this, true);
    for (int i = 0; i < sz1; i++) r._p[i] = a*_p[i]+b*t._p[i];
    return r;
  }

  wstTensor operator+(const wstTensor& t) const {
    return gaxpy_oop(1.0, t, 1.0);
  }

  wstTensor operator-(const wstTensor& t) const {
    return gaxpy_oop(1.0, t, -1.0);
  }
 
  void scale(double a) {
    int sz = this->size();
    for (int i = 0; i < sz; i++) _p[i]*a;
  }
 
  double inner(const wstTensor& t) const {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    double rval = 0.0;
    for (int i = 0; i < sz1; i++) rval += _p[i]*t._p[i]; 
    return rval;
  }

  double norm2() const {
    int sz = this->size();
    double rval = 0.0;
    for (int i = 0; i < sz; i++) rval += _p[i]*_p[i]; 
    return std::sqrt(rval);
  }

  void normalize() {
    double s = this->norm2();
    this->scale(1./s); 
  }
};

wstTensor gaxpy(const double& a, const wstTensor& T1, const double& b, const wstTensor& T2) {
  return T1.gaxpy_oop(a, T2, b);
}

double inner(const wstTensor& t1, const wstTensor& t2) {
  return t1.inner(t2);
}

double norm2(const wstTensor& t) {
  return t.norm2();
}

// create an empty 1-D function
wstTensor empty_function(int d0, bool periodic = false) {
  wstTensor r;
  r.create(d0, periodic);
  r.empty();  
}

// create an empty 2-D function
wstTensor empty_function(int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
  wstTensor r;
  r.create(d0, d1, periodic0, periodic1);
  r.empty();  
}

// create an empty 3-D function
wstTensor empty_function(int d0, int d1, int d2, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
  wstTensor r;
  r.create(d0, d1, d2, periodic0, periodic1, periodic2);
  r.empty();  
}

// create a random 1-D function (obviously cannot be periodic)
wstTensor random_function(int d0, bool periodic = false) {
  wstTensor r;
  r.create(d0, periodic);
  r.fillrandom();  
}

// create a random 2-D function (obviously cannot be periodic)
wstTensor random_function(int d0, int d1, bool periodic0 = false, bool periodic1 = false) {
  wstTensor r;
  r.create(d0, d1, periodic0, periodic1);
  r.fillrandom();  
}

// create a random 3-D function (obviously cannot be periodic)
wstTensor random_function(int d0, int d1, int d2, bool periodic0 = false, bool periodic1 = false, bool periodic2 = false) {
  wstTensor r;
  r.create(d0, d1, d2, periodic0, periodic1, periodic2);
  r.fillrandom();  
}

#endif
