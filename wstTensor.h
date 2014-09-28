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
    if (_allocated) {
      delete [] _dims;
      delete [] _bc;
      delete [] _p;
    }
    _allocated = false;
    _ndim = 0;
  }
  
  wstTensor& operator=(const wstTensor& t) {
    if (this != &t) {
      _ndim = t._ndim;
      _dims = t._dims;
      _bc = t._bc;
      _p = t._p;
      _allocated = t._allocated;
    }
    return *this;
  }

  wstTensor(const wstTensor& t) {
    _ndim = t._ndim;
    _dims = t._dims;
    _bc = t._bc;
    _p = t._p;
    _allocated = t._allocated;
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
    return _p[i0*_dims[1]*_dims[2]+i1*_dims[2]+i2];
  }

  double& operator()(int i0, int i1, int i2) const {
    assert(_ndim == 3);
    int idx0 = (_bc[0]) ? periodic_index(i0, _dims[0]) : i0;
    int idx1 = (_bc[1]) ? periodic_index(i1, _dims[1]) : i1;
    int idx2 = (_bc[2]) ? periodic_index(i2, _dims[2]) : i2;
    return _p[i0*_dims[1]*_dims[2]+i1*_dims[2]+i2];
  }

  // perform inplace a*(*this) + b*t
  void gaxpy(const double& a, const wstTensor& t, const double& b) {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    for (int i = 0; i < sz1; i++) _p[i] = a*_p[i]+b*t._p[i];
  }

  // perform out-of-place a*(*this) + b*t
  wstTensor gaxpy_oop(const double& a, const wstTensor& t, const double& b) {
    int sz1 = this->size();
    int sz2 = t.size();
    assert(sz1 == sz2);
    wstTensor r = copy(*this, true);
    for (int i = 0; i < sz1; i++) r._p[i] = a*_p[i]+b*t._p[i];
    return r;
  }

  wstTensor operator+(const wstTensor& t) {
    return gaxpy_oop(1.0, t, 1.0);
  }

  wstTensor operator-(const wstTensor& t) {
    return gaxpy_oop(1.0, t, -1.0);
  }
  
  double inner(const wstTensor& t) {
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
};

class wstKernel1D {
private:
  struct wstStencil1D {
    int x;
    double c;
    
    wstStencil1D() : x(0), c(0.0) {}
    wstStencil1D(int x, double c) : x(x), c(c) {}
  };

  vector<wstStencil1D> _stencil;
  wstTensor _localf;
  bool _local;

public:

  // default constructor
  wstKernel1D()
   : _local(false) {}

  // constructor with a local function attached
  // remember that we are making a deep copy of localf
  void create(wstTensor localf, 
              vector<int> xoffset,
              vector<double> coeffs) {
    
    _localf = copy(localf);
    _stencil = vector<wstStencil1D>(xoffset.size());
    for (int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil1D(xoffset[i], coeffs[i]);
    _local = true;
  }

  // constructor without a local function
  void create(vector<int> xoffset,
              vector<double> coeffs) {
    _stencil = vector<wstStencil1D>(xoffset.size());
    for (int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil1D(xoffset[i], coeffs[i]);
    _local = false;
  }

  wstTensor apply(const wstTensor& t) {
    wstTensor r = copy(t,true);
    printf("apply: copied tensor t into r\n");
    printf("t.ndim() = %d     r.ndim() = %d\n", t.ndim(), r.ndim());
        int stsz0 = _stencil.size();
        for (int ist = 0; ist < stsz0; ist++) {
          wstStencil1D st = _stencil[ist];
          printf("stencil:     %d   %5.3f\n", st.x, st.c);
        }
    int d0 = t.dim(0); 
    int stsz = _stencil.size();
    // loop over points
    for (int i = 0; i < d0; i++) {
      //double val = (_local) ? _localf(i,j)*t(i,j) : 0.0;
      double val = 0.0;
      for (int ist = 0; ist < stsz; ist++) {
        wstStencil1D st = _stencil[ist];
        val += st.c*t(i+st.x);
      }
      r(i) = val;
    }
    return r;
  }
};

class wstKernel2D {
private:
  struct wstStencil2D {
    int x;
    int y;
    double c;
    
    wstStencil2D() : x(0), y(0), c(0.0) {}
    wstStencil2D(int x, int y, double c) : x(x), y(y), c(c) {}
  };

  vector<wstStencil2D> _stencil;
  wstTensor _localf;
  bool _local;

public:

  // default constructor
  wstKernel2D()
   : _local(false) {}

  // constructor with a local function attached
  // remember that we are making a deep copy of localf
  void create(wstTensor localf, 
              vector<int> xoffset,
              vector<int> yoffset,
              vector<double> coeffs) {
    
    _localf = copy(localf);
    _stencil = vector<wstStencil2D>(xoffset.size());
    for (int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil2D(xoffset[i], yoffset[i],  coeffs[i]);
    _local = true;
  }

  // constructor without a local function
  void create(vector<int> xoffset,
              vector<int> yoffset,
              vector<double> coeffs) {
    _stencil = vector<wstStencil2D>(xoffset.size());
    for (int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil2D(xoffset[i], yoffset[i], coeffs[i]);
    _local = false;
  }

  wstTensor apply(const wstTensor& t) {
    wstTensor r = copy(t,true);
        int stsz0 = _stencil.size();
        for (int ist = 0; ist < stsz0; ist++) {
          wstStencil2D st = _stencil[ist];
          printf("stencil:     %d  %d     %5.3f\n", st.x, st.y, st.c);
        }
    int d0 = t.dim(0); int d1 = t.dim(1); 
    int stsz = _stencil.size();
    // loop over points
    for (int i = 0; i < d0; i++) {
      for (int j = 0; j < d1; j++) {
        //double val = (_local) ? _localf(i,j)*t(i,j) : 0.0;
        double val = 0.0;
        for (int ist = 0; ist < stsz; ist++) {
          wstStencil2D st = _stencil[ist];
          val += st.c*t(i+st.x, j+st.y);
        }
        r(i,j) = val;
      }
    }
    return r;
  }
};

class wstKernel3D {
private:
  struct wstStencil3D {
    int x;
    int y;
    int z;
    double c;
    
    wstStencil3D() : x(0), y(0), z(0), c(0.0) {}
    wstStencil3D(int x, int y, int z, double c) : x(x), y(y), z(z), c(c) {}
  };

  vector<wstStencil3D> _stencil;
  wstTensor _localf;
  bool _local;

public:

  // default constructor
  wstKernel3D()
   : _local(false) {}

  // constructor with a local function attached
  // remember that we are making a deep copy of localf
  void create(wstTensor localf, 
              vector<int> xoffset,
              vector<int> yoffset,
              vector<int> zoffset,
              vector<double> coeffs) {
    
    _localf = copy(localf);
    _stencil = vector<wstStencil3D>(xoffset.size());
    for (int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil3D(xoffset[i], yoffset[i], zoffset[i], coeffs[i]);
    _local = true;
  }

  // constructor without a local function
  void create(vector<int> xoffset,
              vector<int> yoffset,
              vector<int> zoffset,
              vector<double> coeffs) {
    _stencil = vector<wstStencil3D>(xoffset.size());
    for (int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil3D(xoffset[i], yoffset[i], zoffset[i], coeffs[i]);
    _local = false;
  }

  wstTensor apply(const wstTensor& t) {
    wstTensor r = copy(t,true);
    int d0 = t.dim(0); int d1 = t.dim(1); int d2 = t.dim(2);
    int stsz = _stencil.size();
    // loop over points
    for (int i = 0; i < d0; i++) {
      for (int j = 0; j < d1; j++) {
        for (int k = 0; k < d2; k++) {
          printf("%d  %d  %d\n", i, j, k);
          double val = (_local) ? _localf(i,j,k)*t(i,j,k) : 0.0;
          for (int ist = 0; ist < stsz; ist++) {
            wstStencil3D st = _stencil[ist];
            val += st.c*t(i+st.x, j+st.y, k+st.z);
          }
          r(i,j,k) = val;
        }
      }
    }
    return r;
  }
};

