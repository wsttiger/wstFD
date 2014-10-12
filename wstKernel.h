#ifndef WSTKERNEL_H_
#define WSTKERNEL_H_

#include <vector>
#include "wstTensor.h"
#include "wstUtils.h"

using std::vector;

class wstKernel {
  public:
  virtual wstTensor apply(const wstTensor& t) const = 0;
};

class wstKernel1D : public wstKernel {
private:
  struct wstStencil1D {
    // offset
    int x;
    // coefficient
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
  void create(const wstTensor& localf, 
              const vector<int>& xoffset,
              const vector<double>& coeffs) {
    
    _localf = localf;
    _stencil = vector<wstStencil1D>(xoffset.size());
    for (unsigned int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil1D(xoffset[i], coeffs[i]);
    _local = true;
  }

  // constructor without a local function
  void create(const vector<int>& xoffset,
              const vector<double>& coeffs) {
    _stencil = vector<wstStencil1D>(xoffset.size());
    for (unsigned int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil1D(xoffset[i], coeffs[i]);
    _local = false;
  }

  vector<double> make_full_matrix(int sz, bool periodic = true) {
    vector<double> rm(sz*sz,0.0);
    for (int i = 0; i < sz; i++) {
      for (unsigned int ist = 0; ist < _stencil.size(); ist++) {
        wstStencil1D st = _stencil[ist];
        int j = wstUtils::periodic_index(i + st.x, sz);
        rm[i*sz+j] = st.c; 
      }
    }
    return rm;
  }

  virtual wstTensor apply(const wstTensor& t) const {
    wstTensor r = copy(t,true);
    int d0 = t.dim(0); 
    int stsz = _stencil.size();
      for (int ist = 0; ist < stsz; ist++) {
        wstStencil1D st = _stencil[ist];
        //printf("stencil:     %d     %15.4f\n", st.x, st.c);
      }

    //print(_localf, t, r); printf("\n");
    // loop over points
    for (int i = 0; i < d0; i++) {
      double val = (_local) ? _localf(i)*t(i) : 0.0;
      for (int ist = 0; ist < stsz; ist++) {
        wstStencil1D st = _stencil[ist];
        val += st.c*t(i+st.x);
        if (i == 0) {
          //printf("wstKernel1D::apply  ist: %d     st.x: %d     st.c: %15.5f     t(i+st.x): %15.8f     val: %15.8f\n",
          //  ist, st.x, st.c, t(i+st.x), val);
        }
      }
      //printf("wstKernel1D::apply -- %15.8f     %15.8f     %15.8f\n", _localf(i), t(i), val);
      r(i) = val;
    }
    //assert(false);
    return r;
  }
};

class wstKernel2D : public wstKernel {
private:
  struct wstStencil2D {
    // offset
    int x; int y;
    // coefficient
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
  void create(const wstTensor& localf, 
              const vector<int>& xoffset,
              const vector<int>& yoffset,
              const vector<double>& coeffs) {
    
    _localf = localf;
    _stencil = vector<wstStencil2D>(xoffset.size());
    for (unsigned int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil2D(xoffset[i], yoffset[i],  coeffs[i]);
    _local = true;
  }

  // constructor without a local function
  void create(const vector<int>& xoffset,
              const vector<int>& yoffset,
              const vector<double>& coeffs) {
    _stencil = vector<wstStencil2D>(xoffset.size());
    for (unsigned int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil2D(xoffset[i], yoffset[i], coeffs[i]);
    _local = false;
  }

  virtual wstTensor apply(const wstTensor& t) const {
    wstTensor r = copy(t,true);
    int d0 = t.dim(0); int d1 = t.dim(1); 
    int stsz = _stencil.size();
    // loop over points
    for (int i = 0; i < d0; i++) {
      for (int j = 0; j < d1; j++) {
        double val = (_local) ? _localf(i,j)*t(i,j) : 0.0;
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

class wstKernel3D : public wstKernel {
private:
  struct wstStencil3D {
    // offset
    int x; int y; int z;
    // coefficient
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
  void create(const wstTensor& localf, 
              const vector<int>& xoffset,
              const vector<int>& yoffset,
              const vector<int>& zoffset,
              const vector<double>& coeffs) {
    
    _localf = localf;
    _stencil = vector<wstStencil3D>(xoffset.size());
    for (unsigned int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil3D(xoffset[i], yoffset[i], zoffset[i], coeffs[i]);
    _local = true;
  }

  // constructor without a local function
  void create(const vector<int>& xoffset,
              const vector<int>& yoffset,
              const vector<int>& zoffset,
              const vector<double>& coeffs) {
    _stencil = vector<wstStencil3D>(xoffset.size());
    for (unsigned int i = 0; i < xoffset.size(); i++) 
      _stencil[i] = wstStencil3D(xoffset[i], yoffset[i], zoffset[i], coeffs[i]);
    _local = false;
  }

  wstTensor apply(const wstTensor& t) const {
    wstTensor r = copy(t,true);
    int d0 = t.dim(0); int d1 = t.dim(1); int d2 = t.dim(2);
    int stsz = _stencil.size();
    // loop over points
    for (int i = 0; i < d0; i++) {
      for (int j = 0; j < d1; j++) {
        for (int k = 0; k < d2; k++) {
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

  wstTensor apply(const wstTensor& t, int tsize) const {
    wstTensor r = copy(t,true);
    int d0 = t.dim(0); int d1 = t.dim(1); int d2 = t.dim(2);
    int stsz = _stencil.size();
    // loop over points
    for (int itile = 0; itile < d0; itile+=tsize) {
      int iend = std::min(itile+tsize, d0);
      for (int jtile = 0; jtile < d1; jtile+=tsize) {
        int jend = std::min(jtile+tsize, d1);
        for (int ktile = 0; ktile < d2; ktile+=tsize) {
          int kend = std::min(ktile+tsize, d2);
          for (int i = itile; i < iend; i++) {
            for (int j = jtile; j < jend; j++) {
              for (int k = ktile; k < kend; k++) {
                double val = (_local) ? _localf(i,j,k)*t(i,j,k) : 0.0;
                for (int ist = 0; ist < stsz; ist++) {
                  wstStencil3D st = _stencil[ist];
                  val += st.c*t(i+st.x, j+st.y, k+st.z);
                }
                r(i,j,k) = val;
              }
            } 
          }
        }
      }
    }
    return r;
  }
};

wstKernel1D create_laplacian_3p_1d(double hx, double scale = 1.0) {
  // Create the 3-point laplacian stencil
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(3,0); 
  vector<double> vcoeffs3p(3,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = scale*coeffs3p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(xoffset3p, vcoeffs3p);
  return kernel;
};

wstKernel1D create_laplacian_3p_1d(const wstTensor& localf, double hx, double scale = 1.0) {
  // Create the 3-point laplacian stencil
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(3,0); 
  vector<double> vcoeffs3p(3,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = scale*coeffs3p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(localf, xoffset3p, vcoeffs3p);
  return kernel;
};

wstKernel2D create_laplacian_3p_2d(double hx, double hy, double scale = 1.0) {
  // Create the 3-point laplacian stencil
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(6,0); 
  vector<int> yoffset3p(6,0); 
  vector<double> vcoeffs3p(6,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = scale*coeffs3p[i]/hx/hx;
  }
  p += 3;
  for (int i = 0; i < 5; i++) {
    yoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = scale*coeffs3p[i]/hy/hy;
  }
  
  wstKernel2D kernel;
  kernel.create(xoffset3p, yoffset3p, vcoeffs3p);
  return kernel;
};

wstKernel3D create_laplacian_3p_3d(double hx, double hy, double hz, double scale = 1.0) {
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(9,0); 
  vector<int> yoffset3p(9,0); 
  vector<int> zoffset3p(9,0);
  vector<double> vcoeffs3p(9,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = scale*coeffs3p[i]/hx/hx;
  }
  p += 3;
  for (int i = 0; i < 3; i++) {
    yoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = scale*coeffs3p[i]/hy/hy;
  }
  p += 3;
  for (int i = 0; i < 3; i++) {
    zoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = scale*coeffs3p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(xoffset3p, yoffset3p, zoffset3p, vcoeffs3p);
  return kernel;
}

wstKernel1D create_laplacian_5p_1d(double hx, double scale = 1.0) {
  int offsets5p[5] = {-2, -1, 0, 1, 2};
  double coeffs5p[5] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};
  vector<int> xoffset5p(5,0); 
  vector<double> vcoeffs5p(5,0.0);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    xoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = scale*coeffs5p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(xoffset5p, vcoeffs5p);
  return kernel;
}

wstKernel1D create_laplacian_5p_1d(const wstTensor& localf, double hx, double scale = 1.0) {
  int offsets5p[5] = {-2, -1, 0, 1, 2};
  double coeffs5p[5] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};
  vector<int> xoffset5p(5,0); 
  vector<double> vcoeffs5p(5,0.0);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    xoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = scale*coeffs5p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(localf, xoffset5p, vcoeffs5p);
  return kernel;
}

wstKernel2D create_laplacian_5p_2d(double hx, double hy, double scale = 1.0) {
  int offsets5p[5] = {-2, -1, 0, 1, 2};
  double coeffs5p[5] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};
  vector<int> xoffset5p(10,0); 
  vector<int> yoffset5p(10,0); 
  vector<double> vcoeffs5p(10,0.0);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    xoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = scale*coeffs5p[i]/hx/hx;
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    yoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = scale*coeffs5p[i]/hy/hy;
  }

  wstKernel2D kernel;
  kernel.create(xoffset5p, yoffset5p, vcoeffs5p);
  return kernel;
}

wstKernel3D create_laplacian_5p_3d(double hx, double hy, double hz, double scale = 1.0) {
  // Create the 5-point laplacian stencil
  int offsets5p[5] = {-2, -1, 0, 1, 2};
  double coeffs5p[5] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};
  vector<int> xoffset5p(15,0); 
  vector<int> yoffset5p(15,0); 
  vector<int> zoffset5p(15,0);
  vector<double> vcoeffs5p(15,0.0);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    xoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = scale*coeffs5p[i]/hx/hx;
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    yoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = scale*coeffs5p[i]/hy/hy;
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    zoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = scale*coeffs5p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(xoffset5p, yoffset5p, zoffset5p, vcoeffs5p);
  return kernel;
}

wstKernel1D create_laplacian_7p_1d(const wstTensor& localf, double hx, double scale = 1.0) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(7,0); 
  vector<double> vcoeffs7p(7,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(localf, xoffset7p, vcoeffs7p);
  return kernel;
}

wstKernel1D create_laplacian_7p_1d(double hx, double scale = 1.0) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(7,0); 
  vector<double> vcoeffs7p(7,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(xoffset7p, vcoeffs7p);
  return kernel;
}

wstKernel2D create_laplacian_7p_2d(double hx, double hy, double scale = 1.0) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(21,0); 
  vector<int> yoffset7p(21,0); 
  vector<double> vcoeffs7p(21,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hx/hx;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    yoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hy/hy;
  }

  wstKernel2D kernel;
  kernel.create(xoffset7p, yoffset7p, vcoeffs7p);
  return kernel;
}

wstKernel3D create_laplacian_7p_3d(const wstTensor& localf, double hx, double hy, double hz, double scale = 1.0) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(21,0); 
  vector<int> yoffset7p(21,0); 
  vector<int> zoffset7p(21,0);
  vector<double> vcoeffs7p(21,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hx/hx;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    yoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hy/hy;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    zoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(localf, xoffset7p, yoffset7p, zoffset7p, vcoeffs7p);
  return kernel;
}

wstKernel3D create_laplacian_7p_3d(double hx, double hy, double hz, double scale = 1.0) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(21,0); 
  vector<int> yoffset7p(21,0); 
  vector<int> zoffset7p(21,0);
  vector<double> vcoeffs7p(21,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hx/hx;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    yoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hy/hy;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    zoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = scale*coeffs7p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(xoffset7p, yoffset7p, zoffset7p, vcoeffs7p);
  return kernel;
}

// assuming periodic boundary conditions
class wstLanczos1D {
private:
  int _dim0; 
  int _nsize;
  const double& _hx;
  const wstTensor& _localf;
  //wstKernel1D _kernel;
  vector<double> _a;
  vector<double> _b;
  

public:
  wstLanczos1D(const wstTensor& localf, const double& hx, int nsize = 100)
   : _dim0(localf.dim(0)), _nsize(nsize), _localf(localf), _hx(hx) {
    //_kernel = create_laplacian_7p_1d(localf, hx); 
  }

  void run() {
    // assuming periodic boundary conditions
    wstTensor vinit = constant_function(_dim0, 1.0, true);
    vinit.normalize();
    wstTensor vold = empty_function(_dim0, true);

    wstTensor v = copy(vinit,false);
    _a = vector<double>(_nsize,0.0);
    _b = vector<double>(_nsize-1,0.0);

    wstTensor w = copy(vinit,true);
    for (int i = 0; i <_nsize; i++) {
      printf("running iteration %d in Lanczos\n", i);
      // make kernel (do this for debugging)
      wstKernel1D kernel = create_laplacian_7p_1d(_localf, _hx, -0.5); 

      //vector<double> matrix = kernel.make_full_matrix(_dim0, true);
      //wstUtils::print_matrix(matrix, _dim0, _dim0); 
      //assert(false);

      w = kernel.apply(v);
     print(w, v);
      _a[i] = inner(v, w);
     //printf("\nw norm: %15.8f     v norm: %15.8f\n", w.norm2(), v.norm2());
     printf("\n\n");
      w.gaxpy(1.0,v,-_a[i]);
      if (i > 0) w.gaxpy(1.0,vold,-_b[i-1]);
      if (i < _nsize-1) {
        _b[i] = norm2(w);
        w.normalize();
        //printf("w dot vold:  %15.8f\n", inner(w,vold));
        //printf("w dot v:  %15.8f\n", inner(w,v));
        //printf("w norm:  %15.8f\n", norm2(w));
        vold = v;
        v = w;
      }
    }
    printf("\nMatrix elements in Lanczos basis:\n");
    for (int i = 0; i < _nsize; i++) {
      if (i < (_nsize-1))
        printf("%15.8f          %15.8f\n", _a[i], _b[i]);
      else
        printf("%15.8f          %15.8f\n", _a[i], 0.0);
    }
    vector<double> mat(_nsize*_nsize,0.0);
    for (int i = 0; i < _nsize-1; i++)
    {
      mat[i*_nsize+i] = _a[i];
      mat[i*_nsize+i+1] = _b[i];
      mat[(i+1)*_nsize+i] = _b[i];
    }
    mat[_nsize*_nsize-1] = _a[_nsize-1];

    std::vector<double> e = vector<double>(_nsize,0.0);
    std::vector<double> ev = vector<double>(_nsize*_nsize, 0.0);
    wstUtils::diag_matrix(mat,_nsize,e,ev);
    printf("Lanczos: lowest eigenvalue is %15.8f\n\n", e[0]);
    for (int i = 0; i < _nsize; i++) {
      printf("%d:     %15.8f\n", i, e[i]);
    }
  }
};

// assuming periodic boundary conditions
class wstLanczos3D {
private:
  int _dim0, _dim1, _dim2;
  int _nsize;
  wstKernel3D _kernel;
  vector<double> _a;
  vector<double> _b;
  
public:
  wstLanczos3D(const wstTensor& localf, double hx, double hy, double hz, int nsize = 100)
   : _dim0(localf.dim(0)), _dim1(localf.dim(1)), _dim2(localf.dim(2)), _nsize(nsize) {
    _kernel = create_laplacian_7p_3d(localf, hx, hy, hz); 
  }

  void run() {
    // assuming periodic boundary conditions
    wstTensor vinit = random_function(_dim0, _dim1, _dim2, true, true, true);
    wstTensor vold = empty_function(_dim0, _dim1, _dim2, true, true, true);

    wstTensor v = copy(vinit,false);
    _a = vector<double>(_nsize,0.0);
    _b = vector<double>(_nsize-1,0.0);

    wstTensor v2 = copy(vinit,true);
    for (int i = 0; i <_nsize; i++) {
      printf("running iteration %d in Lanczos\n", i);
      if (i > 0) {
        v2 = gaxpy(1.0,_kernel.apply(v),-_b[i-1],vold);
      }
      else {
        v2 = _kernel.apply(v);
      }
      _a[i] = inner(v, v2);
      if (i < (_nsize-1))
      {
        v2 = gaxpy(1.0,v2,-_a[i],v);
        _b[i] = norm2(v2);
        vold = v;
        v = v2;
        v.scale(1./_b[i]);
      }
    }
    printf("\nMatrix elements in Lanczos basis:\n");
    for (int i = 0; i < _nsize; i++) {
      if (i < (_nsize-1))
        printf("%15.8f          %15.8f\n", _a[i], _b[i]);
      else
        printf("%15.8f          %15.8f\n", _a[i], 0.0);
    }
    vector<double> mat(_nsize*_nsize,0.0);
    for (int i = 0; i < _nsize-1; i++)
    {
      mat[i*_nsize+i] = _a[i];
      mat[i*_nsize+i+1] = _b[i];
      mat[(i+1)*_nsize+i] = _b[i];
    }
    mat[_nsize*_nsize-1] = _a[_nsize-1];

    std::vector<double> e = vector<double>(_nsize,0.0);
    std::vector<double> ev = vector<double>(_nsize*_nsize, 0.0);
    wstUtils::diag_matrix(mat,_nsize,e,ev);
    printf("Lanczos: lowest eigenvalue is %15.8f\n\n", e[0]);
  }
};

#endif
