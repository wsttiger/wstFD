#ifndef WSTKERNEL_H_
#define WSTKERNEL_H_

#include <vector>
#include "wstTensor.h"

using std::vector;

class wstKernel {
  public:
  virtual wstTensor apply(const wstTensor& t) const = 0;
};

class wstKernel1D : public wstKernel {
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
    
    _localf = localf;
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

  virtual wstTensor apply(const wstTensor& t) const {
    wstTensor r = copy(t,true);
    int d0 = t.dim(0); 
    int stsz = _stencil.size();
    // loop over points
    for (int i = 0; i < d0; i++) {
      double val = (_local) ? _localf(i)*t(i) : 0.0;
      for (int ist = 0; ist < stsz; ist++) {
        wstStencil1D st = _stencil[ist];
        val += st.c*t(i+st.x);
      }
      r(i) = val;
    }
    return r;
  }
};

class wstKernel2D : public wstKernel {
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
    
    _localf = localf;
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
    
    _localf = localf;
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

  wstTensor apply(const wstTensor& t) const {
    wstTensor r = copy(t,true);
    int d0 = t.dim(0); int d1 = t.dim(1); int d2 = t.dim(2);
    int stsz = _stencil.size();
    printf("t dim0: %d     dim1: %d     dim2: %d\n", t.dim(0), t.dim(1), t.dim(2));
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

wstKernel1D create_laplacian_3p_1d(double hx) {
  // Create the 3-point laplacian stencil
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(3,0); 
  vector<double> vcoeffs3p(3,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(xoffset3p, vcoeffs3p);
  return kernel;
};

wstKernel2D create_laplacian_3p_2d(double hx, double hy) {
  // Create the 3-point laplacian stencil
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(6,0); 
  vector<int> yoffset3p(6,0); 
  vector<double> vcoeffs3p(6,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hx/hx;
  }
  p += 3;
  for (int i = 0; i < 5; i++) {
    yoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hy/hy;
  }
  
  wstKernel2D kernel;
  kernel.create(xoffset3p, yoffset3p, vcoeffs3p);
  return kernel;
};

wstKernel3D create_laplacian_3p_3d(double hx, double hy, double hz) {
  int offsets3p[3] = {-1, 0, 1};
  double coeffs3p[3] = {1.0, -2.0, 1.0};
  vector<int> xoffset3p(9,0); 
  vector<int> yoffset3p(9,0); 
  vector<int> zoffset3p(9,0);
  vector<double> vcoeffs3p(9,0.0);
  int p = 0;
  for (int i = 0; i < 3; i++) {
    xoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hx/hx;
  }
  p += 3;
  for (int i = 0; i < 3; i++) {
    yoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hy/hy;
  }
  p += 3;
  for (int i = 0; i < 3; i++) {
    zoffset3p[i+p] = offsets3p[i];   
    vcoeffs3p[i+p] = coeffs3p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(xoffset3p, yoffset3p, zoffset3p, vcoeffs3p);
  return kernel;
}

wstKernel1D create_laplacian_5p_1d(double hx) {
  int offsets5p[5] = {-2, -1, 0, 1, 2};
  double coeffs5p[5] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};
  vector<int> xoffset5p(15,0); 
  vector<double> vcoeffs5p(15,0.0);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    xoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(xoffset5p, vcoeffs5p);
  return kernel;
}

wstKernel2D create_laplacian_5p_2d(double hx, double hy) {
  int offsets5p[5] = {-2, -1, 0, 1, 2};
  double coeffs5p[5] = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};
  vector<int> xoffset5p(15,0); 
  vector<int> yoffset5p(15,0); 
  vector<double> vcoeffs5p(15,0.0);
  int p = 0;
  for (int i = 0; i < 5; i++) {
    xoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i]/hx/hx;
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    yoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i]/hy/hy;
  }

  wstKernel2D kernel;
  kernel.create(xoffset5p, yoffset5p, vcoeffs5p);
  return kernel;
}

wstKernel3D create_laplacian_5p_3d(double hx, double hy, double hz) {
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
    vcoeffs5p[i+p] = coeffs5p[i]/hx/hx;
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    yoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i]/hy/hy;
  }
  p += 5;
  for (int i = 0; i < 5; i++) {
    zoffset5p[i+p] = offsets5p[i];   
    vcoeffs5p[i+p] = coeffs5p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(xoffset5p, yoffset5p, zoffset5p, vcoeffs5p);
  return kernel;
}

wstKernel1D create_laplacian_7p_1d(double hx) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(21,0); 
  vector<double> vcoeffs7p(21,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hx/hx;
  }

  wstKernel1D kernel;
  kernel.create(xoffset7p, vcoeffs7p);
  return kernel;
}

wstKernel2D create_laplacian_7p_2d(double hx, double hy) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(21,0); 
  vector<int> yoffset7p(21,0); 
  vector<double> vcoeffs7p(21,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hx/hx;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    yoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hy/hy;
  }

  wstKernel2D kernel;
  kernel.create(xoffset7p, yoffset7p, vcoeffs7p);
  return kernel;
}

wstKernel3D create_laplacian_7p_3d(const wstTensor& localf, double hx, double hy, double hz) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(21,0); 
  vector<int> yoffset7p(21,0); 
  vector<int> zoffset7p(21,0);
  vector<double> vcoeffs7p(21,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hx/hx;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    yoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hy/hy;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    zoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(localf, xoffset7p, yoffset7p, zoffset7p, vcoeffs7p);
  return kernel;
}

wstKernel3D create_laplacian_7p_3d(double hx, double hy, double hz) {
  int offsets7p[7] = {-3, -2, -1, 0, 1, 2, 3};
  double coeffs7p[7] = {2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0, 270.0/180.0, -27.0/180.0, 2.0/180.0};
  vector<int> xoffset7p(21,0); 
  vector<int> yoffset7p(21,0); 
  vector<int> zoffset7p(21,0);
  vector<double> vcoeffs7p(21,0.0);
  int p = 0;
  for (int i = 0; i < 7; i++) {
    xoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hx/hx;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    yoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hy/hy;
  }
  p += 7;
  for (int i = 0; i < 7; i++) {
    zoffset7p[i+p] = offsets7p[i];   
    vcoeffs7p[i+p] = coeffs7p[i]/hz/hz;
  }

  wstKernel3D kernel;
  kernel.create(xoffset7p, yoffset7p, zoffset7p, vcoeffs7p);
  return kernel;
}

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

    wstTensor v = copy(vinit,true);
    _a = vector<double>(_nsize,0.0);
    _b = vector<double>(_nsize-1,0.0);

    wstTensor v2;
    for (unsigned int i = 0; i <_nsize; i++) {
      printf("running iteration i in Lanczos\n");
      if (i > 0) {
      printf("1a\n");
        v2 = gaxpy(1.0,_kernel.apply(v),-_b[i-1],vold);
      printf("1b\n");
      }
      else {
      printf("2a\n");
        v2 = _kernel.apply(v);
      printf("2b\n");
      }
      printf("3a\n");
      _a[i] = inner(v, v2);
      printf("3b\n");
      if (i < (_nsize-1))
      {
        printf("4a\n");
        v2 = gaxpy(1.0,v2,-_a[i],v);
        printf("5a\n");
        _b[i] = norm2(v2);
        printf("6a\n");
        vold = v;
        printf("7a\n");
        v = v2;
        printf("8a\n");
        v.scale(1./_b[i]);
        printf("9a\n");
      }
    }
  }

};

#endif