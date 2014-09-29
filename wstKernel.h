#ifndef WSTKERNEL_H_
#define WSTKERNEL_H_

#include <vector>
#include "wstTensor.h"

using std::vector;

class wstKernel {
  public:
  virtual wstTensor apply(const wstTensor& t) = 0;
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

  virtual wstTensor apply(const wstTensor& t) {
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

  virtual wstTensor apply(const wstTensor& t) {
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

#endif
