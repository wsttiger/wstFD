#include <cstdio>
#include "wstTensor.h"
#include "wstKernel.h"
#include "wstMatrix.h"

const double V0 = -10.0;
const double L = 5.0;
const int NPTS = 44;

#define PI 3.141592653589793238462643383279502884197

double V(double L, double x) {
  return -V0;
}

template <typename Q>
class OrbitalCache {
private:
  int _maxorbs;
  double _thresh;
  std::vector<wstTensorT<Q> > _orbs;

public:
  OrbitalCache(int maxorbs = 10, double thresh = 1e-10)
   : _maxorbs(maxorbs), _thresh(thresh) {}

  std::vector<wstTensorT<Q> > append(const std::vector<wstTensorT<Q> >& orbs) {
    unsigned int szorbs = orbs.size();
    unsigned int szorbs2 = _orbs.size();
    std::vector<wstTensorT<Q> > combined_orbs;
    combined_orbs.insert(combined_orbs.begin(), orbs.begin(), orbs.end());
    combined_orbs.insert(combined_orbs.end(), _orbs.begin(), _orbs.end());

    wstMatrixT<Q> S = matrix_inner(combined_orbs, combined_orbs);
    S = 0.5*(S + ctranspose(S));
    std::pair<wstMatrixT<Q>, wstMatrixT<Q> > result = diag(S);
    wstMatrixT<Q> eigs = result.first;
    wstMatrixT<Q> evecs = result.second;

    int indx = -1;
    for (int i = 0; i < S.nrows() && indx < 0; i++) {
      if (std::abs(eigs(i)) > _thresh) {
        indx = i; 
      }
    }
    std::vector<wstTensorT<Q> > rorbs = transform<Q>(orbs,evecs.cols(wstMatrixSlice(indx,S.ncols()-1)));
    normalize(rorbs);
    _orbs = rorbs;
    return rorbs;
  }
};

std::vector<double> klinspace(int npts, double dx) {
  assert(npts % 2 == 0);
  std::vector<double> r(npts);
  int npts2 = npts / 2;
  double dk = 2.0*PI/dx/(double)npts;
  double k0 = -npts2*dk;
  //for (int i : r) {
  for (int i = 0; i < npts; i++) {
    r[i] = k0 + i*dk;
  }
  return r;
}

wstTensorT<std::complex<double> > apply_bsh_1d(const std::vector<double>& x,
                        double hx, 
                        double mu,
                        const wstTensorT<std::complex<double> >& orb) {
  double mu2 = mu*mu;
  int npts = x.size();
  double dx = x[2]-x[1];
  std::vector<double> kx = klinspace(npts, dx);
  wstTensorT<std::complex<double> > r = fft(orb);
  fftshift(r);
  for (int i = 0; i < npts; i++) {
    r(i) = r(i)/(kx[i]*kx[i] + mu2);
  }
  fftshift(r);
  return ifft(r);
}


wstKernel1D<double> build_hamiltonian(const std::vector<double>& x, double hx, int npts) {
  wstTensorT<double> Vpot;
  Vpot.create(std::bind(V, L, std::placeholders::_1), x, npts, true);
  wstKernel1D<double> H = create_laplacian_7p_1d(Vpot, hx, -0.5); 
  return H;
}

std::vector<wstTensorT<double> > make_initial_guess(const wstKernel1D<double>& H, int npts0, int norbs) {
  std::vector<wstTensorT<double> > orbs;
  for (int i = 0; i < norbs; i++) {
    if (i == 0) {
      wstTensorT<double> f = constant_function<double>(npts0, 1.0, true);
      f.normalize();
      orbs.push_back(f);
    } else {
      wstTensorT<double> f = H.apply(orbs[i-1]);
      f.normalize();
      orbs.push_back(f);
    }
  }
  OrbitalCache<double> orbcache(norbs);
  orbs = orbcache.append(orbs);
  return orbs;
}

void doit() {
  int norbs = 10;
  // make grid
  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  // make hamiltonian kernel
  wstKernel1D<double> Hker = build_hamiltonian(x, hx, NPTS);
  // intial guess
  std::vector<wstTensorT<double> > orbs = make_initial_guess(Hker, NPTS, norbs);
  OrbitalCache<double> orbcache(2*norbs);
  orbs = orbcache.append(orbs);

  std::vector<double> e(norbs);
  {
    wstMatrixT<double> H = Hker.sandwich(orbs);
    std::pair<wstMatrixT<double>, wstMatrixT<double> > result = diag(H);
    wstMatrixT<double> eigs = result.first; 
    wstMatrixT<double> evecs = result.second;
    orbs = transform(orbs,evecs.cols());
    for (int i = 0 ; i < norbs; i++) e[i] = eigs(i);
  } 
  
  int maxits = 50;
  // main iteration loop
  for (int iter = 0; iter < maxits; iter++) {
    std::vector<wstTensorT<double> > new_orbs(norbs);
    // loop over orbitals
    for (int iorb = 0; iorb > norbs; iorb++) {
      double shift = 0.0;
      if (e[iorb] > -1e-4) shift = -0.05 + e[iorb];
      double mu = std::sqrt(2.0*(shift-e[iorb]));
      wstTensorT<double> vpsi = (V0-shift)*orbs[iorb];
      new_orbs[iorb] = -2.0*real(apply_bsh_1d(x, hx, mu, vpsi));
    }
    orbs = orbcache.append(new_orbs);
  }
}

int main(int argc, char** argv) {
  doit();
  return 0;
}
