#include <cstdio>
#include "wstTensor.h"
#include "wstKernel.h"
#include "wstMatrix.h"

const double alpha = 2.5;
const double L = 5.0;
const int NPTS = 22;

#define PI 3.141592653589793238462643383279502884197

double V(double L, double x) {
  return -alpha*(std::cos(2.0*PI*x/L)+1.0);
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
  std::vector<double> r;
  int npts2 = npts / 2;
  for (int i = -npts2; i < npts2; i++) {
    r.push_back((double)i/dx/(double)npts);
  }
  return r;
}

wstTensorT<std::complex<double> > apply_bsh(const std::vector<double>& x,
                        double hx, 
                        double mu,
                        const wstTensorT<std::complex<double> >& orb) {
  double mu2 = mu*mu;
  int npts = x.size();
  double dx = x[2]-x[1];
  printf("calling klinspace\n");
  std::vector<double> kx = klinspace(npts, dx);
  printf("calling fft\n");
  wstTensorT<std::complex<double> > r = fft(orb);
  printf("calling fftshift\n");
  fftshift(r);
  printf("applying bsh\n");
  for (int i = 0; i < npts; i++) {
    r(i) = r(i)/(kx[i]*kx[i] + mu2);
  }
  printf("calling fftshift\n");
  fftshift(r);
  printf("returning\n\n");
  return r;
}


wstKernel1D<double> build_hamiltonian(const std::vector<double>& x, double hx, int npts) {
  wstTensorT<double> Vpot;
  Vpot.create(std::bind(V, L, std::placeholders::_1), x, npts, true);
  //Vpot.create(V, x, y, z, npts, npts, npts, true, true, true);
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

void test_orbital_cache() {
  std::vector<wstTensorT<double> > orbs;

  wstTensorT<double> orb1 = empty_function<double>(4, false);
  orb1(0) = 1.0; orb1(1) = 2.0; orb1(2) = 3.0; orb1(3) = 4.0; 
  wstTensorT<double> orb2 = empty_function<double>(4, false);
  orb2(0) = 1.0; orb2(1) = 3.0; orb2(2) = 2.0; orb2(3) = 4.0; 
  wstTensorT<double> orb3 = orb1 + orb2;
  wstTensorT<double> orb4 = empty_function<double>(4, false);
  orb4(0) = 1.0; orb4(1) = 1.0; orb4(2) = 1.0; orb4(3) = 1.0; 

  print(orb1);
  print(orb2);
  print(orb3);

  orbs.push_back(orb1); 
  orbs.push_back(orb2); 
  orbs.push_back(orb3); 

  OrbitalCache<double> orbcache(5);
  orbs = orbcache.append(orbs);

  print(orbs[0]);
  print(orbs[1]);

  orbs.push_back(orb4); 
  orbs = orbcache.append(orbs);
  print(orbs[0]);
  print(orbs[1]);
  print(orbs[2]);
}

bool test_hamiltonian1D() {
  bool passed = true;
  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  wstKernel1D<double> Hker = build_hamiltonian(x, hx, NPTS);
  wstTensorT<double> f0 = constant_function<double>(NPTS, 1.0, true);
  f0.normalize();
  wstTensorT<double> f1 = Hker.apply(f0);
  f1.normalize();
  wstTensorT<double> f2 = Hker.apply(f1);
  f2.normalize();

  std::vector<wstTensorT<double> > orbs;
  orbs.push_back(f0);  
  orbs.push_back(f1);  
  orbs.push_back(f2);  

  OrbitalCache<double> orbcache(3);
  orbs = orbcache.append(orbs);
  wstMatrixT<double> H = Hker.sandwich(orbs);
  std::pair<wstMatrixT<double>, wstMatrixT<double> > result = diag(H);
  wstMatrixT<double> eigs = result.first; 
  passed = passed && (std::abs(eigs[0] + 4.05356273) < 1e-8);
  passed = passed && (std::abs(eigs[1] + 8.50907795e-01) < 1e-8);
  passed = passed && (std::abs(eigs[2] - 1.35212534) < 1e-8);
  return passed;
}

bool test_hamiltonian1D_2() {
  bool passed = true;
  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  wstKernel1D<double> Hker = build_hamiltonian(x, hx, NPTS);
  std::vector<wstTensorT<double> > orbs = make_initial_guess(Hker, NPTS, 3);

  OrbitalCache<double> orbcache(3);
  orbs = orbcache.append(orbs);
  wstMatrixT<double> H = Hker.sandwich(orbs);
  std::pair<wstMatrixT<double>, wstMatrixT<double> > result = diag(H);
  wstMatrixT<double> eigs = result.first; 
  passed = passed && (std::abs(eigs[0] + 4.05356273) < 1e-8);
  passed = passed && (std::abs(eigs[1] + 8.50907795e-01) < 1e-8);
  passed = passed && (std::abs(eigs[2] - 1.35212534) < 1e-8);
  return passed;
}

bool test_bsh() {
  bool passed = true;
  // Do all of this to get an interesting orbital
  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  wstKernel1D<double> Hker = build_hamiltonian(x, hx, NPTS);
  std::vector<wstTensorT<double> > orbs = make_initial_guess(Hker, NPTS, 3);
  OrbitalCache<double> orbcache(3);
  orbs = orbcache.append(orbs);
  //wstMatrixT<double> H = Hker.sandwich(orbs);
  //std::pair<wstMatrixT<double>, wstMatrixT<double> > result = diag(H);
  //wstMatrixT<double> eigs = result.first; 
  //wstMatrixT<double> evecs = result.second;
 
//  orbs = transform(orbs,evecs.cols());
  wstTensorT<std::complex<double> > f = apply_bsh(x, hx, -0.2, orbs[0]);
  wstTensorT<double> freal = real(f);
  wstTensorT<double> fimag = imag(f);
  print(freal, fimag);

  return passed;
}

int main(int argc, char** argv) {
  bool testResult = test_hamiltonian1D();
  if (testResult)
    printf("build_hamiltonian1D -- PASSED\n"); 
  else
    printf("build_hamiltonian1D -- FAILED\n"); 
  testResult = test_hamiltonian1D_2();
  if (testResult)
    printf("build_hamiltonian1D using initial guesses and OrbitalCache -- PASSED\n"); 
  else
    printf("build_hamiltonian1D using initial guesses and OrbitalCache -- FAILED\n"); 
  testResult = test_bsh();
  if (testResult)
    printf("test_bsh -- PASSED\n"); 
  else
    printf("test_bsh -- FAILED\n"); 

  return 0;
}
