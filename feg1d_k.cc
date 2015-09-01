#include <cstdio>
#include <iostream>
#include "wstTensor.h"
#include "wstKernel.h"
#include "wstMatrix.h"

typedef std::pair<std::vector<complex_tensor>, real_matrix> eigenpairT;

const double V0 = -10.0;
const double L = 5.0;
const int NPTS = 44;

#define PI 3.141592653589793238462643383279502884197

//double V(double L, double x) {
//  return V0;
//}

double V(double L, double x) {
  return -2.5*(cos(2*PI*x/L)+1.0);
}

double gauss(double alpha, double x) {
  return std::exp(-alpha*x*x);
}

template <typename Q>
void verify_identity(const wstMatrixT<Q>& S) {
  double tol = 1e-8;
  for (int i = 0; i < S.nrows(); i++) {
    for (int j = 0; j < S.ncols(); j++) {
      if (i == j) {
        if (std::abs(std::real(S(i,j))-1.0) > tol || std::abs(std::imag(S(i,j))-0.0) > tol) {
          printf("ij: (%d,%d)", i, j);
          std::cout << S(i,j) << std::endl;
          //assert(false);
        }
      }
      else {
        if (std::abs(std::real(S(i,j))-0.0) > tol || std::abs(std::imag(S(i,j))-0.0) > tol) {
          printf("ij: (%d,%d)", i, j);
          std::cout << S(i,j) << std::endl;
          //assert(false);
        }
      }
    }
  }
}

template <typename Q>
class OrbitalCache {
private:
  int _maxorbs;
  double _thresh;
  std::vector<wstTensorT<Q> > _orbs;

public:
  OrbitalCache(int maxorbs = 10, double thresh = 1e-8)
   : _maxorbs(maxorbs), _thresh(thresh) {}

  std::vector<wstTensorT<Q> > append(const std::vector<wstTensorT<Q> >& orbs) {
    int szorbs = orbs.size();
    int szorbs2 = _orbs.size();
    std::vector<wstTensorT<Q> > combined_orbs;
    for (int i = 0; i < szorbs; i++) combined_orbs.push_back(orbs[i]);
    for (int i = 0, j = szorbs; i < szorbs2 && j < _maxorbs; i++, j++) combined_orbs.push_back(_orbs[i]);
    //for (int i = 0; i < szorbs2; i++) combined_orbs.push_back(_orbs[i]);

    std::vector<real_tensor> rcombined_orbs = real(combined_orbs);
    auto S = matrix_inner(combined_orbs, combined_orbs);
    //auto S = matrix_inner(rcombined_orbs, rcombined_orbs);
    S = 0.5*(S + ctranspose(S));
    auto result = diag(S);
    auto eigs = result.first;
    auto evecs = result.second;

    printf("real part:\n");
    printf("\n");
    print(real(combined_orbs));
    printf("\n");

    printf("imag part:\n");
    printf("\n");
    print(imag(combined_orbs));
    printf("\n");

    printf("OC S:\n"); print(real(S));
    printf("OC e:\n"); print(eigs);
    //printf("OC v:\n"); print(real(evecs));

    auto indx = -1;
    //for (int i = std::max(0,szorbs+szorbs2-_maxorbs); i < S.nrows() && indx < 0; i++) {
    for (int i = 0; i < S.nrows() && indx < 0; i++) {
      if (std::abs(eigs(i)) > _thresh) {
        indx = i; 
      }
    }
    //printf("OC indx: %d\n\n", indx);
    print(wstSlice(indx,S.ncols()-1));
    auto rorbs = transform<Q>(combined_orbs,evecs.cols(wstSlice(indx,S.ncols()-1)));
    normalize(rorbs);
    _orbs = rorbs;
    S = matrix_inner(_orbs, _orbs);
    S = 0.5*(S + ctranspose(S));
    printf("outgoing S:\n");
    print(real(S));

    printf("\n");
    print(real(_orbs));
    printf("\n");

    return rorbs;
  }
};

std::vector<double> klinspace(int npts, double dx) {
  assert(npts % 2 == 0);
  std::vector<double> r(npts);
  int npts2 = npts / 2;
  double dk = 2.0*PI/dx/(double)npts;
  double k0 = -npts2*dk;
  for (int i = 0; i < npts; i++) {
    r[i] = k0 + i*dk;
  }
  return r;
}

complex_tensor apply_bsh_1d(const std::vector<double>& x,
                        double hx, 
                        double mu,
                        double qx,
                        const complex_tensor& orb) {
  double mu2 = mu*mu;
  int npts = x.size();
  double dx = x[2]-x[1];
  auto kx = klinspace(npts, dx);
  auto r = fft(orb);
  fftshift(r);
  for (int i = 0; i < npts; i++) {
    r(i) = r(i)/((kx[i]+qx)*(kx[i]+qx) + mu2);
  }
  fftshift(r);
  r = ifft(r);
  return r;
}

complex_tensor apply2_bsh_1d(const std::vector<double>& x,
                        double hx, 
                        double mu,
                        double qx,
                        const complex_tensor& orb) {
  double mu2 = mu*mu;
  int npts = x.size();
  double dx = x[2]-x[1];
  auto kx = klinspace(npts, dx);
  auto orb2 = copy(orb);
  for (int i = 0; i < npts; i++) {
    //orb2(i) = std::exp(std::complex<double>(0.0,-qx*x[i]))*orb2(i);
    printf("orb2:    %15.8f     %15.8f\n", std::real(orb2(i)), std::imag(orb2(i)));
  }
  auto r = fft(orb2);
  for (int i = 0; i < npts; i++) {
    printf("orb2:    %15.8f     %15.8f\n", std::real(orb2(i)), std::imag(orb2(i)));
  }
  fftshift(r);
  for (int i = 0; i < npts; i++) {
    r(i) = r(i)/(kx[i]*kx[i] + mu2);
  }
  fftshift(r);
  orb2 = ifft(r);
  for (int i = 0; i < npts; i++) {
    orb2(i) = std::exp(std::complex<double>(0.0,qx*x[i]))*orb2(i);
  }
}


complex_kernel_1d build_hamiltonian_1d(const std::vector<double>& x, double hx, int npts, double k) {
  real_tensor Vpot;
  Vpot.create(std::bind(V, L, std::placeholders::_1), x, npts, true);
  complex_kernel_1d H = create_laplacian_7p_1d(Vpot, hx, -0.5); 
  complex_kernel_1d Hk = H + create_Dx_7p_1d(hx, std::complex<double>(0.0, -k)) + 0.5*k*k;
  return Hk;
}

std::vector<complex_tensor > make_standard_basis(int npts) {
  std::vector<complex_tensor > orbs;
  for (int i = 0; i < npts; i++) {
    complex_tensor f = empty_function<std::complex<double> >(npts, true);
    f[i] = 1.0;
    orbs.push_back(f);
  }
  return orbs;
}

// create the initial guess
// H          --- the Hamiltonian kernel
// npts0      --- the number of points in 1D
// norbs      --- the number of initial orbitals to create
// nst        --- the number of orbitals to keep
eigenpairT initial_guess(const complex_kernel_1d& Hker, 
                         int npts0, int norbs, int nst, 
                         bool random = false) {
  std::vector<complex_tensor > orbs;
  for (int i = 0; i < norbs; i++) {
    if (i == 0) {
      complex_tensor f = (random) ? random_function_double(npts0, true)
       : constant_function<double>(npts0, 1.0, true);
      normalize(f);
      orbs.push_back(f);
    } else {
      complex_tensor f = (random) ? random_function_double(npts0, true) : Hker.apply(orbs[i-1]);
      normalize(f);
      orbs.push_back(f);
    }
  }

  std::vector<double> e(norbs);
  auto H = Hker.sandwich(orbs);
  auto S = matrix_inner(orbs,orbs);
  print(real(H));
  print(real(S));
  eigResultCT result = diag(H,S);
  real_matrix eigs = result.first; 
  complex_matrix evecs = result.second;
  printf("initial eigs: \n");
  for (int i = 0 ; i < norbs; i++) printf("%15.8f\n", eigs(i));
  printf("\n");
  printf("initial evecs: \n");
  orbs = transform(orbs,evecs.cols());
  
  //print(real(orbs[0]), real(orbs[1]), real(orbs[2]), real(orbs[3]), real(orbs[4]), real(orbs[5]), real(orbs[6]));
  //OrbitalCache<std::complex<double> > orbcache(norbs);
  //orbs = orbcache.append(orbs);

  std::vector<complex_tensor> rorbs(orbs.begin(), orbs.begin()+nst);
  
  return eigenpairT(rorbs,eigs(wstSlice(0,nst)));;
}

std::vector<complex_tensor> orth_gs(const std::vector<complex_tensor>& v) {
  int nsize = v.size();
  std::vector<complex_tensor> w(nsize);
  for (auto i = 0; i < nsize; i++) {
    w[i] = copy(v[i]);
    for (auto j = 0; j < i; j++) {
      auto s = inner(w[i],w[j]);
      w[i].gaxpy(1.0,w[j],-s);; 
    }
    normalize(w[i]);
  }
  return w;
}

void doit() {
  // number of orbitals
  int norbs = 13;
  // number of k-points
  int nkpts = 5;
  // make grid
  auto x = wstUtils::linspace(-L/2, L/2, NPTS);
  double hx = std::abs(x[1]-x[0]);
  // make k-grid
  auto kpts = wstUtils::linspace(-PI/L, PI/L, nkpts);
  auto kx = std::abs(kpts[1]-kpts[0]);
  // make hamiltonian kernel
  auto kpt = 0.0;
  auto Hker = build_hamiltonian_1d(x, hx, NPTS, kpt*2.*PI/L);
  // intial guess
  auto result = initial_guess(Hker, NPTS, 3*norbs, norbs, true);
  auto orbs = result.first;
  auto e = result.second;
  OrbitalCache<std::complex<double> > orbcache(2*norbs);
  //orbs = orbcache.append(orbs);


  int maxits = 55;
  // main iteration loop
  for (int iter = 0; iter < maxits; iter++) {
    printf("\n====================\nITERATION #%d\n====================\n", iter);
    norbs = orbs.size();
    printf("norbs:     %d\n\n", norbs);
  
    std::vector<complex_tensor > new_orbs(norbs);
    // loop over orbitals
    for (int iorb = 0; iorb < norbs; iorb++) {
      double shift = 0.0;
      if (e[iorb] > -1e-4) shift = 0.05 + e[iorb];
      double mu = std::sqrt(-2.0*(e[iorb]-shift));
      auto vpsi = std::complex<double>((V0-shift))*orbs[iorb];
      new_orbs[iorb] = -2.0*apply2_bsh_1d(x, hx, mu, kpt*2.*PI/L, vpsi);
    }

    normalize(new_orbs);
//    orbs = orbcache.append(new_orbs);
//    orbs = orth_gs(new_orbs);
    orbs = new_orbs;

    auto H = Hker.sandwich(orbs);
    auto S = matrix_inner(orbs,orbs);
    printf("H real:\n");
    print(real(H));
    printf("H imag:\n");
    print(imag(H));
    printf("S real:\n");
    print(real(S));
    printf("S imag:\n");
    print(imag(H));
    eigResultCT result = diag(H,S);
    //eigResultCT result = diag(H);
    auto e_new = result.first; 
    double ethresh = 0.0;
    for (int i = 0; i < norbs; i++) {
      ethresh += std::sqrt((e_new(i)-e(i))*(e_new(i)-e(i)));
    }
    ethresh = ethresh / (double)norbs;
    printf("\nethresh = %15.8e\n\n", ethresh);
    if (ethresh < 1e-8) exit(0);
    printf("eigs: (old, new)\n");
    for (int i = 0 ; i < norbs; i++) printf("%15.8f     %15.8f\n", e(i), e_new(i));
    printf("\n");
    e = e_new;
    auto c = result.second;
    orbs = transform(orbs,c.cols());
  }
}

void test_bsh() {
  auto x = wstUtils::linspace(-L/2, L/2, NPTS);
  printf("x:\n");
  for (auto xp: x) printf("%15.8f\n", xp); 
  printf("\n\n");
  double hx = std::abs(x[1]-x[0]);
  auto kpt = 0.0;
  auto Hker = build_hamiltonian_1d(x, hx, NPTS, kpt*2.*PI/L);
  complex_tensor orb;
  orb.create(std::bind(gauss, 2.5, std::placeholders::_1), x, NPTS, true);
  auto energy = -0.1;
  auto shift = (energy < -1e-4) ? 0.0 : 0.05 + energy;
  double mu = std::sqrt(-2.0*(energy-shift));
  auto vpsi = std::complex<double>((V0-shift))*orb;
  printf("shift: %15.8f\n\n", shift);
  printf("orb:\n\n");
  print(real(orb), imag(orb));
  printf("vpsi:\n\n");
  print(real(vpsi), imag(vpsi));
  //auto new_orb = -std::complex<double>(2.0,0.0)*(apply_bsh_1d(x, hx, mu, kpt*2.*PI/L, vpsi));
  auto new_orb = apply_bsh_1d(x, hx, mu, kpt*2.*PI/L, vpsi);


  //auto kx = klinspace(NPTS, hx);
  //auto r = fft(vpsi);
  //fftshift(r);
  //for (int i = 0; i < NPTS; i++) {
  //  r(i) = r(i)/((kx[i]+kpt)*(kx[i]+kpt) + mu*mu);
  //}
  //fftshift(r);
  //auto new_orb = -2.0*ifft(r);


  printf("\nnew_orb:\n\n");
  print(real(new_orb), imag(new_orb));

  //complex_tensor rorb;
  //rorb.create(std::bind(gauss, 2.5, std::placeholders::_1), x, NPTS, true);
  //print(real(rorb));
  //auto F = fft(rorb);
  //F = ifft(F);
  //print(real(F), imag(F));
  //complex_tensor crorb = rorb;
  //auto FC = fft(crorb);
  //print(real(FC), imag(FC));
}

void test_ifft() {
  complex_tensor F = linear_function_double(6);
  complex_tensor f = ifft(F);
  print(real(f), imag(f));
}

int main(int argc, char** argv) {
  //test_bsh();
  doit();
  return 0;
}
