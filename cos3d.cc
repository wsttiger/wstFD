#include <cstdio>
#include "wstTensor.h"
#include "wstKernel.h"
#include "wstMatrix.h"

const double alpha = 2.5;
const double L = 5.0;
const int NPTS = 20;

#define PI 3.141592653589793238462643383279502884197

double V(double x, double y, double z) {
  return -alpha*(std::cos(2.0*PI*x/L)*cos(2.0*PI*y/L)*cos(2.0*PI*z/L)+1.0);
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

//  std::vector<wstTensorT<Q> > append(const std::vector<wstTensorT<Q> >& orbs) {
//    unsigned int szorbs = orbs.size();
//    unsigned int szorbs2 = _orbs.size();
//    std::vector<wstTensorT<Q> > combined_orbs;
//    combined_orbs.insert(combined_orbs.begin(), orbs.begin(), orbs.end());
//    combined_orbs.insert(combined_orbs.end(), _orbs.begin(), _orbs.end());
//
//    printf("OrbitalCache::append --> performing outer product ...\n");
//    wstMatrixT<Q> S = outer(combined_orbs, combined_orbs);
//    printf("OrbitalCache::append --> diagonalizing matrix ...\n");
//    std::pair<wstMatrixT<Q>, wstMatrixT<Q> > result = diag(S);
//    printf("OrbitalCache::append --> done  diagonalizing matrix ...\n");
//    wstMatrixT<Q> eigs = result.first;
//    wstMatrixT<Q> evecs = result.second;
//
//    int norbs = 0;
//    for (int i = 0; i < S.nrows(); i++) {
//      if (std::abs(eigs(i)) > _thresh) norbs++;
//    }
//
//    std::vector<wstTensorT<Q> > R(norbs);
//    for (int i = 0, j = 0; i < norbs; j++) {
//      if (std::abs(eigs(j)) > _thresh) {
//        R[i++] = copy_and_fill(orbs[0],evecs.col(j));
//      }
//    }
//    return R; 
//  }

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
    print(S);
    std::vector<wstTensorT<Q> > rorbs = transform<Q>(orbs,evecs.cols(wstMatrixSlice(indx,S.ncols()-1)));
    normalize(rorbs);
    return rorbs;
  }
};

//std::vector<wstKernel3D<double> > apply_bsh(const std::vector<double>& x,
//                                            const std::vector<double>& y,
//                                            const std::vector<double>& z,
//                                            double hx, double hy, double hz,
//                                            const std::vector<wstTensorT<double> > orbs) {
//}


wstKernel3D<double> build_hamiltonian(const std::vector<double>& x, 
                                      const std::vector<double>& y, 
                                      const std::vector<double>& z, 
                                      double hx, double hy, double hz, int npts) {
  wstTensorT<double> Vpot;
  //Vpot.create(std::bind(V, L, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), x, y, z, npts, npts, npts, true, true, true);
  Vpot.create(V, x, y, z, npts, npts, npts, true, true, true);
  wstKernel3D<double> H = create_laplacian_7p_3d(Vpot, hx, hy, hz, -0.5); 
  return H;
}

std::vector<wstTensorT<double> > make_initial_guess(const wstKernel3D<double>& H, int npts0, int npts1, int npts2, int norbs) {
  std::vector<wstTensorT<double> > orbs;
  for (int i = 0; i < norbs; i++) {
    if (i == 0) {
      wstTensorT<double> f = constant_function<double>(npts0, npts1, npts2, 1.0, true, true, true);
      //wstTensorT<double> f = random_function_double(npts0, npts1, npts2, true, true, true);
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

void test_3d() {
  printf("building vectors ...\n");
  vector<double> x = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> y = wstUtils::linspace(-L/2, L/2, NPTS);
  vector<double> z = wstUtils::linspace(-L/2, L/2, NPTS);
  printf("spacing ...\n");
  double hx = std::abs(x[1]-x[0]);
  double hy = std::abs(y[1]-y[0]);
  double hz = std::abs(y[1]-y[0]);
  printf("building hamiltonian kernel ...\n");
  wstKernel3D<double> Hker = build_hamiltonian(x, y, z, hx, hy, hz, NPTS);
  printf("random guess orbitals ...\n");
  std::vector<wstTensorT<double> > orbs = make_initial_guess(Hker, NPTS, NPTS, NPTS, 100);
  printf("making overlap ...\n");
  wstMatrixT<double> S = matrix_inner(orbs, orbs);
  printf("making hamiltonian matrix ...\n");
  wstMatrixT<double> H = Hker.sandwich(orbs);
  H = 0.5*(H + transpose(H));
  printf("diagonalizing hamiltonian matrix ...\n");
  std::pair<wstMatrixT<double>, wstMatrixT<double> > result = diag(H);
  print(result.first);
}

int main(int argc, char** argv) {
  //test_3d();
  test_orbital_cache();
  return 0;
}
