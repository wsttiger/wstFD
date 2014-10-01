#include <vector>

using std::vector;

//*****************************************************************************
template <typename Q, typename typeModel>
struct Lanczos
{
  typeModel* _model;
  unsigned int _nstates; // is the bigger dimension (full size)
  unsigned int _nsize; // is the size of the approximate matrix in the Lanczos basis

  Q _vinit;
  vector<double> _a;
  vector<double> _b;
  vector<double> _e;
  vector<double> _ev;



  Lanczos(typeModel* model, int nsize)
  {
    _model = model;
    _nstates = _model->get_nstates();
    _nsize = nsize;
    create_init_random_vector();
  }

  Lanczos(typeModel* model, int nsize, const Q& vinit)
  {
    _model = model;
    _nstates = _model->get_nstates();
    _nsize = nsize;
    assert(vinit.size() == _nstates);
    _vinit = copy(vinit);
  }

  void create_init_random_vector()
  {
    _vinit = _model->create_random();
  }

  // run() computes the _a and _b parameters (matrix diagonal
  // and off-diagonal) in the Lanczos basis from an initial vector
  // Initial vector can be generated or provided
  void run()
  {
    Q v = copy(_vinit);
    Q vold = _model->empty();
    _a = vector<double>(_nsize,0.0);
    _b = vector<double>(_nsize-1,0.0);
    //Q v2 = _model->empty();
    Q v2;
    for (unsigned int i = 0; i < _nsize; i++)
    {
      printf("lanczos:  i = %d\n", i);
      if (i > 0)
      {
        printf("lanczos: first branch\n");
        v2 = gaxpy(1.0,_model->apply(v),-_b[i-1],vold);
      }
      else
      {
        printf("lanczos: second branch\n");
        v2 = _model->apply(v);
      }
      printf("lanczos:  v2.ndim() = %d     v2.dim(0) = %d     v2.dim(1) = %d     v2.dim(2) = %d\n", v2.ndim(), v2.dim(0), v2.dim(1), v2.dim(2));
      _a[i] = inner(v, v2);
      printf("lanczos: completed inner()\n");
      if (i < (_nsize-1))
      {
        v2 = gaxpy(1.0,v2,-_a[i],v);
        _b[i] = norm2(v2);
        vold = copy(v);
        v = copy(v2);
        v.scale(1./_b[i]);
      }
    }

    vector<double> mat(_nsize*_nsize,0.0);
    for (unsigned int i = 0; i < _nsize-1; i++)
    {
      mat[i*_nsize+i] = _a[i];
      mat[i*_nsize+i+1] = _b[i];
      mat[(i+1)*_nsize+i] = _b[i];
    }
    mat[_nsize*_nsize-1] = _a[_nsize-1];

    _e = vector<double>(_nsize,0.0);
    _ev = vector<double>(_nsize*_nsize, 0.0);
    //diag_matrix(mat,_nsize,_e,_ev);
    printf("Lanczos: lowest eigenvalue is %15.8f\n\n", _e[0]);
  }

  // run(vector) computes the lowest eigenstate using the _a and _b
  // parameters that were already generated from a previous run
  vector<double> run(const Q& vin)
  {
    // create return vector
    Q rv = _model->empty();
    // copy initial vector
    Q v = copy(vin);
    Q vold = _model->empty();
    Q v2 = _model->empty();
    for (unsigned int i = 0; i < _nsize; i++)
    {
      // build up return value
      rv = gaxpy(1.0,rv,_ev[i*_nsize],v); 
      if (i > 0)
      {
        v2 = gaxpy(1.0,_model->apply(v),-_b[i-1],vold);
      }
      else
      {
        v2 = _model->apply(v);
      }
      if (i < (_nsize-1))
      {
        v2 = gaxpy(1.0,v2,-_a[i],v);
        vold = copy(v);
        v = copy(v2);
        v.scale(1./_b[i]);
      }
    }
    normalize(rv);
    return rv;
  }

  vector<double> lowstate()
  {
    return run(_vinit);
  }

};
//*****************************************************************************
