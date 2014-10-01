#ifndef WSTMODEL_H_
#define WSTMODEL_H_

struct wstModel {
 
  const wstKernel& _kernel; 
  const wstTensor& _localf;

  wstModel(const wstKernel& kernel, const wstTensor& localf)
   : _kernel(kernel), _localf(localf) {}

  int get_nstates() {
    return _localf.size();
  }

  wstTensor empty() {
    wstTensor r = copy(_localf, true);
    return r;
  }

  wstTensor create_random() {
    wstTensor r = copy(_localf, true);
    r.fillrandom();
    return r;
  }

  wstTensor apply(const wstTensor& t) {
    return _kernel.apply(t);
  }

};

#endif
