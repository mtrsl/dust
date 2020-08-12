// Generated by cpp11: do not edit by hand
// clang-format off


#include "cpp11/declarations.hpp"

// dust_rng.cpp
SEXP dust_rng_alloc(int seed, int n_generators);
extern "C" SEXP _dust_dust_rng_alloc(SEXP seed, SEXP n_generators) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_alloc(cpp11::as_cpp<cpp11::decay_t<int>>(seed), cpp11::as_cpp<cpp11::decay_t<int>>(n_generators)));
  END_CPP11
}
// dust_rng.cpp
int dust_rng_size(SEXP ptr);
extern "C" SEXP _dust_dust_rng_size(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_size(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// dust_rng.cpp
void dust_rng_jump(SEXP ptr);
extern "C" SEXP _dust_dust_rng_jump(SEXP ptr) {
  BEGIN_CPP11
    dust_rng_jump(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr));
    return R_NilValue;
  END_CPP11
}
// dust_rng.cpp
void dust_rng_long_jump(SEXP ptr);
extern "C" SEXP _dust_dust_rng_long_jump(SEXP ptr) {
  BEGIN_CPP11
    dust_rng_long_jump(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr));
    return R_NilValue;
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_unif_rand(SEXP ptr, int n);
extern "C" SEXP _dust_dust_rng_unif_rand(SEXP ptr, SEXP n) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_unif_rand(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n)));
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_norm_rand(SEXP ptr, int n);
extern "C" SEXP _dust_dust_rng_norm_rand(SEXP ptr, SEXP n) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_norm_rand(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n)));
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_runif(SEXP ptr, int n, std::vector<double> min, std::vector<double> max);
extern "C" SEXP _dust_dust_rng_runif(SEXP ptr, SEXP n, SEXP min, SEXP max) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_runif(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(min), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(max)));
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_rnorm(SEXP ptr, int n, std::vector<double> mean, std::vector<double> sd);
extern "C" SEXP _dust_dust_rng_rnorm(SEXP ptr, SEXP n, SEXP mean, SEXP sd) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_rnorm(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(mean), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(sd)));
  END_CPP11
}
// dust_rng.cpp
std::vector<int> dust_rng_rbinom(SEXP ptr, int n, std::vector<int> size, std::vector<double> prob);
extern "C" SEXP _dust_dust_rng_rbinom(SEXP ptr, SEXP n, SEXP size, SEXP prob) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_rbinom(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<int>>>(size), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(prob)));
  END_CPP11
}
// dust_rng.cpp
std::vector<int> dust_rng_rpois(SEXP ptr, int n, std::vector<double> lambda);
extern "C" SEXP _dust_dust_rng_rpois(SEXP ptr, SEXP n, SEXP lambda) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_rpois(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<int>>(n), cpp11::as_cpp<cpp11::decay_t<std::vector<double>>>(lambda)));
  END_CPP11
}
// dust_rng.cpp
cpp11::writable::raws dust_rng_state(SEXP ptr);
extern "C" SEXP _dust_dust_rng_state(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// openmp.cpp
cpp11::writable::list cpp_openmp_info();
extern "C" SEXP _dust_cpp_openmp_info() {
  BEGIN_CPP11
    return cpp11::as_sexp(cpp_openmp_info());
  END_CPP11
}

extern "C" {
/* .Call calls */
extern SEXP _dust_cpp_openmp_info();
extern SEXP _dust_dust_rng_alloc(SEXP, SEXP);
extern SEXP _dust_dust_rng_jump(SEXP);
extern SEXP _dust_dust_rng_long_jump(SEXP);
extern SEXP _dust_dust_rng_norm_rand(SEXP, SEXP);
extern SEXP _dust_dust_rng_rbinom(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_rnorm(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_rpois(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_runif(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_size(SEXP);
extern SEXP _dust_dust_rng_state(SEXP);
extern SEXP _dust_dust_rng_unif_rand(SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_dust_cpp_openmp_info",    (DL_FUNC) &_dust_cpp_openmp_info,    0},
    {"_dust_dust_rng_alloc",     (DL_FUNC) &_dust_dust_rng_alloc,     2},
    {"_dust_dust_rng_jump",      (DL_FUNC) &_dust_dust_rng_jump,      1},
    {"_dust_dust_rng_long_jump", (DL_FUNC) &_dust_dust_rng_long_jump, 1},
    {"_dust_dust_rng_norm_rand", (DL_FUNC) &_dust_dust_rng_norm_rand, 2},
    {"_dust_dust_rng_rbinom",    (DL_FUNC) &_dust_dust_rng_rbinom,    4},
    {"_dust_dust_rng_rnorm",     (DL_FUNC) &_dust_dust_rng_rnorm,     4},
    {"_dust_dust_rng_rpois",     (DL_FUNC) &_dust_dust_rng_rpois,     3},
    {"_dust_dust_rng_runif",     (DL_FUNC) &_dust_dust_rng_runif,     4},
    {"_dust_dust_rng_size",      (DL_FUNC) &_dust_dust_rng_size,      1},
    {"_dust_dust_rng_state",     (DL_FUNC) &_dust_dust_rng_state,     1},
    {"_dust_dust_rng_unif_rand", (DL_FUNC) &_dust_dust_rng_unif_rand, 2},
    {NULL, NULL, 0}
};
}

extern "C" void R_init_dust(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
