// Generated by cpp11: do not edit by hand

#include <cpp11/R.hpp>
#include <Rcpp.h>
using namespace Rcpp;
#include "cpp11/declarations.hpp"

// dust_rng.cpp
SEXP dust_rng_alloc(int seed, int n_generators);
extern "C" SEXP _dust_dust_rng_alloc(SEXP seed, SEXP n_generators) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_alloc(cpp11::unmove(cpp11::as_cpp<int>(seed)), cpp11::unmove(cpp11::as_cpp<int>(n_generators))));
  END_CPP11
}
// dust_rng.cpp
std::vector<double> dust_rng_unif_rand(SEXP ptr, int n);
extern "C" SEXP _dust_dust_rng_unif_rand(SEXP ptr, SEXP n) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_rng_unif_rand(cpp11::unmove(cpp11::as_cpp<SEXP>(ptr)), cpp11::unmove(cpp11::as_cpp<int>(n))));
  END_CPP11
}

extern "C" {
/* .Call calls */
extern SEXP _dust_dust_rng_alloc(SEXP, SEXP);
extern SEXP _dust_dust_rng_jump(SEXP);
extern SEXP _dust_dust_rng_long_jump(SEXP);
extern SEXP _dust_dust_rng_norm_rand(SEXP, SEXP);
extern SEXP _dust_dust_rng_rbinom(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_rnorm(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_rpois(SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_runif(SEXP, SEXP, SEXP, SEXP);
extern SEXP _dust_dust_rng_size(SEXP);
extern SEXP _dust_dust_rng_unif_rand(SEXP, SEXP);
extern SEXP _dust_openmp_info();

static const R_CallMethodDef CallEntries[] = {
    {"_dust_dust_rng_alloc",     (DL_FUNC) &_dust_dust_rng_alloc,     2},
    {"_dust_dust_rng_jump",      (DL_FUNC) &_dust_dust_rng_jump,      1},
    {"_dust_dust_rng_long_jump", (DL_FUNC) &_dust_dust_rng_long_jump, 1},
    {"_dust_dust_rng_norm_rand", (DL_FUNC) &_dust_dust_rng_norm_rand, 2},
    {"_dust_dust_rng_rbinom",    (DL_FUNC) &_dust_dust_rng_rbinom,    4},
    {"_dust_dust_rng_rnorm",     (DL_FUNC) &_dust_dust_rng_rnorm,     4},
    {"_dust_dust_rng_rpois",     (DL_FUNC) &_dust_dust_rng_rpois,     3},
    {"_dust_dust_rng_runif",     (DL_FUNC) &_dust_dust_rng_runif,     4},
    {"_dust_dust_rng_size",      (DL_FUNC) &_dust_dust_rng_size,      1},
    {"_dust_dust_rng_unif_rand", (DL_FUNC) &_dust_dust_rng_unif_rand, 2},
    {"_dust_openmp_info",        (DL_FUNC) &_dust_openmp_info,        0},
    {NULL, NULL, 0}
};
}

extern "C" void R_init_dust(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}

