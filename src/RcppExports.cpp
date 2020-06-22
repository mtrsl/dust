// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// openmp_info
Rcpp::List openmp_info();
RcppExport SEXP _dust_openmp_info() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    rcpp_result_gen = Rcpp::wrap(openmp_info());
    return rcpp_result_gen;
END_RCPP
}
// test_rng_norm
Rcpp::NumericVector test_rng_norm(int n, int seed, int n_generators);
RcppExport SEXP _dust_test_rng_norm(SEXP nSEXP, SEXP seedSEXP, SEXP n_generatorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type n_generators(n_generatorsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_rng_norm(n, seed, n_generators));
    return rcpp_result_gen;
END_RCPP
}
// test_rng_unif
Rcpp::NumericVector test_rng_unif(int n, double min, double max, int seed, int n_generators);
RcppExport SEXP _dust_test_rng_unif(SEXP nSEXP, SEXP minSEXP, SEXP maxSEXP, SEXP seedSEXP, SEXP n_generatorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type min(minSEXP);
    Rcpp::traits::input_parameter< double >::type max(maxSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type n_generators(n_generatorsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_rng_unif(n, min, max, seed, n_generators));
    return rcpp_result_gen;
END_RCPP
}
// test_rng_binom
Rcpp::IntegerVector test_rng_binom(std::vector<int> n, std::vector<double> p, int seed, int n_generators);
RcppExport SEXP _dust_test_rng_binom(SEXP nSEXP, SEXP pSEXP, SEXP seedSEXP, SEXP n_generatorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< std::vector<int> >::type n(nSEXP);
    Rcpp::traits::input_parameter< std::vector<double> >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type n_generators(n_generatorsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_rng_binom(n, p, seed, n_generators));
    return rcpp_result_gen;
END_RCPP
}
// test_rng_pois
Rcpp::IntegerVector test_rng_pois(std::vector<double> lambda, int seed, int n_generators);
RcppExport SEXP _dust_test_rng_pois(SEXP lambdaSEXP, SEXP seedSEXP, SEXP n_generatorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< std::vector<double> >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type n_generators(n_generatorsSEXP);
    rcpp_result_gen = Rcpp::wrap(test_rng_pois(lambda, seed, n_generators));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_dust_openmp_info", (DL_FUNC) &_dust_openmp_info, 0},
    {"_dust_test_rng_norm", (DL_FUNC) &_dust_test_rng_norm, 3},
    {"_dust_test_rng_unif", (DL_FUNC) &_dust_test_rng_unif, 5},
    {"_dust_test_rng_binom", (DL_FUNC) &_dust_test_rng_binom, 4},
    {"_dust_test_rng_pois", (DL_FUNC) &_dust_test_rng_pois, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_dust(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
