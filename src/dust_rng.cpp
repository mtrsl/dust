#include <cstring>
#include <cpp11/external_pointer.hpp>
#include <cpp11/raws.hpp>
#include <dust/rng2.hpp>

typedef dust::pRNG<double> dust_rng_t;
typedef cpp11::external_pointer<dust_rng_t> dust_rng_ptr_t;

[[cpp11::register]]
SEXP dust_rng_alloc(int seed, int n_generators) {
  dust_rng_t *rng = new dust_rng_t(n_generators, seed);
  return cpp11::external_pointer<dust_rng_t>(rng);
}

[[cpp11::register]]
int dust_rng_size(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  return static_cast<int>(rng->size());
}

[[cpp11::register]]
void dust_rng_jump(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  rng->jump();
}

[[cpp11::register]]
void dust_rng_long_jump(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  rng->long_jump();
}

[[cpp11::register]]
std::vector<double> dust_rng_unif_rand(SEXP ptr, int n) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::unif_rand(rng->state(i % n_generators));
  }
  return y;
}

// NOTE: no special treatment (yet) for this
[[cpp11::register]]
std::vector<double> dust_rng_norm_rand(SEXP ptr, int n) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rnorm(rng->state(i % n_generators), 0, 1);
  }
  return y;
}

[[cpp11::register]]
std::vector<double> dust_rng_runif(SEXP ptr, int n, std::vector<double> min,
                                   std::vector<double> max) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::runif(rng->state(i % n_generators), min[i], max[i]);
  }
  return y;
}

[[cpp11::register]]
std::vector<double> dust_rng_rnorm(SEXP ptr, int n, std::vector<double> mean,
                                   std::vector<double> sd) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();
  std::vector<double> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rnorm(rng->state(i % n_generators), mean[i], sd[i]);
  }
  return y;
}

[[cpp11::register]]
std::vector<int> dust_rng_rbinom(SEXP ptr, int n, std::vector<int> size,
                                 std::vector<double> prob) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();
  std::vector<int> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rbinom(rng->state(i % n_generators), size[i], prob[i]);
  }
  return y;
}

[[cpp11::register]]
std::vector<int> dust_rng_rpois(SEXP ptr, int n, std::vector<double> lambda) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  const size_t n_generators = rng->size();
  std::vector<int> y(n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    y[i] = dust::distr::rpois(rng->state(i % n_generators), lambda[i]);
  }
  return y;
}

[[cpp11::register]]
cpp11::writable::raws dust_rng_state(SEXP ptr) {
  dust_rng_t *rng = cpp11::as_cpp<dust_rng_ptr_t>(ptr).get();
  auto state = rng->export_state();
  size_t len = sizeof(uint64_t) * state.size();
  cpp11::writable::raws ret(len);
  std::memcpy(RAW(ret), state.data(), len);
  return ret;
}
