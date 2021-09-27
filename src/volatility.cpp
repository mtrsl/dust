// Generated by dust (version 0.9.17) - do not edit
#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_volatility_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_config);

[[cpp11::register]]
SEXP dust_volatility_run(SEXP ptr, size_t step_end, bool device,
                       bool deterministic);

[[cpp11::register]]
SEXP dust_volatility_simulate(SEXP ptr, cpp11::sexp step_end, bool device,
                            bool deterministic);

[[cpp11::register]]
SEXP dust_volatility_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_volatility_set_state(SEXP ptr, SEXP r_state, SEXP r_step,
                             bool deterministic);

[[cpp11::register]]
SEXP dust_volatility_reset(SEXP ptr, cpp11::list r_pars, size_t step);

[[cpp11::register]]
SEXP dust_volatility_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_volatility_step(SEXP ptr);

[[cpp11::register]]
void dust_volatility_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_volatility_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_volatility_set_pars(SEXP ptr, cpp11::list r_pars);

[[cpp11::register]]
SEXP dust_volatility_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_volatility_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_volatility_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_volatility_compare_data(SEXP ptr, bool device);

[[cpp11::register]]
SEXP dust_volatility_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device);

[[cpp11::register]]
cpp11::sexp dust_volatility_capabilities();

[[cpp11::register]]
void dust_volatility_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_volatility_n_state(SEXP ptr);

[[cpp11::register]]
cpp11::sexp dust_volatility_device_info();

#include <dust/dust.hpp>
#include <dust/interface.hpp>

class volatility {
public:
  typedef double real_t;
  struct data_t {
    real_t observed;
  };
  typedef dust::no_internal internal_t;

  struct shared_t {
    real_t alpha;
    real_t sigma;
    real_t gamma;
    real_t tau;
    real_t x0;
  };

  volatility(const dust::pars_t<volatility>& pars) : shared(pars.shared) {
  }

  size_t size() {
    return 1;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1);
    state[0] = shared->x0;
    return state;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state, real_t * state_next) {
    const real_t x = state[0];
    state_next[0] = shared->alpha * x +
      shared->sigma * dust::distr::rnorm(rng_state, 0, 1);
  }

  real_t compare_data(const real_t * state, const data_t& data,
                      dust::rng_state_t<real_t>& rng_state) {
    return dust::dnorm(data.observed, shared->gamma * state[0],
                       shared->tau, true);
  }

private:
  dust::shared_ptr<volatility> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {

template <>
dust::pars_t<volatility> dust_pars<volatility>(cpp11::list pars) {
  typedef volatility::real_t real_t;
  real_t x0 = 0;
  real_t alpha = with_default(0.91, pars["alpha"]);
  real_t sigma = with_default(1, pars["sigma"]);
  real_t gamma = with_default(1, pars["gamma"]);
  real_t tau = with_default(1, pars["tau"]);

  volatility::shared_t shared{alpha, sigma, gamma, tau, x0};
  return dust::pars_t<volatility>(shared);
}

template <>
volatility::data_t dust_data<volatility>(cpp11::list data) {
  return volatility::data_t{cpp11::as_cpp<double>(data["observed"])};
}

}

SEXP dust_volatility_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_config) {
  return dust::r::dust_alloc<volatility>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, device_config);
}

SEXP dust_volatility_run(SEXP ptr, size_t step_end, bool device,
                       bool deterministic) {
  return dust::r::dust_run<volatility>(ptr, step_end, device, deterministic);
}

SEXP dust_volatility_simulate(SEXP ptr, cpp11::sexp step_end, bool device,
                            bool deterministic) {
  return dust::r::dust_simulate<volatility>(ptr, step_end, device,
                                           deterministic);
}

SEXP dust_volatility_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<volatility>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_volatility_set_state(SEXP ptr, SEXP r_state, SEXP r_step,
                             bool deterministic) {
  dust::r::dust_set_state<volatility>(ptr, r_state, r_step, deterministic);
  return R_NilValue;
}

SEXP dust_volatility_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust::r::dust_reset<volatility>(ptr, r_pars, step);
}

SEXP dust_volatility_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<volatility>(ptr, r_index);
}

size_t dust_volatility_step(SEXP ptr) {
  return dust::r::dust_step<volatility>(ptr);
}

void dust_volatility_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<volatility>(ptr, r_index);
}

SEXP dust_volatility_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<volatility>(ptr, r_weights);
}

SEXP dust_volatility_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust::r::dust_set_pars<volatility>(ptr, r_pars);
}

SEXP dust_volatility_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<volatility>(ptr, first_only, last_only);
}

SEXP dust_volatility_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<volatility>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_volatility_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<volatility>(ptr, data);
  return R_NilValue;
}

SEXP dust_volatility_compare_data(SEXP ptr, bool device) {
  return dust::r::dust_compare_data<volatility>(ptr, device);
}

SEXP dust_volatility_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device) {
  return dust::r::dust_filter<volatility>(ptr, save_trajectories, step_snapshot, device);
}

cpp11::sexp dust_volatility_capabilities() {
  return dust::r::dust_capabilities<volatility>();
}

void dust_volatility_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<volatility>(ptr, n_threads);
}

int dust_volatility_n_state(SEXP ptr) {
  return dust::r::dust_n_state<volatility>(ptr);
}

cpp11::sexp dust_volatility_device_info() {
  return dust::cuda::device_info<volatility::real_t>();
}
