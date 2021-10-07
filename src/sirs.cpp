// Generated by dust (version 0.9.22) - do not edit
#include <cpp11.hpp>
[[cpp11::register]]
SEXP dust_sirs_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp device_config);

[[cpp11::register]]
SEXP dust_sirs_run(SEXP ptr, size_t step_end, bool device);

[[cpp11::register]]
SEXP dust_sirs_simulate(SEXP ptr, cpp11::sexp step_end, bool device);

[[cpp11::register]]
SEXP dust_sirs_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state);

[[cpp11::register]]
SEXP dust_sirs_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_sirs_step(SEXP ptr);

[[cpp11::register]]
void dust_sirs_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_sirs_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_sirs_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_sirs_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_sirs_compare_data(SEXP ptr, bool device);

[[cpp11::register]]
SEXP dust_sirs_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device);

[[cpp11::register]]
cpp11::sexp dust_sirs_capabilities();

[[cpp11::register]]
void dust_sirs_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_sirs_n_state(SEXP ptr);

[[cpp11::register]]
cpp11::sexp dust_sirs_device_info();

#include <dust/dust.hpp>
#include <dust/interface/dust.hpp>

class sirs {
public:
  typedef double real_t;
  typedef dust::no_internal internal_type;
  typedef dust::random::xoshiro256starstar_state rng_state_type;

  // ALIGN(16) is required before the data_t definition when using NVCC
  // This is so when loaded into shared memory it is aligned correctly
  struct ALIGN(16) data_t {
    double incidence;
  };

  struct shared_type {
    real_t S0;
    real_t I0;
    real_t R0;
    real_t alpha;
    real_t beta;
    real_t gamma;
    real_t dt;
    size_t freq;
    real_t exp_noise;
  };

  sirs(const dust::pars_type<sirs>& pars): shared(pars.shared) {
  }

  size_t size() {
    return 4;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(4);
    state[0] = shared->S0;
    state[1] = shared->I0;
    state[2] = shared->R0;
    state[3] = 0;
    return state;
  }

  void update(size_t step, const real_t * state, rng_state_type& rng_state,
              real_t * state_next) {
    real_t S = state[0];
    real_t I = state[1];
    real_t R = state[2];
    real_t N = S + I + R;

    real_t p_SI = 1 - exp(- shared->beta * I / N);
    real_t p_IR = 1 - exp(-(shared->gamma));
    real_t p_RS = 1 - exp(- shared->alpha);

    real_t dt = shared->dt;
    real_t n_SI = dust::random::binomial<real_t>(rng_state, S, p_SI * dt);
    real_t n_IR = dust::random::binomial<real_t>(rng_state, I, p_IR * dt);
    real_t n_RS = dust::random::binomial<real_t>(rng_state, R, p_RS * dt);

    state_next[0] = S - n_SI + n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR - n_RS;
    state_next[3] = (step % shared->freq == 0) ? n_SI : state[3] + n_SI;
  }

  real_t compare_data(const real_t * state, const data_t& data,
                      rng_state_type& rng_state) {
    const real_t incidence_modelled = state[3];
    const real_t incidence_observed = data.incidence;
    const real_t lambda = incidence_modelled +
      dust::random::exponential<real_t>(rng_state, shared->exp_noise);
    return dust::density::poisson(incidence_observed, lambda, true);
  }

private:
  dust::shared_ptr<sirs> shared;
};

// Helper function for accepting values with defaults
inline double with_default(double default_value, cpp11::sexp value) {
  return value == R_NilValue ? default_value : cpp11::as_cpp<double>(value);
}

namespace dust {
template <>
dust::pars_type<sirs> dust_pars<sirs>(cpp11::list pars) {
  // Initial state values
  sirs::real_t I0 = 10.0;
  sirs::real_t S0 = 1000.0;
  sirs::real_t R0 = 0.0;

  // Time scaling
  // [[dust::param(freq, required = FALSE, default = 1)]]
  size_t freq = std::max(1.0, with_default(1.0, pars["freq"]));
  sirs::real_t dt = 1 / static_cast<sirs::real_t>(freq);

  sirs::real_t exp_noise = 1e6;

  // [[dust::param(alpha, required = FALSE, default = 0.1)]]
  sirs::real_t alpha = with_default(0.1, pars["alpha"]);

  // [[dust::param(beta, required = FALSE, default = 0.2)]]
  sirs::real_t beta = with_default(0.2, pars["beta"]);

  // [[dust::param(gamma, required = FALSE, default = 0.1)]]
  sirs::real_t gamma = with_default(0.1, pars["gamma"]);

  sirs::shared_type shared{S0, I0, R0, alpha, beta, gamma, dt, freq, exp_noise};
  return dust::pars_type<sirs>(shared);
}

template <>
sirs::data_t dust_data<sirs>(cpp11::list data) {
  return sirs::data_t{cpp11::as_cpp<double>(data["incidence"])};
}

template <>
struct has_gpu_support<sirs> : std::true_type {};

namespace cuda {

template <>
size_t device_shared_int_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 1;
}

template <>
size_t device_shared_real_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 5;
}

template <>
void device_shared_copy<sirs>(dust::shared_ptr<sirs> shared,
                              int * shared_int,
                              sirs::real_t * shared_real) {
  typedef sirs::real_t real_t;
  using dust::cuda::shared_copy;
  shared_real = shared_copy<real_t>(shared_real, shared->alpha);
  shared_real = shared_copy<real_t>(shared_real, shared->beta);
  shared_real = shared_copy<real_t>(shared_real, shared->gamma);
  shared_real = shared_copy<real_t>(shared_real, shared->dt);
  shared_real = shared_copy<real_t>(shared_real, shared->exp_noise);
  shared_int = shared_copy<int>(shared_int, shared->freq);
}

template <>
DEVICE
void update_device<sirs>(size_t step,
                         const dust::cuda::interleaved<sirs::real_t> state,
                         dust::cuda::interleaved<int> internal_int,
                         dust::cuda::interleaved<sirs::real_t> internal_real,
                         const int * shared_int,
                         const sirs::real_t * shared_real,
                         sirs::rng_state_type& rng_state,
                         dust::cuda::interleaved<sirs::real_t> state_next) {
  typedef sirs::real_t real_t;
  const real_t alpha = shared_real[0];
  const real_t beta = shared_real[1];
  const real_t gamma = shared_real[2];
  const real_t dt = shared_real[3];
  const int freq = shared_int[0];
  const real_t S = state[0];
  const real_t I = state[1];
  const real_t R = state[2];
  const real_t N = S + I + R;
  const real_t p_SI = 1 - exp(- beta * I / N);
  const real_t p_IR = 1 - exp(- gamma);
  const real_t p_RS = 1 - exp(- alpha);
  const real_t n_SI = dust::random::binomial<real_t>(rng_state, S, p_SI * dt);
  const real_t n_IR = dust::random::binomial<real_t>(rng_state, I, p_IR * dt);
  const real_t n_RS = dust::random::binomial<real_t>(rng_state, R, p_RS * dt);
  state_next[0] = S - n_SI + n_RS;
  state_next[1] = I + n_SI - n_IR;
  state_next[2] = R + n_IR - n_RS;
  state_next[3] = (step % freq == 0) ? n_SI : state[3] + n_SI;
}

template <>
DEVICE
sirs::real_t compare_device<sirs>(const dust::cuda::interleaved<sirs::real_t> state,
                                  const sirs::data_t& data,
                                  dust::cuda::interleaved<int> internal_int,
                                  dust::cuda::interleaved<sirs::real_t> internal_real,
                                  const int * shared_int,
                                  const sirs::real_t * shared_real,
                                  sirs::rng_state_type& rng_state) {
  typedef sirs::real_t real_t;
  const real_t exp_noise = shared_real[4];
  const real_t incidence_modelled = state[3];
  const real_t incidence_observed = data.incidence;
  const real_t lambda = incidence_modelled +
    dust::random::exponential<real_t>(rng_state, exp_noise);
  return dust::density::poisson(incidence_observed, lambda, true);
}

}
}

SEXP dust_sirs_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp device_config) {
  return dust::r::dust_alloc<sirs>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        device_config);
}

SEXP dust_sirs_run(SEXP ptr, size_t step_end, bool device) {
  return dust::r::dust_run<sirs>(ptr, step_end, device);
}

SEXP dust_sirs_simulate(SEXP ptr, cpp11::sexp step_end, bool device) {
  return dust::r::dust_simulate<sirs>(ptr, step_end, device);
}

SEXP dust_sirs_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<sirs>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state) {
  return dust::r::dust_update_state<sirs>(ptr, r_pars, r_state, r_step,
                                               r_set_initial_state);
}

SEXP dust_sirs_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<sirs>(ptr, r_index);
}

size_t dust_sirs_step(SEXP ptr) {
  return dust::r::dust_step<sirs>(ptr);
}

void dust_sirs_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<sirs>(ptr, r_index);
}

SEXP dust_sirs_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<sirs>(ptr, r_weights);
}

SEXP dust_sirs_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<sirs>(ptr, first_only, last_only);
}

SEXP dust_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<sirs>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_sirs_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<sirs>(ptr, data);
  return R_NilValue;
}

SEXP dust_sirs_compare_data(SEXP ptr, bool device) {
  return dust::r::dust_compare_data<sirs>(ptr, device);
}

SEXP dust_sirs_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device) {
  return dust::r::dust_filter<sirs>(ptr, save_trajectories, step_snapshot, device);
}

cpp11::sexp dust_sirs_capabilities() {
  return dust::r::dust_capabilities<sirs>();
}

void dust_sirs_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<sirs>(ptr, n_threads);
}

int dust_sirs_n_state(SEXP ptr) {
  return dust::r::dust_n_state<sirs>(ptr);
}

cpp11::sexp dust_sirs_device_info() {
  return dust::cuda::device_info<sirs::real_t>();
}
