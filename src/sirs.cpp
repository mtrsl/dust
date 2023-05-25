// Generated by dust (version 0.14.4) - do not edit
#include <cpp11.hpp>

[[cpp11::register]]
cpp11::sexp dust_sirs_gpu_info();
[[cpp11::register]]
SEXP dust_cpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                         cpp11::sexp r_n_particles, int n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config, cpp11::sexp ode_control);

[[cpp11::register]]
cpp11::sexp dust_cpu_sirs_capabilities();

[[cpp11::register]]
SEXP dust_cpu_sirs_run(SEXP ptr, cpp11::sexp r_time_end);

[[cpp11::register]]
SEXP dust_cpu_sirs_simulate(SEXP ptr, cpp11::sexp time_end);

[[cpp11::register]]
SEXP dust_cpu_sirs_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state,
                                           SEXP index, SEXP reset_step_size);

[[cpp11::register]]
SEXP dust_cpu_sirs_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
SEXP dust_cpu_sirs_time(SEXP ptr);

[[cpp11::register]]
void dust_cpu_sirs_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_sirs_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_cpu_sirs_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_cpu_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_cpu_sirs_set_data(SEXP ptr, cpp11::list data, bool shared);

[[cpp11::register]]
SEXP dust_cpu_sirs_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_cpu_sirs_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood);

[[cpp11::register]]
void dust_cpu_sirs_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_cpu_sirs_n_state(SEXP ptr);

[[cpp11::register]]
void dust_cpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time);

[[cpp11::register]]
SEXP dust_cpu_sirs_ode_statistics(SEXP ptr);
[[cpp11::register]]
SEXP dust_gpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                         cpp11::sexp r_n_particles, int n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config, cpp11::sexp ode_control);

[[cpp11::register]]
cpp11::sexp dust_gpu_sirs_capabilities();

[[cpp11::register]]
SEXP dust_gpu_sirs_run(SEXP ptr, cpp11::sexp r_time_end);

[[cpp11::register]]
SEXP dust_gpu_sirs_simulate(SEXP ptr, cpp11::sexp time_end);

[[cpp11::register]]
SEXP dust_gpu_sirs_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_gpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state,
                                           SEXP index, SEXP reset_step_size);

[[cpp11::register]]
SEXP dust_gpu_sirs_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
SEXP dust_gpu_sirs_time(SEXP ptr);

[[cpp11::register]]
void dust_gpu_sirs_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_gpu_sirs_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_gpu_sirs_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_gpu_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_gpu_sirs_set_data(SEXP ptr, cpp11::list data, bool shared);

[[cpp11::register]]
SEXP dust_gpu_sirs_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_gpu_sirs_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood);

[[cpp11::register]]
void dust_gpu_sirs_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_gpu_sirs_n_state(SEXP ptr);

[[cpp11::register]]
void dust_gpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time);

[[cpp11::register]]
SEXP dust_gpu_sirs_ode_statistics(SEXP ptr);
#include <dust/r/dust.hpp>

class sirs {
public:
  using real_type = double;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  // __align__(16) is required before the data_type definition when using NVCC
  // This is so when loaded into shared memory it is aligned correctly
  struct __align__(16) data_type {
    double incidence;
  };

  struct shared_type {
    real_type S0;
    real_type I0;
    real_type R0;
    real_type alpha;
    real_type beta;
    real_type gamma;
    real_type dt;
    size_t freq;
    real_type exp_noise;
  };

  sirs(const dust::pars_type<sirs>& pars): shared(pars.shared) {
  }

  size_t size() const {
    return 4;
  }

  std::vector<real_type> initial(size_t time, rng_state_type& rng_state) {
    std::vector<real_type> state(4);
    state[0] = shared->S0;
    state[1] = shared->I0;
    state[2] = shared->R0;
    state[3] = 0;
    return state;
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    real_type S = state[0];
    real_type I = state[1];
    real_type R = state[2];
    real_type N = S + I + R;

    real_type p_SI = 1 - exp(- shared->beta * I / N);
    real_type p_IR = 1 - exp(-(shared->gamma));
    real_type p_RS = 1 - exp(- shared->alpha);

    real_type dt = shared->dt;
    real_type n_SI = dust::random::binomial<real_type>(rng_state, S, p_SI * dt);
    real_type n_IR = dust::random::binomial<real_type>(rng_state, I, p_IR * dt);
    real_type n_RS = dust::random::binomial<real_type>(rng_state, R, p_RS * dt);

    state_next[0] = S - n_SI + n_RS;
    state_next[1] = I + n_SI - n_IR;
    state_next[2] = R + n_IR - n_RS;
    state_next[3] = (time % shared->freq == 0) ? n_SI : state[3] + n_SI;
  }

  real_type compare_data(const real_type * state, const data_type& data,
                         rng_state_type& rng_state) {
    const real_type incidence_modelled = state[3];
    const real_type incidence_observed = data.incidence;
    const real_type lambda = incidence_modelled +
      dust::random::exponential<real_type>(rng_state, shared->exp_noise);
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
  sirs::real_type I0 = 10.0;
  sirs::real_type S0 = 1000.0;
  sirs::real_type R0 = 0.0;

  // Time scaling
  // [[dust::param(freq, required = FALSE, default = 1)]]
  size_t freq = std::max(1.0, with_default(1.0, pars["freq"]));
  sirs::real_type dt = 1 / static_cast<sirs::real_type>(freq);

  sirs::real_type exp_noise = 1e6;

  // [[dust::param(alpha, required = FALSE, default = 0.1)]]
  sirs::real_type alpha = with_default(0.1, pars["alpha"]);

  // [[dust::param(beta, required = FALSE, default = 0.2)]]
  sirs::real_type beta = with_default(0.2, pars["beta"]);

  // [[dust::param(gamma, required = FALSE, default = 0.1)]]
  sirs::real_type gamma = with_default(0.1, pars["gamma"]);

  sirs::shared_type shared{S0, I0, R0, alpha, beta, gamma, dt, freq, exp_noise};
  return dust::pars_type<sirs>(shared);
}

template <>
sirs::data_type dust_data<sirs>(cpp11::list data) {
  return sirs::data_type{cpp11::as_cpp<double>(data["incidence"])};
}

namespace gpu {

template <>
size_t shared_int_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 1;
}

template <>
size_t shared_real_size<sirs>(dust::shared_ptr<sirs> shared) {
  return 5;
}

template <>
void shared_copy<sirs>(dust::shared_ptr<sirs> shared,
                       int * shared_int,
                       sirs::real_type * shared_real) {
  using real_type = sirs::real_type;
  using dust::gpu::shared_copy_data;
  shared_real = shared_copy_data<real_type>(shared_real, shared->alpha);
  shared_real = shared_copy_data<real_type>(shared_real, shared->beta);
  shared_real = shared_copy_data<real_type>(shared_real, shared->gamma);
  shared_real = shared_copy_data<real_type>(shared_real, shared->dt);
  shared_real = shared_copy_data<real_type>(shared_real, shared->exp_noise);
  shared_int = shared_copy_data<int>(shared_int, shared->freq);
}

template <>
__device__
void update_gpu<sirs>(size_t time,
                      const dust::gpu::interleaved<sirs::real_type> state,
                      dust::gpu::interleaved<int> internal_int,
                      dust::gpu::interleaved<sirs::real_type> internal_real,
                      const int * shared_int,
                      const sirs::real_type * shared_real,
                      sirs::rng_state_type& rng_state,
                      dust::gpu::interleaved<sirs::real_type> state_next) {
  using dust::random::binomial;
  using real_type = sirs::real_type;
  const real_type alpha = shared_real[0];
  const real_type beta = shared_real[1];
  const real_type gamma = shared_real[2];
  const real_type dt = shared_real[3];
  const int freq = shared_int[0];
  const real_type S = state[0];
  const real_type I = state[1];
  const real_type R = state[2];
  const real_type N = S + I + R;
  const real_type p_SI = 1 - exp(- beta * I / N);
  const real_type p_IR = 1 - exp(- gamma);
  const real_type p_RS = 1 - exp(- alpha);
  const real_type n_SI = binomial<real_type>(rng_state, S, p_SI * dt);
  const real_type n_IR = binomial<real_type>(rng_state, I, p_IR * dt);
  const real_type n_RS = binomial<real_type>(rng_state, R, p_RS * dt);
  state_next[0] = S - n_SI + n_RS;
  state_next[1] = I + n_SI - n_IR;
  state_next[2] = R + n_IR - n_RS;
  state_next[3] = (time % freq == 0) ? n_SI : state[3] + n_SI;
}

template <>
__device__
sirs::real_type compare_gpu<sirs>(const dust::gpu::interleaved<sirs::real_type> state,
                                  const sirs::data_type& data,
                                  dust::gpu::interleaved<int> internal_int,
                                  dust::gpu::interleaved<sirs::real_type> internal_real,
                                  const int * shared_int,
                                  const sirs::real_type * shared_real,
                                  sirs::rng_state_type& rng_state) {
  using real_type = sirs::real_type;
  const real_type exp_noise = shared_real[4];
  const real_type incidence_modelled = state[3];
  const real_type incidence_observed = data.incidence;
  const real_type lambda = incidence_modelled +
    dust::random::exponential<real_type>(rng_state, exp_noise);
  return dust::density::poisson(incidence_observed, lambda, true);
}

}
}

cpp11::sexp dust_sirs_gpu_info() {
  return dust::gpu::r::gpu_info();
}
using model_cpu = dust::dust_cpu<sirs>;

cpp11::sexp dust_cpu_sirs_capabilities() {
  return dust::r::dust_capabilities<model_cpu>();
}

SEXP dust_cpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                             cpp11::sexp r_n_particles, int n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
  return dust::r::dust_cpu_alloc<sirs>(r_pars, pars_multi, r_time, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config, ode_control);
}

SEXP dust_cpu_sirs_run(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_run<model_cpu>(ptr, r_time_end);
}

SEXP dust_cpu_sirs_simulate(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_simulate<model_cpu>(ptr, r_time_end);
}

SEXP dust_cpu_sirs_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_cpu>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_cpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
  return dust::r::dust_update_state<model_cpu>(ptr, r_pars, r_state, r_time,
                                                      r_set_initial_state, index, reset_step_size);
}

SEXP dust_cpu_sirs_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_cpu>(ptr, r_index);
}

SEXP dust_cpu_sirs_time(SEXP ptr) {
  return dust::r::dust_time<model_cpu>(ptr);
}

void dust_cpu_sirs_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_cpu>(ptr, r_index);
}

SEXP dust_cpu_sirs_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_cpu>(ptr, r_weights);
}

SEXP dust_cpu_sirs_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_cpu>(ptr, first_only, last_only);
}

SEXP dust_cpu_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_cpu>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_cpu_sirs_set_data(SEXP ptr, cpp11::list data,
                                       bool shared) {
  dust::r::dust_set_data<model_cpu>(ptr, data, shared);
  return R_NilValue;
}

SEXP dust_cpu_sirs_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_cpu>(ptr);
}

SEXP dust_cpu_sirs_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood) {
  return dust::r::dust_filter<model_cpu>(ptr, time_end,
                                                save_trajectories,
                                                time_snapshot,
                                                min_log_likelihood);
}

void dust_cpu_sirs_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_cpu>(ptr, n_threads);
}

int dust_cpu_sirs_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_cpu>(ptr);
}

void dust_cpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time) {
  dust::r::dust_set_stochastic_schedule<model_cpu>(ptr, time);
}

SEXP dust_cpu_sirs_ode_statistics(SEXP ptr) {
  return dust::r::dust_ode_statistics<model_cpu>(ptr);
}
using model_gpu = dust::dust_gpu<sirs>;

cpp11::sexp dust_gpu_sirs_capabilities() {
  return dust::r::dust_capabilities<model_gpu>();
}

SEXP dust_gpu_sirs_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time,
                             cpp11::sexp r_n_particles, int n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
  return dust::r::dust_gpu_alloc<sirs>(r_pars, pars_multi, r_time, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config, ode_control);
}

SEXP dust_gpu_sirs_run(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_run<model_gpu>(ptr, r_time_end);
}

SEXP dust_gpu_sirs_simulate(SEXP ptr, cpp11::sexp r_time_end) {
  return dust::r::dust_simulate<model_gpu>(ptr, r_time_end);
}

SEXP dust_gpu_sirs_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_gpu>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_gpu_sirs_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
  return dust::r::dust_update_state<model_gpu>(ptr, r_pars, r_state, r_time,
                                                      r_set_initial_state, index, reset_step_size);
}

SEXP dust_gpu_sirs_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_gpu>(ptr, r_index);
}

SEXP dust_gpu_sirs_time(SEXP ptr) {
  return dust::r::dust_time<model_gpu>(ptr);
}

void dust_gpu_sirs_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_gpu>(ptr, r_index);
}

SEXP dust_gpu_sirs_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_gpu>(ptr, r_weights);
}

SEXP dust_gpu_sirs_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_gpu>(ptr, first_only, last_only);
}

SEXP dust_gpu_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_gpu>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_gpu_sirs_set_data(SEXP ptr, cpp11::list data,
                                       bool shared) {
  dust::r::dust_set_data<model_gpu>(ptr, data, shared);
  return R_NilValue;
}

SEXP dust_gpu_sirs_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_gpu>(ptr);
}

SEXP dust_gpu_sirs_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood) {
  return dust::r::dust_filter<model_gpu>(ptr, time_end,
                                                save_trajectories,
                                                time_snapshot,
                                                min_log_likelihood);
}

void dust_gpu_sirs_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_gpu>(ptr, n_threads);
}

int dust_gpu_sirs_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_gpu>(ptr);
}

void dust_gpu_sirs_set_stochastic_schedule(SEXP ptr, SEXP time) {
  dust::r::dust_set_stochastic_schedule<model_gpu>(ptr, time);
}

SEXP dust_gpu_sirs_ode_statistics(SEXP ptr) {
  return dust::r::dust_ode_statistics<model_gpu>(ptr);
}
