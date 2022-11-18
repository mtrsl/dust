// Generated by dust (version 0.12.1) - do not edit
#include <cpp11.hpp>

[[cpp11::register]]
cpp11::sexp dust_variable_capabilities();

[[cpp11::register]]
cpp11::sexp dust_variable_gpu_info();
[[cpp11::register]]
SEXP dust_cpu_variable_alloc(cpp11::list r_pars, bool pars_multi, size_t time,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config);

[[cpp11::register]]
SEXP dust_cpu_variable_run(SEXP ptr, size_t time_end);

[[cpp11::register]]
SEXP dust_cpu_variable_simulate(SEXP ptr, cpp11::sexp time_end);

[[cpp11::register]]
SEXP dust_cpu_variable_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state,
                                           SEXP index, SEXP reset_step_size);

[[cpp11::register]]
SEXP dust_cpu_variable_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_cpu_variable_time(SEXP ptr);

[[cpp11::register]]
void dust_cpu_variable_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_variable_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_cpu_variable_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_cpu_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_cpu_variable_set_data(SEXP ptr, cpp11::list data, bool shared);

[[cpp11::register]]
SEXP dust_cpu_variable_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_cpu_variable_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood);

[[cpp11::register]]
void dust_cpu_variable_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_cpu_variable_n_state(SEXP ptr);
[[cpp11::register]]
SEXP dust_gpu_variable_alloc(cpp11::list r_pars, bool pars_multi, size_t time,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config);

[[cpp11::register]]
SEXP dust_gpu_variable_run(SEXP ptr, size_t time_end);

[[cpp11::register]]
SEXP dust_gpu_variable_simulate(SEXP ptr, cpp11::sexp time_end);

[[cpp11::register]]
SEXP dust_gpu_variable_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_gpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state,
                                           SEXP index, SEXP reset_step_size);

[[cpp11::register]]
SEXP dust_gpu_variable_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_gpu_variable_time(SEXP ptr);

[[cpp11::register]]
void dust_gpu_variable_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_gpu_variable_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_gpu_variable_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_gpu_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_gpu_variable_set_data(SEXP ptr, cpp11::list data, bool shared);

[[cpp11::register]]
SEXP dust_gpu_variable_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_gpu_variable_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood);

[[cpp11::register]]
void dust_gpu_variable_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_gpu_variable_n_state(SEXP ptr);
#include <dust/r/dust.hpp>

class variable {
public:
  using real_type = double;
  using data_type = dust::no_data;
  using internal_type = dust::no_internal;
  using rng_state_type = dust::random::generator<real_type>;

  struct shared_type {
    size_t len;
    real_type mean;
    real_type sd;
  };

  variable(const dust::pars_type<variable>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return shared->len;
  }

  std::vector<real_type> initial(size_t time) {
    std::vector<real_type> ret;
    for (size_t i = 0; i < shared->len; ++i) {
      ret.push_back(i + 1);
    }
    return ret;
  }

  void update(size_t time, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    for (size_t i = 0; i < shared->len; ++i) {
      state_next[i] = state[i] +
        dust::random::normal<real_type>(rng_state, shared->mean, shared->sd);
    }
  }

private:
  dust::shared_ptr<variable> shared;
};

namespace dust {
template <>
dust::pars_type<variable> dust_pars<variable>(cpp11::list pars) {
  using real_type = variable::real_type;
  const size_t len = cpp11::as_cpp<int>(pars["len"]);
  real_type mean = 0, sd = 1;

  SEXP r_mean = pars["mean"];
  if (r_mean != R_NilValue) {
    mean = cpp11::as_cpp<real_type>(r_mean);
  }

  SEXP r_sd = pars["sd"];
  if (r_sd != R_NilValue) {
    sd = cpp11::as_cpp<real_type>(r_sd);
  }

  variable::shared_type shared{len, mean, sd};
  return dust::pars_type<variable>(shared);
}

template <>
cpp11::sexp dust_info<variable>(const dust::pars_type<variable>& pars) {
  using namespace cpp11::literals;
  return cpp11::writable::list({"len"_nm = pars.shared->len});
}

namespace gpu {

template <>
size_t shared_int_size<variable>(dust::shared_ptr<variable> shared) {
  return 1;
}

template <>
size_t shared_real_size<variable>(dust::shared_ptr<variable> shared) {
  return 2;
}

template <>
void shared_copy<variable>(dust::shared_ptr<variable> shared,
                           int * shared_int,
                           variable::real_type * shared_real) {
  using dust::gpu::shared_copy_data;
  using real_type = variable::real_type;
  shared_int = shared_copy_data<int>(shared_int, shared->len);
  shared_real = shared_copy_data<real_type>(shared_real, shared->mean);
  shared_real = shared_copy_data<real_type>(shared_real, shared->sd);
}

template <>
__device__
void update_gpu<variable>(size_t time,
                          const dust::gpu::interleaved<variable::real_type> state,
                          dust::gpu::interleaved<int> internal_int,
                          dust::gpu::interleaved<variable::real_type> internal_real,
                          const int * shared_int,
                          const variable::real_type * shared_real,
                          variable::rng_state_type& rng_state,
                          dust::gpu::interleaved<variable::real_type> state_next) {
  using real_type = variable::real_type;
  const size_t len = shared_int[0];
  const real_type mean = shared_real[0];
  const real_type sd = shared_real[1];
  for (size_t i = 0; i < len; ++i) {
    state_next[i] = state[i] +
      dust::random::normal<real_type>(rng_state, mean, sd);
  }
}

}
}

cpp11::sexp dust_variable_capabilities() {
  return dust::r::dust_capabilities<variable>();
}

cpp11::sexp dust_variable_gpu_info() {
  return dust::gpu::r::gpu_info();
}
using model_cpu = dust::dust_cpu<variable>;

SEXP dust_cpu_variable_alloc(cpp11::list r_pars, bool pars_multi, size_t time,
                             cpp11::sexp r_n_particles, size_t n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config) {
  return dust::r::dust_cpu_alloc<variable>(r_pars, pars_multi, time, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config);
}

SEXP dust_cpu_variable_run(SEXP ptr, size_t time_end) {
  return dust::r::dust_run<model_cpu>(ptr, time_end);
}

SEXP dust_cpu_variable_simulate(SEXP ptr, cpp11::sexp time_end) {
  return dust::r::dust_simulate<model_cpu>(ptr, time_end);
}

SEXP dust_cpu_variable_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_cpu>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_cpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
  return dust::r::dust_update_state<model_cpu>(ptr, r_pars, r_state, r_time,
                                                      r_set_initial_state, index, reset_step_size);
}

SEXP dust_cpu_variable_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_cpu>(ptr, r_index);
}

size_t dust_cpu_variable_time(SEXP ptr) {
  return dust::r::dust_time<model_cpu>(ptr);
}

void dust_cpu_variable_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_cpu>(ptr, r_index);
}

SEXP dust_cpu_variable_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_cpu>(ptr, r_weights);
}

SEXP dust_cpu_variable_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_cpu>(ptr, first_only, last_only);
}

SEXP dust_cpu_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_cpu>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_cpu_variable_set_data(SEXP ptr, cpp11::list data,
                                       bool shared) {
  dust::r::dust_set_data<model_cpu>(ptr, data, shared);
  return R_NilValue;
}

SEXP dust_cpu_variable_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_cpu>(ptr);
}

SEXP dust_cpu_variable_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood) {
  return dust::r::dust_filter<model_cpu>(ptr, time_end,
                                                save_trajectories,
                                                time_snapshot,
                                                min_log_likelihood);
}

void dust_cpu_variable_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_cpu>(ptr, n_threads);
}

int dust_cpu_variable_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_cpu>(ptr);
}
using model_gpu = dust::dust_gpu<variable>;

SEXP dust_gpu_variable_alloc(cpp11::list r_pars, bool pars_multi, size_t time,
                             cpp11::sexp r_n_particles, size_t n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config) {
  return dust::r::dust_gpu_alloc<variable>(r_pars, pars_multi, time, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config);
}

SEXP dust_gpu_variable_run(SEXP ptr, size_t time_end) {
  return dust::r::dust_run<model_gpu>(ptr, time_end);
}

SEXP dust_gpu_variable_simulate(SEXP ptr, cpp11::sexp time_end) {
  return dust::r::dust_simulate<model_gpu>(ptr, time_end);
}

SEXP dust_gpu_variable_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_gpu>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_gpu_variable_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                           SEXP r_time, SEXP r_set_initial_state, SEXP index, SEXP reset_step_size) {
  return dust::r::dust_update_state<model_gpu>(ptr, r_pars, r_state, r_time,
                                                      r_set_initial_state, index, reset_step_size);
}

SEXP dust_gpu_variable_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_gpu>(ptr, r_index);
}

size_t dust_gpu_variable_time(SEXP ptr) {
  return dust::r::dust_time<model_gpu>(ptr);
}

void dust_gpu_variable_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_gpu>(ptr, r_index);
}

SEXP dust_gpu_variable_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_gpu>(ptr, r_weights);
}

SEXP dust_gpu_variable_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_gpu>(ptr, first_only, last_only);
}

SEXP dust_gpu_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_gpu>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_gpu_variable_set_data(SEXP ptr, cpp11::list data,
                                       bool shared) {
  dust::r::dust_set_data<model_gpu>(ptr, data, shared);
  return R_NilValue;
}

SEXP dust_gpu_variable_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_gpu>(ptr);
}

SEXP dust_gpu_variable_filter(SEXP ptr, SEXP time_end,
                                     bool save_trajectories,
                                     cpp11::sexp time_snapshot,
                                     cpp11::sexp min_log_likelihood) {
  return dust::r::dust_filter<model_gpu>(ptr, time_end,
                                                save_trajectories,
                                                time_snapshot,
                                                min_log_likelihood);
}

void dust_gpu_variable_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_gpu>(ptr, n_threads);
}

int dust_gpu_variable_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_gpu>(ptr);
}
