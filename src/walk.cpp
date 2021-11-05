// Generated by dust (version 0.11.4) - do not edit
#include <cpp11.hpp>

[[cpp11::register]]
cpp11::sexp dust_walk_capabilities();

[[cpp11::register]]
cpp11::sexp dust_walk_gpu_info();
[[cpp11::register]]
SEXP dust_cpu_walk_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, bool deterministic,
                         cpp11::sexp gpu_config);

[[cpp11::register]]
SEXP dust_cpu_walk_run(SEXP ptr, size_t step_end);

[[cpp11::register]]
SEXP dust_cpu_walk_simulate(SEXP ptr, cpp11::sexp step_end);

[[cpp11::register]]
SEXP dust_cpu_walk_set_index(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_walk_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state);

[[cpp11::register]]
SEXP dust_cpu_walk_state(SEXP ptr, SEXP r_index);

[[cpp11::register]]
size_t dust_cpu_walk_step(SEXP ptr);

[[cpp11::register]]
void dust_cpu_walk_reorder(SEXP ptr, cpp11::sexp r_index);

[[cpp11::register]]
SEXP dust_cpu_walk_resample(SEXP ptr, cpp11::doubles r_weights);

[[cpp11::register]]
SEXP dust_cpu_walk_rng_state(SEXP ptr, bool first_only, bool last_only);

[[cpp11::register]]
SEXP dust_cpu_walk_set_rng_state(SEXP ptr, cpp11::raws rng_state);

[[cpp11::register]]
SEXP dust_cpu_walk_set_data(SEXP ptr, cpp11::list data);

[[cpp11::register]]
SEXP dust_cpu_walk_compare_data(SEXP ptr);

[[cpp11::register]]
SEXP dust_cpu_walk_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot);

[[cpp11::register]]
void dust_cpu_walk_set_n_threads(SEXP ptr, int n_threads);

[[cpp11::register]]
int dust_cpu_walk_n_state(SEXP ptr);
#include <dust/interface/dust.hpp>

class walk {
public:
  typedef double real_type;
  typedef dust::no_data data_type;
  typedef dust::no_internal internal_type;
  typedef dust::random::generator<real_type> rng_state_type;

  struct shared_type {
    real_type sd;
  };

  walk(const dust::pars_type<walk>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return 1;
  }

  std::vector<real_type> initial(size_t step) {
    std::vector<real_type> ret = {0};
    return ret;
  }

  void update(size_t step, const real_type * state, rng_state_type& rng_state,
              real_type * state_next) {
    state_next[0] = state[0] +
      dust::random::normal<real_type>(rng_state, 0, shared->sd);
  }

private:
  dust::shared_ptr<walk> shared;
};

namespace dust {

template <>
dust::pars_type<walk> dust_pars<walk>(cpp11::list pars) {
  walk::real_type sd = cpp11::as_cpp<walk::real_type>(pars["sd"]);
  return dust::pars_type<walk>(walk::shared_type{sd});
}

}

cpp11::sexp dust_walk_capabilities() {
  return dust::r::dust_capabilities<walk>();
}

cpp11::sexp dust_walk_gpu_info() {
  return dust::gpu::interface::gpu_info<walk::real_type>();
}
using model_cpu = dust::dust_cpu<walk>;

SEXP dust_cpu_walk_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                             cpp11::sexp r_n_particles, size_t n_threads,
                             cpp11::sexp r_seed, bool deterministic,
                             cpp11::sexp gpu_config) {
  return dust::r::dust_cpu_alloc<walk>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, deterministic,
                                        gpu_config);
}

SEXP dust_cpu_walk_run(SEXP ptr, size_t step_end) {
  return dust::r::dust_run<model_cpu>(ptr, step_end);
}

SEXP dust_cpu_walk_simulate(SEXP ptr, cpp11::sexp step_end) {
  return dust::r::dust_simulate<model_cpu>(ptr, step_end);
}

SEXP dust_cpu_walk_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<model_cpu>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_cpu_walk_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state) {
  return dust::r::dust_update_state<model_cpu>(ptr, r_pars, r_state, r_step,
                                               r_set_initial_state);
}

SEXP dust_cpu_walk_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<model_cpu>(ptr, r_index);
}

size_t dust_cpu_walk_step(SEXP ptr) {
  return dust::r::dust_step<model_cpu>(ptr);
}

void dust_cpu_walk_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<model_cpu>(ptr, r_index);
}

SEXP dust_cpu_walk_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<model_cpu>(ptr, r_weights);
}

SEXP dust_cpu_walk_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<model_cpu>(ptr, first_only, last_only);
}

SEXP dust_cpu_walk_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<model_cpu>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_cpu_walk_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<model_cpu>(ptr, data);
  return R_NilValue;
}

SEXP dust_cpu_walk_compare_data(SEXP ptr) {
  return dust::r::dust_compare_data<model_cpu>(ptr);
}

SEXP dust_cpu_walk_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot) {
  return dust::r::dust_filter<model_cpu>(ptr, save_trajectories, step_snapshot);
}

void dust_cpu_walk_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<model_cpu>(ptr, n_threads);
}

int dust_cpu_walk_n_state(SEXP ptr) {
  return dust::r::dust_n_state<model_cpu>(ptr);
}
