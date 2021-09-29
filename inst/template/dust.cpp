/// IMPORTANT; changes here must be reflected into inst/template/dust.hpp
#include <dust/dust.hpp>
#include <dust/interface.hpp>

{{model}}

SEXP dust_{{name}}_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         cpp11::sexp r_n_particles, size_t n_threads,
                         cpp11::sexp r_seed, cpp11::sexp device_config) {
  return dust::r::dust_alloc<{{class}}>(r_pars, pars_multi, step, r_n_particles,
                                        n_threads, r_seed, device_config);
}

SEXP dust_{{name}}_run(SEXP ptr, size_t step_end, bool device,
                       bool deterministic) {
  return dust::r::dust_run<{{class}}>(ptr, step_end, device, deterministic);
}

SEXP dust_{{name}}_simulate(SEXP ptr, cpp11::sexp step_end, bool device,
                            bool deterministic) {
  return dust::r::dust_simulate<{{class}}>(ptr, step_end, device,
                                           deterministic);
}

SEXP dust_{{name}}_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust::r::dust_set_index<{{class}}>(ptr, r_index);
  return R_NilValue;
}

SEXP dust_{{name}}_update_state(SEXP ptr, SEXP r_pars, SEXP r_state,
                                SEXP r_step, SEXP r_set_initial_state,
                                bool deterministic) {
  return dust::r::dust_update_state<{{class}}>(ptr, r_pars, r_state, r_step,
                                               r_set_initial_state,
                                               deterministic);
}

SEXP dust_{{name}}_state(SEXP ptr, SEXP r_index) {
  return dust::r::dust_state<{{class}}>(ptr, r_index);
}

size_t dust_{{name}}_step(SEXP ptr) {
  return dust::r::dust_step<{{class}}>(ptr);
}

void dust_{{name}}_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust::r::dust_reorder<{{class}}>(ptr, r_index);
}

SEXP dust_{{name}}_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust::r::dust_resample<{{class}}>(ptr, r_weights);
}

SEXP dust_{{name}}_rng_state(SEXP ptr, bool first_only, bool last_only) {
  return dust::r::dust_rng_state<{{class}}>(ptr, first_only, last_only);
}

SEXP dust_{{name}}_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust::r::dust_set_rng_state<{{class}}>(ptr, rng_state);
  return R_NilValue;
}

SEXP dust_{{name}}_set_data(SEXP ptr, cpp11::list data) {
  dust::r::dust_set_data<{{class}}>(ptr, data);
  return R_NilValue;
}

SEXP dust_{{name}}_compare_data(SEXP ptr, bool device) {
  return dust::r::dust_compare_data<{{class}}>(ptr, device);
}

SEXP dust_{{name}}_filter(SEXP ptr, bool save_trajectories,
                          cpp11::sexp step_snapshot,
                          bool device) {
  return dust::r::dust_filter<{{class}}>(ptr, save_trajectories, step_snapshot, device);
}

cpp11::sexp dust_{{name}}_capabilities() {
  return dust::r::dust_capabilities<{{name}}>();
}

void dust_{{name}}_set_n_threads(SEXP ptr, int n_threads) {
  return dust::r::dust_set_n_threads<{{class}}>(ptr, n_threads);
}

int dust_{{name}}_n_state(SEXP ptr) {
  return dust::r::dust_n_state<{{class}}>(ptr);
}

cpp11::sexp dust_{{name}}_device_info() {
  return dust::cuda::device_info<{{class}}::real_t>();
}
