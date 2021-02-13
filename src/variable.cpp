// Generated by dust (version 0.7.7) - do not edit
#include <iostream>
#include <dust/dust.hpp>
#include <dust/interface.hpp>

class variable {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  typedef dust::no_internal internal_t;

  struct shared_t {
    size_t len;
    real_t mean;
    real_t sd;
  };

  variable(const dust::pars_t<variable>& pars) : shared(pars.shared) {
  }

  size_t size() const {
    return shared->len;
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> ret;
    for (size_t i = 0; i < shared->len; ++i) {
      ret.push_back(i + 1);
    }
    return ret;
  }

  void update(size_t step, const real_t * state,
              dust::rng_state_t<real_t>& rng_state,
              real_t * state_next) {
    for (size_t i = 0; i < shared->len; ++i) {
      state_next[i] =
        dust::distr::rnorm(rng_state, state[i] + shared->mean, shared->sd);
    }
  }

private:
  dust::shared_ptr<variable> shared;
};

#include <cpp11/list.hpp>
template <>
dust::pars_t<variable> dust_pars<variable>(cpp11::list pars) {
  typedef variable::real_t real_t;
  const size_t len = cpp11::as_cpp<int>(pars["len"]);
  real_t mean = 0, sd = 1;

  SEXP r_mean = pars["mean"];
  if (r_mean != R_NilValue) {
    mean = cpp11::as_cpp<real_t>(r_mean);
  }

  SEXP r_sd = pars["sd"];
  if (r_sd != R_NilValue) {
    sd = cpp11::as_cpp<real_t>(r_sd);
  }

  variable::shared_t shared{len, mean, sd};
  return dust::pars_t<variable>(shared);
}

[[cpp11::register]]
SEXP dust_variable_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         size_t n_particles, size_t n_threads,
                         cpp11::sexp r_seed) {
  return dust_alloc<variable>(r_pars, pars_multi, step, n_particles,
                               n_threads, r_seed);
}

[[cpp11::register]]
SEXP dust_variable_run(SEXP ptr, size_t step_end) {
  return dust_run<variable>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_variable_simulate2(SEXP ptr, cpp11::sexp step_end) {
  return dust_simulate<variable>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_variable_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<variable>(ptr, r_index);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_variable_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<variable>(ptr, r_state, r_step);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_variable_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust_reset<variable>(ptr, r_pars, step);
}

[[cpp11::register]]
SEXP dust_variable_state(SEXP ptr, SEXP r_index) {
  return dust_state<variable>(ptr, r_index);
}

[[cpp11::register]]
size_t dust_variable_step(SEXP ptr) {
  return dust_step<variable>(ptr);
}

[[cpp11::register]]
void dust_variable_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<variable>(ptr, r_index);
}

[[cpp11::register]]
SEXP dust_variable_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust_resample<variable>(ptr, r_weights);
}

[[cpp11::register]]
SEXP dust_variable_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust_set_pars<variable>(ptr, r_pars);
}

[[cpp11::register]]
SEXP dust_variable_rng_state(SEXP ptr, bool first_only) {
  return dust_rng_state<variable>(ptr, first_only);
}

[[cpp11::register]]
SEXP dust_variable_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust_set_rng_state<variable>(ptr, rng_state);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_variable_set_data(SEXP ptr, cpp11::list data) {
  dust_set_data<variable>(ptr, data);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_variable_compare_data(SEXP ptr) {
  return dust_compare_data<variable>(ptr);
}

[[cpp11::register]]
SEXP dust_variable_filter(SEXP ptr, bool save_history) {
  return dust_filter<variable>(ptr, save_history);
}

[[cpp11::register]]
SEXP dust_variable_simulate(cpp11::sexp r_steps,
                            cpp11::list r_pars,
                            cpp11::doubles_matrix r_state,
                            cpp11::sexp r_index,
                            const size_t n_threads,
                            cpp11::sexp r_seed,
                            bool return_state) {
  return dust_simulate<variable>(r_steps, r_pars, r_state, r_index,
                                  n_threads, r_seed, return_state);
}

[[cpp11::register]]
cpp11::sexp dust_variable_capabilities() {
  return dust_capabilities<variable>();
}

[[cpp11::register]]
void dust_variable_set_n_threads(SEXP ptr, int n_threads) {
  return dust_set_n_threads<variable>(ptr, n_threads);
}

[[cpp11::register]]
int dust_variable_n_state(SEXP ptr) {
  return dust_n_state<variable>(ptr);
}
