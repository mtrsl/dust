# Generated by cpp11: do not edit by hand

density_binomial <- function(x, size, prob, log) {
  .Call(`_dust_density_binomial`, x, size, prob, log)
}

density_normal <- function(x, mu, sd, log) {
  .Call(`_dust_density_normal`, x, mu, sd, log)
}

density_negative_binomial_mu <- function(x, size, mu, log, is_float) {
  .Call(`_dust_density_negative_binomial_mu`, x, size, mu, log, is_float)
}

density_negative_binomial_prob <- function(x, size, prob, log) {
  .Call(`_dust_density_negative_binomial_prob`, x, size, prob, log)
}

density_beta_binomial <- function(x, size, prob, rho, log) {
  .Call(`_dust_density_beta_binomial`, x, size, prob, rho, log)
}

density_poisson <- function(x, lambda, log) {
  .Call(`_dust_density_poisson`, x, lambda, log)
}

dust_rng_alloc <- function(r_seed, n_streams, deterministic, is_float) {
  .Call(`_dust_dust_rng_alloc`, r_seed, n_streams, deterministic, is_float)
}

dust_rng_jump <- function(ptr, is_float) {
  invisible(.Call(`_dust_dust_rng_jump`, ptr, is_float))
}

dust_rng_long_jump <- function(ptr, is_float) {
  invisible(.Call(`_dust_dust_rng_long_jump`, ptr, is_float))
}

dust_rng_random_real <- function(ptr, n, n_threads, is_float) {
  .Call(`_dust_dust_rng_random_real`, ptr, n, n_threads, is_float)
}

dust_rng_random_normal <- function(ptr, n, n_threads, algorithm, is_float) {
  .Call(`_dust_dust_rng_random_normal`, ptr, n, n_threads, algorithm, is_float)
}

dust_rng_uniform <- function(ptr, n, r_min, r_max, n_threads, is_float) {
  .Call(`_dust_dust_rng_uniform`, ptr, n, r_min, r_max, n_threads, is_float)
}

dust_rng_exponential <- function(ptr, n, r_rate, n_threads, is_float) {
  .Call(`_dust_dust_rng_exponential`, ptr, n, r_rate, n_threads, is_float)
}

dust_rng_normal <- function(ptr, n, r_mean, r_sd, n_threads, algorithm, is_float) {
  .Call(`_dust_dust_rng_normal`, ptr, n, r_mean, r_sd, n_threads, algorithm, is_float)
}

dust_rng_binomial <- function(ptr, n, r_size, r_prob, n_threads, is_float) {
  .Call(`_dust_dust_rng_binomial`, ptr, n, r_size, r_prob, n_threads, is_float)
}

dust_rng_nbinomial <- function(ptr, n, r_size, r_prob, n_threads, is_float) {
  .Call(`_dust_dust_rng_nbinomial`, ptr, n, r_size, r_prob, n_threads, is_float)
}

dust_rng_hypergeometric <- function(ptr, n, r_n1, r_n2, r_k, n_threads, is_float) {
  .Call(`_dust_dust_rng_hypergeometric`, ptr, n, r_n1, r_n2, r_k, n_threads, is_float)
}

dust_rng_gamma <- function(ptr, n, r_a, r_b, n_threads, is_float) {
  .Call(`_dust_dust_rng_gamma`, ptr, n, r_a, r_b, n_threads, is_float)
}

dust_rng_poisson <- function(ptr, n, r_lambda, n_threads, is_float) {
  .Call(`_dust_dust_rng_poisson`, ptr, n, r_lambda, n_threads, is_float)
}

dust_rng_cauchy <- function(ptr, n, r_location, r_scale, n_threads, is_float) {
  .Call(`_dust_dust_rng_cauchy`, ptr, n, r_location, r_scale, n_threads, is_float)
}

dust_rng_multinomial <- function(ptr, n, r_size, r_prob, n_threads, is_float) {
  .Call(`_dust_dust_rng_multinomial`, ptr, n, r_size, r_prob, n_threads, is_float)
}

dust_rng_state <- function(ptr, is_float) {
  .Call(`_dust_dust_rng_state`, ptr, is_float)
}

dust_rng_pointer_init <- function(n_streams, seed, long_jump, algorithm) {
  .Call(`_dust_dust_rng_pointer_init`, n_streams, seed, long_jump, algorithm)
}

dust_rng_pointer_sync <- function(obj, algorithm) {
  invisible(.Call(`_dust_dust_rng_pointer_sync`, obj, algorithm))
}

test_rng_pointer_get <- function(obj, n_streams) {
  .Call(`_dust_test_rng_pointer_get`, obj, n_streams)
}

cpp_openmp_info <- function() {
  .Call(`_dust_cpp_openmp_info`)
}

test_cuda_pars <- function(r_gpu_config, n_particles, n_particles_each, n_state, n_state_full, n_shared_int, n_shared_real, data_size, shared_size) {
  .Call(`_dust_test_cuda_pars`, r_gpu_config, n_particles, n_particles_each, n_state, n_state_full, n_shared_int, n_shared_real, data_size, shared_size)
}

test_interpolate_search <- function(target, x) {
  .Call(`_dust_test_interpolate_search`, target, x)
}

test_interpolate_constant1 <- function(t, y, z) {
  .Call(`_dust_test_interpolate_constant1`, t, y, z)
}

test_interpolate_linear1 <- function(t, y, z) {
  .Call(`_dust_test_interpolate_linear1`, t, y, z)
}

test_interpolate_spline1 <- function(t, y, z) {
  .Call(`_dust_test_interpolate_spline1`, t, y, z)
}

test_xoshiro_run <- function(obj) {
  .Call(`_dust_test_xoshiro_run`, obj)
}

cpp_scale_log_weights <- function(w) {
  .Call(`_dust_cpp_scale_log_weights`, w)
}
