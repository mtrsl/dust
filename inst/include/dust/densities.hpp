#ifndef DUST_DENSITIES_HPP
#define DUST_DENSITIES_HPP

#include <cmath>

namespace dust {

template <typename T>
T maybe_log(T x, bool log) {
  return log ? x : std::exp(x);
}

template <typename T>
T lchoose(T n, T k) {
  return std::lgamma(static_cast<T>(n + 1)) -
    std::lgamma(static_cast<T>(k + 1)) -
    std::lgamma(static_cast<T>(n - k + 1));
}

template <typename T>
T lfactorial(T x) {
  return std::lgamma(x + 1);
}

template <typename T>
T lbeta(T x, T y) {
  return lgamma(x) + lgamma(y) - lgamma(x + y);
}

template <typename T>
T dbinom(int x, int size, T prob, bool log) {
  const T ret = lchoose<T>(size, x) +
    x * std::log(prob) +
    (size - x) * std::log(1 - prob);
  return maybe_log(ret, log);
}

template <typename T>
T dnbinom(int x, int size, T mu, bool log) {
  const T prob = size / (size + mu);
  if (x == 0 && size == 0) {
    return maybe_log(0, log);
  }
  const T ret = std::lgamma(static_cast<T>(x + size)) -
    std::lgamma(static_cast<T>(size)) -
    lfactorial(static_cast<T>(x)) +
    size * std::log(prob) + x * std::log(1 - prob);
  return maybe_log(ret, log);
}

// A note on this parametrisation:
//
//   prob = alpha / (alpha + beta)
//   rho = 1 / (alpha + beta + 1)
//
// Where alpha and beta have (0, Inf) support
template <typename T>
T dbetabinom(int x, int size, T prob, T rho, bool log) {
  const T a = prob * (1 / rho - 1);
  const T b = (1 - prob) * (1 / rho - 1);
  const T ret = lchoose<T>(size, x) + lbeta(x + a, size - x + b) - lbeta(a, b);
  return maybe_log(ret, log);
}

template <typename T>
T dpois(int x, T lambda, bool log) {
  const T ret = x * std::log(lambda) - lambda - lfactorial<T>(x);
  return maybe_log(ret, log);
}

}

#endif
