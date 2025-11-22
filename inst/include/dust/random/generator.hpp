#ifndef DUST_RANDOM_GENERATOR_HPP
#define DUST_RANDOM_GENERATOR_HPP

// This is the random generator, holding rng state and providing
// support for generating reals on the interval 0..1
//
// Typically this will be too low level for most applications and you
// should use dust::random::prng which provides a parallel random
// number generator.
//
// The api is:
//
// * the dust::random::xoshiro_state type, plus all the
//   specific versions of it (e.g., xoshiro256starstar); these
//   objects can be created but should be considered opaque.
//
// * dust::random::random_real which yields a real (of the
//   requested type) given a xoshiro_state state
//
// * dust::random::seed which seeds a generator
//
// * dust::random::jump and dust::random::long_jump which "jump" the
//   generator state forward, a key part of the parallel generators.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "dust/random/cuda_compatibility.hpp"
#include "dust/random/utils.hpp"
#include "dust/random/rng_interface.hpp"

// 4x32 10-round Philox counter-based generator
#include "dust/random/philox.hpp"

namespace dust {
namespace random {

/// Generate a real number U(0, 1)
///
/// @tparam T The real type to return, typically `double` or `float`;
/// because this affects the return value only it must be provided.
///
/// @tparam U The random number generator state type; this will be
/// inferred based on the argument
///
/// @param state The random number state, will be updated as a side effect
///
/// @return A real-valued random number on (0, 1]
template <typename T, typename U>
__host__ __device__
T random_real(U& state) {
  const auto value = next(state);
  return int_to_real<T>(value);
}

/// Generate a random integer of a given width
///
/// @tparam T The integer type to generate, such as `uint64_t`. Both
/// signed and unsigned integers can be provided here. Because this
/// affects the return value only it must be provided.
///
/// @tparam U The random number generator state type; this will be
/// inferred based on the argument
///
/// @param state The random number state, will be updated as a side effect
///
/// @return An integer valued number, uniformly distributed anywhere
/// within the range of the given integer type.
template <typename T, typename U>
__host__ __device__
T random_int(U& state) {
  static_assert(sizeof(T) <= sizeof(typename U::int_type),
                "requested integer too wide");
  static_assert(std::is_integral<T>::value,
                "integer type required for T");
  const auto value = next(state);
  return static_cast<T>(value);
}

}
}

#endif
