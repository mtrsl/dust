#ifndef DUST_RANDOM_PHILOX_HPP
#define DUST_RANDOM_PHILOX_HPP

#include "dust/random/cuda_compatibility.hpp"

namespace dust {
namespace random {

// The Philox counter type, analogous to the "state" in a traditional RNG -
// passed to RNG fns etc - but the counter doesn't need to be stored (more than
// instances of this type with a short lifetime).
// This implements the same "interface" as `xoshiro_state`, e.g. `int_type`,
// `size()`, `state` etc (although it's not defined explicitly anywhere, just
// implicitly via templates).
class philox4x32_10 {
public:
  /// Type alias used to find the integer type
  using int_type = uint32_t;

  /// Static method, returning the number of integers per state
  __host__ __device__ static constexpr size_t size() {
    return 4;
  }

  /// Array of state
  int_type state[4];

  // TODO(mjr) store key here? Might make sense, then we can just pass an
  // object of this class to the actual rng functions which should be
  // compatible with their templated definitions? All rng calls will eventually
  // call next and that's the only place where the key actually matters (as
  // it's where the actual, raw rng draws are done)

  /// This flag indicates that the distributions should return the
  /// deterministic expectation of the draw, and not use any random
  /// numbers
  bool deterministic = false;

  /// Accessor method, used to both get and set the underlying state
  __host__ __device__ int_type& operator[](size_t i) {
    return state[i];
  }

  /// Is this RNG counter-based? Implies we don't need to store its state more
  /// permanently, and jumps aren't required.
  static constexpr bool counter_based = true;
};

// Implementing these causes lots of duplicate definition errors, and
// commenting them out makes it compile fine (though while still using xoshiro)

//bool operator==(const philox4x32_10& lhs, const philox4x32_10& rhs) {
  //return lhs.deterministic == rhs.deterministic &&
    //std::equal(std::begin(lhs.state), std::end(lhs.state),
               //std::begin(rhs.state));
//}

//bool operator!=(const philox4x32_10& lhs, const philox4x32_10& rhs) {
  //return !(lhs == rhs);
//}


// Implementation of RNG interface functions

template <>
inline __host__ __device__ uint32_t next(philox4x32_10& state) {
  // TODO(mjr) this is currently a dummy next fn that always returns 123 and
  // the proper one needs to be implemented
  //
  // TODO(mjr) Philox4x32-10 generates blocks of 4 u32s at a time. How to
  // handle this? At first just return the first element of a block and ignore
  // the other 3.
  //
  // TODO(mjr) The existing function template doesn't allow for a key. How
  // should this be implemented? A `key` arg that is a dummy when not needed?
  // Initially get things working by hardcoding a key.
  state[3] += 1;
  return 123;
}


}
}

#endif
