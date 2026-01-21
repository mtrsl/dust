#ifndef DUST_RANDOM_PHILOX_HPP
#define DUST_RANDOM_PHILOX_HPP

#include "dust/random/cuda_compatibility.hpp"

#include <cstdio>

namespace dust {
namespace random {

// Multiplies two 32 bit unsigned ints and keeps only the lowest 32 bits of the result
__host__ __device__
uint32_t umullo_u32(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return a * b;
#else
    return static_cast<uint32_t>(a * b);
#endif
}

// Multiplies two 32 bit unsigned ints and keeps only the highest 32 bits of the result
__host__ __device__
uint32_t umulhi_u32(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> 32);
#endif
}

// Helper types to make it easier to pass arrays by value (because arrays decay
// to pointers when passed to functions)
struct u32x2 {
  uint32_t d[2];

  // Element access (const)
  __host__ __device__
  inline const uint32_t& operator[](int i) const {
    return d[i];
  }

  // Element access (mutable)
  __host__ __device__
  inline uint32_t& operator[](int i) {
    return d[i];
  }
};

struct u32x4 {
  uint32_t d[4];

  // Element access (const)
  __host__ __device__
  inline const uint32_t& operator[](int i) const {
    return d[i];
  }

  // Element access (mutable)
  __host__ __device__
  inline uint32_t& operator[](int i) {
    return d[i];
  }
};

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
  //__host__ __device__
  static constexpr size_t size() {
    return 4;
  }

  /// Counter
  u32x4 ctr = { 0, 0, 0, 0 };

  /// Key
  u32x2 key = { 0, 0 };

  /// This flag indicates that the distributions should return the
  /// deterministic expectation of the draw, and not use any random
  /// numbers
  bool deterministic = false;

  /// Accessor method, used to both get and set the underlying state
  __host__ __device__
  int_type& operator[](size_t i) {
    return ctr[i];
  }

 __host__ __device__
  uint32_t next_u32() {
    if (idx == 4) {
      buf = generate_block(ctr, key);

      //printf("Random block drawn: (%u, %u, %u, %u)\n", buf[0], buf[1], buf[2], buf[3]);

      ctr[3] += 1;
      idx = 0;
    }
    const uint32_t result = buf[idx];
    idx += 1;
    return result;
  }

private:
  static const size_t ROUNDS = 10;

  static const uint32_t M0 = 0xD2511F53;
  static const uint32_t M1 = 0xCD9E8D57;

  // Weyl constants
  static const uint32_t W0 = 0x9E3779B9;
  static const uint32_t W1 = 0xBB67AE85;

  int idx = 4;
  u32x4 buf;

  /// Generates the next block of 4 u32s in the stream. `ctr` and `key` copied
  /// into the function arguments and local copies are modified by this
  /// function - class members left unchanged
  __host__ __device__
  static u32x4 generate_block(
    u32x4 ctr,
    u32x2 key
  ) {
    for (size_t i = 0; i < ROUNDS; i += 1) {
      uint32_t hi0 = umulhi_u32(ctr[0], M0);
      uint32_t lo0 = umullo_u32(ctr[0], M0);

      uint32_t hi1 = umulhi_u32(ctr[2], M1);
      uint32_t lo1 = umullo_u32(ctr[2], M1);

      u32x4 ctr_new = {
        hi1 ^ ctr[1] ^ key[0],
        lo1,
        hi0 ^ ctr[3] ^ key[1],
        lo0
      };

      ctr = ctr_new;

      key[0] += W0;
      key[1] += W1;
    }

    return ctr;
  }
};

// Implementation of RNG interface functions

template <>
inline
__host__ __device__
uint32_t next(philox4x32_10& state) {
  //printf("    ctr: (%u, %u, %u, %u)", state.ctr[0], state.ctr[1], state.ctr[2], state.ctr[3]);
  //printf("    key: (%u, %u)", state.key[0], state.key[1]);
  //printf("\n");

  uint32_t n = state.next_u32();
  //printf("Random u32 drawn: %u", n);

  return n;
}


}
}

#endif
