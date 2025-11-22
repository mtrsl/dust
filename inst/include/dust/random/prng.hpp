#ifndef DUST_RANDOM_PRNG_HPP
#define DUST_RANDOM_PRNG_HPP

#include <algorithm>
#include <vector>

#include "dust/random/generator.hpp"

namespace dust {
namespace random {

/// Container class for parallel random number streams. This class
/// does not do any actual running of random numbers (and nothing in
/// parallel) but acts to hold together the state and ease some
/// bookkeeping.
///
/// @tparam T Random number state type to use
class prng {
public:
  /// Construct a new `prng` object from a vector of seed data. We
  /// will consume as many items of `seed` as possible, then start
  /// jumping
  ///
  /// @param seed A vector of integers to seed the generator with
  // TODO(mjr) maybe delete the n param here? But also, maybe just delete the
  // whole class as it does nothing
  prng(const size_t n, const bool deterministic = false)
    : deterministic_{deterministic}
  {
    // TODO(mjr) where should we now store this flag since deleting the
    // rng_state? Maybe just as member data in this class? See below
    //s.deterministic = deterministic;
  }

  /// Indicates if the generators are deterministic
  bool deterministic() const {
    //return state_[0].deterministic;
    return deterministic_;
  }

private:
  // TODO(mjr)
  bool deterministic_;
};

}
}

#endif
