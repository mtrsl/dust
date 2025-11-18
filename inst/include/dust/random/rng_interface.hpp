#ifndef DUST_RANDOM_RNG_INTERFACE_HPP
#define DUST_RANDOM_RNG_INTERFACE_HPP

#include <array>

namespace dust {
namespace random {

// TODO(mjr) do we need to decouple the state's underlying int type from the
// type that the RNG produces? Is this hardcoded for xoshiro family? Philox
// returns a 4xu32 block
template <typename T>
typename T::int_type next(T& state);

template <typename T>
std::array<typename T::int_type, T::size()> jump_constants();

template <typename T>
std::array<typename T::int_type, T::size()> long_jump_constants();

}
}

#endif
