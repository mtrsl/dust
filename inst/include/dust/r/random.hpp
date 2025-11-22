#ifndef DUST_R_RANDOM_HPP
#define DUST_R_RANDOM_HPP

#include <cstring> // memcpy

#include <cpp11/environment.hpp>
#include <cpp11/external_pointer.hpp>
#include <cpp11/list.hpp>
#include <cpp11/raws.hpp>

#include <R_ext/Random.h>

#include "dust/random/generator.hpp"
#include "dust/random/prng.hpp"

namespace dust {
namespace random {
namespace r {

namespace {

std::string algorithm_name() {
  std::string ret = "philox4x32_10";
  return ret;
}

}

SEXP rng_pointer_init(int n_streams) {
  auto *rng = new prng(n_streams);
  auto r_ptr = cpp11::external_pointer<prng>(rng);
  return cpp11::writable::list({r_ptr});
}

/// Recieve and check the pointer to rng state.  This checks that the
/// object is valid, is of the correct state type, has sufficient
/// streams and has not been invalidated by serialisation.
///
/// @tparam rng_state_type The random number state type to use
///
/// @param obj An `dust_rng_pointer` object, created in R with
/// ``dust::dust_rng_pointer``
///
/// @param n_streams The number of required streams. Set this to 0 to
/// disable the check.  If you are going to use 100 streams pass 100
/// here, and a runtime error will be thrown if the object does not
/// contain enough streams, which is nicer than a crash when
/// `prng::state` fails.
prng* rng_pointer_get(cpp11::environment obj,
                      int n_streams = 0) {
  // We could probably do this more efficiently if we store an enum
  // in the object but this is probably ok.
  const auto algorithm_given = cpp11::as_cpp<std::string>(obj["algorithm"]);
  const auto algorithm_expected = algorithm_name();
  if (algorithm_given != algorithm_expected) {
    cpp11::stop("Incorrect rng type: given %s, expected %s",
                algorithm_given.c_str(), algorithm_expected.c_str());
  }

  cpp11::environment env_enclos =
    cpp11::as_cpp<cpp11::environment>(obj[".__enclos_env__"]);
  cpp11::environment env =
    cpp11::as_cpp<cpp11::environment>(env_enclos["private"]);

  using ptr_type = cpp11::external_pointer<prng>;
  auto ptr = cpp11::as_cpp<ptr_type>(env["ptr_"]);

  auto * rng = ptr.get();
  if (rng == nullptr) {
    if (!cpp11::as_cpp<bool>(env["is_current_"])) {
      cpp11::stop("Can't unserialise an rng pointer that was not synced");
    }
    //cpp11::raws seed_data = cpp11::as_cpp<cpp11::raws>(env["state_"]);
    //auto seed = raw_seed(seed_data);
    //const auto n_streams_orig = seed.size() / rng_state_type::size();
    // TODO(mjr) what to do about the seeds etc here?
    // prng does nothing now, so its `n` arg (here `n_streams_orig`) isn't
    // needed - just remove the arg entirely?
    //rng = new prng(n_streams_orig);
    rng = new prng();
    env["ptr_"] = cpp11::external_pointer<prng>(rng);
  }

  if (n_streams > 0 && static_cast<int>(rng->size()) < n_streams) {
    cpp11::stop("Requested a rng with %d streams but only have %d",
                n_streams, rng->size());
  }
  env["is_current_"] = cpp11::as_sexp(false);

  return rng;
}

void rng_pointer_sync(cpp11::environment obj) {
  using ptr_type = cpp11::external_pointer<prng>;
  if (!cpp11::as_cpp<bool>(obj["is_current_"])) {
    auto ptr = cpp11::as_cpp<ptr_type>(obj["ptr_"]);
    obj["is_current_"] = cpp11::as_sexp(true);
  }
}

}
}
}

#endif
