#ifndef DUST_PARTICLE_HPP
#define DUST_PARTICLE_HPP

#include <sstream>
#include <vector>

#include "dust/types.hpp"

namespace dust {

template <typename T>
class particle {
public:
  using pars_type = dust::pars_type<T>;
  using time_type = size_t;
  using real_type = typename T::real_type;
  using data_type = typename T::data_type;
  using rng_state_type = typename T::rng_state_type;

  particle(pars_type pars, time_type time) :
    model_(pars),
    time_(time),
    timestep_count(0),
    y_(model_.initial(time_)),
    y_swap_(model_.size()) {
  }

  // `particle_id` is used to produce independent RNG streams for each particle
  void run(const time_type time_end, size_t particle_id) {
    while (time_ < time_end) {
      rng_state_type rng_state;
      rng_state.ctr[0] = timestep_count;
      rng_state.ctr[1] = particle_id;
      // Unlike the new GPU graph code, in the CPU version there's just one
      // update fn so just fix "equation number" here to zero for now
      rng_state.ctr[2] = 0;
      rng_state.ctr[3] = 0;
      // TODO(mjr) make it possible to set the key - re-add some of the "seed" code?
      rng_state.key[0] = 0;
      rng_state.key[1] = 0;
      model_.update(time_, y_.data(), rng_state, y_swap_.data());
      time_++;
      timestep_count++;
      std::swap(y_, y_swap_);
    }
  }

  void state(const std::vector<size_t>& index,
             typename std::vector<real_type>::iterator end_state) const {
    for (size_t i = 0; i < index.size(); ++i) {
      *(end_state + i) = y_[index[i]];
    }
  }

  void state_full(typename std::vector<real_type>::iterator end_state) const {
    for (size_t i = 0; i < y_.size(); ++i) {
      *(end_state + i) = y_[i];
    }
  }

  size_t size() const {
    return y_.size();
  }

  time_type time() const {
    return time_;
  }

  void swap() {
    std::swap(y_, y_swap_);
  }

  void set_time(const time_type time) {
    time_ = time;
  }

  void set_state(const particle<T>& other) {
    y_swap_ = other.y_;
  }

  void set_pars(const pars_type pars, const time_type time, bool set_state) {
    const auto m = T(pars);
    if (m.size() != size()) {
      std::stringstream msg;
      msg << "'pars' created inconsistent state size: " <<
        "expected length " << size() << " but created length " <<
        m.size();
      throw std::invalid_argument(msg.str());
    }
    model_ = m;
    time_ = time;
    if (set_state) {
      y_ = model_.initial(time_);
    }
  }

  void set_state(typename std::vector<real_type>::const_iterator state) {
    for (size_t i = 0; i < y_.size(); ++i, ++state) {
      y_[i] = *state;
    }
  }

  void set_state(typename std::vector<real_type>::const_iterator state,
                 const std::vector<size_t>& index) {
    for (size_t i = 0; i < index.size(); ++i, ++state) {
      y_[index[i]] = *state;
    }
  }

  real_type compare_data(const data_type& data, size_t particle_id) {
    // TODO(mjr) What should we set the ctr to here? See also gpu equivalent in
    // kernels.hpp
    rng_state_type rng_state;
    rng_state.ctr[0] = 0;
    rng_state.ctr[1] = particle_id;
    // Unlike the new GPU graph code, in the CPU version there's just one
    // update fn so just fix "equation number" here to zero for now
    rng_state.ctr[2] = 0;
    rng_state.ctr[3] = 0;
    // TODO(mjr) make it possible to set the key - re-add some of the "seed" code?
    rng_state.key[0] = 0;
    rng_state.key[1] = 0;
    return model_.compare_data(y_.data(), data, rng_state);
  }

  T& model() {
    return model_;
  }

  std::vector<real_type> state() {
    return y_;
  }

private:
  T model_;
  time_type time_;

  // Count of the number of timesteps performed, used for setting the RNG
  // counter. It might be possible to remove this and just use time_, but in
  // principle it's possible to run some timesteps, set time_ back to zero
  // (using set_time()), and then run some more timesteps. The second set would
  // then end up getting the same random draws as the first set, which isn't
  // what we want
  size_t timestep_count;

  std::vector<real_type> y_;
  std::vector<real_type> y_swap_;
};

}

#endif
