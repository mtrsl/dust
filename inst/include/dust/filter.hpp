#ifndef DUST_FILTER_HPP
#define DUST_FILTER_HPP

#include "dust/filter_state.hpp"
#include "dust/filter_tools.hpp"

namespace dust {
namespace filter {

// Host version
template <typename T>
std::vector<typename T::real_type>
filter(T * obj,
       size_t step_end,
       filter_state_host<typename T::real_type>& state,
       bool save_trajectories,
       std::vector<size_t> step_snapshot) {
  using real_type = typename T::real_type;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  const size_t n_particles_each = n_particles / n_pars;
  std::vector<real_type> log_likelihood(n_pars);
  std::vector<real_type> log_likelihood_step(n_pars);
  std::vector<real_type> weights(n_particles);
  std::vector<size_t> kappa(n_particles);

  // TODO: we should fill these really?
  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);

    // TODO: consider filling with NA, somehow
    // std::fill(state.trajectories.begin(), state.trajectories.end(), NA_REAL);

    // TODO: only add the starting point if we need to.
    //
    // if (obj->step() <= obj->data().begin()->first()) {
    obj->state(state.trajectories.value_iterator());
    // }

    state.trajectories.advance();
  }

  bool save_snapshots = false;
  if (step_snapshot.size() > 0) {
    save_snapshots = true;
    state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);
    // std::fill(state.snapshots.begin(), state.snapshots.end(), NA_REAL);
  }

  // TODO: we can jump more efficiently here I belive by jumping ahead
  // but I imagine that this is going to be fine for now.  The most
  // efficient approach would be to do a binary search for both
  // beginning and end iterators, then use std::range() here with
  // those.
  for (const auto& d : obj->data()) {
    const auto step = d.first;
    if (step <= obj->step()) {
      // TODO: we could do this more efficiently in one go
      if (save_trajectories) {
        state.trajectories.advance();
      }
      continue;
    }
    obj->run(step);
    obj->compare_data(weights, d.second);

    // TODO: we should cope better with the case where all weights
    // are 0; I think that is the behaviour in the model (or rather
    // the case where there is no data and so we do not resample)
    //
    // TODO: we should cope better with the case where one filter
    // has become impossible but others continue, but that's hard!
    auto wi = weights.begin();
    for (size_t i = 0; i < n_pars; ++i) {
      log_likelihood_step[i] =
        scale_log_weights<real_type>(wi, n_particles_each);
      log_likelihood[i] += log_likelihood_step[i];
      wi += n_particles_each;
    }

    // We could move this below if wanted but we'd have to rewrite
    // the re-sort algorithm; that would be worth doing I think
    // https://github.com/mrc-ide/dust/issues/202
    if (save_trajectories) {
      obj->state(state.trajectories.value_iterator());
    }

    obj->resample(weights, kappa);

    if (save_trajectories) {
      std::copy(kappa.begin(), kappa.end(),
                state.trajectories.order_iterator());
      state.trajectories.advance();
    }

    if (save_snapshots && state.snapshots.is_snapshot_step(d.first)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }

    if (step >= step_end) {
      break;
    }
  }

  return log_likelihood;
}



}
}

#endif
