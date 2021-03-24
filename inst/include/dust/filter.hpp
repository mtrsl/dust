#ifndef DUST_FILTER_HPP
#define DUST_FILTER_HPP

#include <dust/filter_state.hpp>
#include <dust/filter_tools.hpp>

namespace dust {
namespace filter {

template <typename T>
std::vector<typename T::real_t> filter(Dust<T> * obj,
                                       filter_state<typename T::real_t>& state,
                                       bool save_trajectories,
                                       std::vector<size_t> step_snapshot) {
  typedef typename T::real_t real_t;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  const size_t n_particles_each = n_particles / n_pars;
  std::vector<real_t> log_likelihood(n_pars);
  std::vector<real_t> log_likelihood_step(n_pars);
  std::vector<real_t> weights(n_particles);
  std::vector<size_t> kappa(n_particles);

  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);
    obj->state(state.trajectories.value_iterator());
    state.trajectories.advance();
  }

  state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);

  for (auto & d : obj->data()) {
    obj->run(d.first);
    obj->compare_data(weights, d.second);

    // TODO: we should cope better with the case where all weights
    // are 0; I think that is the behaviour in the model (or rather
    // the case where there is no data and so we do not resample)
    //
    // TODO: we should cope better with the case where one filter
    // has become impossible but others continue, but that's hard!
    auto wi = weights.begin();
    for (size_t i = 0; i < n_pars; ++i) {
      log_likelihood_step[i] = scale_log_weights<real_t>(wi, n_particles_each);
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

    if (state.snapshots.is_snapshot_step(d.first)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }
  }

  return log_likelihood;
}


template <typename T>
std::vector<typename T::real_t> filter_device(Dust<T> * obj,
                                       filter_state_device<typename T::real_t>& state
                                       bool save_trajectories,
                                       std::vector<size_t> step_snapshot) {
  typedef typename T::real_t real_t;

  const size_t n_particles = obj->n_particles();
  const size_t n_data = obj->n_data();
  const size_t n_pars = obj->n_pars_effective();
  const size_t n_particles_each = n_particles / n_pars;
  dust::device_array<real_t> log_likelihood(n_pars);
  dust::device_array<real_t> log_likelihood_step(n_pars);
  dust::device_array<real_t> weights(n_particles);
  dust::device_array<size_t> kappa(n_particles);

  // NOTES
  // 1) state.trajectories: this is a device_array which copies state (y_selected)
  // at each step. order needs kappa copied in at each step
  // 2) state.snapshots: this is a device_array which copies state_full at some
  // steps
  // 1 & 2) both of these will then be deinterleaved in their class
  // 3) obj->run: obj->run_device should just work
  // 4) obj->compare_data: will be like run_particles, but data needs to be flattened
  // and copied to the device. d.second will then be a pointer in to this
  // 5) scale_log_weights(): use cub::DeviceSegmentedReduce::Max and a simple kernel which calls exp()
  // NB: this is only used to calculate the total likelihood
  // 6) resample: calls resample_weight with a single RNG draw
  // see https://github.com/mrc-ide/mcstate/blob/master/R/particle_filter.R#L426-L431
  // or https://github.com/NVIDIA/cuda-samples/blob/master/Samples/shfl_scan/shfl_scan.cu
  // for prefix sum version
  // or https://arxiv.org/pdf/1301.4019.pdf for rejection sampler version


  if (save_trajectories) {
    state.trajectories.resize(obj->n_state(), n_particles, n_data);
    obj->state(state.trajectories.value_iterator());
    state.trajectories.advance();
  }

  state.snapshots.resize(obj->n_state_full(), n_particles, step_snapshot);

  for (auto & d : obj->data()) {
    obj->run(d.first);
    obj->compare_data(weights, d.second);

    // TODO: we should cope better with the case where all weights
    // are 0; I think that is the behaviour in the model (or rather
    // the case where there is no data and so we do not resample)
    //
    // TODO: we should cope better with the case where one filter
    // has become impossible but others continue, but that's hard!
    auto wi = weights.begin();
    for (size_t i = 0; i < n_pars; ++i) {
      log_likelihood_step[i] = scale_log_weights<real_t>(wi, n_particles_each);
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

    if (state.snapshots.is_snapshot_step(d.first)) {
      obj->state_full(state.snapshots.value_iterator());
      state.snapshots.advance();
    }
  }

  return log_likelihood;
}

}
}

#endif
