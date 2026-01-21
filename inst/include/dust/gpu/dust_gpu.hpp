#ifndef DUST_GPU_DUST_GPU_HPP
#define DUST_GPU_DUST_GPU_HPP

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dust/gpu/call.hpp"
#include "dust/gpu/cuda.hpp"
#include "dust/gpu/device_resample.hpp"
#include "dust/gpu/filter_state.hpp"
#include "dust/gpu/kernels.hpp"
#include "dust/gpu/launch_control.hpp"
#include "dust/gpu/types.hpp"
#include "dust/filter_tools.hpp"
#include "dust/particle.hpp"
#include "dust/random/density.hpp"
#include "dust/utils.hpp"

namespace dust {

template <typename T>
class dust_gpu {
public:
  using model_type = T;
  using time_type = size_t;
  using pars_type = dust::pars_type<T>;
  using real_type = typename T::real_type;
  using data_type = typename T::data_type;
  using internal_type = typename T::internal_type;
  using shared_type = typename T::shared_type;
  using rng_state_type = typename T::rng_state_type;

  // TODO: fix this elsewhere, perhaps (see also dust/dust_cpu.hpp)
  using filter_state_type = dust::filter::filter_state_device<real_type>;

  dust_gpu(const pars_type& pars,
           const size_t time,
           const size_t n_particles,
           const size_t n_threads,
           //const std::vector<rng_int_type>& seed,
           const gpu::gpu_config& gpu_config) :
    n_pars_(0),
    n_particles_each_(n_particles),
    n_particles_total_(n_particles),
    n_state_full_(0),
    n_state_(0),
    pars_are_shared_(true),
    n_threads_(n_threads),
    gpu_config_(gpu_config),
    select_needed_(true),
    select_scatter_(false),
    time_(time),
    resample_calls_(0) {
    // TODO(mjr) replace seed with key for philox? Need a way to set it (for
    // reproducibility). Maybe at first just hardcode a key then work out a
    // good way to set it (idea - pass key via kernel arg? only idea I can
    // think of that would allow changing key without recompiling anything)
    //initialise_device_state(std::vector<pars_type>(1, pars), seed);
    initialise_device_state(std::vector<pars_type>(1, pars));
    shape_ = {n_particles};
  }

  dust_gpu(const std::vector<pars_type>& pars, const size_t time,
           const size_t n_particles, const size_t n_threads,
           //const std::vector<rng_int_type>& seed,
           const std::vector<size_t>& shape,
           const gpu::gpu_config& gpu_config) :
    n_pars_(pars.size()),
    n_particles_each_(n_particles == 0 ? 1 : n_particles),
    n_particles_total_(n_particles_each_ * pars.size()),
    n_state_full_(0), // needed for malloc size
    n_state_(0),
    pars_are_shared_(n_particles != 0),
    n_threads_(n_threads),
    gpu_config_(gpu_config),
    select_needed_(true),
    select_scatter_(false),
    time_(time),
    resample_calls_(0) {
    initialise_device_state(pars);
    // constructing the shape here is harder than above.
    if (n_particles > 0) {
      shape_.push_back(n_particles);
    }
    for (auto i : shape) {
      shape_.push_back(i);
    }
  }

  // We only need a destructor when running with cuda profiling; don't
  // include ond otherwise because we don't actually follow the rule
  // of 3/5/0
#ifdef DUST_ENABLE_CUDA_PROFILER
  ~dust_gpu() {
    cuda_profiler_stop(gpu_config_);
  }
#endif

  size_t n_threads() const {
    return n_threads_;
  }

  size_t n_particles() const {
    return n_particles_total_;
  }

  size_t n_state() const {
    return n_state_;
  }

  size_t n_state_full() const {
    return n_state_full_;
  }

  size_t n_variables() const {
    return n_state_full();
  }

  size_t n_pars() const {
    return n_pars_;
  }

  size_t n_pars_effective() const {
    return n_pars_ == 0 ? 1 : n_pars_;
  }

  bool pars_are_shared() const {
    return pars_are_shared_;
  }

  size_t n_data() const {
    return device_data_offsets_.size();
  }

  // NOTE: this is _just_ the offsets. However, when this is used
  // within the filter code we only care about the first part of the
  // map (i.e., the time to stop at) so it's ok.
  const std::map<size_t, size_t>& data() const {
    return device_data_offsets_;
  }

  size_t time() const {
    return time_;
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

  // These two (check_errors, reset_errors) don't really exist for
  // this model because we always (currently) run particles serially
  // on the cpu (and errors on the gpu are unrecoverable). If we ever
  // set up to do openmp running of the particles (and we probably
  // should), then these need the same implementation as for dust_cpu.
  void check_errors() {
  }

  void reset_errors() {
  }

  void set_pars(const pars_type& pars, bool set_state) {
    set_pars(std::vector<pars_type>(1, pars), set_state);
  }

  void set_pars(const std::vector<pars_type>& pars, bool set_state) {
    set_device_shared(pars);
    if (set_state) {
      set_state_from_pars(pars);
    }
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_type>& state,
                 const std::vector<size_t>& index) {
    if (index.size() != 0) {
      throw std::runtime_error("Can't use index with gpu models");
    }
    const size_t n_particles = n_particles_total_;
    const size_t n_state = n_state_full();
    const bool individual = state.size() == n_state * n_particles;
    const size_t n = individual ? 1 : n_particles_each_;
    auto it = state.begin();
    // Interleaved state of all particles
    std::vector<real_type> y(n_particles * n_state);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_particles; ++i) {
      size_t at = i;
      for (size_t j = 0; j < n_state; ++j) {
        at = dust::utils::stride_copy(y.data(),
                                      *(it + (i / n) * n_state + j),
                                      at,
                                      n_particles);
      }
    }
    device_state_.y.set_array(y);
    select_needed_ = true;
  }

  void set_time(const size_t time) {
    time_ = time;
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    const size_t n_particles = n_particles_total_;
    n_state_ = index.size();
    device_state_.set_device_index(index, n_particles, n_state_full());

    select_needed_ = true;
    select_scatter_ = !std::is_sorted(index.cbegin(), index.cend());

    const auto block_size = cuda_pars_.index_scatter.block_size;
    cuda_pars_.index_scatter =
      dust::gpu::launch_control_simple(block_size, n_particles * n_state());
  }

  // TODO(mjr) move most of the graph stuff to member data so it can be
  // reused across multiple calls to `run`
  void run(const size_t time_end) {
#ifdef __NVCC__
    if (time_end > time_) {
      const size_t time_start = time_;
      // Device copy of the time
      size_t *d_time;

      // Allocated the device time variables. Don't bother with initialising
      // values as they will be written before they're used anyway
      CUDA_CALL(cudaMalloc(&d_time, sizeof(size_t)));

      size_t time = time_start;

      // Get the number of update kernels
      const size_t n_update_kernels = dust::gpu::get_num_update_gpu_kernels<T>();

      // Declare graph handle
      cudaGraph_t graph;

      // Create an empty graph
      CUDA_CALL(cudaGraphCreate(&graph, 0));

      // Create a kernel node for each of the update kernels
      std::vector<cudaGraphNode_t> nodes(n_update_kernels);

      // Storage for the kernel args
      // TODO(mjr) change the hardcoded number of args here when needed. Maybe
      // better (definitely safer) to just use `.push_back()` or
      // `.emplace_back()`
      std::vector<std::vector<void *>> kernel_args(n_update_kernels, std::vector<void *>(14, nullptr));
      std::vector<cudaKernelNodeParams> kernel_node_params(n_update_kernels);

      const size_t n_pars_effective_local = n_pars_effective();
      const real_type *y_local = device_state_.y.data();
      const real_type *y_next_local = device_state_.y_next.data();
      const int *internal_int_local = device_state_.internal_int.data();
      const real_type *internal_real_local = device_state_.internal_real.data();
      const int *shared_int_local = device_state_.shared_int.data();
      const real_type *shared_real_local = device_state_.shared_real.data();

      void **kernels = dust::gpu::get_update_gpu_kernels<T>();

      // Create nodes with the appropriate params (copied from the original
      // kernel launch params etc) and add them to the graph
      for (size_t k = 0; k < n_update_kernels; k += 1) {
        // TODO(mjr) make a struct for most of the kernel arguments and then
        // just pass a ptr to the struct? Also see
        // https://github.com/mrc-ide/dust/issues/319

        kernel_args[k][0] = (void *) &time_start;
        kernel_args[k][1] = (void *) &d_time;
        kernel_args[k][2] = (void *) &n_particles_total_;
        kernel_args[k][3] = (void *) &n_pars_effective_local;
        kernel_args[k][4] = (void *) &y_local;
        kernel_args[k][5] = (void *) &y_next_local;
        kernel_args[k][6] = (void *) &internal_int_local;
        kernel_args[k][7] = (void *) &internal_real_local;
        kernel_args[k][8] = (void *) &device_state_.n_shared_int;
        kernel_args[k][9] = (void *) &device_state_.n_shared_real;
        kernel_args[k][10] = (void *) &shared_int_local;
        kernel_args[k][11] = (void *) &shared_real_local;
        kernel_args[k][12] = (void *) &cuda_pars_.run.shared_int;
        kernel_args[k][13] = (void *) &cuda_pars_.run.shared_real;

        kernel_node_params[k] = {
          .func = (void*) kernels[k],
          .gridDim = cuda_pars_.run.block_count,
          .blockDim = cuda_pars_.run.block_size,
          .sharedMemBytes = (unsigned int) cuda_pars_.run.shared_size_bytes,
          .kernelParams = (void **) kernel_args[k].data(),
          .extra = nullptr
        };

        CUDA_CALL(cudaGraphAddKernelNode(&nodes[k], graph, nullptr, 0, &kernel_node_params[k]));
      }

      // Get the number of update deps 
      const size_t n_update_deps = dust::gpu::get_num_update_gpu_dependencies<T>();

      // Get the array of deps
      const size_t (*update_deps)[2] = dust::gpu::get_update_gpu_dependencies<T>();

      // Add node dependencies
      for (size_t dep = 0; dep < n_update_deps; dep += 1) {
        CUDA_CALL(
          cudaGraphAddDependencies(
            graph,
            &nodes[update_deps[dep][0]],
            &nodes[update_deps[dep][1]],
            1
          )
        );
      }

      // Output the graph to a dot file for debugging
      //std::cout << "Outputting dot file of graph\n";
      //CUDA_CALL(cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose));
      //CUDA_CALL(cudaGraphDebugDotPrint(graph, "graph.dot", 0));

      // Declare and create an executable instance of the graph
      cudaGraphExec_t graph_exec;
      CUDA_CALL(cudaGraphInstantiate(&graph_exec, graph, 0));

      for (time = time_start; time < time_end; time += 1) {
        // Set the device copy of the time
        CUDA_CALL(cudaMemcpyAsync(d_time, &time, sizeof(size_t), cudaMemcpyHostToDevice, kernel_stream_.stream()));

        // Launch the graph
        CUDA_CALL(cudaGraphLaunch(graph_exec, kernel_stream_.stream()));

        // Synchronise the kernel stream
        kernel_stream_.sync();
      }

      // Swap the device state pointers if we've done an odd number of
      // timesteps in this go
      if ((time_end - time_start) % 2 == 1) {
        device_state_.swap();
      }

      select_needed_ = true;
      time_ = time_end;

      // TODO(mjr) probably a stupid place to put this
      // Should we make a destructor for `dust_gpu` so we can free stuff there?
      CUDA_CALL(cudaFree(d_time));

      // Destroy graph objects
      CUDA_CALL(cudaGraphExecDestroy(graph_exec));
      CUDA_CALL(cudaGraphDestroy(graph));
    }
#else
    printf("CUDA not enabled!\n");
#endif
  }

  std::vector<real_type> simulate(const std::vector<size_t>& time_end) {
    const size_t n_time = time_end.size();
    // The filter snapshot class can be used to store the indexed state
    // (implements async copy, swap space, and deinterleaving)
    // Filter trajectories not used as we don't need order here
    dust::filter::filter_snapshots_device<real_type> state_store;
    state_store.resize(n_state(), n_particles(), time_end);
    for (size_t t = 0; t < n_time; ++t) {
      run(time_end[t]);
      state_store.store(device_state_selected());
      state_store.advance();
    }
    std::vector<real_type> ret(n_state() * n_particles() * n_time);
    state_store.history(ret.data());
    return ret;
  }

  void state(std::vector<real_type>& end_state) {
    return state(end_state.begin());
  }

  void state(typename std::vector<real_type>::iterator end_state) {
    size_t np = n_particles_total_;
    size_t index_size = n_state_;

    // Run the selection and copy items back
    run_select();
    std::vector<real_type> y_selected(np * index_size);
    device_state_.y_selected.get_array(y_selected);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      dust::utils::destride_copy(end_state + i * index_size, y_selected, i,
                                  np);
    }
  }

  // TODO: we should really do this via a kernel I think? Currently we
  // grab the whole state back from the device to the host, then
  // filter through it. See https://github.com/mrc-ide/dust/pull/302
  void state(std::vector<size_t> index,
             std::vector<real_type>& end_state) {
    std::vector<real_type> full_state(n_state_full_ * n_particles_total_);
    get_device_state(full_state.begin());
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_particles_total_; ++i) {
      size_t particle_start = i * index.size();
      for (size_t j = 0; j < index.size(); ++j) {
        end_state[particle_start + j] =
          full_state[i * n_state_full_ + index[j]];
      }
    }
  }

  void state_full(std::vector<real_type>& end_state) {
    state_full(end_state.begin());
  }

  void state_full(typename std::vector<real_type>::iterator end_state) {
    get_device_state(end_state);
  }

  void state_full(dust::gpu::device_array<real_type>& device_state,
                  size_t dst_offset) {
    device_state.set_array(device_state_.y.data(),
                           device_state_.y.size(), dst_offset);
  }

  void reorder(const std::vector<size_t>& index) {
    size_t n_particles = n_particles_total_;
    size_t n_state = n_state_full();
    device_state_.scatter_index.set_array(index);
    bool select_kernel = false;
#ifdef __NVCC__
    dust::gpu::scatter_device<real_type><<<cuda_pars_.reorder.block_count,
                                          cuda_pars_.reorder.block_size,
                                          0,
                                          kernel_stream_.stream()>>>(
      device_state_.scatter_index.data(),
      device_state_.y.data(),
      device_state_.y_next.data(),
      n_state,
      n_particles,
      select_kernel);
    kernel_stream_.sync();
#else
    dust::gpu::scatter_device<real_type>(
      device_state_.scatter_index.data(),
      device_state_.y.data(),
      device_state_.y_next.data(),
      n_state,
      n_particles,
      select_kernel);
#endif

    device_state_.swap();
    select_needed_ = true;
  }

  // NOTE: this is only used for debugging/testing, otherwise we would
  // make device_weights and scan class members.
  std::vector<size_t> resample(const std::vector<real_type>& weights) {
    dust::gpu::device_weights<real_type>
      device_weights(n_particles(), n_pars_effective());
    device_weights.weights() = weights;

    dust::gpu::device_scan_state<real_type> scan;
    scan.initialise(n_particles_total_, device_weights.weights());
    resample(device_weights.weights(), scan);

    std::vector<size_t> index(n_particles());
    device_state_.scatter_index.get_array(index);
    return index;
  }

  // Functions used in the device filter
  void resample(dust::gpu::device_array<real_type>& weights,
                dust::gpu::device_scan_state<real_type>& scan) {
    rng_state_type resample_rng;
    resample_rng.ctr[0] = resample_calls_;
    resample_rng.ctr[1] = n_particles_total_;
    resample_rng.ctr[2] = 0;
    resample_rng.ctr[3] = 0;
    // TODO(mjr) key
    resample_rng.key[0] = 0;
    resample_rng.key[1] = 0;
    dust::filter::run_device_resample(n_particles(),
                                      n_pars_effective(),
                                      n_state_full(),
                                      cuda_pars_,
                                      kernel_stream_,
                                      resample_stream_,
                                      resample_rng,
                                      device_state_,
                                      weights,
                                      scan);
        resample_calls_++;
  }

  // For the particle filter only
  dust::gpu::device_array<size_t>& filter_kappa() {
    return device_state_.scatter_index;
  }

  dust::gpu::device_array<real_type>& device_state_full() {
    kernel_stream_.sync();
    return device_state_.y;
  }

  dust::gpu::device_array<real_type>& device_state_selected() {
    run_select();
    return device_state_.y_selected;
  }

  void set_n_threads(size_t n_threads) {
    n_threads_ = n_threads;
  }

  void set_data(std::map<size_t, std::vector<data_type>>& data,
                bool data_is_shared) {
    std::vector<data_type> flattened_data;
    size_t i = 0;
    for (auto & d_time : data) {
      device_data_offsets_[d_time.first] = i;
      for (auto & d : d_time.second) {
        flattened_data.push_back(d);
        i++;
      }
    }
    device_data_ = dust::gpu::device_array<data_type>(flattened_data.size());
    device_data_.set_array(flattened_data);
    data_is_shared_ = data_is_shared;
  }

  std::vector<real_type> compare_data() {
    std::vector<real_type> res;
    auto d = device_data_offsets_.find(time());
    if (d != device_data_offsets_.end()) {
      res.resize(n_particles());
      compare_data(device_state_.compare_res, d->second);
      device_state_.compare_res.get_array(res);
    }
    return res;
  }

  void compare_data(dust::gpu::device_array<real_type>& res,
                    const size_t data_offset) {
#ifdef __NVCC__
    dust::gpu::compare_particles<T><<<cuda_pars_.compare.block_count,
                                       cuda_pars_.compare.block_size,
                                       cuda_pars_.compare.shared_size_bytes,
                                       kernel_stream_.stream()>>>(
                     n_particles(),
                     n_pars_effective(),
                     device_state_.y.data(),
                     res.data(),
                     device_state_.internal_int.data(),
                     device_state_.internal_real.data(),
                     device_state_.n_shared_int,
                     device_state_.n_shared_real,
                     device_state_.shared_int.data(),
                     device_state_.shared_real.data(),
                     device_data_.data() + data_offset,
                     cuda_pars_.compare.shared_int,
                     cuda_pars_.compare.shared_real,
                     data_is_shared_);
    kernel_stream_.sync();
#else
    const bool use_shared_int = false;
    const bool use_shared_real = false;
    dust::gpu::compare_particles<T>(
                     n_particles(),
                     n_pars_effective(),
                     device_state_.y.data(),
                     res.data(),
                     device_state_.internal_int.data(),
                     device_state_.internal_real.data(),
                     device_state_.n_shared_int,
                     device_state_.n_shared_real,
                     device_state_.shared_int.data(),
                     device_state_.shared_real.data(),
                     device_data_.data() + data_offset,
                     use_shared_int,
                     use_shared_real,
                     data_is_shared_);
#endif
  }

private:
  // delete move and copy to avoid accidentally using them
  dust_gpu(const dust_gpu &) = delete;
  dust_gpu(dust_gpu &&) = delete;

  // Host quantities
  const size_t n_pars_; // 0 in the "single" case, >=1 otherwise
  const size_t n_particles_each_; // Particles per parameter set
  const size_t n_particles_total_; // Total number of particles
  size_t n_state_full_; // State size of a particle
  size_t n_state_; // State size of a particle with an index
  const bool pars_are_shared_; // Does the n_particles dimension exist in shape?

  std::vector<size_t> shape_; // shape of output
  size_t n_threads_;
  // TODO(mjr) maybe replace with a semi-magic index (n_kernels + 1 for that component of the index tuple?)
  // Wouldn't need to store it here though - just need to also change odin.dust
  // to use this index. Maybe similar applies elsewhere? Are there any other
  // "extra" RNGs anywhere?
  //rng_state_type resample_rng_; // for the filter
  gpu::gpu_config gpu_config_;

  // GPU support
  dust::gpu::launch_control_dust cuda_pars_;
  dust::gpu::device_state<real_type> device_state_;
  dust::gpu::device_array<data_type> device_data_;
  std::map<size_t, size_t> device_data_offsets_;
  dust::gpu::cuda_stream kernel_stream_;
  dust::gpu::cuda_stream resample_stream_;
  bool data_is_shared_;

  bool select_needed_;
  bool select_scatter_;
  size_t time_;
  size_t resample_calls_;

  // Naming of functions:
  //
  // Initialise called once (device_state and device_memory), to
  // allocate memory on device. Because the size of the problem
  // (number of particles, state size, number of parameter sets) is
  // constant, this never needs doing again.
  //
  // Set can be called multiple times, sets device memory from host

  // Sets state from model + pars, called from the constructors
  void initialise_device_state(const std::vector<pars_type>& pars) {
    if (n_state_full_ == 0) {
      //auto r = rng_state_blank_;
      // TODO: it would be nice to enforce that the rng was not
      // accessed here; it will not work. We could error?
      const dust::particle<T> p(pars[0], time_);
      n_state_full_ = p.size();
      n_state_ = n_state_full_;
      // TODO(mjr) can't detect rng use by testing if the state has changed any
      // more. Not super important to implement this though - just don't
      // use rng in initial conditions
      //if (r != rng_state_blank_) {
        //throw std::runtime_error("GPU models cannot use rng in initial");
      //}
    }

    initialise_device_memory(pars[0].shared);
    set_device_shared(pars);
    set_state_from_pars(pars);

    set_cuda_launch();

    select_needed_ = true;

    // ifdef guards not really needed here but helps guarantee
    // symmetry with destructor.
#ifdef DUST_ENABLE_CUDA_PROFILER
    cuda_profiler_start(gpu_config_);
#endif
  }

  // This only gets called on construction; the size of these never
  // changes.  Could be merged with the above really.
  void initialise_device_memory(typename dust::shared_ptr<T> s) {
    const size_t n_pars = n_pars_effective();
    const size_t n_internal_int = dust::gpu::internal_int_size<T>(s);
    const size_t n_internal_real = dust::gpu::internal_real_size<T>(s);
    const size_t n_shared_int = dust::gpu::shared_int_size<T>(s);
    const size_t n_shared_real = dust::gpu::shared_real_size<T>(s);
    device_state_.initialise(n_particles_total_, n_state_full_, n_pars,
                             n_internal_int, n_internal_real,
                             n_shared_int, n_shared_real);
  }

  void set_device_shared(const std::vector<pars_type>& pars) {
    size_t n = n_particles() == 0 ? 0 : n_state_full();
    std::vector<dust::particle<T>> p;
    for (size_t i = 0; i < n_pars_effective(); ++i) {
      p.push_back(dust::particle<T>(pars[i], time_));
      if (n > 0 && p.back().size() != n) {
        std::stringstream msg;
        msg << "'pars' created inconsistent state size: " <<
          "expected length " << n << " but parameter set " << i + 1 <<
          " created length " << p.back().size();
        throw std::invalid_argument(msg.str());
      }
      n = p.back().size(); // ensures all particles have same size
    }

    const size_t n_shared_int = device_state_.n_shared_int;
    const size_t n_shared_real = device_state_.n_shared_real;
    std::vector<int> shared_int(n_shared_int * n_pars_effective());
    std::vector<real_type> shared_real(n_shared_real * n_pars_effective());
    for (size_t i = 0; i < pars.size(); ++i) {
      int * dest_int = shared_int.data() + n_shared_int * i;
      real_type * dest_real = shared_real.data() + n_shared_real * i;
      dust::gpu::shared_copy<T>(pars[i].shared, dest_int, dest_real);
    }
    device_state_.shared_int.set_array(shared_int);
    device_state_.shared_real.set_array(shared_real);
  }

  // TODO: This update function is wildly inefficient; we should
  // probably support things like "copy one state to all the particles
  // of that parameter index", possibly as a kernel. Detecting when
  // that is the case would be important as we're probably doing the
  // wrong thing here sometimes:
  // https://github.com/mrc-ide/dust/issues/310
  //
  // TODO: We don't check again that the rng was not accessed, but as
  // the first time we check this we'll catch most cases. We don't try
  // and sync state back from the device to the host here as that will
  // be a fairly big copy that most of the time is not needed.
  void set_state_from_pars(const std::vector<pars_type>& pars) {
    const size_t n_pars = pars.size(); // or n_pars_effective();
    std::vector<std::vector<real_type>>
      state_host(n_particles() * n_pars,
                 std::vector<real_type>(n_state_full_));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < n_pars; ++i) {
      for (size_t j = 0; j < n_particles(); ++j) {
        dust::particle<T> p(pars[i], time_);
        p.state_full(state_host[i * n_particles() + j].begin());
      }
    }

    set_device_state(state_host);
  }

  // Interleaves and copies a 2D state vector
  void set_device_state(std::vector<std::vector<real_type>>& state_full) {
    const size_t np = n_particles(), ny = n_state_full();
    std::vector<real_type> y(np * ny); // Interleaved state of all particles
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      // Interleave state
      dust::utils::stride_copy(y.data(), state_full[i], i, np);
    }
    // H -> D copy
    device_state_.y.set_array(y);
    select_needed_ = true;
  }

  // set_device_state and get_device_state are inverses of each other
  void set_device_state(std::vector<real_type>& state_full) {
    const size_t np = n_particles(), ny = n_state_full();
    std::vector<real_type> y(np * ny); // Interleaved state of all particles
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      // Interleave state
      dust::utils::stride_copy(y.data(), state_full[i], i, np);
    }
    // H -> D copy
    device_state_.y.set_array(y);
    select_needed_ = true;
  }

  void get_device_state(typename std::vector<real_type>::iterator state_full) {
    const size_t np = n_particles(), ny = n_state_full();
    std::vector<real_type> y(np * ny); // Interleaved state of all particles
    device_state_.y.get_array(y);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads_)
#endif
    for (size_t i = 0; i < np; ++i) {
      dust::utils::destride_copy(state_full + i * ny, y, i, np);
    }
  }

  void run_select() {
    if (!select_needed_) {
      return;
    }
    const bool select_kernel = true;
#ifdef __NVCC__
    size_t size_select_tmp = device_state_.select_tmp.size();
    cub::DeviceSelect::Flagged(device_state_.select_tmp.data(),
                                size_select_tmp,
                                device_state_.y.data(),
                                device_state_.index.data(),
                                device_state_.y_selected.data(),
                                device_state_.n_selected.data(),
                                device_state_.y.size(),
                                kernel_stream_.stream());
    kernel_stream_.sync();
    if (select_scatter_) {
      dust::gpu::scatter_device<real_type><<<cuda_pars_.index_scatter.block_count,
                                           cuda_pars_.index_scatter.block_size,
                                           0,
                                           kernel_stream_.stream()>>>(
        device_state_.index_state_scatter.data(),
        device_state_.y_selected.data(),
        device_state_.y_selected_swap.data(),
        n_state(),
        n_particles(),
        select_kernel);
      kernel_stream_.sync();
      device_state_.swap_selected();
    }
#else
    size_t selected_idx = 0;
    for (size_t i = 0; i < device_state_.y.size(); i++) {
      if (device_state_.index.data()[i] == 1) {
        device_state_.y_selected.data()[selected_idx] =
          device_state_.y.data()[i];
        selected_idx++;
      }
    }
    if (select_scatter_) {
      dust::gpu::scatter_device<real_type>(
        device_state_.index_state_scatter.data(),
        device_state_.y_selected.data(),
        device_state_.y_selected_swap.data(),
        n_state(),
        n_particles(),
        select_kernel);
      device_state_.swap_selected();
    }
#endif
    select_needed_ = false;
  }

  // Set up CUDA block sizes and shared memory preferences
  void set_cuda_launch() {
    cuda_pars_ = dust::gpu::launch_control_dust(gpu_config_,
                                                n_particles(),
                                                n_particles_each_,
                                                n_state(),
                                                n_state_full(),
                                                device_state_.n_shared_int,
                                                device_state_.n_shared_real,
                                                sizeof(real_type),
                                                sizeof(data_type));
  }
};

}

#endif
