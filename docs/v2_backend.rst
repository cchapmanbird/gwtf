The v2 compiled backend
=======================

From version 2, the numerical work in ``pygwtf`` is done by a compiled
C++/HIP core rather than numba. The public Python interface
(:class:`~pygwtf.generator.AnalyticTimeFrequencyWaveform`) is largely the
same as before. This page describes how the backend is organised, and
how to extend it.

Architecture
------------

.. code-block:: text

   Python (pygwtf/)                         compiled core (src/cpp/)
   ---------------------------------        ------------------------------------------
   generator.py / multimode.py              bindings/module.cpp  (nanobind, DLPack)
     - array-API inputs -> backend arrays       -> validates arrays, fills POD args
     - allocates every output (NumPy/CuPy)   include/gwtf/api.hpp  (run_* entry points)
     - noise preparation, precision          cpu/cpu_ops.cpp       serial host loop
     - converts outputs to the input format  gpu/gpu_ops.hip       HIP kernels (CUDA via shim)
                                             include/gwtf/ops.hpp  shared pipeline
                                               models -> response -> Fresnel window -> sink

*  Outputs are allocated with NumPy (CPU) or CuPy
  (GPU) and handed to the extension zero-copy. 
  GPU kernels are launched on CuPy's current stream.
* Parameters, data and noise may be NumPy, CuPy, PyTorch, JAX or any 
  DLPack-capable array. Results are returned in the format of the inputs.
* All models -- analytic or numerical -- go through the same pipeline in 
``ops.hpp``. A source produces a per-segment mode track 
  (phase, frequency, fdot, amplitude) in double precision; a response 
  (TT polarisations or TDI-2 A/E/T) turns each mode
  into channel coefficients; and a *sink* either writes the waveform or accumulates
  ``<d|h>`` and ``<h|h>``.
* One task per (source, time segment): one GPU thread each, or one loop iteration 
  on the CPU. Unlike v1 (one thread per source), this keeps the GPU busy even for a 
  single source.

Building
--------

``pip install .`` (or ``pip install -e . --no-build-isolation`` for
development) builds the extensions with CMake via ``scikit-build-core``.
A C++17 compiler is required for the CPU backend. The GPU backend is
selected with the ``GWTF_GPU`` environment variable:

=========  ===============================================================
``AUTO``   (default) CUDA if ``nvcc`` is found, else HIP if ``hipcc`` is
           found, else CPU only.
``CUDA``   NVIDIA. Architectures from ``CMAKE_CUDA_ARCHITECTURES`` /
           ``CUDAARCHS`` (default: the GPU of the build machine, ``native``).
``HIP``    AMD/ROCm. Set ``CMAKE_HIP_ARCHITECTURES`` (e.g. ``gfx90a``).
``OFF``    CPU backend only.
=========  ===============================================================

For example::

    GWTF_GPU=HIP CMAKE_HIP_ARCHITECTURES=gfx90a pip install .
    CUDAARCHS="80;90" pip install .

Precision
---------

``precision="single"`` (the default) is the fast path; ``precision="double"``
keeps float64 throughout.

* Mode phases reach 1e7--1e8 rad, so the analytic model (and numerical tracks) 
  stay in float64.
* The mode phase and the ``f T`` term are reduced modulo 2 pi in double; 
  frequencies enter the per-bin arithmetic as offsets from the nearest bin. 
  Single precision only ever sees O(1)-cycle quantities, suppressing round-off.
* Far from the track, the Fresnel arguments are large and the quadratic phase 
  ``pi fdot delta^2`` and the Fresnel phase ``pi v^2 / 2`` are individually huge 
  but cancel. The kernel uses the asymptotic branch of the Fresnel approximation 
  to combine them analytically (see ``fresnel.hpp``), which is algebraically identical 
  to v1 but keeps float32 accurate.

Measured against v2 double precision, single-precision waveforms agree to
~2e-5 of the peak and segment-summed statistics to ~1e-6.

Adding an analytic model
------------------------

1. Write ``src/cpp/include/gwtf/models/<name>.hpp`` implementing the model
   concept documented in ``model_base.hpp``: a struct with static
   ``compute_derived`` (unit conversion and derived parameters, in place),
   ``time_bounds``, ``load`` (per-source state), ``track`` (phase, f, fdot,
   amplitude at time t) and ``ylm`` (spherical harmonics per mode). Use
   ``taylort2ecc.hpp`` as a template. Long PN expressions can be translated
   from Python with ``tools/translate_pn.py``.
2. Add it to ``GWTF_ANALYTIC_MODELS`` in ``models/registry.hpp`` with a new id
   and rebuild.
3. Subclass :class:`pygwtf.models.AnalyticModel` with ``kernel_name`` equal to
   the C++ ``name`` and the parameter names in the order of the C++ indices.
   The constructor checks the parameter counts against the compiled model.

Models with several harmonics set ``n_modes`` accordingly; statistics then
include cross-mode terms automatically.

Numerical multi-mode sources
----------------------------

:class:`pygwtf.multimode.MultipleHarmonicTimeFrequencyWaveform` takes
per-segment tracks ``(n_sources, n_modes, nT, 4)`` of ``(amp, phi, f, fdot)``
and spherical harmonics ``(n_sources, n_modes, 2)`` instead of a model, so
any code can drive the same kernels (``AnalyticTimeFrequencyWaveform.tracks``
produces this format from an analytic model). Entries with non-finite values,
zero amplitude or ``f <= 0`` are skipped, and ``fdot`` may be negative.
Statistics are those of the summed signal, cross-mode terms included; up to
``max_fused_modes`` (16) modes are fused in one kernel, larger counts fall
back to generating the waveform and taking dense inner products (to be improved!).

``cross_terms=False`` treats the modes as independent
(``<h|h> = sum_m <h_m|h_m>``; ``<d|h>`` is unchanged) and evaluates the
statistic with one thread per (source, mode, segment), accumulated
atomically, with no limit on the mode count. Cross terms are exactly zero
for modes more than about ``2 * kernel_width`` bins apart. For closer modes
they are a few per cent of ``<h|h>`` per segment, and typically 1e-4 to 1e-3
once summed over segments (more if the spacing stays near a whole number of
bins), so only drop them when that is acceptable.

Not yet implemented (extension points)
--------------------------------------

* **Global-fit mode** (per-walker data and noise, ``gf_mode`` /
  ``data_indices``): ``NoiseView`` in ``ops.hpp`` is the single place where
  data and weights are indexed; a per-source batch offset slots in there.
* **Fused semi-coherent search statistic** (``N_seg``): needs a per-source
  reduction over segments, i.e. a sink that combines the per-segment results
  of all segment tasks of a source (or a source-major launch).
* **Runtime (JIT) models**: the headers avoid heavy std dependencies so they
  could be compiled with hipRTC/NVRTC in future.
