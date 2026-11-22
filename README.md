# gwtf

[![Documentation Status](https://app.readthedocs.org/projects/pygwtf/badge/?version=stable)](https://pygwtf.readthedocs.io/en/stable/) [![DOI](https://zenodo.org/badge/1118152669.svg)](https://doi.org/10.5281/zenodo.20531363)

Waveform generation and likelihood approximations for slowly-evolving GW sources in the time-frequency domain.

Usage of waveform generator in an end-to-end parameter estimation provided in the notebook [Parameter_estimation.ipynb](examples/Parameter_estimation.ipynb)

Currently being used to analyse the stellar-origin binaries in the LISA data-challenge Mojito lite.

Basic documentation available <a href="https://pygwtf.readthedocs.io/en/latest/index.html">here</a>.

Features:
--------
- Compiled C++/HIP backend (v2): one source runs on the CPU and on AMD or NVIDIA GPUs, with NumPy/CuPy/PyTorch/JAX
  inputs via DLPack and a fast single-precision path (double precision optional). See
  [docs/v2_backend.rst](docs/v2_backend.rst).
- Support both CPU and GPU based computations of both waveforms and likelihoods.
- Supports inner-product/waveform batching over GPU threads.
- Contains search-specific implementation of semi-coherent detection statistic, designed to minimise memory use and maximise batching of sources (only valid for single mode waveforms).
- Response function that generalises to slowly varying arm-lengths.
- Response can use either analytic or supplied orbits (Mojito-orbit reading in functionality is present and functional but somewhat crude).
- Supports modelling of both box-car and tukey-window in the time-frequency Fresnel representation.

Waveforms Supported:
--------
- TaylorT2Ecc (non-spinning)
- TaylorT3 (spinning)
- Any multi-harmonic signal supplied as numerical per-segment tracks (`pygwtf.multimode`)

Installation from source needs a C++17 compiler; the GPU backend is built automatically when `nvcc` or `hipcc` is
available (override with `GWTF_GPU=CUDA|HIP|OFF`). GPU execution also needs the CuPy build for your platform.

Contributors
---------
- Christian Chapman-Bird
- Diganta Bandopadhyay

If you use ``pygwtf`` in your work, please consider citing both the software [doi](10.5281/zenodo.20531363) and the associated publication

```bibtex
@article{Bandopadhyay:2025fyx,
    author = "Bandopadhyay, Diganta and Chapman-Bird, Christian E. A. and Vecchio, Alberto",
    title = "{Global time-frequency search for stellar-mass binary black holes in LISA}",
    eprint = "2510.19047",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/339z-w985",
    journal = "Phys. Rev. D",
    volume = "114",
    number = "6",
    pages = "064015",
    year = "2026"
}
```
