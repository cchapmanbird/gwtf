# gwtf

[![Documentation Status](https://app.readthedocs.org/projects/pygwtf/badge/?version=stable)](https://pygwtf.readthedocs.io/en/stable/) [![DOI](https://zenodo.org/badge/1118152669.svg)](https://doi.org/10.5281/zenodo.20531363)

Waveform generation and likelihood approximations for slowly-evolving GW sources in the time-frequency domain in <b>O(10) microseconds</b>.

Usage of waveform generator in an end-to-end parameter estimation provided in the notebook [Parameter_estimation.ipynb](examples/Parameter_estimation.ipynb)

Currently being used to analyse the stellar-origin binaries in the LISA data-challenge Mojito lite.

Features:
--------
- Support both CPU and GPU based computations of both waveforms and likelihoods.
- Supports inner-product/waveform batching over GPU threads.
- Contains search-specific implementation of semi-coherent detection statistic, designed to minimise memory use and maximise batching of sources (only valid for single mode waveforms).
- Response function that generalises to slowly varying arm-lengths.
- Response can use either analytic or supplied orbits (Mojito-orbit reading in functionality is present and functional but somewhat crude).

Waveforms Supported:
--------
- TaylorT2Ecc (non-spinning)
- TaylorT3 (spinning)

Contributors
---------
- Christian Chapman-Bird
- Diganta Bandopadhyay
