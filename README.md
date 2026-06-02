# gwtf
Waveform generation and likelihood approximations for slowly-evolving GW sources in the time-frequency domain in <b>O(10) microseconds</b>.

Usage of waveform generator in an end-to-end parameter estimation provided in the notebook [Parameter_estimation.ipynb](https://github.com/cchapmanbird/gwtf/blob/cleanup_and_document/examples/Parameter_estimation.ipynb)

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
