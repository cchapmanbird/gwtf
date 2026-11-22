"""
Legacy (v1) numba implementation of pygwtf.

This subpackage holds the original numba CPU/CUDA kernels. It is kept only as a
reference for validating the v2 compiled backend (see ``tests/``) and will be
removed once v2 has been fully validated. It requires ``numba``.

Usage mirrors the v1 public API::

    from pygwtf.legacy.generator import AnalyticTimeFrequencyWaveform
    from pygwtf.legacy.models import TaylorT2Ecc
"""
