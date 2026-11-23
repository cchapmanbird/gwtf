"""
Model-agnostic multi-harmonic time-frequency generator (v2 compiled backend).

Instead of an analytic model, each source is described numerically by the
per-segment track of every harmonic -- amplitude, phase, frequency and
frequency derivative -- plus the spin-weighted spherical harmonics of each
mode.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ._base import (
    COMPLEX_DTYPE,
    NOISE_DIAGONAL,
    NOISE_FULL,
    PRECISION_CODE,
    WINDOW_BOXCAR,
    WINDOW_TUKEY,
    TimeFrequencyBase,
    parse_n_seg,
)
from ._interop import ArrayFormat, to_backend
from .backend import Backend

__all__ = ["MultipleHarmonicTimeFrequencyWaveform"]


class MultipleHarmonicTimeFrequencyWaveform(TimeFrequencyBase):
    """
    Time-frequency waveforms and inner products from numerical mode tracks.

    Parameters are as for
    :class:`~pygwtf.generator.AnalyticTimeFrequencyWaveform` (minus the model).

    Track conventions
    -----------------
    ``phase_amp_information[s, m, t] = (amp, phi, f, fdot)`` of mode ``m`` of
    source ``s`` at the evaluation time of segment ``t`` (its midpoint when
    ``use_midpoint``, else its start). Entries with non-finite values, zero
    amplitude or ``f <= 0`` are skipped, so tracks can be padded with NaN or 0
    after a source leaves the band. ``fdot`` may be negative.

    ``y_lms[s, m] = (Y_lm, Y_l-m)``: the negative-m entry must include any
    symmetry factor such as ``(-1)^l``; for ``m = 0`` split the harmonic
    equally between the two entries. The same polarisation convention as
    the analytic models is applied, so feeding the (2,2) track of an analytic
    model with ``Y_22, Y_2-2`` reproduces its output exactly.

    Statistics
    ----------
    By default ``<h|h>`` is that of the summed signal, including the
    cross-mode terms ``2 Re <h_i|h_j>`` wherever two modes' Fresnel kernels
    share bins. With ``cross_terms=False`` the modes are treated as
    independent (``<h|h> = sum_m <h_m|h_m>``) and evaluated with one GPU
    thread per (source, mode, segment), which is faster and has no mode limit.
    ``<d|h>`` is identical in both cases.
    """

    # Kernel used for cross_terms=False: "mode_parallel" (one thread per
    # source, mode and segment) or "segment_loop" (one thread per source and
    # segment, looping over modes). Exposed for benchmarking only.
    _self_term_kernel = "mode_parallel"

    def __init__(
        self,
        config: dict,
        backend: str | Backend = "cpu",
        tdi_type: int | None = None,
        channels: Any = None,
        psds: Any = None,
        spacecraft_orbits: Any = None,
        spacecraft_ltts: Any = None,
        use_tukey: bool = False,
        invC: Any = None,
        full_invC: bool | None = None,
        precision: str = "single",
        prescription: str = "fresnel",
    ):
        if prescription != "fresnel":
            raise ValueError("Only the 'fresnel' prescription is available.")
        super().__init__(
            config,
            backend=backend,
            tdi_type=tdi_type,
            channels=channels,
            psds=psds,
            invC=invC,
            full_invC=full_invC,
            spacecraft_orbits=spacecraft_orbits,
            spacecraft_ltts=spacecraft_ltts,
            use_tukey=use_tukey,
            precision=precision,
        )

    @property
    def max_fused_modes(self) -> int:
        """Mode count above which cross-term statistics fall back to waveform + inner product."""
        return int(self.backend.core.max_fused_modes)

    def _inputs(self, phase_amp_information, y_lms):
        tracks = to_backend(phase_amp_information, self.backend, np.float64)
        ylms = to_backend(y_lms, self.backend, np.complex128)
        single = tracks.ndim == 3
        if single:
            tracks, ylms = tracks[None], ylms[None]
        nT = self.config["nT"]
        if tracks.ndim != 4 or tracks.shape[2] != nT or tracks.shape[3] != 4:
            raise ValueError(
                f"phase_amp_information must have shape (n_sources, n_modes, {nT}, 4)"
            )
        n, m = tracks.shape[:2]
        if ylms.shape != (n, m, 2):
            raise ValueError(f"y_lms must have shape {(n, m, 2)}")
        return (
            self.backend.xp.ascontiguousarray(tracks),
            self.backend.xp.ascontiguousarray(ylms),
            single,
        )

    def _track_segments(self, tracks):
        """
        In-band tranche range per source from backend tracks: first and last
        tranche where any mode has a usable entry (finite, non-zero amplitude)
        with its frequency in the grid's band ``[fmin, fmin + (nF - 1/2) dF]``
        -- the band the analytic models' time bounds use.
        """
        xp = self.backend.xp
        c = self.config
        nT = c["nT"]
        f = tracks[..., 2]
        usable = (
            xp.isfinite(tracks).all(axis=-1)
            & (tracks[..., 0] != 0)
            & (f >= c["fmin"])
            & (f <= c["fmin"] + (c["nF"] - 0.5) * c["dF"])
        ).any(axis=1)  # (n_sources, nT)
        has = usable.any(axis=1)
        first = xp.argmax(usable, axis=1)
        last = nT - 1 - xp.argmax(usable[:, ::-1], axis=1)
        return (
            xp.where(has, first, 0).astype(np.int32),
            xp.where(has, last, -1).astype(np.int32),
        )

    def segment_indices(self, phase_amp_information):
        """First and last in-band tranche per source, as used for ``N_seg``
        (see ``_track_segments``); ``end < start`` if none."""
        fmt = ArrayFormat.from_inputs(phase_amp_information)
        tracks = to_backend(phase_amp_information, self.backend, np.float64)
        single = tracks.ndim == 3
        if single:
            tracks = tracks[None]
        s0, s1 = self._track_segments(tracks)
        if single:
            s0, s1 = s0[0], s1[0]
        return fmt.wrap(s0, self.backend), fmt.wrap(s1, self.backend)

    def __call__(
        self,
        phase_amp_information: Any,
        y_lms: Any,
        channels: Any = None,
        psds: Any = None,
        parameters_response: Any = None,
        out: Any = None,
        compute_statistic: bool = False,
        mixed_precision: bool | None = None,
        use_midpoint: bool = True,
        invC: Any = None,
        invC_prefolded: bool = False,
        cross_terms: bool = True,
        data_indices: Any = None,
        N_seg: int | None = None,
        search_statistic: Any = None,
        segment_split: str = "spread",
        segment_ranges: Any = None,
    ):
        """
        Evaluate waveforms (summed over modes) or per-segment inner products.

        Parameters
        ----------
        phase_amp_information : array_like, shape (n_sources, n_modes, nT, 4)
            Per-segment ``(amp, phi, f, fdot)`` of each mode (leading axis
            optional for a single source).
        y_lms : array_like, shape (n_sources, n_modes, 2)
            ``(Y_lm, Y_l-m)`` per mode.
        parameters_response : array_like, shape (n_sources, 4), optional
            ``[cosi, pol, ecliptic_long, ecliptic_lat]`` (required for TDI;
            ``cosi`` is unused here since ``y_lms`` carry the inclination).
        compute_statistic : bool
            Return ``(n_sources, nT, 2)`` per-segment ``<d|h>`` and ``<h|h>`` of
            the full multi-mode signal, cross-mode terms included.
        cross_terms : bool
            With ``False``, neglect the cross-mode terms of ``<h|h>``
            (modes treated as independent). Faster, and appropriate when the
            modes are many kernel widths apart in frequency; see the class
            notes.
        data_indices : array_like of int, shape (n_sources,), optional
            Global-fit mode: walker of each source when the data and/or noise
            carry a leading per-walker axis (see
            :class:`~pygwtf.generator.AnalyticTimeFrequencyWaveform`).
        N_seg, search_statistic, segment_split :
            Semi-coherent statistic, as for the analytic generator.
        segment_ranges : (array_like, array_like), optional
            ``(start, end)`` in-band tranche range per source used to define
            the ``N_seg`` segments. Default: derived from the tracks (see
            :meth:`segment_indices`).

        Other arguments are as for the analytic generator.
        """
        if self.use_tukey and not use_midpoint:
            raise ValueError(
                "The Tukey Fresnel kernel is midpoint only, use_midpoint must be True."
            )
        if data_indices is not None and not compute_statistic:
            raise ValueError(
                "data_indices only applies to statistic calls (compute_statistic=True)."
            )
        if N_seg is not None:
            N_seg = parse_n_seg(N_seg)
            if not compute_statistic:
                raise ValueError(
                    "N_seg (semi-coherent statistic) requires compute_statistic=True."
                )
        precision = self._resolve_precision(mixed_precision)
        fmt = ArrayFormat.from_inputs(phase_amp_information, channels, y_lms)
        xp = self.backend.xp
        core = self.backend.core
        stream = self.backend.stream_ptr()
        grid = self._grid(use_midpoint)

        tracks, ylms, single = self._inputs(phase_amp_information, y_lms)
        n, n_modes = tracks.shape[:2]
        nT, nF = self.config["nT"], self.config["nF"]
        seg_start = xp.zeros(n, np.int32)
        seg_end = xp.full(n, nT - 1, np.int32)
        if parameters_response is None:
            if self.tdi_type is not None:
                raise ValueError(
                    "parameters_response must be supplied for TDI generation."
                )
            response = xp.zeros((n, 4), np.float64)
        else:
            response = to_backend(
                parameters_response, self.backend, np.float64
            )
            if response.ndim == 1:
                response = response[None]
            if response.shape != (n, 4):
                raise ValueError(
                    f"parameters_response must have shape {(n, 4)}"
                )
        orbits, ltts = self._detector(precision)
        window = WINDOW_TUKEY if self.use_tukey else WINDOW_BOXCAR
        cdt = COMPLEX_DTYPE[precision]
        noise_code = NOISE_FULL if self.full_invC else NOISE_DIAGONAL
        model = int(core.numerical_model_id)

        def waveform_into(buf, amp_scale):
            core.waveform(
                grid,
                model,
                n,
                None,
                tracks,
                ylms,
                seg_start,
                seg_end,
                response,
                self._response_kind,
                orbits,
                ltts,
                window,
                PRECISION_CODE[precision],
                buf,
                amp_scale,
                stream,
            )

        if compute_statistic:
            data, w, scale, index = self._noise_inputs(
                channels,
                psds,
                invC,
                invC_prefolded,
                precision,
                data_indices,
                n,
            )
            shape = (n, nT, 2)
            out = (
                xp.empty(shape, cdt)
                if out is None
                else self._check_out(out, shape, cdt)
            )
            if not cross_terms or n_modes <= self.max_fused_modes:
                mode = core.statistic_modes[
                    "cross_terms" if cross_terms else self._self_term_kernel
                ]
                core.statistic(
                    grid,
                    model,
                    n,
                    None,
                    tracks,
                    ylms,
                    seg_start,
                    seg_end,
                    response,
                    self._response_kind,
                    orbits,
                    ltts,
                    window,
                    PRECISION_CODE[precision],
                    noise_code,
                    data,
                    w,
                    out,
                    scale,
                    stream,
                    mode,
                    data_index=index,
                )
            else:
                # Too many modes to keep cross terms in registers: materialise
                # the waveform and take dense inner products (which apply the
                # single-precision rescaling to h and the data on load).
                nbytes = n * nT * nF * self.n_channels * np.dtype(cdt).itemsize
                if nbytes > 2**32:
                    warnings.warn(
                        f"{n_modes} modes exceeds the fused limit ({self.max_fused_modes}); "
                        f"allocating a {nbytes / 2**30:.1f} GiB intermediate waveform.",
                        stacklevel=2,
                    )
                h = xp.zeros((n, nT, nF, self.n_channels), cdt)
                waveform_into(h, 1.0)
                core.inner_product(
                    h,
                    data,
                    w,
                    PRECISION_CODE[precision],
                    noise_code,
                    out,
                    stream,
                    data_index=index,
                    scale=scale,
                )
            if N_seg is not None:
                if segment_ranges is None:
                    s0, s1 = self._track_segments(tracks)
                else:
                    s0, s1 = (
                        xp.clip(
                            to_backend(r, self.backend, np.int32).reshape(-1),
                            lo,
                            nT - 1,
                        ).astype(np.int32)
                        for r, lo in zip(segment_ranges, (0, -1))
                    )
                    if s0.shape != (n,) or s1.shape != (n,):
                        raise ValueError(
                            f"segment_ranges must be two arrays of shape ({n},)"
                        )
                out = self._semicoherent(
                    out,
                    s0,
                    s1,
                    N_seg,
                    segment_split,
                    precision,
                    search_statistic,
                )
        else:
            shape = (n, nT, nF, self.n_channels)
            if out is None:
                out = xp.zeros(shape, cdt)
            else:
                self._check_out(out, shape, cdt).fill(0)
            waveform_into(out, 1.0)
        return fmt.wrap(out[0] if single else out, self.backend)
