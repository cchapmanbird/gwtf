"""
Analytic time-frequency waveform generator (v2 compiled backend).
"""

from __future__ import annotations

import warnings
from typing import Any, Type

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
from .models.base import AnalyticModel

__all__ = ["AnalyticTimeFrequencyWaveform"]


class AnalyticTimeFrequencyWaveform(TimeFrequencyBase):
    """
    Analytic time-frequency waveform generator.

    Generates Fresnel time-frequency waveforms, or per-segment inner products
    against data, for an analytic model compiled into the backend.

    Parameters
    ----------
    model_class : type[AnalyticModel]
        Model class (e.g. ``TaylorT2Ecc``); instantiated with the backend.
    config : dict
        Grid configuration: ``nT``, ``nF``, ``dT``, ``dF``, ``kernel_width``,
        optionally ``fmin`` (central frequency of bin 0, default ``dF``) and
        ``alpha`` (Tukey taper fraction, required if ``use_tukey``).
    backend : {'cpu', 'gpu'} or Backend
        ``'cpu'`` runs the serial host build, ``'gpu'`` the HIP/CUDA build.
    tdi_type : {None, 2}
        ``None`` produces TT polarisations (h+, hx); ``2`` second-generation
        TDI (A, E, T).
    channels, psds, invC :
        Data ``(nT, nF, n_channels)`` and either diagonal PSDs (same shape) or
        a full inverse covariance ``(nT, nF, n_channels, n_channels)``, used
        when not supplied at call time. Each may instead carry a leading
        per-walker axis (global-fit mode, see below). Reassigning the
        attributes (e.g. ``gen.psds = new``) refreshes the prepared copies.
    spacecraft_orbits, spacecraft_ltts :
        ``(nT, 3, 3)`` positions [m] and ``(nT, 3)`` arm lengths [m] for links
        (12, 23, 31). Analytic orbits are used if omitted (TDI only).
    use_tukey : bool
        Tukey-windowed Fresnel kernel (midpoint only) instead of box-car.
    full_invC : bool, optional
        Noise model; inferred from whether ``invC`` is given.
    precision : {'single', 'double'}
        Default precision. ``'single'`` (default, fast path) evaluates the
        model in double, then builds the time-frequency kernel, response and
        inner products in float32; ``'double'`` keeps float64 throughout.
        Overridable per call with ``mixed_precision``.
    block_vectorised_gpu : bool
        Ignored (v1 launch option; v2 always parallelises over source x segment).
    gf_mode : bool
        Require ``data_indices`` on every statistic call (global-fit mode).
        Per-walker data work without it: see below.

    Global-fit mode
    ---------------
    A global fit scores ``n_sources = n_walkers x n_temps`` proposals against
    per-walker residuals and/or PSDs. Pass ``channels`` and/or the noise with
    a leading ``n_walkers`` axis -- either may also be shared, e.g. per-walker
    residuals with one common PSD -- and map each source to its walker with
    ``data_indices`` (shape ``(n_sources,)``; walker-fastest layout gives
    ``np.tile(np.arange(n_walkers), n_temps)``). The data are read in place
    when already C-contiguous in the compute precision (complex64 for
    ``precision='single'``), so per-walker residuals cost no copy per call;
    noise weights are prepared once per stored array.
    """

    def __init__(
        self,
        model_class: Type[AnalyticModel],
        config: dict,
        backend: str | Backend = "cpu",
        tdi_type: int | None = None,
        channels: Any = None,
        psds: Any = None,
        spacecraft_orbits: Any = None,
        spacecraft_ltts: Any = None,
        block_vectorised_gpu: bool = False,
        gf_mode: bool = False,
        use_tukey: bool = False,
        invC: Any = None,
        full_invC: bool | None = None,
        precision: str = "single",
    ):
        if block_vectorised_gpu:
            warnings.warn(
                "block_vectorised_gpu has no effect in the v2 backend (kernels always "
                "parallelise over source x time segment).",
                stacklevel=2,
            )
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
        self.model = model_class(backend=self.backend)
        self.gf_mode = bool(gf_mode)
        self.block_vectorised_gpu = False

    # ------------------------------------------------------------------ inputs
    def _raw_parameters(self, parameters) -> tuple[Any, bool]:
        """User parameters -> (n_sources, n_params) float64 backend array."""
        names = self.model.parameters
        if isinstance(parameters, dict):
            missing = [n for n in names if n not in parameters]
            if missing:
                raise KeyError(
                    f"Missing parameters for {type(self.model).__name__}: {missing}"
                )
            first = parameters[names[0]]
            single = np.ndim(first) == 0
            cols = [
                to_backend(parameters[n], self.backend, np.float64).reshape(-1)
                for n in names
            ]
            raw = self.backend.xp.stack(cols, axis=1)
        else:
            raw = to_backend(parameters, self.backend, np.float64)
            single = raw.ndim == 1
            if single:
                raw = raw[None]
        if raw.ndim != 2 or raw.shape[1] != len(names):
            raise ValueError(
                f"parameters must have shape (n_sources, {len(names)}) in order {names}"
            )
        return self.backend.xp.ascontiguousarray(raw), single

    def _prepare(self, raw, grid):
        """Run the model's unit conversion, derived parameters and time bounds."""
        xp = self.backend.xp
        n = raw.shape[0]
        stride = self.model.num_parameters + self.model.num_derived_parameters
        params = xp.empty((n, stride), np.float64)
        seg_start = xp.empty(n, np.int32)
        seg_end = xp.empty(n, np.int32)
        c = self.config
        # fmin is the centre of bin 0; the last bin's upper edge is fmin + (nF - 1/2) dF.
        f_lo, f_hi = c["fmin"], c["fmin"] + (c["nF"] - 0.5) * c["dF"]
        self.backend.core.prepare(
            self.model.kernel_id,
            raw,
            params,
            seg_start,
            seg_end,
            grid,
            f_lo,
            f_hi,
            self.backend.stream_ptr(),
        )
        return params, seg_start, seg_end

    def _response_parameters(self, parameters_response, raw, n_sources):
        if parameters_response is None:
            if self.tdi_type is not None:
                raise ValueError(
                    "parameters_response must be supplied for TDI generation."
                )
            # TT: only the inclination is used; take it from the model's 'cosi'.
            resp = self.backend.xp.zeros((n_sources, 4), np.float64)
            if "cosi" in self.model.parameters:
                resp[:, 0] = raw[:, self.model.parameters.index("cosi")]
            return resp
        resp = to_backend(parameters_response, self.backend, np.float64)
        if resp.ndim == 1:
            resp = resp[None]
        if resp.shape != (n_sources, 4):
            raise ValueError(
                f"parameters_response must have shape {(n_sources, 4)}"
            )
        return resp

    # ------------------------------------------------------------------ call
    def __call__(
        self,
        parameters: dict | Any,
        channels: Any = None,
        psds: Any = None,
        parameters_response: Any = None,
        out: Any = None,
        compute_statistic: bool = False,
        search_statistic: Any = None,
        N_seg: int | None = None,
        mixed_precision: bool | None = None,
        use_midpoint: bool = True,
        data_indices: Any = None,
        invC: Any = None,
        invC_prefolded: bool = False,
        segment_split: str = "spread",
    ):
        """
        Evaluate waveforms, per-segment inner products or the semi-coherent
        statistic.

        Parameters
        ----------
        parameters : dict or array_like, shape (n_sources, n_params)
            Source parameters as a dict keyed by ``model.parameters`` or an
            array in that order. Scalars / 1-D give a single source (leading
            axis dropped from the output).
        channels, psds, invC :
            Data and noise for ``compute_statistic`` (defaults: stored arrays),
            each shared or with a leading per-walker axis.
        parameters_response : array_like, shape (n_sources, 4), optional
            ``[cosi, pol, ecliptic_long, ecliptic_lat]``; required for TDI. The
            response ``cosi`` sets the mode's spherical harmonics. For TT output
            without it, the model's ``cosi`` parameter is used.
        out : backend array, optional
            Pre-allocated output (waveform: ``(n_sources, nT, nF, n_channels)``,
            statistic: ``(n_sources, nT, 2)``), complex64/complex128 according
            to precision. Overwritten.
        compute_statistic : bool
            Return per-segment ``<d|h>`` (``[..., 0]``) and ``<h|h>``
            (``[..., 1]``) instead of the waveform.
        mixed_precision : bool, optional
            ``True`` forces single precision, ``False`` double; default is the
            generator's ``precision``.
        use_midpoint : bool
            Evaluate mode tracks at segment midpoints (default, recommended).
        invC_prefolded : bool
            Full-invC only: ``invC`` already includes the ``4 dF`` factor.
        data_indices : array_like of int, shape (n_sources,), optional
            Global-fit mode: the walker (index along the leading axis of the
            per-walker data/noise) each source is scored against. Required
            when the data or noise is per-walker. Host arrays are
            range-checked; device arrays are not, and an out-of-range index
            gives NaN statistics for that source.
        N_seg : int, optional
            With ``compute_statistic``, return the semi-coherent statistic
            ``Upsilon = sum_n |<d|h>_n|^2 / <h|h>_n`` over ``N_seg`` segments
            of each source's in-band tranches instead of the per-tranche
            statistic (see :meth:`semi_coherent_statistic`). ``out`` is then
            used as the per-tranche scratch buffer, if given.
        search_statistic : backend array, optional
            Pre-allocated float64 ``(n_sources,)`` output for ``Upsilon``.
        segment_split : {'spread', 'truncate'}
            How in-band tranches are split into segments: ``'spread'``
            (default) uses every tranche, ``'truncate'`` reproduces v1 (drops
            the remainder). See :meth:`semi_coherent_statistic`.

        Returns
        -------
        Array in the format of the inputs (NumPy, CuPy, ...): the waveform
        ``(n_sources, nT, nF, n_channels)``, the statistic
        ``(n_sources, nT, 2)`` or, with ``N_seg``, ``Upsilon`` (float64,
        ``(n_sources,)``).
        """
        if N_seg is not None:
            N_seg = parse_n_seg(N_seg)
            if not compute_statistic:
                raise ValueError(
                    "N_seg (semi-coherent statistic) requires compute_statistic=True."
                )
        if data_indices is not None and not compute_statistic:
            raise ValueError(
                "data_indices only applies to statistic calls (compute_statistic=True); "
                "waveforms do not depend on the data."
            )
        if self.gf_mode and compute_statistic and data_indices is None:
            raise ValueError(
                "In global-fit mode (gf_mode=True), data_indices must be supplied to map sources to walkers."
            )
        if self.use_tukey and not use_midpoint:
            raise ValueError(
                "The Tukey Fresnel kernel is midpoint only, use_midpoint must be True when use_tukey is True."
            )

        precision = self._resolve_precision(mixed_precision)
        fmt = ArrayFormat.from_inputs(
            parameters, channels, psds, invC, parameters_response
        )
        xp = self.backend.xp
        core = self.backend.core
        stream = self.backend.stream_ptr()
        grid = self._grid(use_midpoint)

        raw, single = self._raw_parameters(parameters)
        n = raw.shape[0]
        params, seg_start, seg_end = self._prepare(raw, grid)
        response = self._response_parameters(parameters_response, raw, n)
        orbits, ltts = self._detector(precision)
        window = WINDOW_TUKEY if self.use_tukey else WINDOW_BOXCAR
        cdt = COMPLEX_DTYPE[precision]
        nT, nF = self.config["nT"], self.config["nF"]

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
            core.statistic(
                grid,
                self.model.kernel_id,
                n,
                params,
                None,
                None,
                seg_start,
                seg_end,
                response,
                self._response_kind,
                orbits,
                ltts,
                window,
                PRECISION_CODE[precision],
                NOISE_FULL if self.full_invC else NOISE_DIAGONAL,
                data,
                w,
                out,
                scale,
                stream,
                data_index=index,
            )
            if N_seg is not None:
                out = self._semicoherent(
                    out,
                    seg_start,
                    seg_end,
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
            core.waveform(
                grid,
                self.model.kernel_id,
                n,
                params,
                None,
                None,
                seg_start,
                seg_end,
                response,
                self._response_kind,
                orbits,
                ltts,
                window,
                PRECISION_CODE[precision],
                out,
                1.0,
                stream,
            )
        return fmt.wrap(out[0] if single else out, self.backend)

    # ------------------------------------------------------------------ extras
    def tracks(self, parameters: dict | Any, use_midpoint: bool = True):
        """
        Mode tracks of the analytic model on the segment grid.

        Returns an array of shape ``(n_sources, n_modes, nT, 4)`` holding
        ``(amp, phi, f, fdot)`` at each segment's evaluation time -- the input
        format of :class:`~pygwtf.multimode.MultipleHarmonicTimeFrequencyWaveform`.
        """
        fmt = ArrayFormat.from_inputs(parameters)
        grid = self._grid(use_midpoint)
        raw, single = self._raw_parameters(parameters)
        params, _, _ = self._prepare(raw, grid)
        out = self.backend.xp.empty(
            (raw.shape[0], self.model.num_modes, self.config["nT"], 4),
            np.float64,
        )
        self.backend.core.tracks(
            self.model.kernel_id, params, grid, out, self.backend.stream_ptr()
        )
        return fmt.wrap(out[0] if single else out, self.backend)

    def segment_indices(
        self, parameters: dict | Any, use_midpoint: bool = True
    ):
        """First and last in-band segment index per source (as used by the kernels)."""
        fmt = ArrayFormat.from_inputs(parameters)
        raw, single = self._raw_parameters(parameters)
        _, s0, s1 = self._prepare(raw, self._grid(use_midpoint))
        if single:
            s0, s1 = s0[0], s1[0]
        return fmt.wrap(s0, self.backend), fmt.wrap(s1, self.backend)
