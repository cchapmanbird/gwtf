"""
Shared machinery for the v2 time-frequency generators: configuration, grid,
detector geometry, data/noise preparation (including the single-precision
rescaling and global-fit per-walker data), precision handling and the dense
inner-product helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._interop import ArrayFormat, _on_gpu, to_backend
from .backend import Backend, get_backend
from .response.orbits import get_analytic_ltts, get_analytic_orbits

# Integer codes understood by the compiled core (see api.hpp).
PRECISION_CODE = {"single": 0, "double": 1}
RESPONSE_TT, RESPONSE_TDI2 = 0, 1
WINDOW_BOXCAR, WINDOW_TUKEY = 0, 1
NOISE_DIAGONAL, NOISE_FULL = 0, 1

REAL_DTYPE = {"single": np.float32, "double": np.float64}
COMPLEX_DTYPE = {"single": np.complex64, "double": np.complex128}

# How in-band tranches are split into semi-coherent segments (see api.hpp).
SEGMENT_SPLITS = ("spread", "truncate")

# Number of noise-weight entries sampled to choose the single-precision scale.
_SCALE_SAMPLES = 1 << 16


def _sample_rows(x, xp):
    """At most _SCALE_SAMPLES evenly spaced rows of ``x`` (along axis 0)."""
    n = x.shape[0]
    if n <= _SCALE_SAMPLES:
        return x
    return x[(xp.arange(_SCALE_SAMPLES, dtype=np.int64) * n) // _SCALE_SAMPLES]


def normalise_config(config: dict, use_tukey: bool) -> dict:
    """Validate and copy a grid configuration dictionary."""
    missing = [
        k for k in ("nT", "nF", "dT", "dF", "kernel_width") if k not in config
    ]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    cfg = dict(config)
    cfg["nT"] = int(cfg["nT"])
    cfg["nF"] = int(cfg["nF"])
    cfg["dT"] = float(cfg["dT"])
    cfg["dF"] = float(cfg["dF"])
    cfg["kernel_width"] = int(cfg["kernel_width"])
    # Central frequency of bin 0; defaults to dF (the DC bin is dropped).
    cfg["fmin"] = float(cfg.get("fmin", cfg["dF"]))
    cfg["dt"] = 1 / (cfg["nF"] * cfg["dF"])
    if use_tukey:
        if "alpha" not in cfg:
            raise ValueError(
                "Config must contain 'alpha' when use_tukey is True."
            )
        cfg["alpha"] = float(cfg["alpha"])
        if not 0 < cfg["alpha"] <= 1:
            raise ValueError(
                f"Tukey alpha must be in (0, 1], but got alpha={cfg['alpha']}."
            )
    return cfg


def parse_n_seg(N_seg) -> int:
    n = int(N_seg)
    if n != N_seg or n < 1:
        raise ValueError(f"N_seg must be a positive integer, got {N_seg!r}")
    return n


def parse_precision(precision: str) -> str:
    p = str(precision).lower()
    if p in ("single", "float32", "fp32", "mixed"):
        return "single"
    if p in ("double", "float64", "fp64"):
        return "double"
    raise ValueError(
        f"precision must be 'single' or 'double', got {precision!r}"
    )


class TimeFrequencyBase:
    """Common state for the analytic and multi-mode generators."""

    def __init__(
        self,
        config: dict,
        backend: str | Backend = "cpu",
        tdi_type: int | None = None,
        channels: Any = None,
        psds: Any = None,
        invC: Any = None,
        full_invC: bool | None = None,
        spacecraft_orbits: Any = None,
        spacecraft_ltts: Any = None,
        use_tukey: bool = False,
        precision: str = "single",
    ):
        self.backend = get_backend(backend)
        self.config = normalise_config(config, use_tukey)
        self.use_tukey = bool(use_tukey)
        self.precision = parse_precision(precision)
        xp = self.backend.xp
        nT, nF = self.config["nT"], self.config["nF"]

        # Segment start times and bin central frequencies.
        self.t_tranche = xp.arange(nT) * self.config["dT"]
        self.f_tranche = (
            self.config["fmin"] + xp.arange(nF) * self.config["dF"]
        )

        if tdi_type not in (None, 2):
            raise ValueError(
                "Only TDI-2 generation is currently supported. Set tdi_type to 2 or None."
            )
        self.tdi_type = tdi_type
        if tdi_type is None:
            self.n_channels = 2
            self._response_kind = RESPONSE_TT
            self.spacecraft_orbits = None
            self.spacecraft_ltts = None
        else:
            self.n_channels = 3
            self._response_kind = RESPONSE_TDI2
            if spacecraft_orbits is None:
                print(
                    "Spacecraft orbits not supplied. Falling back to analytic orbits"
                )
                spacecraft_orbits = get_analytic_orbits(
                    self.backend.asnumpy(self.t_tranche)
                )
            orbits = to_backend(spacecraft_orbits, self.backend, np.float64)
            if orbits.shape != (nT, 3, 3):
                raise ValueError(
                    f"Spacecraft orbits array must have shape {(nT, 3, 3)}"
                )
            if spacecraft_ltts is None:  # metres
                print(
                    "Spacecraft light travel times not supplied. Falling back to analytic calculation"
                )
                spacecraft_ltts = get_analytic_ltts(
                    self.backend.asnumpy(orbits)
                )
            ltts = to_backend(spacecraft_ltts, self.backend, np.float64)
            if ltts.shape != (nT, 3):
                raise ValueError(
                    f"Spacecraft light travel times array must have shape {(nT, 3)}"
                )
            self.spacecraft_orbits = orbits
            self.spacecraft_ltts = ltts
        self._detector_cache: dict[str, tuple] = {}

        # Noise model: diagonal PSDs (fast path) or full inverse covariance.
        if psds is not None and invC is not None:
            raise ValueError(
                "Pass either psds (diagonal noise) or invC (full inverse covariance), not both."
            )
        if full_invC is None:
            full_invC = invC is not None
        if full_invC and psds is not None:
            raise ValueError(
                "full_invC=True needs a full inverse covariance: pass invC, not psds."
            )
        if not full_invC and invC is not None:
            raise ValueError(
                "full_invC=False uses diagonal noise: pass psds, not invC."
            )
        self.full_invC = bool(full_invC)

        self._data_cache: dict = {}
        self._weight_cache: dict = {}
        self._channels = self._psds = self._invC = None
        self.channels = channels
        if self.full_invC:
            self.invC = invC
        else:
            self.psds = psds

    @property
    def channels(self):
        """Stored data, ``([n_walkers,] nT, nF, n_channels)``."""
        return self._channels

    @channels.setter
    def channels(self, value):
        if value is not None:
            self._walker_axis("channels", value, self._data_shape)
        self._channels = value
        self._data_cache.clear()

    @property
    def psds(self):
        """Stored diagonal noise PSDs, ``([n_walkers,] nT, nF, n_channels)``."""
        return self._psds

    @psds.setter
    def psds(self, value):
        if value is not None:
            if self.full_invC:
                raise ValueError(
                    "This generator uses the full inverse covariance (full_invC=True): set invC, not psds."
                )
            self._walker_axis("psds", value, self._noise_shape)
        self._psds = value
        self._weight_cache.clear()

    @property
    def invC(self):
        """Stored inverse covariance, ``([n_walkers,] nT, nF, n_channels, n_channels)``."""
        return self._invC

    @invC.setter
    def invC(self, value):
        if value is not None:
            if not self.full_invC:
                raise ValueError(
                    "This generator uses diagonal noise (full_invC=False): set psds, not invC."
                )
            self._walker_axis("invC", value, self._noise_shape)
        self._invC = value
        self._weight_cache.clear()

    @property
    def _data_shape(self):
        return (self.config["nT"], self.config["nF"], self.n_channels)

    @property
    def _noise_shape(self):
        return self._data_shape + ((self.n_channels,) if self.full_invC else ())

    @staticmethod
    def _walker_axis(name, arr, shape):
        """Number of per-walker copies in ``arr`` (None if shared); validates its shape."""
        got = tuple(arr.shape)
        if got == shape:
            return None
        if len(got) == len(shape) + 1 and got[1:] == shape and got[0] > 0:
            return got[0]
        raise ValueError(
            f"{name} has shape {got}, expected {shape} or (n_walkers, {', '.join(map(str, shape))}) from config."
        )

    # ------------------------------------------------------------------ helpers
    def _grid(self, use_midpoint: bool):
        g = self.backend.core.Grid()
        c = self.config
        g.nT, g.nF, g.kernel_width = c["nT"], c["nF"], c["kernel_width"]
        g.midpoint = int(bool(use_midpoint))
        g.dT, g.dF, g.fmin = c["dT"], c["dF"], c["fmin"]
        g.tukey_alpha = c.get("alpha", 1.0)
        return g

    def _resolve_precision(self, mixed_precision: bool | None) -> str:
        if mixed_precision is None:
            return self.precision
        return "single" if mixed_precision else "double"

    def _detector(self, precision: str):
        """(orbits, ltts) as Real arrays on the backend, or (None, None) for TT."""
        if self._response_kind == RESPONSE_TT:
            return None, None
        if precision not in self._detector_cache:
            dt = REAL_DTYPE[precision]
            self._detector_cache[precision] = (
                self.backend.xp.ascontiguousarray(
                    self.spacecraft_orbits, dtype=dt
                ),
                self.backend.xp.ascontiguousarray(
                    self.spacecraft_ltts, dtype=dt
                ),
            )
        return self._detector_cache[precision]

    def _prepare_data(self, channels, precision: str):
        """C-contiguous complex in the compute precision. Zero-copy if the array 
        already is (the single-precision rescaling is applied inside the kernels)."""
        return to_backend(channels, self.backend, COMPLEX_DTYPE[precision])

    def _prepare_weights(self, noise, prefolded: bool, precision: str):
        """
        Noise weights for the kernels and the single-precision scale.

        weights = 4 dF / psd (diagonal) or 4 dF invC (full; invC as-is if
        ``prefolded``), divided by s^2. In single precision s = sqrt(median |w|)
        (estimated from a sample of the diagonal) and the kernels multiply
        amplitudes and data by s, which leaves <d|h> and <h|h> unchanged but
        keeps every float32 quantity O(1). In double precision s = 1.
        """
        xp = self.backend.xp
        nc = self.n_channels
        fold = 1.0 if (self.full_invC and prefolded) else 4.0 * self.config["dF"]
        if self.full_invC:
            src = to_backend(noise, self.backend, np.complex128)
            rows = _sample_rows(src.reshape(-1, nc * nc), xp)
            sample = fold * xp.abs(rows[:, :: nc + 1])
        else:
            src = to_backend(noise, self.backend, np.float64)
            sample = fold / xp.abs(_sample_rows(src.reshape(-1), xp))
        scale = 1.0
        if precision == "single":
            good = sample[xp.isfinite(sample) & (sample > 0)]
            if good.size:
                scale = float(xp.sqrt(xp.median(good)))
        factor = fold / (scale * scale)
        if self.full_invC:
            dtype = COMPLEX_DTYPE[precision]
            if factor == 1.0 and src.dtype == dtype:
                return src, scale
            w = xp.empty(src.shape, dtype)
            xp.multiply(src, factor, out=w)
        else:
            w = xp.empty(src.shape, REAL_DTYPE[precision])
            xp.divide(factor, src, out=w)
        return w, scale

    def _data_index(self, data_indices, n_src: int, n_data):
        """
        Source -> walker map as an int32 backend array (None if not used).

        Host-side indices are range-checked here; device-side ones are not (to
        avoid a synchronisation) and sources with an out-of-range index get
        NaN statistics instead.
        """
        if data_indices is None:
            if n_data is not None:
                raise ValueError(
                    "The data or noise carry a leading per-walker axis (global-fit mode): pass "
                    "data_indices mapping each source to its walker."
                )
            return None
        if n_data is None:
            raise ValueError(
                "data_indices selects per-walker data, but neither channels nor the noise has a "
                "leading per-walker axis."
            )
        if _on_gpu(data_indices):
            idx = to_backend(data_indices, self.backend)
        else:
            idx = np.asarray(data_indices)
        if idx.dtype.kind not in "iu":
            raise TypeError(
                f"data_indices must be integers, got dtype {idx.dtype}."
            )
        idx = idx.reshape(-1)
        if idx.shape != (n_src,):
            raise ValueError(
                f"data_indices must have shape ({n_src},) (one walker per source), got {tuple(np.shape(data_indices))}."
            )
        if isinstance(idx, np.ndarray) and n_src and (
            idx.min() < 0 or idx.max() >= n_data
        ):
            raise ValueError(
                f"data_indices must lie in [0, {n_data - 1}]: there are only {n_data} walkers."
            )
        return to_backend(idx, self.backend, np.int32)

    def _noise_inputs(
        self,
        channels,
        psds,
        invC,
        invC_prefolded: bool,
        precision: str,
        data_indices=None,
        n_src: int = 0,
    ):
        """
        Resolve call-time vs stored data/noise and prepare them for the kernels.

        Returns ``(data, weights, scale, data_index)``. 
        Kernel copies of the stored arrays are cached.
        """
        if self.full_invC:
            if psds is not None:
                raise ValueError(
                    "This generator uses the full inverse covariance (full_invC=True): pass invC, not psds."
                )
            noise, name = invC, "invC"
        else:
            if invC is not None:
                raise ValueError(
                    "This generator uses diagonal noise (full_invC=False): pass psds, not invC. "
                    "Construct it with invC=... or full_invC=True for the full inverse covariance."
                )
            if invC_prefolded:
                raise ValueError(
                    "invC_prefolded only applies to full-invC generators."
                )
            noise, name = psds, "psds"
        stored_data, stored_noise = channels is None, noise is None
        if stored_data:
            channels = self.channels
        if stored_noise:
            noise = self.invC if self.full_invC else self.psds
        if channels is None:
            raise ValueError(
                "Channels must be supplied to compute the statistic."
            )
        if noise is None:
            raise ValueError(
                f"{name} must be supplied to compute the statistic."
            )

        # The kernels index bins up to nF without bounds checks: shapes must match.
        nd = self._walker_axis("channels", channels, self._data_shape)
        nw = self._walker_axis(name, noise, self._noise_shape)
        if nd is not None and nw is not None and nd != nw:
            raise ValueError(
                f"channels has {nd} walkers but {name} has {nw}."
            )
        index = self._data_index(data_indices, n_src, nd or nw)

        if stored_data:
            if precision not in self._data_cache:
                self._data_cache[precision] = self._prepare_data(
                    channels, precision
                )
            data = self._data_cache[precision]
        else:
            data = self._prepare_data(channels, precision)
        if stored_noise:
            key = (precision, bool(invC_prefolded))
            if key not in self._weight_cache:
                self._weight_cache[key] = self._prepare_weights(
                    noise, invC_prefolded, precision
                )
            w, scale = self._weight_cache[key]
        else:
            w, scale = self._prepare_weights(noise, invC_prefolded, precision)
        return data, w, scale, index

    def _check_out(self, out, shape, dtype):
        xp = self.backend.xp
        if not isinstance(out, xp.ndarray):
            raise TypeError(
                f"`out` must be a {xp.__name__} array for the {self.backend.name} backend "
                "(results are written in place)."
            )
        if (
            tuple(out.shape) != tuple(shape)
            or out.dtype != dtype
            or not out.flags.c_contiguous
        ):
            raise ValueError(
                f"`out` must be a C-contiguous {np.dtype(dtype)} array of shape {tuple(shape)}"
            )
        return out

    def _semicoherent(
        self,
        stat,
        seg_start,
        seg_end,
        N_seg,
        segment_split: str,
        precision: str,
        out=None,
    ):
        """Upsilon (float64, (n_sources,)) from a backend per-tranche statistic."""
        if segment_split not in SEGMENT_SPLITS:
            raise ValueError(
                f"segment_split must be one of {SEGMENT_SPLITS}, got {segment_split!r}"
            )
        n = stat.shape[0]
        out = (
            self.backend.xp.empty(n, np.float64)
            if out is None
            else self._check_out(out, (n,), np.float64)
        )
        core = self.backend.core
        core.semicoherent(
            stat,
            seg_start,
            seg_end,
            parse_n_seg(N_seg),
            PRECISION_CODE[precision],
            core.segment_splits[segment_split],
            out,
            self.backend.stream_ptr(),
        )
        return out

    # ------------------------------------------------------------------ public
    def semi_coherent_statistic(
        self,
        statistic,
        N_seg: int,
        segment_start,
        segment_end,
        segment_split: str = "spread",
    ):
        """
        Semi-coherent statistic from per-tranche inner products.

        ``Upsilon = sum_n |<d|h>_n|^2 / <h|h>_n``, where ``<.|.>_n`` sums the
        tranches of segment ``n`` of the source's in-band range
        ``[segment_start, segment_end]`` (arXiv:2510.19047, Eq. 8). ``N_seg=1``
        returns the coherent ``|<d|h>|^2 / <h|h>``.

        Parameters
        ----------
        statistic : array_like, shape (n_sources, nT, 2) or (nT, 2)
            Per-tranche ``<d|h>``, ``<h|h>``.
        N_seg : int
            Number of semi-coherent segments.
        segment_start, segment_end : array_like of int, shape (n_sources,)
            In-band tranche range per source (inclusive; ``end < start`` for
            none), e.g. from ``segment_indices``.
        segment_split : {'spread', 'truncate'}
            How the ``L`` in-band tranches are split. ``'spread'`` (default):
            segment ``n`` covers ``[n L // N_seg, (n + 1) L // N_seg)``, so all
            tranches are used. ``'truncate'``: the v1 convention, ``L // N_seg``
            tranches each and the last ``L % N_seg`` dropped. Segments with
            ``<h|h>_n = 0`` do not contribute in either case.

        Returns
        -------
        float64 array of shape (n_sources,) (or a scalar array), in the format
        of ``statistic``.
        """
        fmt = ArrayFormat.from_inputs(statistic)
        stat = to_backend(statistic, self.backend)
        if stat.dtype not in (np.complex64, np.complex128):
            stat = stat.astype(np.complex128)
        precision = "single" if stat.dtype == np.complex64 else "double"
        single = stat.ndim == 2
        if single:
            stat = stat[None]
        nT = self.config["nT"]
        if stat.ndim != 3 or stat.shape[1:] != (nT, 2):
            raise ValueError(
                f"statistic must have shape (n_sources, {nT}, 2), got {tuple(np.shape(statistic))}"
            )
        n = stat.shape[0]

        # Clipped to the grid, as the kernels read the statistic unchecked.
        seg = []
        for name, arr, lo in (
            ("segment_start", segment_start, 0),
            ("segment_end", segment_end, -1),
        ):
            a = to_backend(arr, self.backend, np.int32).reshape(-1)
            if a.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
            seg.append(self.backend.xp.clip(a, lo, nT - 1).astype(np.int32))
        out = self._semicoherent(
            stat, seg[0], seg[1], N_seg, segment_split, precision
        )
        return fmt.wrap(out[0] if single else out, self.backend)

    def inner_product(
        self,
        h,
        channels=None,
        psds=None,
        invC=None,
        invC_prefolded: bool = False,
        mixed_precision: bool | None = None,
        data_indices=None,
    ):
        """
        Per-segment ``<d|h>`` and ``<h|h>`` of dense time-frequency waveforms.

        Parameters
        ----------
        h : array_like, shape (n_sources, nT, nF, n_channels) or (nT, nF, n_channels)
            Waveforms, e.g. the output of a waveform call.
        channels, psds, invC, invC_prefolded, data_indices :
            Data and noise as for a statistic call (defaults: stored arrays),
            including per-walker (global-fit) data selected by ``data_indices``.
        mixed_precision : bool, optional
            Override the generator precision for this call.

        Returns
        -------
        array, shape (n_sources, nT, 2) (or (nT, 2)) with ``[..., 0] = <d|h>``
        and ``[..., 1] = <h|h>``, in the format of ``h``.
        """
        precision = self._resolve_precision(mixed_precision)
        fmt = ArrayFormat.from_inputs(h, channels)
        hb = to_backend(h, self.backend, COMPLEX_DTYPE[precision])
        single = hb.ndim == 3
        if single:
            hb = hb[None]
        data, w, scale, index = self._noise_inputs(
            channels,
            psds,
            invC,
            invC_prefolded,
            precision,
            data_indices,
            hb.shape[0],
        )
        out = self.backend.xp.empty(
            (hb.shape[0], hb.shape[1], 2), COMPLEX_DTYPE[precision]
        )
        self.backend.core.inner_product(
            hb,
            data,
            w,
            PRECISION_CODE[precision],
            NOISE_FULL if self.full_invC else NOISE_DIAGONAL,
            out,
            self.backend.stream_ptr(),
            data_index=index,
            scale=scale,
        )
        return fmt.wrap(out[0] if single else out, self.backend)
