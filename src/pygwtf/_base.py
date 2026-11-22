"""
Shared machinery for the v2 time-frequency generators: configuration, grid,
detector geometry, noise preparation (including the single-precision
rescaling), precision handling and the dense inner-product helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._interop import ArrayFormat, to_backend
from .backend import Backend, get_backend
from .response.orbits import get_analytic_ltts, get_analytic_orbits

# Integer codes understood by the compiled core (see api.hpp).
PRECISION_CODE = {"single": 0, "double": 1}
RESPONSE_TT, RESPONSE_TDI2 = 0, 1
WINDOW_BOXCAR, WINDOW_TUKEY = 0, 1
NOISE_DIAGONAL, NOISE_FULL = 0, 1

REAL_DTYPE = {"single": np.float32, "double": np.float64}
COMPLEX_DTYPE = {"single": np.complex64, "double": np.complex128}


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

        data_shape = (nT, nF, self.n_channels)
        if channels is not None and tuple(channels.shape) != data_shape:
            raise ValueError(f"Channels array must have shape {data_shape}")
        noise = invC if self.full_invC else psds
        noise_shape = data_shape + (
            (self.n_channels,) if self.full_invC else ()
        )
        if noise is not None and tuple(noise.shape) != noise_shape:
            raise ValueError(
                f"{'invC' if self.full_invC else 'psds'} array must have shape {noise_shape}"
            )
        self.channels = channels
        self.psds = psds
        self.invC = invC
        self._noise_cache: dict[tuple, tuple] = {}

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

    def _prepare_noise(self, channels, noise, prefolded: bool, precision: str):
        """
        Convert data and noise to kernel inputs: (data, weights, amp_scale).

        weights = 4 dF / psd (diagonal) or 4 dF invC (full, unless prefolded).
        In single precision the data and weights are rescaled by s and 1/s^2
        with s = sqrt(median |w|), and the kernel amplitudes by s, which leaves
        <d|h> and <h|h> unchanged but keeps every float32 quantity O(1).
        """
        xp = self.backend.xp
        dF = self.config["dF"]
        data = to_backend(channels, self.backend, np.complex128)
        if self.full_invC:
            w = to_backend(noise, self.backend, np.complex128)
            if not prefolded:
                w = w * (4.0 * dF)
            diag = xp.abs(xp.diagonal(w, axis1=-2, axis2=-1))
        else:
            w = (4.0 * dF) / to_backend(noise, self.backend, np.float64)
            diag = xp.abs(w)
        scale = 1.0
        if precision == "single":
            good = diag[xp.isfinite(diag) & (diag > 0)]
            if good.size:
                scale = float(xp.sqrt(xp.median(good)))
            data = data * scale
            w = w / (scale * scale)
        cd, rd = COMPLEX_DTYPE[precision], REAL_DTYPE[precision]
        data = xp.ascontiguousarray(data, dtype=cd)
        w = xp.ascontiguousarray(w, dtype=cd if self.full_invC else rd)
        return data, w, scale

    def _noise_inputs(
        self, channels, psds, invC, invC_prefolded: bool, precision: str
    ):
        """Resolve call-time vs stored data/noise and prepare them (stored ones are cached)."""
        if self.full_invC:
            if psds is not None:
                raise ValueError(
                    "This generator uses the full inverse covariance (full_invC=True): pass invC, not psds."
                )
            noise, stored_noise, name = invC, self.invC, "invC"
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
            noise, stored_noise, name = psds, self.psds, "psds"
        use_stored = channels is None and noise is None
        if channels is None:
            channels = self.channels
        if noise is None:
            noise = stored_noise
        if channels is None:
            raise ValueError(
                "Channels must be supplied to compute the statistic."
            )
        if noise is None:
            raise ValueError(
                f"{name} must be supplied to compute the statistic."
            )

        # The kernels index bins up to nF without bounds checks: shapes must match.
        nT, nF, nc = self.config["nT"], self.config["nF"], self.n_channels
        data_shape = (nT, nF, nc)
        noise_shape = data_shape + ((nc,) if self.full_invC else ())
        for arr_name, arr, shape in (
            ("channels", channels, data_shape),
            (name, noise, noise_shape),
        ):
            if tuple(arr.shape) != shape:
                raise ValueError(
                    f"{arr_name} has shape {tuple(arr.shape)}, expected {shape} from config."
                )

        if use_stored:
            key = (precision, bool(invC_prefolded))
            if key not in self._noise_cache:
                self._noise_cache[key] = self._prepare_noise(
                    channels, noise, invC_prefolded, precision
                )
            return self._noise_cache[key]
        return self._prepare_noise(channels, noise, invC_prefolded, precision)

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

    # ------------------------------------------------------------------ public
    def inner_product(
        self,
        h,
        channels=None,
        psds=None,
        invC=None,
        invC_prefolded: bool = False,
        mixed_precision: bool | None = None,
    ):
        """
        Per-segment ``<d|h>`` and ``<h|h>`` of dense time-frequency waveforms.

        Parameters
        ----------
        h : array_like, shape (n_sources, nT, nF, n_channels) or (nT, nF, n_channels)
            Waveforms, e.g. the output of a waveform call.
        channels, psds, invC, invC_prefolded :
            Data and noise as for a statistic call (defaults: stored arrays).
        mixed_precision : bool, optional
            Override the generator precision for this call.

        Returns
        -------
        array, shape (n_sources, nT, 2) (or (nT, 2)) with ``[..., 0] = <d|h>``
        and ``[..., 1] = <h|h>``, in the format of ``h``.
        """
        precision = self._resolve_precision(mixed_precision)
        fmt = ArrayFormat.from_inputs(h, channels)
        data, w, scale = self._noise_inputs(
            channels, psds, invC, invC_prefolded, precision
        )
        hb = to_backend(h, self.backend, np.complex128)
        single = hb.ndim == 3
        if single:
            hb = hb[None]
        if scale != 1.0:
            hb = hb * scale
        hb = self.backend.xp.ascontiguousarray(
            hb, dtype=COMPLEX_DTYPE[precision]
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
        )
        return fmt.wrap(out[0] if single else out, self.backend)
