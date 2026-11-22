"""
pygwtf: time-frequency waveforms and likelihoods for slowly-evolving GW sources.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pygwtf")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0+unknown"
