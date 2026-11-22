// Physical constants. Values are those of lisaconstants (see
// src/pygwtf/constants.py); tests/test_core.py asserts they agree.
#pragma once

namespace gwtf {
namespace constants {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double two_pi = 2.0 * pi;
constexpr double clight = 299792458.0;                 // m / s
constexpr double gamma_e = 0.5772156649015329;         // Euler-Mascheroni
constexpr double MTsun = 4.9254909476412675e-06;      // GM_sun / c^3, s
constexpr double pc = 3.085677581491367e+16;           // m

} // namespace constants
} // namespace gwtf
