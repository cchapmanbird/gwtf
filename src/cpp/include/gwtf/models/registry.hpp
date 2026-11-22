// Registry of compiled analytic models. Each entry is X(id, Type); ids are
// stable integers used by the Python layer (AnalyticModel.kernel_name is
// resolved to an id through pygwtf._core_*.models()).
//
// To add a model: include its header here and append X(<next id>, Type).
#pragma once

#include "gwtf/models/taylort2ecc.hpp"
#include "gwtf/models/taylort3spin.hpp"

#define GWTF_ANALYTIC_MODELS(X)            \
    X(0, ::gwtf::models::TaylorT2Ecc)      \
    X(1, ::gwtf::models::TaylorT3Spin)

namespace gwtf {
// Model id reserved for the numerical (precomputed track) multi-mode source.
constexpr int32_t numerical_model_id = -1;
} // namespace gwtf
