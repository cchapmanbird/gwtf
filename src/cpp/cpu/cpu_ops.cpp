// Host backend: the shared pipeline (ops.hpp) executed by a plain serial
// loop. Deliberately single-threaded (user can parallelise in Python); 
// it doubles as the reference implementation for testing the GPU backend.
#include <cstring>

#include "gwtf/api.hpp"
#include "gwtf/ops.hpp"

namespace gwtf {
namespace backend {

namespace {
struct CpuExec {
    template <class Body>
    void launch(int64_t n, const Body& body) const
    {
        for (int64_t i = 0; i < n; ++i)
            body(i);
    }
    void zero(void* p, size_t bytes) const { std::memset(p, 0, bytes); }
};
} // namespace

void run_prepare(const PrepareArgs& a) { CpuExec ex; dispatch_prepare(ex, a); }
void run_waveform(const WaveformArgs& a) { CpuExec ex; dispatch_waveform(ex, a); }
void run_statistic(const StatisticArgs& a) { CpuExec ex; dispatch_statistic(ex, a); }
void run_inner_product(const InnerProductArgs& a) { CpuExec ex; dispatch_inner_product(ex, a); }
void run_semicoherent(const SemiCoherentArgs& a) { CpuExec ex; dispatch_semicoherent(ex, a); }
void run_tracks(const TrackArgs& a) { CpuExec ex; dispatch_tracks(ex, a); }
void synchronize(void*) {}

} // namespace backend
} // namespace gwtf
