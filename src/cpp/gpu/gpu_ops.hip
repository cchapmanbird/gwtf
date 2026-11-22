// GPU backend (HIP; builds on NVIDIA through the shim in gpu_runtime.hpp).
// One thread per task (see ops.hpp); kernels are launched on the
// stream handed over from Python (CuPy's current stream), so they order
// correctly with surrounding CuPy work without extra synchronisation.
#include "gwtf/api.hpp"
#include "gwtf/gpu_runtime.hpp"
#include "gwtf/ops.hpp"

namespace gwtf {
namespace backend {

namespace {

template <class Body>
__global__ void task_kernel(Body body, int64_t n)
{
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        body(i);
}

struct GpuExec {
    hipStream_t stream;
    static constexpr int threads_per_block = 128;

    template <class Body>
    void launch(int64_t n, const Body& body) const
    {
        if (n <= 0)
            return;
        const int64_t blocks = (n + threads_per_block - 1) / threads_per_block;
        hipLaunchKernelGGL(task_kernel<Body>, dim3(unsigned(blocks)), dim3(threads_per_block), 0,
                           stream, body, n);
        gpu::check(hipGetLastError(), "kernel launch");
    }

    void zero(void* p, size_t bytes) const
    {
        gpu::check(hipMemsetAsync(p, 0, bytes, stream), "memset");
    }
};

GpuExec exec_for(void* stream) { return GpuExec{static_cast<hipStream_t>(stream)}; }

} // namespace

void run_prepare(const PrepareArgs& a) { auto ex = exec_for(a.stream); dispatch_prepare(ex, a); }
void run_waveform(const WaveformArgs& a) { auto ex = exec_for(a.stream); dispatch_waveform(ex, a); }
void run_statistic(const StatisticArgs& a) { auto ex = exec_for(a.stream); dispatch_statistic(ex, a); }
void run_inner_product(const InnerProductArgs& a) { auto ex = exec_for(a.stream); dispatch_inner_product(ex, a); }
void run_tracks(const TrackArgs& a) { auto ex = exec_for(a.stream); dispatch_tracks(ex, a); }
void synchronize(void* stream)
{
    gpu::check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "stream synchronize");
}

} // namespace backend
} // namespace gwtf
