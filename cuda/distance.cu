#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void pairwise_distance_kernel(
    const float* __restrict__ positions,
    float* __restrict__ distances,
    int batch_size,
    int num_hits)
{
    const long long total =
        static_cast<long long>(batch_size) *
        num_hits *
        num_hits;

    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (idx >= total)
        return;

    const int j = idx % num_hits;
    const long long temp = idx / num_hits;
    const int i = temp % num_hits;
    const int b = temp / num_hits;

    const long long i_offset =
        (static_cast<long long>(b) * num_hits + i) * 3;

    const long long j_offset =
        (static_cast<long long>(b) * num_hits + j) * 3;

    const float dx =
        positions[i_offset + 0] - positions[j_offset + 0];
    const float dy =
        positions[i_offset + 1] - positions[j_offset + 1];
    const float dz =
        positions[i_offset + 2] - positions[j_offset + 2];

    distances[idx] = dx * dx + dy * dy + dz * dz;
}

} // namespace

torch::Tensor pairwise_distance_cuda(torch::Tensor positions)
{
    const int batch_size = static_cast<int>(positions.size(0));
    const int num_hits = static_cast<int>(positions.size(1));

    auto distances = torch::empty(
        {batch_size, num_hits, num_hits},
        positions.options()
    );

    const long long total =
        static_cast<long long>(batch_size) *
        num_hits *
        num_hits;

    constexpr int threads = 256;

    const int blocks =
        static_cast<int>((total + threads - 1) / threads);

    pairwise_distance_kernel<<<blocks, threads>>>(
        positions.data_ptr<float>(),
        distances.data_ptr<float>(),
        batch_size,
        num_hits
    );

    return distances;
}
