#include <torch/extension.h>

torch::Tensor pairwise_distance_cuda(torch::Tensor positions);

torch::Tensor pairwise_distance(torch::Tensor positions) {
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(positions.scalar_type() == torch::kFloat32,
              "positions must have dtype torch.float32");
  TORCH_CHECK(positions.dim() == 3,
              "positions must have shape [batch, hits, 3]");
  TORCH_CHECK(positions.size(2) == 3, "last dimension must contain x, y, z");

  return pairwise_distance_cuda(positions.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pairwise_distance", &pairwise_distance,
        "Pairwise squared hit distance (CUDA)");
}
