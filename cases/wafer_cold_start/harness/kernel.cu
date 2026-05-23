#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void {fn}_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 1.0001f + {salt}f;
}}

torch::Tensor {fn}(torch::Tensor x) {{
    auto y = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    {fn}_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{fn}", &{fn}, "tiny launch-overhead probe ({fn})");
}}
