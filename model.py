"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    // TODO: implement warp-level sum reduction using shuffle intrinsics
    constexpr unsigned FULL_MASK = 0xffffffffu;

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }

    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    // TODO: implement warp-level max reduction using shuffle intrinsics
    constexpr unsigned FULL_MASK = 0xffffffffu;

    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(FULL_MASK, val, offset);
        val = val > other ? val : other;
    }

    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    __shared__ float tail[32];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int full_warps = blockDim.x / 32;
    const int tail_size = blockDim.x % 32;

    if (tail_size > 0) {
        if (warp_id == full_warps)
            tail[lane] = val;

        __syncthreads();

        if (full_warps == 0) {
            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride && lane + stride < tail_size)
                    tail[lane] += tail[lane + stride];

                __syncthreads();
            }
            return tid == 0 ? tail[0] : 0.0f;
        }

        if (warp_id == 0 && lane < tail_size)
            val += tail[lane];
    }

    if (warp_id < full_warps) {
        float warp_max = warp_reduce_sum(val);
        if (lane == 0)
            shared[warp_id] = warp_max;
    }

    __syncthreads();

    if (warp_id == 0) {
        float partial = lane < full_warps ? shared[lane] : 0.0f;
        return warp_reduce_sum(partial);
    }

    return 0.0f;
}

# Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    __shared__ float tail[32];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int full_warps = blockDim.x / 32;
    const int tail_size = blockDim.x % 32;

    if (tail_size > 0) {
        if (warp_id == full_warps)
            tail[lane] = val;

        __syncthreads();

        // For blocks smaller than a warp, reduce the tail in parallel.
        if (full_warps == 0) {
            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride && lane + stride < tail_size)
                    tail[lane] = fmaxf(tail[lane], tail[lane + stride]);

                __syncthreads();
            }
            return tid == 0 ? tail[0] : -INFINITY;
        }

        if (warp_id == 0 && lane < tail_size)
            val = fmaxf(val, tail[lane]);
    }

    if (warp_id < full_warps) {
        float warp_max = warp_reduce_max(val);
        if (lane == 0)
            shared[warp_id] = warp_max;
    }

    __syncthreads();

    if (warp_id == 0) {
        float partial = lane < full_warps ? shared[lane] : -INFINITY;
        return warp_reduce_max(partial);
    }

    return -INFINITY;
}

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
  // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]
  (void)x; (void)residual; (void)out; (void)n;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < n) {
    out[idx] = x[idx] + residual[idx];
  }
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // TODO: Apply GELU (tanh approximation) elementwise to x, write into out
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        out[idx] = 0.5f * x[idx] * (1 + tanhf(sqrtf(2 / 3.14159) * (x[idx] + 0.044715f * x[idx]*x[idx]*x[idx])));
    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    // TODO: apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        out[idx] = x[idx] / (1 + expf(-x[idx]));
    }
}

# Step 8 - swiglu_kernel
__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    // TODO: out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        out[idx] = silu(gate[idx]) * up[idx];
    }
}

# Step 9 - rmsnorm_kernel
__global__ void rmsnorm_kernel(
    const float* x, const float* weight, float* out, int n, float eps)
{
    __shared__ float scratch[32];
    __shared__ float inv_rms;

    const size_t row_offset = static_cast<size_t>(blockIdx.x) * n;
    const float* x_row = x + row_offset;
    float* out_row = out + row_offset;

    // Each thread sums the squares of its assigned columns.
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float value = x_row[i];
        sum_sq += value * value;
    }

    // Every thread contributes. Thread 0 receives the row total.
    float row_sum_sq = block_reduce_sum(sum_sq, scratch);

    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(row_sum_sq / n + eps);
    }
    __syncthreads();

    // Each thread revisits its columns and writes the result.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

# Step 10 - layernorm_kernel (not yet solved)
# TODO: implement

# Step 11 - fused_add_rmsnorm_kernel (not yet solved)
# TODO: implement

# Step 12 - softmax_row_kernel (not yet solved)
# TODO: implement

# Step 13 - causal_softmax_kernel (not yet solved)
# TODO: implement

# Step 14 - embedding_lookup_kernel (not yet solved)
# TODO: implement

# Step 15 - rope_kernel (not yet solved)
# TODO: implement

# Step 16 - linear_kernel
__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // TODO: compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int row = idx / N;
    int col = idx % N;

    if (row < M && col < N) {
        float acc = 0.f;

        for (int k=0; k < K; k++) {
            acc += x[row * K + k] * weight[col * K + k];
        }
        out[row * N + col] = acc;
        if (bias != nullptr) {
            out[row * N + col] += bias[col];
        }
    }
}

# Step 17 - fused_linear_bias_gelu_kernel
__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
    // TODO: fuse matmul, bias add, and GELU tanh approx into one kernel
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int row = idx / N, col = idx % N;

    if (row < M && col < N) {
        float acc = 0.0f;

        for (int k=0; k < K; k++) {
            acc += x[row * K + k] * weight[col * K + k];
        }
        if (bias != nullptr) {
            acc += bias[col];
        }
        out[row * N + col] = 0.5f * acc * (1 + tanhf(sqrtf(2.0f / (float)M_PI) * (acc + 0.044715f*acc*acc*acc)));
    }
}

# Step 18 - mlp_swiglu_forward
void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up,
                        const float* w_down, float* out,
                        int M, int hidden_dim, int intermediate_dim) {
    // TODO: allocate temps, run gate/up linears, swiglu, then down projection
    float *d_gate = nullptr, *d_up = nullptr, *d_act = nullptr;
    float* bias = nullptr;
    cudaMalloc(&d_gate, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_up, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_act, M * intermediate_dim * sizeof(float));
    
    int threads = 256;
    int blocks = (M * intermediate_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(x, w_gate, bias, d_gate, M, intermediate_dim, hidden_dim);

    linear_kernel<<<blocks, threads>>>(x, w_up, bias, d_up, M, intermediate_dim, hidden_dim);

    swiglu_kernel<<<blocks, threads>>>(d_gate, d_up, d_act, M * intermediate_dim);

    blocks = (M * hidden_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(d_act, w_down, bias, out, M, hidden_dim, intermediate_dim);

    cudaDeviceSynchronize();
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_act);
}

# Step 19 - rmsnorm_residual_block (not yet solved)
# TODO: implement

# Step 20 - run_transformer_ffn (not yet solved)
# TODO: implement

