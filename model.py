"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = expf(a[i]);
    }
}

# Step 4 - row_max
#include <cmath>

__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float m = -INFINITY;
        for (int c = 0; c < cols; c++) {
            float v = matrix[r * cols + c];
            if (v > m) m = v;
        }
        out[r] = m;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    __shared__ float sdata[1024];

    int r = blockIdx.x;
    int tid = threadIdx.x;
    if (r >= rows) return;

    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        sum += matrix[(size_t)r * cols + c];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[r] = sdata[0];
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += a[row * k + l] * b[l * n + col];
        }
        c[row * n + col] = sum;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        float dot = dot_product(q + i * head_dim, k + j * head_dim, head_dim);
        scores[i * seq_len + j] = dot * rsqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
#include <cmath>

__global__ void softmax_rows(float* matrix, int rows, int cols) {
    __shared__ float sdata[1024];

    int r = blockIdx.x;
    int tid = threadIdx.x;
    if (r >= rows) return;

    float* row = matrix + (size_t)r * cols;

    // Pass 1: row max
    float m = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
        m = fmaxf(m, row[c]);
    }
    sdata[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Pass 2: exp(x - max), store it back, and sum
    float s = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float e = expf(row[c] - row_max);
        row[c] = e;
        s += e;
    }
    sdata[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float total = sdata[0];

    // Pass 3: normalize
    for (int c = tid; c < cols; c += blockDim.x) {
        row[c] /= total;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += p[i * seq_len + j] * v[j * head_dim + d];
        }
        out[i * head_dim + d] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    float* d_scores = nullptr;
    cudaMalloc(&d_scores, (size_t)seq_len * seq_len * sizeof(float));

    // Stage 1: S = Q K^T / sqrt(d)
    dim3 block2d(16, 16);
    dim3 grid_qk((seq_len + block2d.x - 1) / block2d.x,
                 (seq_len + block2d.y - 1) / block2d.y);
    qk_scores<<<grid_qk, block2d>>>(d_q, d_k, d_scores, seq_len, head_dim);

    // Stage 2: row-wise softmax in place, one block per row
    int sm_block = 32;
    while (sm_block < seq_len && sm_block < 1024) sm_block <<= 1;
    softmax_rows<<<seq_len, sm_block>>>(d_scores, seq_len, seq_len);

    // Stage 3: O = P V
    dim3 grid_pv((head_dim + block2d.x - 1) / block2d.x,
                 (seq_len + block2d.y - 1) / block2d.y);
    pv_matmul<<<grid_pv, block2d>>>(d_scores, d_v, d_out, seq_len, head_dim);

    cudaDeviceSynchronize();
    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return correction * old_sum + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int d = 0; d < head_dim; d++) {
        out_row[d] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    int total = tile_rows * tile_cols;
    for (int i = thread_id; i < total; i += num_threads) {
        int r = i / tile_cols;
        int c = i % tile_cols;
        int gr = src_row_start + r;
        int gc = src_col_start + c;

        if (gr < src_rows && gc < src_cols) {
            shared_dst[i] = src[(size_t)gr * src_cols + gc];
        } else {
            shared_dst[i] = 0.0f;
        }
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    int total = tile_q * tile_k;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;

        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            acc += q_tile[i * head_dim + d] * k_tile[j * head_dim + d];
        }
        s_tile[i * tile_k + j] = acc * scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float m = -INFINITY;
        for (int c = 0; c < tile_k; c++) {
            m = fmaxf(m, s_tile[r * tile_k + c]);
        }
        row_max_out[r] = m;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int total = tile_q * tile_k;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int r = idx / tile_k;
        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float s = 0.0f;
        for (int c = 0; c < tile_k; c++) {
            s += p_tile[r * tile_k + c];
        }
        row_sum_out[r] = s;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc,
                              int tile_q, int tile_k, int head_dim,
                              int thread_id, int num_threads) {
    int total = tile_q * head_dim;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int r = idx / head_dim;
        int d = idx % head_dim;

        float acc = 0.0f;
        for (int c = 0; c < tile_k; c++) {
            acc += p_tile[r * tile_k + c] * v_tile[c * head_dim + d];
        }
        out_acc[idx] += acc;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    extern __shared__ float smem[];
    float* q_s   = smem;                         // tile_q * head_dim
    float* k_s   = q_s + tile_q * head_dim;      // tile_k * head_dim
    float* v_s   = k_s + tile_k * head_dim;      // tile_k * head_dim
    float* s_s   = v_s + tile_k * head_dim;      // tile_q * tile_k
    float* o_s   = s_s + tile_q * tile_k;        // tile_q * head_dim
    float* m_s   = o_s + tile_q * head_dim;      // running max per row
    float* l_s   = m_s + tile_q;                 // running sum per row
    float* tmax  = l_s + tile_q;                 // tile max, then new max, then tile sum
    float* alpha = tmax + tile_q;                // correction factor per row

    int tid = threadIdx.x;
    int nt = blockDim.x;
    int q_start = blockIdx.x * tile_q;

    load_tile(q, q_s, q_start, 0, seq_len, head_dim, tile_q, head_dim, tid, nt);
    for (int idx = tid; idx < tile_q * head_dim; idx += nt) {
        o_s[idx] = 0.0f;
    }
    for (int r = tid; r < tile_q; r += nt) {
        m_s[r] = -INFINITY;
        l_s[r] = 0.0f;
    }
    __syncthreads();

    for (int k_start = 0; k_start < seq_len; k_start += tile_k) {
        load_tile(k, k_s, k_start, 0, seq_len, head_dim, tile_k, head_dim, tid, nt);
        load_tile(v, v_s, k_start, 0, seq_len, head_dim, tile_k, head_dim, tid, nt);
        __syncthreads();

        tile_scores(q_s, k_s, s_s, tile_q, tile_k, head_dim, scale, tid, nt);
        __syncthreads();

        for (int idx = tid; idx < tile_q * tile_k; idx += nt) {
            int c = idx % tile_k;
            if (k_start + c >= seq_len) {
                s_s[idx] = -INFINITY;
            }
        }
        __syncthreads();

        tile_rowmax(s_s, tmax, tile_q, tile_k, tid, nt);
        __syncthreads();

        for (int r = tid; r < tile_q; r += nt) {
            float m_old = m_s[r];
            float m_new = online_max(m_old, tmax[r]);
            alpha[r] = correction_factor(m_old, m_new);
            m_s[r] = m_new;
            tmax[r] = m_new;
        }
        __syncthreads();

        tile_exp(s_s, tmax, tile_q, tile_k, tid, nt);
        __syncthreads();

        // tmax is no longer needed, so reuse it for the tile row sums
        tile_rowsum(s_s, tmax, tile_q, tile_k, tid, nt);
        __syncthreads();

        for (int r = tid; r < tile_q; r += nt) {
            rescale_output(o_s + r * head_dim, head_dim, alpha[r]);
            l_s[r] = update_running_sum(l_s[r], alpha[r], tmax[r]);
        }
        __syncthreads();

        accumulate_pv(s_s, v_s, o_s, tile_q, tile_k, head_dim, tid, nt);
        __syncthreads();
    }

    for (int idx = tid; idx < tile_q * head_dim; idx += nt) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        int gr = q_start + r;
        if (gr < seq_len) {
            out[gr * head_dim + d] = o_s[idx] / l_s[r];
        }
    }
}

# Step 24 - flash_attention_launcher
#include <cmath>

void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    float scale = 1.0f / sqrtf((float)head_dim);

    int threads = 128;
    int blocks = (seq_len + tile_q - 1) / tile_q;

    size_t smem_floats = (size_t)2 * tile_q * head_dim
                       + (size_t)2 * tile_k * head_dim
                       + (size_t)tile_q * tile_k
                       + (size_t)4 * tile_q;
    size_t smem_bytes = smem_floats * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
    }

    flash_attention_kernel<<<blocks, threads, smem_bytes>>>(
        d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale);
    cudaDeviceSynchronize();
}

# Step 25 - causal_mask
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    int total = tile_q * tile_k;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int r = idx / tile_k;
        int c = idx % tile_k;
        if (k_col_start + c > q_row_start + r) {
            s_tile[idx] = -INFINITY;
        }
    }
}

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

