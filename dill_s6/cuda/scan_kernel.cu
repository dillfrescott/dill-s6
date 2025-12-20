#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void selective_scan_fwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ D,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ x_state_save, 
    const int B_size, const int L_size, const int D_size, const int N_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = B_size * D_size;
    if (idx >= total_channels) return;

    int b_idx = idx / D_size;
    int d_idx = idx % D_size;

    float state[16]; 
    for (int n = 0; n < N_size; ++n) state[n] = 0.0f;

    const scalar_t* u_ptr = u + (b_idx * L_size * D_size) + d_idx;
    const scalar_t* delta_ptr = delta + (b_idx * L_size * D_size) + d_idx;
    scalar_t* out_ptr = out + (b_idx * L_size * D_size) + d_idx;
    scalar_t* state_save_ptr = x_state_save + (b_idx * L_size * D_size * N_size) + (d_idx * N_size);

    const scalar_t* A_ptr = A + (d_idx * N_size);
    scalar_t D_val = D[d_idx];

    int stride = D_size;
    int B_stride = L_size * N_size;
    int C_stride = L_size * N_size;

    for (int t = 0; t < L_size; ++t) {
        scalar_t u_val = *u_ptr;
        scalar_t dt = *delta_ptr;

        const scalar_t* B_t = B + (b_idx * B_stride) + (t * N_size);
        const scalar_t* C_t = C + (b_idx * C_stride) + (t * N_size);

        scalar_t y_val = 0.0f;

        for (int n = 0; n < N_size; ++n) {
            scalar_t A_val = A_ptr[n];
            scalar_t B_val = B_t[n];
            scalar_t C_val = C_t[n];

            scalar_t dA = exp(dt * A_val);
            scalar_t dB = dt * B_val;

            state[n] = state[n] * dA + dB * u_val;
            state_save_ptr[n] = state[n];

            y_val += state[n] * C_val;
        }
        
        *out_ptr = y_val + D_val * u_val;

        u_ptr += stride;
        delta_ptr += stride;
        out_ptr += stride;
        state_save_ptr += stride * N_size; 
    }
}

template <typename scalar_t>
__global__ void selective_scan_bwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ x_state_save,
    const scalar_t* __restrict__ dout,
    
    scalar_t* __restrict__ du,
    scalar_t* __restrict__ ddelta,
    scalar_t* __restrict__ dA,
    scalar_t* __restrict__ dB,
    scalar_t* __restrict__ dC,
    scalar_t* __restrict__ dD,
    
    const int B_size, const int L_size, const int D_size, const int N_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = B_size * D_size;
    if (idx >= total_channels) return;

    int b_idx = idx / D_size;
    int d_idx = idx % D_size;

    float dstate[16];
    for (int n = 0; n < N_size; ++n) dstate[n] = 0.0f;
    
    int offset = (b_idx * L_size * D_size) + d_idx;
    const scalar_t* u_ptr = u + offset + (L_size - 1) * D_size;
    const scalar_t* delta_ptr = delta + offset + (L_size - 1) * D_size;
    const scalar_t* dout_ptr = dout + offset + (L_size - 1) * D_size;
    const scalar_t* state_ptr = x_state_save + (b_idx * L_size * D_size * N_size) + (d_idx * N_size) + (L_size - 1) * D_size * N_size;

    scalar_t* du_ptr = du + offset + (L_size - 1) * D_size;
    scalar_t* ddelta_ptr = ddelta + offset + (L_size - 1) * D_size;

    const scalar_t* A_ptr = A + (d_idx * N_size);
    scalar_t* dA_ptr = dA + (d_idx * N_size);
    scalar_t D_val = D[d_idx];
    scalar_t* dD_ptr = dD + d_idx;

    int B_stride = L_size * N_size;
    int C_stride = L_size * N_size;

    for (int t = L_size - 1; t >= 0; --t) {
        scalar_t u_val = *u_ptr;
        scalar_t dt = *delta_ptr;
        scalar_t dy = *dout_ptr;
        
        scalar_t du_val = dy * D_val;
        scalar_t ddelta_val = 0.0f;
        
        atomicAdd(dD_ptr, dy * u_val);

        const scalar_t* B_t = B + (b_idx * B_stride) + (t * N_size);
        const scalar_t* C_t = C + (b_idx * C_stride) + (t * N_size);
        
        scalar_t* dB_t = dB + (b_idx * B_stride) + (t * N_size);
        scalar_t* dC_t = dC + (b_idx * C_stride) + (t * N_size);

        for (int n = 0; n < N_size; ++n) {
            scalar_t A_val = A_ptr[n];
            scalar_t B_val = B_t[n];
            scalar_t C_val = C_t[n];
            
            scalar_t h_t = state_ptr[n]; 
            scalar_t h_prev = (t > 0) ? (state_ptr - D_size * N_size)[n] : 0.0f;

            dstate[n] += dy * C_val;
            atomicAdd(dC_t + n, dy * h_t); 
            
            scalar_t dA_exp = exp(dt * A_val);
            scalar_t d_h_prev = dstate[n] * dA_exp;
            du_val += dstate[n] * dt * B_val;
            atomicAdd(dB_t + n, dstate[n] * u_val * dt); 
            
            scalar_t ddt_1 = dstate[n] * h_prev * dA_exp * A_val;
            scalar_t ddt_2 = dstate[n] * u_val * B_val;
            ddelta_val += ddt_1 + ddt_2;
            
            scalar_t d_A_val = dstate[n] * h_prev * dA_exp * dt;
            atomicAdd(dA_ptr + n, d_A_val);

            dstate[n] = d_h_prev;
        }
        
        *du_ptr = du_val;
        *ddelta_ptr = ddelta_val;

        u_ptr -= D_size;
        delta_ptr -= D_size;
        dout_ptr -= D_size;
        du_ptr -= D_size;
        ddelta_ptr -= D_size;
        state_ptr -= D_size * N_size;
    }
}

void launch_fwd(
    const float* u, const float* delta, const float* A, const float* B, const float* C, const float* D,
    float* out, float* x_save,
    int B_sz, int L, int Dim, int N, cudaStream_t stream
) {
    const int threads = 128;
    const int blocks = (B_sz * Dim + threads - 1) / threads;
    selective_scan_fwd_kernel<float><<<blocks, threads, 0, stream>>>(u, delta, A, B, C, D, out, x_save, B_sz, L, Dim, N);
}

void launch_bwd(
    const float* u, const float* delta, const float* A, const float* B, const float* C, const float* D,
    const float* x_save, const float* dout,
    float* du, float* ddelta, float* dA, float* dB, float* dC, float* dD,
    int B_sz, int L, int Dim, int N, cudaStream_t stream
) {
    const int threads = 128;
    const int blocks = (B_sz * Dim + threads - 1) / threads;
    selective_scan_bwd_kernel<float><<<blocks, threads, 0, stream>>>(u, delta, A, B, C, D, x_save, dout, du, ddelta, dA, dB, dC, dD, B_sz, L, Dim, N);
}