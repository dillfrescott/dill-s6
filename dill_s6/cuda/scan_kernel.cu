#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ float my_exp(float x) { return expf(x); }
__device__ __forceinline__ double my_exp(double x) { return exp(x); }
__device__ __forceinline__ __half my_exp(__half x) { return hexp(x); }

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void my_atomic_add(float* address, float val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void my_atomic_add(double* address, double val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void my_atomic_add(__half* address, __half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(address, val);
#else
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned long long>(address) & ~0x3UL);
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    __half2 old_val = *reinterpret_cast<__half2*>(&old);
    __half new_val;
    if (reinterpret_cast<unsigned long long>(address) & 0x2) {
        new_val = __float2half(__half2float(old_val.y) + __half2float(val));
        old_val.y = new_val;
    } else {
        new_val = __float2half(__half2float(old_val.x) + __half2float(val));
        old_val.x = new_val;
    }
    __half2 new_val2 = old_val;
    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed, *reinterpret_cast<unsigned int*>(&new_val2));
        old_val = *reinterpret_cast<__half2*>(&old);
        if (reinterpret_cast<unsigned long long>(address) & 0x2) {
            new_val = __float2half(__half2float(old_val.y) + __half2float(val));
            old_val.y = new_val;
        } else {
            new_val = __float2half(__half2float(old_val.x) + __half2float(val));
            old_val.x = new_val;
        }
        new_val2 = old_val;
    } while (assumed != old);
#endif
}

template <typename scalar_t>
__global__ __launch_bounds__(128, 4) void selective_scan_fwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ h0,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ x_state_save,
    scalar_t* __restrict__ ht,
    const int B_size, const int L_size, const int D_size, const int N_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = B_size * D_size;
    if (idx >= total_channels) return;

    int b_idx = idx / D_size;
    int d_idx = idx % D_size;

    const scalar_t* u_ptr = u + (b_idx * L_size * D_size) + d_idx;
    const scalar_t* delta_ptr = delta + (b_idx * L_size * D_size) + d_idx;
    scalar_t* out_ptr = out + (b_idx * L_size * D_size) + d_idx;
    scalar_t* state_save_ptr = x_state_save + (b_idx * L_size * D_size * N_size) + (d_idx * N_size);

    const scalar_t* A_ptr = A + (d_idx * N_size);
    scalar_t D_val = __ldg(D + d_idx);

    int stride = D_size;
    int B_stride = L_size * N_size;
    int C_stride = L_size * N_size;

    for (int t = 0; t < L_size; ++t) {
        scalar_t u_val = __ldg(u_ptr);
        scalar_t dt = __ldg(delta_ptr);

        const scalar_t* B_t = B + (b_idx * B_stride) + (t * N_size);
        const scalar_t* C_t = C + (b_idx * C_stride) + (t * N_size);

        scalar_t y_val = static_cast<scalar_t>(0.0f);

        #pragma unroll
        for (int n = 0; n < N_size; ++n) {
            scalar_t A_val = __ldg(A_ptr + n);
            scalar_t B_val = __ldg(B_t + n);
            scalar_t C_val = __ldg(C_t + n);

            scalar_t dA = my_exp(dt * A_val);
            scalar_t dB = dt * B_val;

            scalar_t prev_s;
            if (t == 0) {
                if (h0 != nullptr) {
                    prev_s = __ldg(h0 + (b_idx * D_size * N_size) + (d_idx * N_size) + n);
                } else {
                    prev_s = static_cast<scalar_t>(0.0f);
                }
            } else {
                prev_s = (state_save_ptr - stride * N_size)[n];
            }

            scalar_t s = prev_s * dA + dB * u_val;
            state_save_ptr[n] = s;

            y_val += s * C_val;
        }
        
        *out_ptr = y_val + D_val * u_val;

        u_ptr += stride;
        delta_ptr += stride;
        out_ptr += stride;
        state_save_ptr += stride * N_size; 
    }

    if (ht != nullptr) {
        scalar_t* ht_ptr = ht + (b_idx * D_size * N_size) + (d_idx * N_size);
        const scalar_t* src_ptr = state_save_ptr - stride * N_size;
        #pragma unroll
        for (int n = 0; n < N_size; ++n) ht_ptr[n] = src_ptr[n];
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(128, 4) void selective_scan_bwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ h0,
    const scalar_t* __restrict__ x_save,
    const scalar_t* __restrict__ dout,

    scalar_t* __restrict__ du,
    scalar_t* __restrict__ ddelta,
    scalar_t* __restrict__ dA,
    scalar_t* __restrict__ dB,
    scalar_t* __restrict__ dC,
    scalar_t* __restrict__ dD,
    scalar_t* __restrict__ dh0,

    const int B_size, const int L_size, const int D_size, const int N_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_channels = B_size * D_size;
    if (idx >= total_channels) return;

    int b_idx = idx / D_size;
    int d_idx = idx % D_size;

    scalar_t* dstate_ptr = dh0 + (b_idx * D_size * N_size) + (d_idx * N_size);
    for (int n = 0; n < N_size; ++n) dstate_ptr[n] = static_cast<scalar_t>(0.0f);
    
    int offset = (b_idx * L_size * D_size) + d_idx;
    const scalar_t* u_ptr = u + offset + (L_size - 1) * D_size;
    const scalar_t* delta_ptr = delta + offset + (L_size - 1) * D_size;
    const scalar_t* dout_ptr = dout + offset + (L_size - 1) * D_size;
    const scalar_t* state_ptr = x_save + (b_idx * L_size * D_size * N_size) + (d_idx * N_size) + (L_size - 1) * D_size * N_size;

    scalar_t* du_ptr = du + offset + (L_size - 1) * D_size;
    scalar_t* ddelta_ptr = ddelta + offset + (L_size - 1) * D_size;

    const scalar_t* A_ptr = A + (d_idx * N_size);
    scalar_t* dA_ptr = dA + (d_idx * N_size);
    scalar_t D_val = __ldg(D + d_idx);
    scalar_t* dD_ptr = dD + d_idx;

    int B_stride = L_size * N_size;
    int C_stride = L_size * N_size;

    scalar_t dD_accum = static_cast<scalar_t>(0.0f);

    for (int t = L_size - 1; t >= 0; --t) {
        scalar_t u_val = __ldg(u_ptr);
        scalar_t dt = __ldg(delta_ptr);
        scalar_t dy = __ldg(dout_ptr);
        
        scalar_t du_val = dy * D_val;
        scalar_t ddelta_val = static_cast<scalar_t>(0.0f);
        
        dD_accum += dy * u_val;

        const scalar_t* B_t = B + (b_idx * B_stride) + (t * N_size);
        const scalar_t* C_t = C + (b_idx * C_stride) + (t * N_size);
        
        scalar_t* dB_t = dB + (b_idx * B_stride) + (t * N_size);
        scalar_t* dC_t = dC + (b_idx * C_stride) + (t * N_size);

        #pragma unroll
        for (int n = 0; n < N_size; ++n) {
            scalar_t A_val = __ldg(A_ptr + n);
            scalar_t B_val = __ldg(B_t + n);
            scalar_t C_val = __ldg(C_t + n);
            
            scalar_t h_t = __ldg(state_ptr + n);
            scalar_t h_prev;
            if (t > 0) {
                 h_prev = __ldg(state_ptr - D_size * N_size + n);
            } else {
                 if (h0 != nullptr) {
                     h_prev = __ldg(h0 + (b_idx * D_size * N_size) + (d_idx * N_size) + n);
                 } else {
                     h_prev = static_cast<scalar_t>(0.0f);
                 }
            }

            scalar_t ds = dstate_ptr[n];
            ds += dy * C_val;
            
            float dC_contrib = static_cast<float>(dy * h_t);
            dC_contrib = warpReduceSum(dC_contrib);
            if ((threadIdx.x % 32) == 0) {
                 my_atomic_add(dC_t + n, static_cast<scalar_t>(dC_contrib));
            }
            
            scalar_t dA_exp = my_exp(dt * A_val);
            scalar_t d_h_prev = ds * dA_exp;
            du_val += ds * dt * B_val;
            
            float dB_contrib = static_cast<float>(ds * u_val * dt);
            dB_contrib = warpReduceSum(dB_contrib);
            if ((threadIdx.x % 32) == 0) {
                my_atomic_add(dB_t + n, static_cast<scalar_t>(dB_contrib));
            }
            
            scalar_t ddt_1 = ds * h_prev * dA_exp * A_val;
            scalar_t ddt_2 = ds * u_val * B_val;
            ddelta_val += ddt_1 + ddt_2;
            
            scalar_t d_A_val = ds * h_prev * dA_exp * dt;
            my_atomic_add(dA_ptr + n, d_A_val);

            dstate_ptr[n] = d_h_prev;
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
    
    my_atomic_add(dD_ptr, dD_accum);
}

extern "C" {
    void selective_scan_fwd_cuda(
        int dtype, 
        void* u, void* delta, void* A, void* B, void* C, void* D, void* h0,
        void* out, void* x_save, void* ht,
        int B_sz, int L, int Dim, int N, cudaStream_t stream
    ) {
        const int threads = 128;
        const int blocks = (B_sz * Dim + threads - 1) / threads;
        
        if (dtype == 0) {
            selective_scan_fwd_kernel<float><<<blocks, threads, 0, stream>>>(
                (float*)u, (float*)delta, (float*)A, (float*)B, (float*)C, (float*)D, (float*)h0,
                (float*)out, (float*)x_save, (float*)ht,
                B_sz, L, Dim, N
            );
        } else if (dtype == 1) {
            selective_scan_fwd_kernel<__half><<<blocks, threads, 0, stream>>>(
                (__half*)u, (__half*)delta, (__half*)A, (__half*)B, (__half*)C, (__half*)D, (__half*)h0,
                (__half*)out, (__half*)x_save, (__half*)ht,
                B_sz, L, Dim, N
            );
        }
    }

    void selective_scan_bwd_cuda(
        int dtype,
        void* u, void* delta, void* A, void* B, void* C, void* D, void* h0,
        void* x_save, void* dout,
        void* du, void* ddelta, void* dA, void* dB, void* dC, void* dD, void* dh0,
        int B_sz, int L, int Dim, int N, cudaStream_t stream
    ) {
        const int threads = 128;
        const int blocks = (B_sz * Dim + threads - 1) / threads;

        if (dtype == 0) {
            selective_scan_bwd_kernel<float><<<blocks, threads, 0, stream>>>(
                (float*)u, (float*)delta, (float*)A, (float*)B, (float*)C, (float*)D, (float*)h0, (float*)x_save, (float*)dout,
                (float*)du, (float*)ddelta, (float*)dA, (float*)dB, (float*)dC, (float*)dD, (float*)dh0,
                B_sz, L, Dim, N
            );
        } else if (dtype == 1) {
            selective_scan_bwd_kernel<__half><<<blocks, threads, 0, stream>>>(
                (__half*)u, (__half*)delta, (__half*)A, (__half*)B, (__half*)C, (__half*)D, (__half*)h0, (__half*)x_save, (__half*)dout,
                (__half*)du, (__half*)ddelta, (__half*)dA, (__half*)dB, (__half*)dC, (__half*)dD, (__half*)dh0,
                B_sz, L, Dim, N
            );
        }
    }
}
