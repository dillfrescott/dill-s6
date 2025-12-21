#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" {
    void selective_scan_fwd_cuda(
        int dtype, 
        void* u, void* delta, void* A, void* B, void* C, void* D, void* h0,
        void* out, void* x_save, void* ht,
        int B_sz, int L, int Dim, int N, cudaStream_t stream
    );

    void selective_scan_bwd_cuda(
        int dtype,
        void* u, void* delta, void* A, void* B, void* C, void* D, void* h0,
        void* x_save, void* dout,
        void* du, void* ddelta, void* dA, void* dB, void* dC, void* dD, void* dh0,
        int B_sz, int L, int Dim, int N, cudaStream_t stream
    );
}

std::vector<torch::Tensor> fwd(torch::Tensor u, torch::Tensor delta, torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D, c10::optional<torch::Tensor> h0) {
    auto B_sz = u.size(0); auto L = u.size(1); auto Dim = u.size(2); auto N = A.size(1);
    auto out = torch::zeros_like(u);
    auto x_save = torch::zeros({(int)B_sz, (int)L, (int)Dim, (int)N}, u.options());
    
    torch::Tensor ht;
    void* h0_ptr = nullptr;
    void* ht_ptr = nullptr;

    if (h0.has_value() && h0->defined()) {
        h0_ptr = h0->data_ptr();
        ht = torch::zeros_like(*h0);
        ht_ptr = ht.data_ptr();
    }

    int dtype_id = -1;
    if (u.scalar_type() == torch::kFloat) dtype_id = 0;
    else if (u.scalar_type() == torch::kHalf) dtype_id = 1;
    else TORCH_CHECK(false, "Unsupported dtype");

    selective_scan_fwd_cuda(
        dtype_id,
        u.data_ptr(), delta.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(), h0_ptr,
        out.data_ptr(), x_save.data_ptr(), ht_ptr,
        (int)B_sz, (int)L, (int)Dim, (int)N, at::cuda::getCurrentCUDAStream().stream()
    );
    
    if (h0.has_value() && h0->defined()) {
        return {out, x_save, ht};
    } else {
        return {out, x_save};
    }
}

std::vector<torch::Tensor> bwd(
    torch::Tensor u, torch::Tensor delta, torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D, 
    c10::optional<torch::Tensor> h0, torch::Tensor x_save, torch::Tensor dout
) {
    auto B_sz = u.size(0); auto L = u.size(1); auto Dim = u.size(2); auto N = A.size(1);
    
    auto du = torch::zeros_like(u);
    auto ddelta = torch::zeros_like(delta);
    auto dA = torch::zeros_like(A);
    auto dB = torch::zeros_like(B);
    auto dC = torch::zeros_like(C);
    auto dD = torch::zeros_like(D);
    
    void* h0_ptr = nullptr;
    if (h0.has_value() && h0->defined()) {
        h0_ptr = h0->data_ptr();
    }
    
    auto dh0 = torch::zeros({(int)B_sz, (int)Dim, (int)N}, u.options());
    void* dh0_ptr = dh0.data_ptr();

    int dtype_id = -1;
    if (u.scalar_type() == torch::kFloat) dtype_id = 0;
    else if (u.scalar_type() == torch::kHalf) dtype_id = 1;
    else TORCH_CHECK(false, "Unsupported dtype");

    selective_scan_bwd_cuda(
        dtype_id,
        u.data_ptr(), delta.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(), h0_ptr,
        x_save.data_ptr(), dout.data_ptr(),
        du.data_ptr(), ddelta.data_ptr(), dA.data_ptr(), dB.data_ptr(), dC.data_ptr(), dD.data_ptr(), dh0_ptr,
        (int)B_sz, (int)L, (int)Dim, (int)N, at::cuda::getCurrentCUDAStream().stream()
    );

    if (h0.has_value() && h0->defined()) {
        return {du, ddelta, dA, dB, dC, dD, dh0};
    } else {
        return {du, ddelta, dA, dB, dC, dD};
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "Selective Scan Forward", py::arg("u"), py::arg("delta"), py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D"), py::arg("h0") = c10::nullopt);
    m.def("bwd", &bwd, "Selective Scan Backward", py::arg("u"), py::arg("delta"), py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D"), py::arg("h0") = c10::nullopt, py::arg("x_save"), py::arg("dout"));
}