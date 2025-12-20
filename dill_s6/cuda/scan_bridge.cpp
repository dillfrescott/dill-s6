#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

void launch_fwd(
    const float* u, const float* delta, const float* A, const float* B, const float* C, const float* D, const float* h0,
    float* out, float* x_save, float* ht,
    int B_sz, int L, int Dim, int N, cudaStream_t stream
);

void launch_bwd(
    const float* u, const float* delta, const float* A, const float* B, const float* C, const float* D, const float* h0,
    const float* x_save, const float* dout,
    float* du, float* ddelta, float* dA, float* dB, float* dC, float* dD, float* dh0,
    int B_sz, int L, int Dim, int N, cudaStream_t stream
);

std::vector<torch::Tensor> fwd(torch::Tensor u, torch::Tensor delta, torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D, c10::optional<torch::Tensor> h0) {
    auto B_sz = u.size(0); auto L = u.size(1); auto Dim = u.size(2); auto N = A.size(1);
    TORCH_CHECK(N <= 64, "d_state must be <= 64");
    auto out = torch::zeros_like(u);
    auto x_save = torch::zeros({(int)B_sz, (int)L, (int)Dim, (int)N}, u.options());
    
    torch::Tensor ht;
    const float* h0_ptr = nullptr;
    float* ht_ptr = nullptr;

    if (h0.has_value() && h0->defined()) {
        h0_ptr = h0->data_ptr<float>();
        ht = torch::zeros_like(*h0);
        ht_ptr = ht.data_ptr<float>();
    }

    launch_fwd(
        u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
        B.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        h0_ptr,
        out.data_ptr<float>(), x_save.data_ptr<float>(),
        ht_ptr,
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
    
    const float* h0_ptr = nullptr;
    float* dh0_ptr = nullptr;
    torch::Tensor dh0;
    
    if (h0.has_value() && h0->defined()) {
        h0_ptr = h0->data_ptr<float>();
        dh0 = torch::zeros_like(*h0);
        dh0_ptr = dh0.data_ptr<float>();
    }

    launch_bwd(
        u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
        B.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(),
        h0_ptr,
        x_save.data_ptr<float>(), dout.data_ptr<float>(),
        du.data_ptr<float>(), ddelta.data_ptr<float>(), dA.data_ptr<float>(),
        dB.data_ptr<float>(), dC.data_ptr<float>(), dD.data_ptr<float>(),
        dh0_ptr,
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