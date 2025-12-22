# Dill S6

**Dill S6** is a high-performance PyTorch extension implementing the S6 Selective Scan mechanism (key component of Mamba architectures) with custom CUDA kernels for efficient training and inference.

## Features

- **Custom CUDA Kernels**: Optimized forward and backward passes for the selective scan operation.
- **PyTorch Integration**: Seamless integration as a `torch.nn.Module`.
- **Automatic Compilation**: The C++/CUDA extensions are compiled automatically when you install the package.

## Prerequisites

- **OS**: Linux or Windows (with CUDA environment set up).
- **Python**: 3.7+
- **PyTorch**: >= 1.10 (with CUDA support).
- **CUDA Toolkit**: Version matching your PyTorch installation (e.g., 11.8, 12.1). The `nvcc` compiler must be available in your system path.
- **C++ Build Tools**:
    - **Windows**: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload installed.
    - **Linux**: GCC/G++ (typically `build-essential`).

## Installation

Since this package contains C++/CUDA extensions, it must be compiled during installation. Ensure you have the prerequisites installed, then run:

```bash
pip install git+https://github.com/dillfrescott/dill-s6 --no-build-isolation
```

*Note: This process may take a few minutes as it compiles the CUDA kernels.*

## Usage

You can use the `S6` module in your PyTorch models just like any other layer.

```python
import torch
from dill_s6 import S6

# Configuration
batch_size = 4
seq_len = 128
d_model = 64

# Initialize the module
# d_state: State expansion factor (default 16)
# d_conv: Local convolution kernel size (default 4)
# expand: Block expansion factor (default 2)
# depth: Number of S6 layers (default 1)
# bidirectional: Whether to process sequence in both directions (default False)
model = S6(d_model=d_model, d_state=16, d_conv=4, expand=2, depth=1, bidirectional=True).cuda()

# Input tensor: (Batch, Length, Dimension)
x = torch.randn(batch_size, seq_len, d_model).cuda()

# Forward pass
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

The `depth` parameter controls the number of S6 layers stacked together. Each layer contains the same components (normalization, projections, convolutions, etc.) and processes the input sequentially. A higher depth creates a deeper network with more processing stages, potentially improving model capacity and performance for complex tasks.

## Troubleshooting

### `ImportError: mamba_cuda_core extension is not available`

If you see this error, it means the compiled C++ extension could not be loaded.
1. Ensure the installation command completed successfully without errors.
2. Check that `nvcc` is in your PATH (`nvcc --version`).
3. If on Windows, ensure your Visual Studio C++ build tools are compatible with your CUDA version.

### `CUDA_HOME` not found

The `setup.py` attempts to auto-detect `CUDA_HOME` using `nvcc`. If this fails, set the environment variable manually:

- **Linux**: `export CUDA_HOME=/usr/local/cuda`
- **Windows**: `set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
