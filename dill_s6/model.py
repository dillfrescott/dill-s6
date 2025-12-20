import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from . import mamba_cuda_core as mamba_cuda
except ImportError:
    mamba_cuda = None
    pass

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, initial_state=None):
        if mamba_cuda is None:
            raise ImportError("mamba_cuda_core extension is not available. Please ensure the package is installed correctly.")
        
        if u.dtype != torch.float32:
            u, delta, A, B, C, D = [x.float() for x in [u, delta, A, B, C, D]]
            if initial_state is not None:
                initial_state = initial_state.float()
            
        u, delta, A, B, C, D = [x.contiguous() for x in [u, delta, A, B, C, D]]
        if initial_state is not None:
            initial_state = initial_state.contiguous()

        out_tuple = mamba_cuda.fwd(u, delta, A, B, C, D, initial_state)
        out = out_tuple[0]
        x_save = out_tuple[1]
        
        has_initial_state = initial_state is not None
        ctx.save_for_backward(u, delta, A, B, C, D, initial_state, x_save)
        ctx.has_initial_state = has_initial_state
        
        if has_initial_state:
            return out, out_tuple[2] # Return ht
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        if mamba_cuda is None:
            raise ImportError("mamba_cuda_core extension is not available. Please ensure the package is installed correctly.")

        u, delta, A, B, C, D, initial_state, x_save = ctx.saved_tensors
        dout = dout.contiguous()
        
        grads = mamba_cuda.bwd(u, delta, A, B, C, D, initial_state, x_save, dout)
        
        du, ddelta, dA, dB, dC, dD = grads[0], grads[1], grads[2], grads[3], grads[4], grads[5]
        
        dh0 = None
        if ctx.has_initial_state:
            dh0 = grads[6]
            
        return du, ddelta, dA, dB, dC, dD, dh0

def selective_scan_fn(u, delta, A, B, C, D, initial_state=None):
    return SelectiveScanFn.apply(u, delta, A, B, C, D, initial_state)

class S6(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.dt_rank = max(1, self.d_model // 16)
        self.x_proj = nn.Linear(self.d_inner, (self.d_state * 2) + self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        nn.init.kaiming_uniform_(self.dt_proj.weight, a=math.sqrt(5))

        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            self.d_inner,
            dim=0
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, initial_state=None):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        y = selective_scan_fn(x_conv, dt, A, B_ssm, C_ssm, self.D, initial_state)

        if initial_state is not None:
            y, final_state = y

        y = y * self.act(z_branch)
        
        out = self.out_proj(y)
        if initial_state is not None:
            return out, final_state
        return out