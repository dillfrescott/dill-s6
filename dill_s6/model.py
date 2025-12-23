import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from . import mamba_cuda_core as mamba_cuda
except ImportError:
    mamba_cuda = None

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, initial_state=None):
        if mamba_cuda is not None:
            u, delta, A, B, C, D = [x if x.is_contiguous() else x.contiguous() for x in (u, delta, A, B, C, D)]
            if initial_state is not None:
                initial_state = initial_state if initial_state.is_contiguous() else initial_state.contiguous()
            out, x_save, *rest = mamba_cuda.fwd(u, delta, A, B, C, D, initial_state)
            ctx.save_for_backward(u, delta, A, B, C, D, initial_state, x_save)
            ctx.has_initial_state = initial_state is not None
            if ctx.has_initial_state:
                return out, rest[0]
            return out
        return selective_scan_fn_fallback(u, delta, A, B, C, D, initial_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if mamba_cuda is not None:
            u, delta, A, B, C, D, initial_state, x_save = ctx.saved_tensors
            dout = dout if dout.is_contiguous() else dout.contiguous()
            grads = mamba_cuda.bwd(u, delta, A, B, C, D, initial_state, x_save, dout)
            du, ddelta, dA, dB, dC, dD = grads[:6]
            dh0 = grads[6] if ctx.has_initial_state else None
            return du, ddelta, dA, dB, dC, dD, dh0
        return selective_scan_fn_fallback_backward(ctx, dout, *args)

def selective_scan_fn(u, delta, A, B, C, D, initial_state=None):
    if mamba_cuda is not None:
        return SelectiveScanFn.apply(u, delta, A, B, C, D, initial_state)
    return selective_scan_fn_fallback(u, delta, A, B, C, D, initial_state)

def selective_scan_fn_fallback(u, delta, A, B, C, D, initial_state=None):
    batch, seqlen, dim = u.shape
    _, n_state = A.shape
    u = u.contiguous()
    delta = delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    if initial_state is not None:
        x = initial_state
    else:
        x = torch.zeros((batch, dim, n_state), dtype=u.dtype, device=u.device)
    A_expanded = A.unsqueeze(0).unsqueeze(0)
    y = torch.empty((batch, seqlen, dim), dtype=u.dtype, device=u.device)
    for i in range(seqlen):
        u_i = u[:, i].unsqueeze(-1)
        delta_i = delta[:, i].unsqueeze(-1)
        B_i = B[:, i]
        C_i = C[:, i]
        x = torch.exp(delta_i * A_expanded) * x + B_i * u_i * delta_i
        y[:, i] = torch.sum(x * C_i, dim=-1)
    return y + u * D

def selective_scan_fn_fallback_backward(ctx, dout, *args):
    raise NotImplementedError

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class S6(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_proj_bias=True, dt_soft_plus=True, conv_bias=True, bias=False, depth=1, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.depth = depth
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList()
        for _ in range(depth):
            in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
            conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
            rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank
            x_proj = nn.Linear(self.d_inner, rank + 2 * d_state, bias=False)
            dt_proj = nn.Linear(rank, self.d_inner, bias=dt_proj_bias)
            std = rank / d_model if dt_init == "constant" else 0.1
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -std, std)
            else:
                raise NotImplementedError
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            A = torch.arange(1, d_state + 1, dtype=torch.float32, device=in_proj.weight.device).repeat(self.d_inner, 1)
            A_log = nn.Parameter(torch.log(A))
            with torch.no_grad():
                A_negexp = -torch.exp(A_log)

            layer = nn.Module()
            layer.norm = RMSNorm(d_model)
            layer.in_proj = in_proj
            layer.conv1d = conv1d
            layer.x_proj = x_proj
            layer.dt_proj = dt_proj
            layer.act = nn.SiLU()
            layer.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
            layer.register_parameter("A_log", A_log)
            layer.register_buffer("A_negexp", A_negexp)
            layer.register_parameter("D", nn.Parameter(torch.ones(self.d_inner, device=in_proj.weight.device)))
            layer.dt_rank = rank
            self.layers.append(layer)
        self.norm_f = RMSNorm(d_model)

    def forward(self, x, initial_state=None):
        out = x
        for layer in self.layers:
            residual = out
            out = layer.norm(out)
            B, L, _ = out.shape
            xz = layer.in_proj(out)
            x_branch, z_branch = xz.chunk(2, dim=-1)
            x_conv = layer.conv1d(x_branch.transpose(1, 2))[..., :L]
            x_conv = layer.act(x_conv.transpose(1, 2))
            x_dbl = layer.x_proj(x_conv)
            dt_rank = layer.dt_rank
            dt, B_ssm, C_ssm = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(layer.dt_proj(dt))
            B_ssm = B_ssm.view(B, L, 1, self.d_state).expand(B, L, self.d_inner, self.d_state)
            C_ssm = C_ssm.view(B, L, 1, self.d_state).expand(B, L, self.d_inner, self.d_state)
            A = layer.A_negexp.to(x_conv.dtype)
            y = selective_scan_fn(x_conv, dt, A, B_ssm, C_ssm, layer.D)

            if self.bidirectional:
                x_conv_bwd = x_conv.flip([1])
                dt_bwd = dt.flip([1])
                B_ssm_bwd = B_ssm.flip([1])
                C_ssm_bwd = C_ssm.flip([1])
                y_bwd = selective_scan_fn(x_conv_bwd, dt_bwd, A, B_ssm_bwd, C_ssm_bwd, layer.D)
                y = y + y_bwd.flip([1])

            y = y * layer.act(z_branch)
            out = layer.out_proj(y) + residual
        return self.norm_f(out)