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
        if mamba_cuda is not None:
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
                return out, out_tuple[2]
            return out
        else:
            return selective_scan_fn_fallback(u, delta, A, B, C, D, initial_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if mamba_cuda is not None:
            u, delta, A, B, C, D, initial_state, x_save = ctx.saved_tensors
            dout = dout.contiguous()

            grads = mamba_cuda.bwd(u, delta, A, B, C, D, initial_state, x_save, dout)

            du, ddelta, dA, dB, dC, dD = grads[0], grads[1], grads[2], grads[3], grads[4], grads[5]

            dh0 = None
            if ctx.has_initial_state:
                dh0 = grads[6]

            return du, ddelta, dA, dB, dC, dD, dh0
        else:
            return selective_scan_fn_fallback_backward(ctx, dout, *args)

def selective_scan_fn(u, delta, A, B, C, D, initial_state=None):
    return SelectiveScanFn.apply(u, delta, A, B, C, D, initial_state)

def selective_scan_fn_fallback(u, delta, A, B, C, D, initial_state=None):
    batch, seqlen, dim = u.shape
    _, _, n_state = A.shape

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
        u_i = u[:, i, :].unsqueeze(-1)
        delta_i = delta[:, i, :].unsqueeze(-1)
        B_i = B[:, i, :, :]
        C_i = C[:, i, :, :]

        delta_A_i = torch.exp(delta_i * A_expanded)

        x = delta_A_i * x + B_i * u_i * delta_i

        y_i = torch.sum(x * C_i, dim=-1)
        y[:, i, :] = y_i

    y = y + u * D

    return y

def selective_scan_fn_fallback_backward(ctx, dout, *args):
    raise NotImplementedError("Fallback backward not fully implemented. Use CUDA version.")

class S6(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random",
                 dt_scale=1.0, dt_proj_bias=True, dt_soft_plus=True, conv_bias=True, bias=False, use_fast_path=True, depth=1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.depth = depth
        self.use_fast_path = use_fast_path

        self.layers = nn.ModuleList()
        for i in range(depth):
            layer_in_proj = nn.Linear(d_model if i == 0 else self.d_inner, self.d_inner * 2, bias=bias)

            layer_conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )

            layer_dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank
            layer_x_proj = nn.Linear(self.d_inner, layer_dt_rank + self.d_state * 2, bias=False)
            layer_dt_proj = nn.Linear(layer_dt_rank, self.d_inner, bias=dt_proj_bias)

            dt_init_std = layer_dt_rank / d_model if dt_init == "constant" else 0.1
            if dt_init == "constant":
                nn.init.constant_(layer_dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(layer_dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=1e-4)

            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                layer_dt_proj.bias.copy_(inv_dt)

            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)

            layer_modules = {
                'in_proj': layer_in_proj,
                'conv1d': layer_conv1d,
                'x_proj': layer_x_proj,
                'dt_proj': layer_dt_proj,
                'act': nn.SiLU(),
                'out_proj': nn.Linear(self.d_inner, d_model if i == depth - 1 else self.d_inner, bias=bias),
            }

            layer_dict = nn.ModuleDict(layer_modules)

            layer_dict.register_parameter('A_log', nn.Parameter(torch.log(A)))
            layer_dict.register_parameter('D', nn.Parameter(torch.ones(self.d_inner)))
            layer_dict.dt_rank = layer_dt_rank

            self.layers.append(layer_dict)

    def forward(self, x, initial_state=None):
        output = x

        for i, layer in enumerate(self.layers):
            B, L, _ = output.shape

            xz = layer['in_proj'](output)
            x_branch, z_branch = xz.chunk(2, dim=-1)

            x_conv = x_branch.transpose(1, 2)
            x_conv = layer['conv1d'](x_conv)[..., :L]
            x_conv = x_conv.transpose(1, 2)
            x_conv = layer['act'](x_conv)

            x_dbl = layer['x_proj'](x_conv)
            dt_rank = layer.dt_rank
            dt, B_ssm, C_ssm = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)

            dt = layer['dt_proj'](dt)
            dt = F.softplus(dt)

            B_ssm = B_ssm.view(B, L, 1, self.d_state).expand(B, L, self.d_inner, self.d_state).contiguous()
            C_ssm = C_ssm.view(B, L, 1, self.d_state).expand(B, L, self.d_inner, self.d_state).contiguous()

            A = -torch.exp(layer.A_log.float()).to(x_conv.dtype)

            y = selective_scan_fn(x_conv, dt, A, B_ssm, C_ssm, layer.D)

            y = y * layer['act'](z_branch)

            output = layer['out_proj'](y)

        return output