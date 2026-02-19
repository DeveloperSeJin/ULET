from functools import partial
import numpy as np
import torch
from inspect import isfunction

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if x_start.dtype == torch.int64:
        noise = default(noise, lambda: torch.randn(x_start.shape, device=x_start.device))
    elif x_start.dtype == torch.float32:
        noise = default(noise, lambda: torch.randn_like(x_start))
    
    return (extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def exists(x):
    return x is not None

def register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, device='cuda'):
    if exists(given_betas):
        betas = given_betas
    else:
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                    cosine_s=cosine_s)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    timesteps, = betas.shape
    num_timesteps = int(timesteps)
    linear_start = linear_start
    linear_end = linear_end
    assert alphas_cumprod.shape[0] == num_timesteps, 'alphas have to be defined for each timestep'

    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

    # betas =  to_torch(betas)
    # alphas_cumprod = to_torch(alphas_cumprod)
    # alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
    v_posterior=0
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + v_posterior * betas
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

    parameterization = 'eps'
    if parameterization == "eps":
        lvlb_weights = betas ** 2 / (
                    2 * posterior_variance * alphas * (1 - alphas_cumprod))
    elif parameterization == "x0":
        lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
    else:
        raise NotImplementedError("mu not supported")
    # TODO how to choose this term
    lvlb_weights[0] = lvlb_weights[1]
    lvlb_weights = to_torch(lvlb_weights)
    assert not torch.isnan(lvlb_weights).all()

    return {
        'betas': to_torch(betas),
        'alphas_cumprod': to_torch(alphas_cumprod),
        'alphas_cumprod_prev': to_torch(alphas_cumprod_prev),
        'sqrt_alphas_cumprod': to_torch(np.sqrt(alphas_cumprod)),
        'sqrt_one_minus_alphas_cumprod': to_torch(np.sqrt(1. - alphas_cumprod)),
        'log_one_minus_alphas_cumprod': to_torch(np.log(1. - alphas_cumprod)),
        'sqrt_recip_alphas_cumprod': to_torch(np.sqrt(1. / alphas_cumprod)),
        'sqrt_recipm1_alphas_cumprod': to_torch(np.sqrt(1. / alphas_cumprod - 1)),
        'posterior_variance': to_torch(posterior_variance),
        'posterior_log_variance_clipped': to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        'posterior_mean_coef1': to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)),
        'posterior_mean_coef2': to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)),
        'lvlb_weights': lvlb_weights
    }