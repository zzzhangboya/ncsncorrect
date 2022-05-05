import torch

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    # grad_outputs作用
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    # 为什么是mean，不应该是sum么
    tr_dfdx = (eps_dfdx * eps).mean([-1,-2,-3])
    return tr_dfdx


def exact_jacobian_trace(fx, x):
    vals = []
    # x.size(1)是channel那一维，那一维是1，fx和x都是256*1*28*28
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            for k in range(x.size(3)):
                fxi = fx[:, i, j, k]
                # fxi.sum()是scalar
                dfxi_dxi = keep_grad(fxi.sum(), x)[:, i, j, k][:, None]
                vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.mean(dim=1)


def stein_stats(logp, x, critic, approx_jcb=True, n_samples=1):


    lp = logp
    sq = keep_grad(lp.sum(), x)

    fx = critic(x)
    sq_fx = (sq * fx).mean([-1,-2,-3])

    if approx_jcb==False:
        tr_dfdx = exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(n_samples)], dim=1).mean(
            dim=1)

    stats = sq_fx + tr_dfdx
    norms = (fx * fx).mean([-1,-2,-3])
    grad_norms = (sq * sq).mean([-1,-2,-3])
    return stats, norms, grad_norms, lp


def stein_stats_withscore(score, x, y, critic, approx_jcb=True, n_samples=1):

    sq = score

    fx = critic(x,y)
    sq_fx = (sq * fx).mean([-1,-2,-3])

    if approx_jcb==False:
        tr_dfdx = exact_jacobian_trace(fx, x)
    else:
        tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(n_samples)], dim=1).mean(
            dim=1)

    stats = sq_fx + tr_dfdx
    norms = (fx * fx).mean([-1,-2,-3])
    return stats, norms
