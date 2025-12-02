# cleanrl/armijo.py
from typing import Iterable, List, Tuple

import torch
from torch import Tensor


def get_grad_list(params: Iterable[torch.nn.Parameter]) -> List[Tensor]:
    """
    Collect gradients (or zeros if no grad) into a list of tensors
    with the same shapes as parameters.
    """
    params = list(params)
    grads: List[Tensor] = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p.data))
        else:
            grads.append(p.grad.detach().clone())
    return grads


def grad_norm_sq(grads: List[Tensor]) -> Tensor:
    """
    Compute squared L2 norm of a list of gradient tensors.
    """
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    total = torch.tensor(0.0, device=device)
    for g in grads:
        total = total + torch.sum(g * g)
    return total


def armijo_line_search(
    params: Iterable[torch.nn.Parameter],
    grads: List[Tensor],
    loss_closure,
    f_init: float,
    grad_norm_sq_val: float,
    eta_max: float,
    c: float = 1e-4,
    beta: float = 0.5,
    max_iters: int = 50,
) -> Tuple[float, float]:
    """
    Armijo backtracking line search along the negative gradient direction.

    We search for step size eta in (0, eta_max] such that:

        f(theta_new) <= f(theta) - c * eta * ||∇f||^2

    where theta_new = theta - eta * grad.
    """
    params = list(params)
    params_current = [p.data.clone() for p in params]
    direction = [-g for g in grads]  # gradient descent direction

    eta = eta_max

    for _ in range(max_iters):
        # theta_new = theta_current + eta * direction
        with torch.no_grad():
            for p, p0, d in zip(params, params_current, direction):
                p.data.copy_(p0 + eta * d)

        with torch.no_grad():
            f_new = loss_closure().item()

        rhs = f_init - c * eta * grad_norm_sq_val
        if f_new <= rhs:
            return eta, f_new

        eta *= beta  # shrink step size and retry

    # Fallback to tiny step
    eta = 1e-6
    with torch.no_grad():
        for p, p0, d in zip(params, params_current, direction):
            p.data.copy_(p0 + eta * d)
    with torch.no_grad():
        f_new = loss_closure().item()
    return eta, f_new
