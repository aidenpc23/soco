import torch
import numpy as np


def soco_step_loss(x_t: torch.Tensor, y_t: torch.Tensor, x_prev: torch.Tensor, m: float):
    hit = 0.5 * m * (x_t - y_t).pow(2)
    move = 0.5 * (x_t - x_prev).pow(2)
    return (hit, move, hit + move)


# incorrect; cannot be done as a step loss
def ec_l2o_step_loss(u: float, x_ml: torch.Tensor, x_o: torch.Tensor,
                     x_t: torch.Tensor, x_prev: torch.Tensor, y_t: torch.Tensor,
                     cost_o: float, m: float):
    _hit, _move, c = soco_step_loss(x_t, y_t, x_prev, m)
    term = x_ml - x_o
    lf = max(0, term * term / cost_o)
    return u * lf + (1 - u) * c


def soco_seq_loss(xs: torch.Tensor, ys: torch.Tensor, m: float) -> torch.Tensor:
    x_prev = torch.cat([torch.as_tensor([0.0]), xs[:-1]], dim=0)
    hit = 0.5 * m * (xs - ys).pow(2)
    move = 0.5 * (xs - x_prev).pow(2)
    return (hit + move).sum()


def ec_l2o_seq_loss(
    xs: torch.Tensor,
    ys: torch.Tensor,
    x_mls: torch.Tensor,
    m: float,
    u: float,
    Dlen: float,
    p_bar: float = 0.1,
):
    o_hist = oracle_history(m, ys)

    base_cost = soco_seq_loss(xs, ys, m)
    oracle_cost = sum(o_hist["total"])

    div = (x_mls - torch.as_tensor(o_hist["x"])).pow(2).sum()
    den = max(oracle_cost, 0.0001)
    lf = max((div / den) - p_bar, 0)

    loss = (1.0 - u) * base_cost + u * lf

    return loss / Dlen


def create_tridiagonal(T, a, b, c):
    A = np.zeros([T, T])
    np.fill_diagonal(A, b)
    np.fill_diagonal(A[:-1, 1:], a)
    np.fill_diagonal(A[1:, :-1], c)
    return A


def oracle_with_zero(m: float, ys):
    res = oracle(m, ys[1:])
    return np.insert(res, 0, 0)


def oracle(m: float, ys):
    T = len(ys)
    A = create_tridiagonal(T, -1, m + 2, -1)
    A[-1, -1] = m + 1
    R = m * ys
    return np.linalg.solve(A, R)


def oracle_history(m: float, ys):
    ys = np.asarray(ys)
    xs = oracle_with_zero(m, ys)

    hit = 0.5 * m * (xs - ys) ** 2
    dx = np.concatenate(([xs[0] - 0.0], np.diff(xs)))
    move = 0.5 * dx ** 2
    total = hit + move

    history = {
        "x": xs.tolist(),
        "y": ys.tolist(),
        "hitting": hit.tolist(),
        "movement": move.tolist(),
        "total": total.tolist(),
    }
    return history
