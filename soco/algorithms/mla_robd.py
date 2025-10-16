import math
import torch


class MLA_ROBD:
    def __init__(
        self,
        m: float,
        lam1: float = 1.0,
        theta: float = 0.5,
    ):
        self.m = m
        self.set_params(lam1, theta)

    def set_params(self, lam1: float, theta: float):
        self.theta = theta
        self.lam1 = float(lam1)
        self.lam3 = float(theta) * self.lam1
        self.lam2 = self._optimal_lambda2(m=self.m, lam1=self.lam1, theta=theta)

    def _optimal_lambda2(self, m: float, lam1: float, theta: float) -> float:
        term = 1.0 + (1.0 / m) * theta
        sqrt_term = math.sqrt(term * term + 4.0 / m)
        lam2 = m * lam1 * 0.5 * (sqrt_term + 1.0 - (2.0 / lam1) - theta / m)
        return lam2

    @torch.no_grad()
    def step(self, y_t: float, history, x_ml) -> float:
        x_prev = history["x"][-1]
        v_t = y_t

        num = self.m * y_t + self.lam1 * x_prev + self.lam2 * v_t + self.lam3 * x_ml
        den = self.m + self.lam1 + self.lam2 + self.lam3
        return num / den

    def step_tensor(
        self,
        y_t: torch.Tensor,
        x_prev: torch.Tensor,
        x_ml: torch.Tensor,
    ) -> torch.Tensor:
        v_t = y_t
        # in the case of quadratic, v_t = y_t so seems like lam2 would be redundant
        num = self.m * y_t + self.lam1 * x_prev + self.lam2 * v_t + self.lam3 * x_ml
        den = self.m + self.lam1 + self.lam2 + self.lam3
        return num / den
