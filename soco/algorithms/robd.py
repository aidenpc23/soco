import math


class R_OBD_L2:
    def __init__(self, m: float):
        self.m = m
        self.lam1 = 2 / (1 + math.sqrt(1 + 4 / m))
        print(f"Lamda1 set at: {self.lam1}")
        # self.lam2 = 0.0 - dont need because l2

    def step(self, y_t: float, history) -> float:
        x_prev = history["x"][-1]

        # using derivative to find argmin
        # interesting that when written like this it is the same as
        # a weighted average where m and lam1 are weights and
        # y_t and prev_x are values
        x_t = (self.m * y_t + self.lam1 * x_prev) / (self.m + self.lam1)
        return x_t
