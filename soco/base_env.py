from soco.algorithms.util import oracle_history


class SOCOTestEnv:
    def __init__(self, algorithm, m=1.0):
        self.algorithm = algorithm
        self.m = m
        self._reset_history()
        self.cum_history = {"x": [], "y": [],
                            "hitting": [], "movement": [],
                            "total": [], "cr": []}

    def _reset_history(self):
        self.history = {
            "x": [],
            "y": [],
            "hitting": [],
            "movement": [],
            "total": [],
            "cr": []
        }

    def hit_cost(self, x_t, y_t):
        return 0.5 * self.m * (x_t - y_t) ** 2

    def move_cost(self, x_t, x_prev):
        return 0.5 * (x_t - x_prev) ** 2

    def total_cost(self):
        return sum(self.history["total"])

    def run(self, data):
        cr = 0.0
        for ys in data:
            labels = ["x", "y", "hitting", "movement", "total"]

            o_hist = oracle_history(self.m, ys)

            y0 = ys[0]
            hit = self.hit_cost(0.0, y0)
            for k, v in zip(labels, [0.0, y0, hit, 0.0, hit]):
                self.history[k].append(v)

            for t, y_t in enumerate(ys):
                if t == 0:
                    continue

                x_prev = self.history["x"][-1]
                x_t = self.algorithm.step(y_t, self.history)

                hit = self.hit_cost(x_t, y_t)
                move = self.move_cost(x_t, x_prev)
                total = hit + move
                for k, v in zip(labels, [x_t, y_t, hit, move, total]):
                    self.history[k].append(v)

            ncr = sum(self.history["total"]) / sum(o_hist["total"])
            cr = max(cr, ncr)

            for key in self.history:
                vals = self.history[key]
                self.cum_history[key].extend(vals)

            self._reset_history()

        self.cum_history["cr"] = cr

        return self.cum_history


class OracleEnv:
    def __init__(self, m=1.0):
        self.m = m
        self.cum_history = {"x": [], "y": [],
                            "hitting": [], "movement": [],
                            "total": [], "cr": []}

    def run(self, data):
        for ys in data:
            history = oracle_history(self.m, ys)
            for key in history:
                vals = history[key]
                self.cum_history[key].extend(vals)
        self.cum_history["cr"] = 1.0
        return self.cum_history
