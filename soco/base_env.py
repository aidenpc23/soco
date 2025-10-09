class SOCOEnvironment:
    def __init__(self, algorithm, m=5.0):
        self.algorithm = algorithm
        self.m = m
        self._reset_history()
        self.cum_history = {"x": [], "y": [], "hitting": [], "movement": [], "total": []}

    def _reset_history(self):
        self.history = {
            "x": [0.0],
            "y": [],
            "hitting": [],
            "movement": [],
            "total": [],
        }

    def hit_cost(self, x_t, y_t):
        return 0.5 * self.m * (x_t - y_t) ** 2

    def move_cost(self, x_t, x_prev):
        return 0.5 * (x_t - x_prev) ** 2

    def total_cost(self):
        return sum(self.history["total"])

    def run(self, data):
        for seq in data:
            for t, y_t in enumerate(seq):
                x_prev = self.history["x"][-1]
                x_t = self.algorithm.step(y_t, self.history)

                hit = self.hit_cost(x_t, y_t)
                move = self.move_cost(x_t, x_prev)
                total = hit + move
                labels = ["x", "y", "hitting", "movement", "total"]
                for k, v in zip(labels, [x_t, y_t, hit, move, total]):
                    self.history[k].append(v)

            for key in self.history:
                self.cum_history[key].extend(self.history[key])

            self._reset_history()

        return self.cum_history
