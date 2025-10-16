from typing import List
from . import util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .mla_robd import MLA_ROBD


class EC_L2O_STEP:
    def __init__(
        self,
        m: float = 1.0,
        hidden: int = 64,
        num_lstm_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        max_norm: float = 1.0,
        lr_patience: int = 3,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        plat_threshold: float = 1e-3,
        lr_cooldown: int = 0,
        input_window: int = 1,
        lam1: float = 1.0,
        theta: float = 0.5,
        u: float = 0.5,
    ):
        self.m = float(m)
        self.input_window = int(input_window)
        self.max_norm = float(max_norm)
        self.u = u

        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden,
                            num_layers=num_lstm_layers,
                            dropout=(dropout if num_lstm_layers > 1 else 0.0),
                            batch_first=True)
        self.head = nn.Linear(hidden, 1)

        self.params = list(self.lstm.parameters()) + \
            list(self.head.parameters())
        self.opt = optim.Adam(self.params, lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=plat_threshold,
            threshold_mode="rel",
            cooldown=lr_cooldown,
            min_lr=min_lr,
        )

        self.mla_robd = MLA_ROBD(
            m=self.m,
            lam1=lam1,
            theta=theta,
        )

        self.reset_mem()

    def reset_mem(self):
        self.mem = None

    def clip_grads(self):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_norm)

    @torch.no_grad()
    def step(self, y_t: float, history) -> float:
        if len(history["y"]) == 0:
            self.reset_mem()

        feats = self._build_window(
            y_hist=history["y"],
            x_hist=history["x"],
            y_t=y_t,
        )  # [1, L,2]

        out, self.mem = self.lstm(feats, self.mem)  # [1, L, H]
        x_ml = self.head(out[:, -1, :]).view(1)

        x_prev = torch.tensor([history["x"][-1]])
        y_t_tensor = torch.tensor([y_t])

        x_t = self.mla_robd.step_tensor(y_t_tensor, x_prev, x_ml)
        return float(x_t.item())

    def _build_window(self, y_hist, x_hist, y_t) -> torch.Tensor:
        n_completed = len(y_hist)
        max_hist_rows = max(0, self.input_window - 1)
        first_j = max(0, n_completed - max_hist_rows)

        rows = []
        for j in range(first_j, max(0, n_completed)):
            rows.append([float(y_hist[j]), float(x_hist[j + 1])])

        rows.append([float(y_t), float(x_hist[-1])])

        feats = torch.tensor(
            rows, dtype=torch.float32).unsqueeze(0)  # [1, L, 2]
        return feats

    def _current_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def fit(self, sequences: List[np.ndarray], epochs: int = 30, unroll: int = 8):
        self.lstm.train()
        self.head.train()

        for epoch in range(1, epochs + 1):
            np.random.shuffle(sequences)
            total_epoch_loss = 0.0
            total_steps_epoch = 0

            for seq in sequences:
                ys = torch.tensor(seq, dtype=torch.float32)
                T = int(ys.shape[0])
                o_hist = util.oracle_history(self.m, seq)

                y_hist: list[float] = []
                x_hist: list[float] = []

                self.reset_mem()

                loss_buf: list[torch.Tensor] = []

                for t in range(T):
                    y_t = ys[t].view(1)

                    if t == 0:
                        x_prev = torch.as_tensor(0.0, dtype=torch.float32)
                        x_t = torch.as_tensor([0.0], dtype=torch.float32)
                        x_ml = torch.as_tensor([0.0], dtype=torch.float32)
                    else:
                        feats = self._build_window(
                            y_hist=y_hist,
                            x_hist=x_hist,
                            y_t=float(y_t.item()),
                        )

                        out, self.mem = self.lstm(feats, self.mem)  # [1, L, H]
                        x_ml = self.head(out[:, -1, :]).view(1)

                        x_prev = torch.as_tensor(
                            [x_hist[-1]], dtype=torch.float32)
                        x_t = self.mla_robd.step_tensor(y_t, x_prev, x_ml)

                    loss = util.ec_l2o_step_loss(
                        self.u, x_ml, o_hist["x"][t], x_t, x_prev,
                        y_t, o_hist["total"][t], self.m)
                    loss_buf.append(loss)

                    y_hist.append(float(y_t.item()))
                    x_hist.append(float(x_t.item()))

                    if (t + 1) % unroll == 0 or (t + 1) == T:
                        batch_loss = torch.stack(loss_buf).sum()
                        self.opt.zero_grad()
                        batch_loss.backward()
                        self.clip_grads()
                        self.opt.step()

                        total_steps_epoch += len(loss_buf)
                        total_epoch_loss += float(batch_loss.item())
                        loss_buf.clear()

                        self.mem = (self.mem[0].detach(), self.mem[1].detach())

            avg_epoch_loss = total_epoch_loss / max(1, total_steps_epoch)
            self.scheduler.step(avg_epoch_loss)

            print(f"[EC-L2O] epoch {epoch:02d}  loss {
                  avg_epoch_loss:.4f}  lr {self._current_lr():.6g}")
