from typing import List
from . import util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LSTM:
    def __init__(
        self,
        hidden: int = 64,
        num_lstm_layers: int = 1,
        m: float = 1.0,
        dropout=0.0,
        lr: float = 1e-3,
        max_norm=1.0,
        lr_patience: int = 3,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        plat_threshold: float = 1e-3,
        lr_cooldown: int = 0,
    ):
        self.m = m
        self.max_norm = max_norm

        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden,
                            num_layers=num_lstm_layers,
                            dropout=(dropout if num_lstm_layers > 1 else 0.0),
                            batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

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

        self.reset_mem()

    def reset_mem(self):
        self.mem = None

    def clip_grads(self):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_norm)

    @torch.no_grad()
    def step(self, y_t: float, history) -> float:
        if len(history["y"]) == 0:
            self.reset_mem()

        inputs = torch.as_tensor(
            [y_t, history["x"][-1]], dtype=torch.float32).view([1, 1, 2])

        out, self.mem = self.lstm(inputs, self.mem)  # [1, L, H]
        x_t = self.head(out[:, -1, :]).view(1)

        return float(x_t.item())

    def _current_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def fit(self, sequences: List[np.ndarray],
            epochs: int = 30, unroll: int = 8):
        self.lstm.train()
        self.head.train()

        for epoch in range(1, epochs + 1):
            np.random.shuffle(sequences)
            total_epoch_loss = 0.0

            for seq in sequences:
                ys = torch.tensor(seq, dtype=torch.float32)  # [T]
                T = int(ys.shape[0])

                x_hist: list[float] = []
                hit_hist: list[float] = []
                mov_hist: list[float] = []
                tot_hist: list[float] = []

                self.reset_mem()

                loss_buf: list[torch.Tensor] = []

                for t in range(T):
                    y_t = ys[t]

                    # [1, L, 5]

                    if t == 0:
                        x_prev = torch.as_tensor(0.0, dtype=torch.float32)
                        x_t = torch.as_tensor([0.0], dtype=torch.float32)
                    else:
                        inputs = torch.as_tensor(
                            [y_t, x_hist[-1]]).view([1, 1, 2])

                        x_prev = torch.tensor([x_hist[-1]])
                        out, self.mem = self.lstm(
                            inputs, self.mem)          # [1, L, H]
                        x_t = self.head(out[:, -1, :]).view(1)          # [1]

                    hit, move, tot = util.soco_step_loss(x_t, y_t, x_prev, self.m)
                    loss_buf.append(tot)

                    x_hist.append(float(x_t.item()))
                    hit_hist.append(float(hit.item()))
                    mov_hist.append(float(move.item()))
                    tot_hist.append(float(tot.item()))

                    if (t + 1) % unroll == 0 or (t + 1) == T:
                        batch_loss = torch.stack(loss_buf).sum()
                        self.opt.zero_grad()
                        batch_loss.backward()
                        self.clip_grads()
                        self.opt.step()

                        total_epoch_loss += float(batch_loss.item())
                        loss_buf.clear()

                        self.mem = (self.mem[0].detach(), self.mem[1].detach())

                    # print(
                    #     f"[LSTM] epoch {epoch:02d} | "
                    #     f"loss={total_epoch_loss/len(sequences):.4f} | "
                    #     f"hit={sum(hit_hist):.4f} move={
                    #         sum(mov_hist):.4f} total={sum(tot_hist):.4f}"
                    # )

            avg_epoch_loss = total_epoch_loss / max(1, len(sequences))
            self.scheduler.step(avg_epoch_loss)

            print(f"[LSTM] epoch {epoch:02d}  loss {
                  avg_epoch_loss:.4f}  lr {self._current_lr():.6g}")
