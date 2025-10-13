from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def soco_seq_loss(xs: torch.Tensor, ys: torch.Tensor, m: float) -> torch.Tensor:
    hit = 0.5 * m * (xs - ys).pow(2)
    x_prev = torch.cat([torch.zeros_like(xs[:1]), xs[:-1]], dim=0)
    move = 0.5 * (xs - x_prev).pow(2)
    return (hit + move).sum()


def soco_step_loss(x_t: torch.Tensor, y_t: torch.Tensor, x_prev: torch.Tensor, m: float):
    hit = 0.5 * m * (x_t - y_t).pow(2)
    move = 0.5 * (x_t - x_prev).pow(2)
    return (hit, move, hit + move)


class LSTM:
    def __init__(
        self,
        hidden: int = 64,
        num_lstm_layers: int = 1,
        m: float = 5.0,
        dropout=0.0,
        lr: float = 1e-3,
        input_window: int = 8,
        max_norm=1.0,
        lr_patience: int = 3,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        plat_threshold: float = 1e-3,
        lr_cooldown: int = 0,
    ):
        self.m = m
        self.input_window = input_window
        self.max_norm = max_norm
        self.device = torch.device("cpu")

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

        feats = self._build_window(
            y_hist=history["y"],
            x_hist=history["x"],
            y_t=y_t,
        )
        out, self.mem = self.lstm(feats, self.mem)
        x_t = self.head(out[:, -1, :]).view(1)  # [1]

        return float(x_t.item())

    # figured out that its best to not use a window input
    # presumably because LSTM has a built in memory so it would be redundant
    def _build_window(
            self,
            y_hist,
            x_hist,
            y_t,
    ) -> torch.Tensor:
        n_completed = len(y_hist)
        max_hist_rows = max(0, self.input_window-1)
        first_j = max(0, n_completed - max_hist_rows)

        rows = []
        for j in range(first_j, max(0, n_completed)):
            rows.append([
                float(y_hist[j]),
                float(x_hist[j + 1]),
            ])

        rows.append([
            float(y_t),
            float(x_hist[-1]),
        ])

        feats = torch.tensor(
            rows, dtype=torch.float32).unsqueeze(0)  # [1, L, 5]
        return feats

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
                ys = torch.tensor(seq, dtype=torch.float32,
                                  device=self.device)  # [T]
                T = int(ys.shape[0])

                y_hist: list[float] = []
                x_hist: list[float] = [0.0]
                hit_hist: list[float] = []
                mov_hist: list[float] = []
                tot_hist: list[float] = []

                self.reset_mem()

                loss_buf: list[torch.Tensor] = []

                for t in range(T):
                    y_t = ys[t]

                    # [1, L, 5]
                    feats = self._build_window(
                        y_hist=y_hist,
                        x_hist=x_hist,
                        y_t=float(y_t.item()),
                    )

                    out, self.mem = self.lstm(
                        feats, self.mem)          # [1, L, H]
                    x_t = self.head(out[:, -1, :]).view(1)          # [1]

                    x_prev = torch.tensor([x_hist[-1]])
                    hit, move, tot = soco_step_loss(x_t, y_t, x_prev, self.m)
                    loss_buf.append(tot)

                    y_hist.append(float(y_t.item()))
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
