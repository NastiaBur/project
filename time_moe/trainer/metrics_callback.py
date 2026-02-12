import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainerCallback


class LogPlotCallback(TrainerCallback):
    """
    Collects metrics from HuggingFace Trainer logs and plots them.
    Works best when Trainer is created with `compute_metrics=...`,
    so eval logs contain: eval_mse, eval_mae (and usually eval_loss).
    """

    def __init__(self, plot_path: str | None = None, plot_every_log: bool = True):
        self.plot_path = plot_path
        self.plot_every_log = plot_every_log
        self.rows = []  # each row: {"step":..., "epoch":..., "eval_mse":..., ...}

    def on_train_begin(self, args, state, control, **kwargs):
        self.rows = []
        if self.plot_path:
            os.makedirs(os.path.dirname(self.plot_path) or ".", exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        # We only care about eval metrics (produced by Trainer.evaluate with compute_metrics)
        has_eval = any(k.startswith("eval_") for k in logs.keys())
        if not has_eval:
            return

        row = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else np.nan,
        }
        for k, v in logs.items():
            if k.startswith("eval_"):
                row[k] = float(v) if v is not None else np.nan

        self.rows.append(row)

        if self.plot_path and self.plot_every_log:
            self._plot()

    def on_train_end(self, args, state, control, **kwargs):
        if self.plot_path:
            self._plot()

    def _plot(self):
        if not self.rows:
            return

        steps = [r["step"] for r in self.rows]

        def series(key):
            return [r.get(key, np.nan) for r in self.rows]

        # MSE
        if any(np.isfinite(series("eval_mse"))):
            plt.figure(figsize=(10, 6))
            plt.plot(steps, series("eval_mse"), marker="o", label="eval_mse")
            plt.xlabel("Step")
            plt.ylabel("MSE")
            plt.title("Eval MSE over training")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path.replace(".png", "_eval_mse.png"), dpi=200)
            plt.close()

        # MAE
        if any(np.isfinite(series("eval_mae"))):
            plt.figure(figsize=(10, 6))
            plt.plot(steps, series("eval_mae"), marker="o", label="eval_mae")
            plt.xlabel("Step")
            plt.ylabel("MAE")
            plt.title("Eval MAE over training")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path.replace(".png", "_eval_mae.png"), dpi=200)
            plt.close()

        # Optional: eval_loss
        if any(np.isfinite(series("eval_loss"))):
            plt.figure(figsize=(10, 6))
            plt.plot(steps, series("eval_loss"), marker="o", label="eval_loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Eval loss over training")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path.replace(".png", "_eval_loss.png"), dpi=200)
            plt.close()
