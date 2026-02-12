import numpy as np
import torch

class StreamingRegressionMetrics:
    """
    Streaming accumulator for MSE/MAE with optional mask.
    Aggregates across many batches -> returns epoch-level metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None):
        """
        preds:  [B, T] or [B, T, H]
        labels: same broadcastable shape
        mask:   [B, T] or broadcastable; values >0 are valid
        """
        preds = preds.detach()
        labels = labels.detach()

        # Move mask to correct device if present
        if mask is not None:
            mask = mask.to(device=preds.device)

            # Expand mask to match preds dims
            while mask.dim() < preds.dim():
                mask = mask.unsqueeze(-1)
            if mask.shape[-1] == 1 and preds.shape[-1] != 1:
                mask = mask.expand_as(preds)

            valid = mask > 0
            if valid.any():
                diff = (preds - labels)[valid]
            else:
                return
        else:
            diff = (preds - labels).reshape(-1)

        # Accumulate
        self.sum_sq += torch.sum(diff * diff).item()
        self.sum_abs += torch.sum(torch.abs(diff)).item()
        self.count += diff.numel()

    def compute(self):
        if self.count == 0:
            return {"mse": np.nan, "mae": np.nan}
        return {"mse": self.sum_sq / self.count, "mae": self.sum_abs / self.count}
