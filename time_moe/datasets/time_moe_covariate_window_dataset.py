import numpy as np
from time_moe.datasets.ts_dataset import TimeSeriesDataset


class TimeMoECovariateWindowDataset:
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0, stride: int = None):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size

        self.sub_seq_indexes = []
        for seq_idx in range(len(self.dataset)):
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            if n_points < 2:
                continue

            self.sub_seq_indexes.append((seq_idx, 0))
            for offset_idx in range(self.stride, n_points - self.window_size_plus_one + 1, self.stride):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        item = self.dataset[seq_i]

        seq = item["sequence"][offset_i: offset_i + self.window_size_plus_one]                # [W+1]
        main = item["main_features"][offset_i: offset_i + self.window_size_plus_one]          # [W+1, d_main]
        macro = item["macro_features"][offset_i: offset_i + self.window_size_plus_one]        # [W+1, d_macro]

        seq = np.asarray(seq, dtype=np.float32)
        main = np.asarray(main, dtype=np.float32)
        macro = np.asarray(macro, dtype=np.float32)

        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)

        n_pad = self.window_size_plus_one - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), mode="constant", constant_values=0)
            main = np.pad(main, ((0, n_pad), (0, 0)), mode="constant", constant_values=0)
            macro = np.pad(macro, ((0, n_pad), (0, 0)), mode="constant", constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), mode="constant", constant_values=0)

        return {
            "main_features": main[:-1],   # [W, d_main]
            "macro_features": macro[:-1], # [W, d_macro]
            "labels": seq[1:],            # [W]
            "loss_masks": loss_mask,      # [W]
        }
