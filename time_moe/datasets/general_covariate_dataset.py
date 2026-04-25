import json
import os
import numpy as np
from .ts_dataset import TimeSeriesDataset
from torch.utils.data import Dataset



class GeneralCovariateDataset(TimeSeriesDataset):
    def __init__(self, data_path):
        self.data = self.read_jsonl_to_list(data_path)
        self.num_tokens = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        item = self.data[seq_idx]

        sequence = np.asarray(item["sequence"], dtype=np.float32)  # target/main AR series
        main_features = np.asarray(item["main_features"], dtype=np.float32)
        macro_features = np.asarray(item["macro_features"], dtype=np.float32)

        if len(sequence) != len(main_features) or len(sequence) != len(macro_features):
            raise ValueError(
                f"Length mismatch at idx={seq_idx}: "
                f"len(sequence)={len(sequence)}, "
                f"len(main_features)={len(main_features)}, "
                f"len(macro_features)={len(macro_features)}"
            )

        return {
            "sequence": sequence,
            "main_features": main_features,
            "macro_features": macro_features,
        }

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum(len(self[i]["sequence"]) for i in range(len(self)))
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        return len(self[seq_idx]["sequence"])

    @staticmethod
    def is_valid_path(data_path):
        return os.path.isfile(data_path) and data_path.endswith(".jsonl")

    @staticmethod
    def read_jsonl_to_list(jsonl_fn):
        with open(jsonl_fn, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]


class GeneralCovariateEvalDataset(Dataset):
    def __init__(self, data_path, context_length: int, prediction_length: int = 1):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.items = []

        with open(data_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        for row in rows:
            seq = np.asarray(row["sequence"], dtype=np.float32)
            main = np.asarray(row["main_features"], dtype=np.float32)
            macro = np.asarray(row["macro_features"], dtype=np.float32)
            dates = row["dates"]

            if len(seq) != len(main) or len(seq) != len(macro) or len(seq) != len(dates):
                raise ValueError(
                    f"Length mismatch in row for ticker={row.get('ticker')}"
                )

            total_window = context_length + prediction_length
            if len(seq) < total_window:
                continue

            for start in range(0, len(seq) - total_window + 1):
                end_ctx = start + context_length
                end_all = end_ctx + prediction_length

                self.items.append({
                    "main_features": main[start:end_ctx],
                    "macro_features": macro[start:end_ctx],
                    "labels": seq[end_ctx:end_all],
                    "attention_mask": np.ones(context_length, dtype=np.int64),
                    "target_dates": dates[end_ctx:end_all],
                    "ticker": row.get("ticker", "UNK"),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



class GeneralNoCovariateEvalDataset(Dataset):
    def __init__(self, data_path, context_length: int, prediction_length: int = 1):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.items = []

        with open(data_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        for row in rows:
            seq = np.asarray(row["sequence"], dtype=np.float32)
            main = np.asarray(row["main_features"], dtype=np.float32)
            dates = row["dates"]

            if len(seq) != len(main) or len(seq) != len(dates):
                raise ValueError(
                    f"Length mismatch in row for ticker={row.get('ticker')}"
                )

            total_window = context_length + prediction_length
            if len(seq) < total_window:
                continue

            for start in range(0, len(seq) - total_window + 1):
                end_ctx = start + context_length
                end_all = end_ctx + prediction_length

                self.items.append({
                    "main_features": main[start:end_ctx],
                    "labels": seq[end_ctx:end_all],
                    "attention_mask": np.ones(context_length, dtype=np.int64),
                    "target_dates": dates[end_ctx:end_all],
                    "ticker": row.get("ticker", "UNK"),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
