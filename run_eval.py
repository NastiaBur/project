#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import sys
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.models.configuration_time_moe import TimeMoeConfig

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset
from time_moe.datasets.general_covariate_dataset import GeneralCovariateEvalDataset


def plot_eval_metrics(mse_per_batch, mae_per_batch, loss_per_batch, save_path):
    """Plot MSE, MAE and loss per batch and save to save_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return

    steps = np.arange(1, len(mse_per_batch) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mse_per_batch, label="MSE", color="C0", alpha=0.9)
    ax.plot(steps, mae_per_batch, label="MAE", color="C1", alpha=0.9)
    ax.plot(steps, loss_per_batch, label="Loss (Smooth L1)", color="C2", alpha=0.9)
    ax.set_xlabel("Batch (step)")
    ax.set_ylabel("Metric value")
    ax.set_title("Evaluation metrics per batch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_dir = os.path.dirname(save_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Saved metrics plot to {save_path}")


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class CovariateEvalCollator:
    def __call__(self, features):

        batch = {}

        tensor_keys = {"main_features", "macro_features", "labels", "attention_mask"}
        list_keys = {
            "target_dates",
            "ticker",
            "forecast_origin_date",
            "window_start_date",
            "eval_start_date",
        }

        for key in features[0].keys():
            vals = [f[key] for f in features]
            if key in tensor_keys:
                dtype = torch.float32 if key != "attention_mask" else torch.long
                batch[key] = torch.tensor(np.array(vals), dtype=dtype)
            elif key in list_keys:
                batch[key] = vals
            else:
                batch[key] = vals

        return batch


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, temporal_mixer=None, **kwargs):
        # Load config first
        if temporal_mixer is not None:
            config = TimeMoeConfig.from_pretrained(model_path)
            config.temporal_mixer = temporal_mixer
            # keep HF-compatible attention implementation
            if temporal_mixer == "mamba":
                config._attn_implementation = "eager"
        else:
            config = TimeMoeConfig.from_pretrained(model_path)

        try:
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                config=config,
                dtype="auto",
            )
        except Exception as e:
            logging.warning(f"Falling back to AutoModelForCausalLM due to: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                trust_remote_code=True,
            )

        model = model.to(device)
        model.eval()

        logging.info(
            f'>>> Model dtype: {model.dtype}; '
            f'Temporal mixer: {getattr(model.config, "temporal_mixer", "N/A")}; '
            f'Attention: {getattr(model.config, "_attn_implementation", "N/A")}'
        )

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.temporal_mixer = getattr(model.config, "temporal_mixer", None)

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        if "main_features" in batch:
            main_features = batch["main_features"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Fix for models trained with a different main_input_size than dataset provides
            expected_main_dim = getattr(model.config, "main_input_size", None)
            if expected_main_dim is not None and main_features.shape[-1] != expected_main_dim:
                if expected_main_dim == 1 and main_features.shape[-1] >= 1:
                    # keep only the first channel, usually firm return
                    main_features = main_features[..., :1]
                else:
                    raise ValueError(
                        f"Model expects main_input_size={expected_main_dim}, "
                        f"but batch main_features has shape {main_features.shape}"
                    )

            if torch.is_floating_point(main_features):
                main_features = main_features.to(model.dtype)

            macro_features = batch.get("macro_features", None)
            if macro_features is not None:
                macro_features = macro_features.to(device)
                if torch.is_floating_point(macro_features):
                    macro_features = macro_features.to(model.dtype)

            outputs = model(
                main_features=main_features,
                macro_features=macro_features,
                attention_mask=attention_mask,
                max_horizon_length=prediction_length,
                return_dict=True,
                use_cache=False,
            )

            preds = outputs.logits  # expected [B, seq_len, prediction_length * input_size]
            preds = preds[:, -1, :]  # last context position

            # input_size = 1 -> [B, prediction_length]
            if preds.ndim == 2:
                if preds.shape[-1] == prediction_length:
                    pass
                elif preds.shape[-1] % prediction_length == 0:
                    preds = preds.view(preds.shape[0], prediction_length, -1)
                    if preds.shape[-1] == 1:
                        preds = preds.squeeze(-1)
                else:
                    raise ValueError(
                        f"Unexpected preds shape {preds.shape} for prediction_length={prediction_length}"
                    )

            if labels.ndim == 3 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)

            return preds, labels

        # old path
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)

        if torch.is_floating_point(inputs):
            inputs = inputs.to(model.dtype)

        outputs = model(
            input_ids=inputs,
            max_horizon_length=prediction_length,
            return_dict=True,
            use_cache=False,
        )

        preds = outputs.logits
        preds = preds[:, -1, :]

        if preds.ndim == 2:
            if preds.shape[-1] == prediction_length:
                pass
            elif preds.shape[-1] % prediction_length == 0:
                preds = preds.view(preds.shape[0], prediction_length, -1)
                if preds.shape[-1] == 1:
                    preds = preds.squeeze(-1)

        if labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)

        return preds, labels

def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length

    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    if torch.cuda.is_available():
        try:
            setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        except Exception as e:
            print('Error: ', f'Setup nccl fail, so set device to cpu: {e}')
            device = 'cpu'
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # evaluation
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    model = TimeMoE(
        args.model,
        device,
        context_length=context_length,
        prediction_length=prediction_length,
        temporal_mixer=getattr(args, 'temporal_mixer', None),
    )
    if args.data.endswith(".jsonl") and args.use_covariates:
        dataset = GeneralCovariateEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
            window_stride=args.window_stride if args.window_stride is not None else prediction_length,
            eval_only=True,
        )
        collator = CovariateEvalCollator()
    elif args.data.endswith('.csv'):
        dataset = BenchmarkEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        collator = None
    else:
        dataset = GeneralEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        collator = None

    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
    else:
        sampler = None
    num_workers = 2 if device != "cpu" else 0

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    test_dl = test_dl = DataLoader(
        **dl_kwargs,
        collate_fn=collator,
    )

    acc_count = 0
    mse_per_batch = []
    mae_per_batch = []
    loss_per_batch = []
    all_preds = [] if (getattr(args, 'predictions_path', None) and (not is_dist or rank == 0)) else None
    all_labels = [] if all_preds is not None else None
    all_dates = [] if all_preds is not None else None
    all_tickers = [] if all_preds is not None else None
    all_forecast_origin_dates = [] if all_preds is not None else None
    all_window_start_dates = [] if all_preds is not None else None
    all_eval_start_dates = [] if all_preds is not None else None
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            preds, labels = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            # Per-batch metrics for plotting (rank 0 only when distributed)
            if not is_dist or rank == 0:
                mse_b = torch.mean((preds - labels) ** 2).item()
                mae_b = torch.mean(torch.abs(preds - labels)).item()
                loss_b = torch.nn.functional.smooth_l1_loss(preds, labels, reduction="mean").item()
                mse_per_batch.append(mse_b)
                mae_per_batch.append(mae_b)
                loss_per_batch.append(loss_b)
                if all_preds is not None:
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

                    if "target_dates" in batch:
                        all_dates.extend(batch["target_dates"])
                    if "ticker" in batch:
                        all_tickers.extend(batch["ticker"])
                    if "forecast_origin_date" in batch:
                        all_forecast_origin_dates.extend(batch["forecast_origin_date"])
                    if "window_start_date" in batch:
                        all_window_start_dates.extend(batch["window_start_date"])
                    if "eval_start_date" in batch:
                        all_eval_start_dates.extend(batch["eval_start_date"])
                    

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'{rank} - {ret_metric}')

    metric_tensors = []
    for metric in metric_list:
        if isinstance(metric.value, torch.Tensor):
            metric_tensors.append(metric.value.detach().float().to(device))
        else:
            metric_tensors.append(torch.tensor(metric.value, dtype=torch.float32, device=device))

    metric_tensors.append(torch.tensor(acc_count, dtype=torch.float32, device=device))

    stat_tensor = torch.stack(metric_tensors)

    if is_dist:
        dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)

    all_stat = stat_tensor

    if rank == 0:
        item = {
            'model': args.model,
            'data': args.data,
            'context_length': args.context_length,
            'prediction_length': args.prediction_length,
        }

        count = all_stat[-1]
        for i, metric in enumerate(metric_list):
            val = all_stat[i] / count
            item[metric.name] = float(val.cpu().numpy())
        logging.info(item)

        if getattr(args, 'plot_path', None):
            plot_eval_metrics(mse_per_batch, mae_per_batch, loss_per_batch, args.plot_path)

        if getattr(args, 'predictions_path', None) and all_preds is not None:
            preds_arr = np.concatenate([p.double().numpy() for p in all_preds], axis=0)
            labels_arr = np.concatenate([l.double().numpy() for l in all_labels], axis=0)

            if all_dates is not None:
                flat_dates = []
                for x in all_dates:
                    if isinstance(x, (list, tuple)):
                        flat_dates.extend(x)
                    else:
                        flat_dates.append(x)
                dates_arr = np.array(flat_dates, dtype=object)
            else:
                dates_arr = None
            
            if all_tickers is not None:
                tickers_arr = np.array(all_tickers, dtype=object)
            else:
                tickers_arr = None
            if all_forecast_origin_dates is not None:
                forecast_origin_dates_arr = np.array(all_forecast_origin_dates, dtype=object)
            else:
                forecast_origin_dates_arr = None

            if all_window_start_dates is not None:
                window_start_dates_arr = np.array(all_window_start_dates, dtype=object)
            else:
                window_start_dates_arr = None

            if all_eval_start_dates is not None:
                eval_start_dates_arr = np.array(all_eval_start_dates, dtype=object)
            else:
                eval_start_dates_arr = None

            out_path = args.predictions_path.strip()
            if not out_path.endswith('.npz'):
                out_path = out_path + '.npz'
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            save_dict = {
                "predictions": preds_arr,
                "labels": labels_arr,
            }
            if dates_arr is not None:
                save_dict["dates"] = dates_arr
            if tickers_arr is not None:
                save_dict["tickers"] = tickers_arr
            if forecast_origin_dates_arr is not None:
                save_dict["forecast_origin_dates"] = forecast_origin_dates_arr
            if window_start_dates_arr is not None:
                save_dict["window_start_dates"] = window_start_dates_arr
            if eval_start_dates_arr is not None:
                save_dict["eval_start_dates"] = eval_start_dates_arr

            np.savez(out_path, **save_dict)
            logging.info(
                f"Saved predictions/labels to {out_path} "
                f"(predictions {preds_arr.shape}, labels {labels_arr.shape})"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Maple728/TimeMoE-50M',
        help='Model path'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Benchmark data path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size of evaluation'
    )
    parser.add_argument(
        '--context_length', '-c',
        type=int,
        help='Context length'
    )
    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=96,
        help='Prediction length'
    )
    parser.add_argument(
        '--plot_path',
        type=str,
        default=None,
        help='If set, save a plot of MSE, MAE and loss per batch to this path (e.g. plots/eval_metrics.png)'
    )
    parser.add_argument(
        '--predictions_path',
        type=str,
        default=None,
        help='If set, save predictions and labels to this path as a .npz file (e.g. results/preds.npz). Load with np.load(path)[\"predictions\"] and [\"labels\"]'
    )
    parser.add_argument(
        '--temporal_mixer',
        type=str,
        default=None,
        choices=['attn', 'mamba'],
        help='Override architecture when loading the checkpoint. Usually not needed: the model uses the config saved in the checkpoint. Set to "mamba" if the checkpoint was trained with Mamba but config.json has temporal_mixer="attn".'
    )
    parser.add_argument(
        "--use_covariates",
        action="store_true",
        help="Use covariate-aware evaluation dataset and model inputs"
    )

    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Stride between evaluation windows. Defaults to prediction_length when use_covariates eval_only is enabled."
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="If set, only create windows whose targets lie in the eval period defined by eval_start_date/eval_length."
    )
    args = parser.parse_args()
    if args.context_length is None:
        if args.prediction_length == 96:
            args.context_length = 512
        elif args.prediction_length == 192:
            args.context_length = 1024
        elif args.prediction_length == 336:
            args.context_length = 2048
        elif args.prediction_length == 720:
            args.context_length = 3072
        else:
            args.context_length = args.prediction_length * 4
    evaluate(args)
