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

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


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


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, temporal_mixer=None, **kwargs):
        try:
            # Ensure black_mamba is on sys.path before loading Mamba model (must be before first time_moe import)
            if temporal_mixer == 'mamba':
                _eval_dir = os.path.dirname(os.path.abspath(__file__))
                _black_mamba = os.path.join(_eval_dir, 'black_mamba')
                if os.path.exists(_black_mamba) and _black_mamba not in sys.path:
                    sys.path.insert(0, _black_mamba)
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction, MAMBA_AVAILABLE
            from time_moe.models.configuration_time_moe import TimeMoeConfig
            if temporal_mixer == 'mamba' and not MAMBA_AVAILABLE:
                raise ImportError(
                    "Mamba model requested (--temporal_mixer mamba) but black_mamba could not be loaded. "
                    "Run from the project root so 'black_mamba' is found, install black_mamba deps (e.g. causal_conv1d, einops), "
                    "and ensure CUDA extensions are built if needed."
                )
            load_kw = dict(device_map=device, dtype='auto')
            if temporal_mixer is not None:
                config = TimeMoeConfig.from_pretrained(model_path, temporal_mixer=temporal_mixer)
                load_kw['config'] = config
            model = TimeMoeForPrediction.from_pretrained(model_path, **load_kw)
        except ImportError as e:
            if "mamba" in str(e).lower() or "MAMBA" in str(e):
                raise
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                dtype='auto',
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                dtype='auto',
                trust_remote_code=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Temporal mixer: {getattr(model.config, "temporal_mixer", "N/A")}; Attention: {model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        outputs = model.generate(
            inputs=batch['inputs'].to(device).to(model.dtype),
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
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
    if args.data.endswith('.csv'):
        dataset = BenchmarkEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
    else:
        dataset = GeneralEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )

    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
    else:
        sampler = None
    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
    )

    acc_count = 0
    mse_per_batch = []
    mae_per_batch = []
    loss_per_batch = []
    all_preds = [] if (getattr(args, 'predictions_path', None) and (not is_dist or rank == 0)) else None
    all_labels = [] if all_preds is not None else None
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

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'{rank} - {ret_metric}')

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

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
            preds_arr = np.concatenate([p.numpy() for p in all_preds], axis=0)
            labels_arr = np.concatenate([l.numpy() for l in all_labels], axis=0)
            out_path = args.predictions_path.strip()
            if not out_path.endswith('.npz'):
                out_path = out_path + '.npz'
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            np.savez(out_path, predictions=preds_arr, labels=labels_arr)
            logging.info(f'Saved predictions and labels to {out_path} (shape: predictions {preds_arr.shape}, labels {labels_arr.shape})')


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
