import os
import math
import random
from functools import reduce
from operator import mul

# Import comet_ml before torch to avoid warnings
import comet_ml  # noqa: F401

import torch

from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
from time_moe.datasets.general_covariate_dataset import GeneralCovariateDataset
from time_moe.datasets.time_moe_covariate_window_dataset import TimeMoECovariateWindowDataset
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
from time_moe.utils.dist_util import get_world_size
from time_moe.utils.log_util import logger, log_in_local_rank_0

import numpy as np


class CovariateDataCollator:
    def __call__(self, features):
        batch = {}
        keys = features[0].keys()

        for key in keys:
            values = [f[key] for f in features]

            if key in {"main_features", "labels"}:
                batch[key] = torch.tensor(np.array(values), dtype=torch.float32)
            elif key == "macro_features":
                batch[key] = torch.tensor(np.array(values), dtype=torch.float32)
            elif key in {"loss_masks", "attention_mask"}:
                batch[key] = torch.tensor(np.array(values), dtype=torch.long)
            else:
                batch[key] = torch.tensor(np.array(values))

        if "attention_mask" not in batch:
            if "loss_masks" in batch:
                batch["attention_mask"] = (batch["loss_masks"] > 0).long()
            elif "labels" in batch:
                batch["attention_mask"] = torch.ones(
                    batch["labels"].shape[:2], dtype=torch.long
                )

        return batch


class TimeMoeRunner:
    def __init__(
            self,
            model_path: str = None,
            output_path: str = 'logs/time_moe',
            seed: int = 9899
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.seed = seed

    def load_model(self, model_path: str = None, from_scatch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path

        temporal_mixer = kwargs.pop('temporal_mixer', None)
        attn = kwargs.pop('attn_implementation', None)

        main_input_size = kwargs.pop('main_input_size', None)
        macro_input_size = kwargs.pop('macro_input_size', None)
        use_macro_covariates = kwargs.pop('use_macro_covariates', None)
        macro_hidden_dropout = kwargs.pop('macro_hidden_dropout', None)
        macro_fusion = kwargs.pop('macro_fusion', None)

        if temporal_mixer == 'mamba':
            attn = 'mamba'
            log_in_local_rank_0('Use Mamba temporal mixer')
        elif attn is None:
            attn = 'eager'
        elif attn == 'auto':
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
                attn = 'flash_attention_2'
            except:
                log_in_local_rank_0('Flash attention import failed, switching to eager attention.', type='warn')
                attn = 'eager'

        if attn == 'eager':
            log_in_local_rank_0('Use Eager Attention')
        elif attn == 'flash_attention_2':
            log_in_local_rank_0('Use Flash Attention 2')
        elif attn != 'mamba':
            raise ValueError(f'Unknown attention method: {attn}')

        kwargs['attn_implementation'] = attn

        config_kw = {}
        if temporal_mixer is not None:
            config_kw['temporal_mixer'] = temporal_mixer
        if attn != 'mamba':
            config_kw['_attn_implementation'] = attn

        if main_input_size is not None:
            config_kw['main_input_size'] = main_input_size
        if macro_input_size is not None:
            config_kw['macro_input_size'] = macro_input_size
        if use_macro_covariates is not None:
            config_kw['use_macro_covariates'] = use_macro_covariates
        if macro_hidden_dropout is not None:
            config_kw['macro_hidden_dropout'] = macro_hidden_dropout
        if macro_fusion is not None:
            config_kw['macro_fusion'] = macro_fusion

        if from_scatch:
            config = TimeMoeConfig.from_pretrained(model_path, **config_kw)
            model = TimeMoeForPrediction(config)
        else:
            model = TimeMoeForPrediction.from_pretrained(model_path, **config_kw, **kwargs)

        return model

    def train_model(self, from_scratch: bool = False, **kwargs):
        setup_seed(self.seed)

        train_config = kwargs

        num_devices = get_world_size()

        global_batch_size = train_config.get('global_batch_size', None)
        micro_batch_size = train_config.get('micro_batch_size', None)

        if global_batch_size is None and micro_batch_size is None:
            raise ValueError('Must set at lease one argument: "global_batch_size" or "micro_batch_size"')
        elif global_batch_size is None:
            gradient_accumulation_steps = 1
            global_batch_size = micro_batch_size * num_devices
        elif micro_batch_size is None:
            micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = 1
        else:
            if micro_batch_size * num_devices > global_batch_size:
                if num_devices > global_batch_size:
                    micro_batch_size = 1
                    global_batch_size = num_devices
                else:
                    micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = math.ceil(global_batch_size / num_devices / micro_batch_size)
            global_batch_size = int(gradient_accumulation_steps * num_devices * micro_batch_size)

        if ('train_steps' in train_config
                and train_config['train_steps'] is not None
                and train_config['train_steps'] > 0):
            train_steps = int(train_config["train_steps"])
            num_train_epochs = -1
        else:
            train_steps = -1
            num_train_epochs = _safe_float(train_config.get("num_train_epochs", 1))

        precision = train_config.get('precision', 'bf16')
        if precision not in ['bf16', 'fp16', 'fp32']:
            log_in_local_rank_0(f'Precision {precision} is not set, use fp32 default!', type='warn')
            precision = 'fp32'

        if precision == 'bf16':
            torch_dtype = torch.bfloat16
        elif precision == 'fp16':
            # use fp32 to load model but uses fp15 to train model
            torch_dtype = torch.float32
        elif precision == 'fp32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f'Unsupported precision {precision}')

        log_in_local_rank_0(f'Set global_batch_size to {global_batch_size}')
        log_in_local_rank_0(f'Set micro_batch_size to {micro_batch_size}')
        log_in_local_rank_0(f'Set gradient_accumulation_steps to {gradient_accumulation_steps}')
        log_in_local_rank_0(f'Set precision to {precision}')
        log_in_local_rank_0(f'Set normalization to {train_config["normalization_method"]}')

        comet_api_key = (os.getenv("COMET_API_KEY") or "").strip()
        enable_comet = bool(comet_api_key) and comet_api_key not in {"YOUR_KEY_HERE", "..."}
        report_to = ["comet_ml"] if enable_comet else []

        training_args = TimeMoETrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=num_train_epochs,
            # use_cpu=True,
            max_steps=train_steps,
            # evaluation_strategy=train_config.get("evaluation_strategy", 'no'),
            eval_steps=_safe_float(train_config.get("eval_steps", None)),
            save_strategy=train_config.get("save_strategy", "no"),
            save_steps=_safe_float(train_config.get("save_steps", None)),
            learning_rate=float(train_config.get("learning_rate", 1e-5)),
            min_learning_rate=float(train_config.get("min_learning_rate", 0)),
            adam_beta1=float(train_config.get("adam_beta1", 0.9)),
            adam_beta2=float(train_config.get("adam_beta2", 0.95)),
            adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
            lr_scheduler_type=train_config.get("lr_scheduler_type", 'constant'),
            warmup_ratio=float(train_config.get("warmup_ratio") or 0.0),
            warmup_steps=int(train_config.get("warmup_steps", 0)),
            weight_decay=float(train_config.get("weight_decay", 0.1)),
            per_device_train_batch_size=int(micro_batch_size),
            per_device_eval_batch_size=int(micro_batch_size * 2),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            gradient_checkpointing=train_config.get("gradient_checkpointing", False),
            bf16=True if precision == 'bf16' else False,
            fp16=True if precision == 'fp16' else False,
            deepspeed=train_config.get("deepspeed"),
            push_to_hub=False,
            logging_first_step=True,
            log_on_each_node=False,
            logging_steps=int(train_config.get('logging_steps', 1)),
            report_to=report_to,
            run_name=os.getenv("COMET_EXPERIMENT_NAME") if enable_comet else None,
            seed=self.seed,
            data_seed=self.seed,
            max_grad_norm=train_config.get('max_grad_norm', 1.0),
            optim=train_config.get('optim', 'adamw_torch'),
            torch_compile=train_config.get('torch_compile', False),
            dataloader_num_workers=train_config.get('dataloader_num_workers') or 2,
            ddp_find_unused_parameters=False,

            logging_dir=os.path.join(self.output_path, 'tb_logs'),
            save_only_model=train_config.get('save_only_model', True),
            save_total_limit=train_config.get('save_total_limit'),
        )

        model_path = train_config.pop('model_path', None) or self.model_path
        if model_path is not None:
            model = self.load_model(
                model_path=model_path,
                from_scatch=from_scratch,
                torch_dtype=torch_dtype,
                attn_implementation=train_config.get('attn_implementation', 'eager'),
                temporal_mixer=train_config.get('temporal_mixer'),
                main_input_size=train_config.get('main_input_size'),
                macro_input_size=train_config.get('macro_input_size'),
                use_macro_covariates=train_config.get('use_macro_covariates'),
                macro_hidden_dropout=train_config.get('macro_hidden_dropout'),
                macro_fusion=train_config.get('macro_fusion'),
            )
            log_in_local_rank_0(f'Load model parameters from: {model_path}')
        else:
            raise ValueError('Model path is None')

        num_total_params = 0
        for p in model.parameters():
            num_total_params += reduce(mul, p.shape)

        # print statistics info
        log_in_local_rank_0(train_config)
        log_in_local_rank_0(training_args)
        log_in_local_rank_0(model.config)
        # Verify temporal mixer is set correctly
        tm = getattr(model.config, 'temporal_mixer', None)
        log_in_local_rank_0(f'>>> Model temporal_mixer: {tm} (should be "mamba" for Mamba training)')
        log_in_local_rank_0(f'Number of the model parameters: {length_to_str(num_total_params)}')

        if train_steps > 0:
            total_train_tokens = train_steps * global_batch_size * train_config['max_length']
            log_in_local_rank_0(f'Tokens will consume: {length_to_str(total_train_tokens)}')

        # Training
        train_ds = self.get_train_dataset(
            train_config["data_path"],
            max_length=train_config["max_length"],
            stride=train_config["stride"],
            normalization_method=train_config["normalization_method"],
            use_covariates=train_config.get("use_covariates", False),
        )
        trainer = TimeMoeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=CovariateDataCollator() if train_config.get("use_covariates", False) else None,
        )

        trainer.train()
        trainer.save_model(self.output_path)
        log_in_local_rank_0(f'Saving model to {self.output_path}')

        return trainer.model

    def get_train_dataset(
        self,
        data_path,
        max_length,
        stride,
        normalization_method,
        use_covariates=False,
    ):
        log_in_local_rank_0('Loading dataset...')

        if use_covariates:
            dataset = GeneralCovariateDataset(data_path)
            log_in_local_rank_0('Processing covariate-aware dataset to fixed-size sub-sequences...')
            window_dataset = TimeMoECovariateWindowDataset(
                dataset,
                context_length=max_length,
                prediction_length=0,
                stride=stride,
            )
        else:
            dataset = TimeMoEDataset(data_path, normalization_method=normalization_method)
            log_in_local_rank_0('Processing dataset to fixed-size sub-sequences...')
            window_dataset = TimeMoEWindowDataset(
                dataset,
                context_length=max_length,
                prediction_length=0,
                stride=stride,
                shuffle=False,
            )
        return window_dataset


def setup_seed(seed: int = 9899):
    """
    Setup seed for all known operations.

    Args:
        seed (int): seed number.

    Returns:

    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def length_to_str(length):
    if length >= 1e12:
        return f'{length / 1e12:.3f}T'
    if length >= 1e9:
        return f'{length / 1e9:.3f}B'
    elif length >= 1e6:
        return f'{length / 1e6:.3f}M'
    else:
        return f'{length / 1e3:.3f}K'


def _safe_float(number):
    if number is None:
        return None
    else:
        return float(number)
