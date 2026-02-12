#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: peter.sxm
@project: TimeMOE
@time: 2024/5/22 20:04
@desc:
"""
import argparse
import os
import sys
import torch


def parse_arbitrary_args(argv):
    """Parse arbitrary command-line arguments in the format --key value.

    Args:
        argv: List of arguments (typically sys.argv).

    Returns:
        Dictionary of key-value pairs.
    """
    args = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if key.find('=') > 0:
                tmp_idx = key.find('=')
                value = key[tmp_idx + 1:]
                key = key[:tmp_idx]
                i += 1
            else:
                if i + 1 >= len(argv) or argv[i + 1].startswith('--'):
                    value = True
                    i += 1
                else:
                    value = argv[i + 1]
                    i += 2
            args[key] = value

        else:
            i += 1
    return args


def obtain_dist_env_dict():
    # SLURM-provided world info
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))

    # GPUs allocated to this job should be visible here:
    nproc_per_node = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else torch.cuda.device_count()

    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr is None:
        # pick first node in job as master
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "localhost")
        # simplest: let SLURM resolve it
        master_addr = os.popen(f"scontrol show hostnames {nodelist} | head -n 1").read().strip() or "localhost"

    master_port = os.environ.get("MASTER_PORT", "9899")

    return {
        "master_addr": master_addr,
        "master_port": master_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "nproc_per_node": nproc_per_node,
    }



def auto_dist_run(main_file: str, argv: str):
    if torch.cuda.is_available():
        env_dict = obtain_dist_env_dict()
        env_dict = obtain_dist_env_dict()
        launch_cmd = " ".join([
            # "torchrun"
            # Use the current interpreter to avoid relying on `torchrun` being on PATH
            sys.executable, "-m", "torch.distributed.run",
            f'--master_addr={env_dict["master_addr"]}',
            f'--master_port={env_dict["master_port"]}',
            f'--node_rank={env_dict["node_rank"]}',
            f'--nproc_per_node={env_dict["nproc_per_node"]}',
            f'--nnodes={env_dict["nnodes"]}',
        ])


        executed_cmd = launch_cmd + f' {main_file} {argv}'
    else:
        executed_cmd = f'python {main_file} {argv}'

    os.system(f'echo "{executed_cmd}"')

    if os.system(executed_cmd) != 0:
        raise RuntimeError(f'Error occurred when execute: {executed_cmd}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'main_file',
        help='Main file path'
    )
    parser.add_argument(
        '--port', '-p',
        default=9899,
        type=int,
        help='Port to use for distributed training'
    )

    args, unknown = parser.parse_known_args()
    argv = ' '.join(unknown)

    unique_job_name = args.main_file + argv
    os.environ['MASTER_PORT'] = str(args.port)

    auto_dist_run(
        main_file=args.main_file,
        argv=argv
    )
