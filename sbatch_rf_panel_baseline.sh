#!/bin/bash
# Panel RF baseline (compare.baselines.run_panel_baseline) — CPU only, no Comet / no GPU training.
# Submit: sbatch sbatch_rf_panel_baseline.sh
#
# Tune --cpus-per-task to be >= n_jobs_outer + a few spare cores (notebook uses n_jobs_outer=8).

#SBATCH --job-name=rf_panel_gfc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=aaburkova_1@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-rf-panel-%j.out
#SBATCH --error=slurm-rf-panel-%j.err

set -euo pipefail
set -x

ENV=/home/aaburkova_1/.conda/envs/mama
PY=$ENV/bin/python
PROJECT=/home/aaburkova_1/project-1

cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

"$PY" - <<'PY'
from compare.baselines import run_panel_baseline

df_rf, metrics_rf = run_panel_baseline(
    csv_path="dataset_finance_fred2/eval_csv_with_context/gfc_2008_2010_with_context.csv",
    model_name="rf",
    context_length=256,
    horizon=1,
    step=1,
    include_market=True,
    include_macro=True,
    aggregate_mode="none",
    rf_n_estimators=100,
    rf_max_depth=8,
    rf_min_samples_leaf=5,
    rf_n_jobs=1,
    n_jobs_outer=8,
    save_path="results/baselines/rf/gfc_h1.csv",
)
print(metrics_rf)
PY
