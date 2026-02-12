#!/bin/bash
#SBATCH --job-name=debug_paths
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00


#!/bin/bash
set -euo pipefail
set -x

echo "=============================="
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "=============================="

echo
echo ">>> Which conda is visible?"
which conda || true
conda --version || true

echo
echo ">>> Available conda environments"
conda env list || true

echo
echo ">>> Activating mama environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/aaburkova_1/.conda/envs/mama

echo
echo ">>> Python executable"
which python
python -c "import sys; print(sys.executable); print(sys.version)"

echo
echo ">>> Core ML packages versions"
python - <<'PY'
pkgs = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "numpy",
]
for p in pkgs:
    try:
        m = __import__(p)
        print(f"{p:15s} {getattr(m,'__version__','?')}  ({m.__file__})")
    except Exception as e:
        print(f"{p:15s} NOT IMPORTABLE -> {e}")
PY

echo
echo ">>> Conda list (full)"
conda list

source deactivate # На всякий случай
source activate /home/aaburkova_1/.conda/envs/mama

echo ">>> Conda list (full)"
conda list

echo
echo ">>> pip list (top 50)"
pip list | head -n 50



echo
echo ">>> conda list (top 50)"
conda list | head -n 50
