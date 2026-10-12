#!/usr/bin/env bash
# The whole 18-client Fed-TCGA-BRCA study.  Everything here is seconds-to-minutes:
# the model is 40 parameters and the dataset is 1088 patients.
#
#   bash cfl_tcga_brca_client/run_all.sh            # ~10 min total on CPU or GPU
set -euo pipefail
cd "$(dirname "$0")/.."       # project root
ROUNDS=${ROUNDS:-100}
UPDATES=${UPDATES:-20}
RES=${RES:-results_brca18}
export PYTHONWARNINGS=ignore
mkdir -p "$RES"
EPS1=${EPS1:-}
EPS2=${EPS2:-}

echo "=== 1/5 baselines ==="
python -m cfl_tcga_brca_client.baselines                            2>&1 | tee -a "$RES/baselines_stdout.txt"

echo "=== 2/5 FedAvg (calibration + reference) ==="
python -m cfl_tcga_brca_client.run_experiment --run-name fedavg18 --strategy fedavg \
    --rounds "$ROUNDS" --local-updates "$UPDATES" --gamma-probe-every 5 \
                                                        2>&1 | tee -a "$RES/fedavg18_stdout.txt"
python -m cfl_tcga_brca_client.calibrate --run fedavg18             2>&1 | tee -a "$RES/calibration.txt"
if [ -z "$EPS1" ]; then
    EPS1=$(python -c "import json;print(json.load(open('$RES/fedavg18/calibration.json'))['eps_1'])")
    EPS2=$(python -c "import json;print(json.load(open('$RES/fedavg18/calibration.json'))['eps_2'])")
fi
echo "using eps_1=$EPS1 eps_2=$EPS2"

echo "=== 3/5 oracle partition (the 6 regions) ==="
python -m cfl_tcga_brca_client.run_experiment --run-name oracle_centers --strategy partition \
    --partition "0,1,2|3,4,5|6,7,8|9,10,11|12,13,14|15,16,17" \
    --rounds "$ROUNDS" --local-updates "$UPDATES"        2>&1 | tee -a "$RES/oracle_stdout.txt"

echo "=== 4/5 CFL ==="
python -m cfl_tcga_brca_client.run_experiment --run-name cfl18 --strategy cfl \
    --rounds "$ROUNDS" --local-updates "$UPDATES" --eps-1 "$EPS1" --eps-2 "$EPS2" \
    --gamma-auto --gamma-probe-every 5                  2>&1 | tee -a "$RES/cfl18_stdout.txt"

echo "=== 5/5 CFL with stragglers ==="
python -m cfl_tcga_brca_client.run_experiment --run-name cfl18_strag --strategy cfl \
    --rounds "$ROUNDS" --local-updates "$UPDATES" --eps-1 "$EPS1" --eps-2 "$EPS2" \
    --gamma-auto --gamma-probe-every 5 \
    --straggler-mode partial --straggler-profile tiers  2>&1 | tee -a "$RES/strag_stdout.txt"

echo "ALL DONE"
