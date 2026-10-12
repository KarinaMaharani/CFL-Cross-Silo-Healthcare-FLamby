#!/usr/bin/env bash
# The whole 8-client Fed-Heart-Disease study.  Everything is seconds: the model is
# 14 parameters and the dataset is ~740 patients.
#
#   python -m cfl_heart_disease_client.download     # once, ~40 KB from UCI
#   bash cfl_heart_disease_client/run_all.sh                  # a few minutes total
set -euo pipefail
cd "$(dirname "$0")/.."       # project root
ROUNDS=${ROUNDS:-100}
UPDATES=${UPDATES:-20}
RES=${RES:-results_heart8}
export PYTHONWARNINGS=ignore
mkdir -p "$RES"
EPS1=${EPS1:-}
EPS2=${EPS2:-}

echo "=== 1/5 baselines ==="
python -m cfl_heart_disease_client.baselines                           2>&1 | tee -a "$RES/baselines_stdout.txt"

echo "=== 2/5 FedAvg (calibration + reference) ==="
python -m cfl_heart_disease_client.run_experiment --run-name fedavg8 --strategy fedavg \
    --rounds "$ROUNDS" --local-updates "$UPDATES" --gamma-probe-every 5 \
                                                        2>&1 | tee -a "$RES/fedavg8_stdout.txt"
python -m cfl_heart_disease_client.calibrate --run fedavg8             2>&1 | tee -a "$RES/calibration.txt"
if [ -z "$EPS1" ]; then
    EPS1=$(python -c "import json;print(json.load(open('$RES/fedavg8/calibration.json'))['eps_1'])")
    EPS2=$(python -c "import json;print(json.load(open('$RES/fedavg8/calibration.json'))['eps_2'])")
fi
echo "using eps_1=$EPS1 eps_2=$EPS2"

echo "=== 3/5 oracle partition (the 4 hospitals) ==="
python -m cfl_heart_disease_client.run_experiment --run-name oracle_centers --strategy partition \
    --partition "0,1|2,3|4,5|6,7" --rounds "$ROUNDS" --local-updates "$UPDATES" \
                                                        2>&1 | tee -a "$RES/oracle_stdout.txt"

echo "=== 4/5 CFL ==="
python -m cfl_heart_disease_client.run_experiment --run-name cfl8 --strategy cfl \
    --rounds "$ROUNDS" --local-updates "$UPDATES" --eps-1 "$EPS1" --eps-2 "$EPS2" \
    --gamma-auto --gamma-probe-every 5                  2>&1 | tee -a "$RES/cfl8_stdout.txt"

echo "=== 5/5 CFL with stragglers ==="
python -m cfl_heart_disease_client.run_experiment --run-name cfl8_strag --strategy cfl \
    --rounds "$ROUNDS" --local-updates "$UPDATES" --eps-1 "$EPS1" --eps-2 "$EPS2" \
    --gamma-auto --gamma-probe-every 5 \
    --straggler-mode partial --straggler-profile tiers --straggler-tiers 1.0,0.4 \
                                                        2>&1 | tee -a "$RES/strag_stdout.txt"

echo "ALL DONE"
