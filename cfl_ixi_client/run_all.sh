#!/usr/bin/env bash
# One full study for 9-client Fed-IXI, in the order the runs depend on each other.
# Meant to be started inside tmux on the server:  bash cfl_ixi_client/run_all.sh
#
#   1. baselines      pooled / per-centre / per-client   (reference points)
#   2. fedavg9        never splits                       (calibrates eps_1, eps_2)
#   3. oracle_centers the ground-truth partition         (what a perfect CFL would get)
#   4. cfl9           CFL, paper criteria only
#   5. cfl9_strag     the same with partial stragglers
#
# Steps 4 and 5 need eps_1 / eps_2 read off step 2 -- edit EPS1/EPS2 below after it
# finishes (tools/calibrate.py prints the values), or export them before running.
set -euo pipefail
cd "$(dirname "$0")/.."                 # project root (cfl-replicate-flamby)

ROUNDS=${ROUNDS:-60}
RES=${RES:-results_ixi9}
export PYTHONWARNINGS=ignore
mkdir -p "$RES"
# EPS1/EPS2 are read from the calibration of step 2 unless you set them here.
EPS1=${EPS1:-}
EPS2=${EPS2:-}

echo "=== 1/5 baselines ==="
python -m cfl_ixi_client.baselines                       2>&1 | tee -a "$RES/baselines_stdout.txt"

echo "=== 2/5 FedAvg (calibration + reference) ==="
python -m cfl_ixi_client.run_experiment --run-name fedavg9 --strategy fedavg \
    --rounds "$ROUNDS" --gamma-probe-every 5             2>&1 | tee -a "$RES/fedavg9_stdout.txt"
python -m cfl_ixi_client.calibrate --run fedavg9         2>&1 | tee -a "$RES/calibration.txt"
if [ -z "$EPS1" ]; then
    EPS1=$(python -c "import json;print(json.load(open('$RES/fedavg9/calibration.json'))['eps_1'])")
    EPS2=$(python -c "import json;print(json.load(open('$RES/fedavg9/calibration.json'))['eps_2'])")
fi
echo "using eps_1=$EPS1 eps_2=$EPS2"

echo "=== 3/5 oracle partition (centres) ==="
python -m cfl_ixi_client.run_experiment --run-name oracle_centers --strategy partition \
    --partition "0,1,2|3,4,5|6,7,8" --rounds "$ROUNDS"   2>&1 | tee -a "$RES/oracle_stdout.txt"

echo "=== 4/5 CFL ==="
python -m cfl_ixi_client.run_experiment --run-name cfl9 --strategy cfl --rounds "$ROUNDS" \
    --eps-1 "$EPS1" --eps-2 "$EPS2" --gamma-auto --gamma-probe-every 5 \
                                                         2>&1 | tee -a "$RES/cfl9_stdout.txt"

echo "=== 5/5 CFL with stragglers ==="
python -m cfl_ixi_client.run_experiment --run-name cfl9_strag --strategy cfl --rounds "$ROUNDS" \
    --eps-1 "$EPS1" --eps-2 "$EPS2" --gamma-auto --gamma-probe-every 5 \
    --straggler-mode partial --straggler-profile tiers --straggler-tiers 1.0,0.5,0.25 \
                                                         2>&1 | tee -a "$RES/cfl9_strag_stdout.txt"

echo "ALL DONE"
