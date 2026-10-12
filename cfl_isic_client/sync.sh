#!/usr/bin/env bash
# Move this project between the laptop (run from WSL2) and the GPU server.
#
#   bash cfl_isic_client/sync.sh code    # just the python package (fast, do this often)
#   bash cfl_isic_client/sync.sh data    # the preprocessed jpgs + split csv (~0.6 GB, once)
#   bash cfl_isic_client/sync.sh raw     # the 3.2 GB uint8 cache (optional: the server can
#                                       #   rebuild it from the jpgs in ~5 min)
#   bash cfl_isic_client/sync.sh all     # code + data
#   bash cfl_isic_client/sync.sh pull    # bring results_isic18/ back to the laptop
#   bash cfl_isic_client/sync.sh ssh     # open a shell on the server, in the project dir
#
# Override any of these from the environment:
#   REMOTE=karina@152.118.31.34  PORT=51022  RDIR=skripsi/cfl-replicate-flamby
set -euo pipefail

REMOTE=${REMOTE:-karina@152.118.31.34}
PORT=${PORT:-51022}
RDIR=${RDIR:-skripsi/cfl-replicate-flamby}        # relative to the remote home dir
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"         # the project root on this machine
SSH="ssh -p $PORT"
RS="rsync -avzP --human-readable -e ${SSH}"

remote_mkdir() { ssh -p "$PORT" "$REMOTE" "mkdir -p '$RDIR'"; }

case "${1:-help}" in
  code)
    remote_mkdir
    $RS --exclude '__pycache__' --exclude '*.pyc' \
        "$LOCAL/cfl_isic_client" "$REMOTE:$RDIR/"
    ;;
  data)
    remote_mkdir
    $RS --exclude "*.npy" --exclude "*.done" "$LOCAL/isic/train_test_split.csv" "$LOCAL/isic/ISIC_2019_Training_Input_preprocessed" "$REMOTE:$RDIR/isic/"
    ;;
  raw)
    remote_mkdir
    $RS "$LOCAL/isic/cache_220.npy" "$LOCAL/isic/cache_220.npy.done" "$REMOTE:$RDIR/isic/"
    ;;
  all)
    "$0" code
    "$0" data
    ;;
  pull)
    mkdir -p "$LOCAL/results_isic18"
    $RS "$REMOTE:$RDIR/results_isic18/" "$LOCAL/results_isic18/"
    ;;
  ssh)
    exec ssh -p "$PORT" -t "$REMOTE" "cd '$RDIR' && exec \$SHELL -l"
    ;;
  *)
    sed -n '2,18p' "$0"
    ;;
esac
