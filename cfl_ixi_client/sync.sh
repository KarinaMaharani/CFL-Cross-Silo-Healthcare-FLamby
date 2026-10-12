#!/usr/bin/env bash
# Move this project between the laptop (run from WSL2) and the GPU server.
#
#   bash cfl_ixi_client/sync.sh code    # just the python package (fast, do this often)
#   bash cfl_ixi_client/sync.sh data    # the 374 MB preprocessed cache (once)
#   bash cfl_ixi_client/sync.sh raw     # the 447 MB raw IXI volumes (only if you want
#                                       #   the server to redo the preprocessing itself)
#   bash cfl_ixi_client/sync.sh all     # code + data
#   bash cfl_ixi_client/sync.sh pull    # bring results_ixi9/ back to the laptop
#   bash cfl_ixi_client/sync.sh ssh     # open a shell on the server, in the project dir
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
        "$LOCAL/cfl_ixi_client" "$REMOTE:$RDIR/"
    ;;
  data)
    remote_mkdir
    $RS "$LOCAL/cache" "$REMOTE:$RDIR/"
    ;;
  raw)
    remote_mkdir
    $RS "$LOCAL/ixi" "$REMOTE:$RDIR/"
    ;;
  all)
    "$0" code
    "$0" data
    ;;
  pull)
    mkdir -p "$LOCAL/results_ixi9"
    $RS "$REMOTE:$RDIR/results_ixi9/" "$LOCAL/results_ixi9/"
    ;;
  ssh)
    exec ssh -p "$PORT" -t "$REMOTE" "cd '$RDIR' && exec \$SHELL -l"
    ;;
  *)
    sed -n '2,18p' "$0"
    ;;
esac
