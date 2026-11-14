#!/usr/bin/env bash
# Run a command in the dev container on the GPUs held by a slurm/dev-gpu.sh job, and on
# no others. The container sees every GPU of the node, and the cluster's GPU monitor
# only checks that a GPU is allocated to *some* job, so this is the one place that keeps
# container work on its own allocation.
#
#   slurm/with-gpu.sh [--job <id>] <command> [args...]
#   slurm/with-gpu.sh --status
#
# Without --job there must be exactly one live lease.
set -euo pipefail

usage="usage: slurm/with-gpu.sh [--job <id>] <command> [args...] | --status"
dir=${LABRAG_RUN_ROOT:-$HOME/labrag-run}/dev-gpu
max_age=180 # seconds since the job last touched its lease (it does so every 60)

# Sets job_id node gpu_index gpu_uuid end_time state left from lease file $1.
read_lease() {
  job_id='' node='' gpu_index='' gpu_uuid='' end_time=0
  local k v
  while IFS='=' read -r k v; do
    case $k in job_id | node | gpu_index | gpu_uuid | end_time) printf -v "$k" %s "$v" ;; esac
  done <"$1"
  local now age
  now=$(date +%s)
  age=$((now - $(stat -c %Y "$1")))
  if ((end_time > 0 && now >= end_time)); then
    state=expired
  elif ((age > max_age)); then
    state="stale (last heartbeat ${age}s ago)"
  else
    state=live
  fi
  if ((end_time > 0)); then
    left=$(((end_time - now) / 3600))h$(((end_time - now) % 3600 / 60))m
  else
    left=unlimited
  fi
}

leases=()
[[ -d $dir ]] && for f in "$dir"/*; do [[ -f $f && $f != *.tmp ]] && leases+=("$f"); done

if [[ ${1:-} == --status ]]; then
  ((${#leases[@]})) || { echo "no leases in $dir"; exit 0; }
  for f in "${leases[@]}"; do
    read_lease "$f"
    echo "job $job_id on $node: GPU $gpu_index ($gpu_uuid), $state, $left left"
  done
  exit 0
fi

want=''
if [[ ${1:-} == --job ]]; then
  want=${2:?$usage}
  shift 2
fi
(($#)) || { echo "$usage" >&2; exit 1; }
[[ -e /dev/nvidiactl ]] || { echo "this container has no GPU access (devcontainer runArgs --gpus)" >&2; exit 1; }

live=()
for f in "${leases[@]}"; do
  read_lease "$f"
  [[ -n $want && $job_id != "$want" ]] && continue
  [[ $state == live ]] && live+=("$f")
done
if ((${#live[@]} != 1)); then
  if ((${#live[@]} == 0)); then
    echo "no live lease${want:+ for job $want} in $dir; ask Steven to submit slurm/dev-gpu.sh" >&2
  else
    echo "${#live[@]} live leases; pick one with --job (see --status)" >&2
  fi
  exit 1
fi

read_lease "${live[0]}"
echo "with-gpu: job $job_id, GPU $gpu_index ($gpu_uuid), $left left" >&2
export CUDA_VISIBLE_DEVICES=$gpu_uuid CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@"
