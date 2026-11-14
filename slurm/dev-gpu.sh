#!/usr/bin/env bash
# Hold GPUs for the dev container, which runs on kg35-nvl01 and has no Slurm client. Submit
# by hand from nvl01's host shell; scancel the job to give the GPUs back.
#
#   sbatch slurm/dev-gpu.sh     (1 GPU for 8 hours; override with --gres / --time)
#
# While the job runs, ~/labrag-run/dev-gpu/<job-id> (mounted read-only in the container)
# names the allocated GPUs by UUID; slurm/with-gpu.sh reads it there and runs commands on
# exactly those GPUs. The job touches the file every minute and removes it on exit, so a
# lease left behind by a job that died without cleaning up goes stale and is refused.
#SBATCH --job-name=dev-gpu
#SBATCH --nodelist=kg35-nvl01
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/tmp/dev-gpu-%j.out
set -euo pipefail

[[ -n ${SLURM_JOB_ID:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]] ||
  { echo "submit with sbatch and --gres=gpu:N" >&2; exit 1; }

dir=${LABRAG_RUN_ROOT:-$HOME/labrag-run}/dev-gpu
lease=$dir/$SLURM_JOB_ID
mkdir -p "$dir"

# nvidia-smi -i takes NVML (PCI bus order) indices, the same numbering Slurm's
# CUDA_VISIBLE_DEVICES uses here. UUIDs make the container side independent of numbering.
uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" | paste -sd,)
# 0 = no time limit.
end=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || echo 0)}

cleanup() { rm -f "$lease" "$lease.tmp"; }
trap cleanup EXIT
trap 'exit 0' TERM INT

printf 'job_id=%s\nnode=%s\ngpu_index=%s\ngpu_uuid=%s\nend_time=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$CUDA_VISIBLE_DEVICES" "$uuids" "$end" >"$lease.tmp"
mv "$lease.tmp" "$lease"
echo "job $SLURM_JOB_ID on $(hostname): GPUs $CUDA_VISIBLE_DEVICES ($uuids) leased in $lease"

# Backgrounded so the TERM trap runs as soon as Slurm signals, not after the sleep.
while :; do
  sleep 60 &
  wait $!
  touch "$lease"
done
