#!/usr/bin/env bash
# Batch script for every pipeline job. Submit it with slurm/submit.sh, not by hand.
#
#   job.sh <commit-sha> <repo-url> <command> [args...]
#
# On every node of the job, makes sure ~/labrag-run/commits/<sha>/ holds a shallow clone
# of that commit with its uv env (never modified once built, so commits can't
# interfere), then runs <command> there once. Without Slurm, the same steps run on the
# local host.
#
#   ~/labrag-run/ (LABRAG_RUN_ROOT): commits/<sha>/, uv-cache/ python/ bin/ (same
#   filesystem, so venvs hard-link), bin/hf (for pre-pulling weights by hand),
#   logs/<job-id>.out, setup.lock
#   LABRAG_SHARED_ROOT/logs/<job-id>.out: the log, copied once when the job ends
#
# Slurm doesn't confine jobs, so --cpus-per-task is the job's CPU share only by
# agreement. Default 72: a GPU's even share of a node (labrag evaluate is sized for it).
#SBATCH --output=/tmp/labrag-%j.out
#SBATCH --cpus-per-task=72
set -euo pipefail

usage="usage: job.sh <commit-sha> <repo-url> <command> [args...]"
sha=${1:?$usage}
url=${2:?$usage}
shift 2
(($#)) || { echo "$usage" >&2; exit 1; }
[[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "not a full commit sha: $sha" >&2; exit 1; }

root=${LABRAG_RUN_ROOT:-$HOME/labrag-run}
log=$root/logs/${SLURM_JOB_ID:-local-$$}.out
mkdir -p "$root/logs"
# Slurm's log is node-local /tmp; from here on everything also goes to the run root.
exec 3>&1 4>&2
exec > >(tee -a "$log") 2>&1
tee_pid=$!

# Copy the finished log to shared storage, once per job (node-local logs vanish with
# their node's run root). Atomic rename, so a partial copy is never visible.
publish_log() {
  local rc=$?
  exec 1>&3 2>&4 # back to Slurm's log; tee sees EOF and drains
  wait "$tee_pid" || true
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    local dest=${LABRAG_SHARED_ROOT:-/opt/gpu_working/steven/labrag-shared}/logs
    mkdir -p "$dest" && cp "$log" "$dest/.$SLURM_JOB_ID.out.tmp" &&
      mv "$dest/.$SLURM_JOB_ID.out.tmp" "$dest/$SLURM_JOB_ID.out" ||
      echo "could not copy $log to $dest" >&2
  fi
  exit "$rc"
}
trap publish_log EXIT
# scancel and time limits send SIGTERM; without a trap bash would exit without EXIT's.
trap 'exit 143' TERM

# A function so srun can run it on every node (the others don't have this file).
setup_commit() {
  local sha=$1 url=$2 root=$3
  local dir=$root/commits/$sha
  export UV_INSTALL_DIR=$root/bin UV_CACHE_DIR=$root/uv-cache UV_PYTHON_INSTALL_DIR=$root/python
  export UV_TOOL_DIR=$root/tools UV_TOOL_BIN_DIR=$root/bin
  mkdir -p "$root/commits"
  exec 9>"$root/setup.lock"
  flock 9
  if [[ ! -e $dir/.ready ]]; then
    echo "$(hostname): building $dir"
    rm -rf "$dir" # left half-built by a crashed setup; no job runs from it without .ready
    git init -q --template= "$dir"
    git -C "$dir" fetch -q --depth 1 "$url" "$sha"
    git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
    git -C "$dir" -c init.templateDir= submodule -q update --init --depth 1
    local uv
    uv=$("$dir/envs/install-uv.sh")
    (cd "$dir" && unset VIRTUAL_ENV && "$uv" sync --locked --no-dev --group test)
    touch "$dir/.ready"
  fi
  # Not used by jobs; kept here so every node that runs a job has it.
  echo "$(hostname): hf $("$dir/envs/install-hf.sh")"
  flock -u 9
  echo "$(hostname): $dir ready"
}

if [[ -n ${SLURM_JOB_ID:-} ]]; then
  echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST at commit $sha"
  srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -c "set -euo pipefail; $(declare -f setup_commit); setup_commit $(printf '%q ' "$sha" "$url" "$root")"
else
  echo "local run on $(hostname) at commit $sha"
  setup_commit "$sha" "$url" "$root"
fi

dir=$root/commits/$sha
cd "$dir"
export VIRTUAL_ENV=$dir/.venv PATH=$dir/.venv/bin:$PATH LABRAG_COMMIT=$sha
# Cap BLAS/OpenMP thread pools and joblib workers at --cpus-per-task (they default to
# the whole node). Per pool: an n_jobs > 1 call must also pass
# joblib.parallel_config(backend="loky", inner_max_num_threads=1).
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  ncpu=$SLURM_CPUS_PER_TASK
  export OPENBLAS_NUM_THREADS=$ncpu OMP_NUM_THREADS=$ncpu LOKY_MAX_CPU_COUNT=$ncpu
  echo "thread pools and joblib workers capped at $ncpu (--cpus-per-task)"
fi
"$@"
