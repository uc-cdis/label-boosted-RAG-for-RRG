#!/usr/bin/env bash
# Batch script for every pipeline job. Submit it with slurm/submit.sh, not by hand.
#
#   job.sh <commit-sha> <repo-url> <command> [args...]
#
# sbatch ships only this file to the node, so it brings its own code: on every node of the
# job it makes sure ~/labrag-run/commits/<sha>/ holds a shallow clone of exactly that
# commit with its uv env (.venv, from that commit's uv.lock), then runs <command> once,
# from that directory, with the env on PATH. Nothing that exists is ever modified: a later
# commit gets its own directory, so jobs at different commits on one node can't interfere.
#
#   ~/labrag-run/            (LABRAG_RUN_ROOT)
#     commits/<sha>/         one complete, independent clone per commit, plus .venv/
#     uv-cache/ python/ bin/ uv's package cache, Python builds and binary, kept on the
#                            same filesystem as commits/ so venvs hard-link from the cache
#     bin/hf, tools/         the Hugging Face CLI, for pre-pulling weights by hand
#                            (envs/install-hf.sh; at the versions in the latest job's lock)
#     logs/<job-id>.out      this job's log
#     setup.lock             held while a commit dir is being built
#
# Without Slurm (SLURM_JOB_ID unset) it runs the same steps on the local host.
#SBATCH --output=/tmp/labrag-%j.out
set -euo pipefail

usage="usage: job.sh <commit-sha> <repo-url> <command> [args...]"
sha=${1:?$usage}
url=${2:?$usage}
shift 2
(($#)) || { echo "$usage" >&2; exit 1; }
[[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "not a full commit sha: $sha" >&2; exit 1; }

root=${LABRAG_RUN_ROOT:-$HOME/labrag-run}
mkdir -p "$root/logs"
# Slurm's own log is node-local /tmp (the only dir sure to exist before this runs);
# from here on everything also goes to the run root.
exec > >(tee -a "$root/logs/${SLURM_JOB_ID:-local-$$}.out") 2>&1

# Build (once per node and commit) the commit dir and its env. Defined as a function so
# srun can run it on every node of the job: the other nodes don't have this file.
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
    (cd "$dir" && unset VIRTUAL_ENV && "$uv" sync --locked --no-dev)
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
# Our Slurm doesn't confine jobs with cgroups, so a process can use every core of the
# node. OpenBLAS (numpy/scipy/sklearn) and OpenMP (torch) size their thread pools to the
# whole machine by default, and joblib (sklearn n_jobs=-1) its worker count; cap all three
# at the job's --cpus-per-task (Slurm's default is 1). These are per pool: joblib workers
# inherit the BLAS cap, so any n_jobs > 1 call must also pass
# joblib.parallel_config(backend="loky", inner_max_num_threads=1).
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  ncpu=${SLURM_CPUS_PER_TASK:-1}
  export OPENBLAS_NUM_THREADS=$ncpu OMP_NUM_THREADS=$ncpu LOKY_MAX_CPU_COUNT=$ncpu
  echo "thread pools and joblib workers capped at $ncpu (--cpus-per-task)"
fi
"$@"
