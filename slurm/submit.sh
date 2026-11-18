#!/usr/bin/env bash
# Submit a pipeline job at the current commit. Run from a clone of this repo on a host
# with the Slurm client.
#
#   slurm/submit.sh [sbatch options...] -- <command> [args...]
#   e.g. slurm/submit.sh --gres=gpu:1 --cpus-per-task=16 -- labrag labels --dataset chexpertplus --section findings
#   (request a rail type, gpu:h200_rail0/1, only when the job needs rail alignment;
#   --cpus-per-task defaults to 72, slurm/job.sh's #SBATCH default)
#
# The job runs HEAD, cloned from GitHub on each node: clean tree, HEAD pushed.
set -euo pipefail

usage="usage: slurm/submit.sh [sbatch options...] -- <command> [args...]"
sbatch_args=()
while (($#)) && [[ $1 != -- ]]; do
  sbatch_args+=("$1")
  shift
done
[[ ${1:-} == -- ]] && shift || { echo "$usage" >&2; exit 1; }
(($#)) || { echo "$usage" >&2; exit 1; }

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$repo"
if [[ -n $(git status --porcelain) ]]; then
  echo "refusing to submit: uncommitted or untracked files (the job would not see them)" >&2
  git status --short >&2
  exit 1
fi
sha=$(git rev-parse HEAD)

# The nodes clone over public HTTPS (they have no GitHub credentials).
url=$(git remote get-url origin)
url=${url/#git@github.com:/https:\/\/github.com\/}
# Same fetch the nodes will do, into a throwaway repo (never into this clone: a --depth
# fetch here would make it shallow).
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
git init -q --template= "$probe"
if ! GIT_TERMINAL_PROMPT=0 git -C "$probe" fetch -q --depth 1 "$url" "$sha" 2>/dev/null; then
  echo "refusing to submit: $sha is not on $url (push it first)" >&2
  exit 1
fi

sbatch --job-name=labrag "${sbatch_args[@]}" slurm/job.sh "$sha" "$url" "$@"
