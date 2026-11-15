#!/usr/bin/env bash
# Remove labrag's Docker leftovers from one node. Docker is per node and the dev container
# has none, so run it as a job, once per node (kg35-nvl02 included):
#
#   slurm/submit.sh -w <node> --with-nvl02 --cpus-per-task=1 -- slurm/clean-docker.sh [--apply]
#
# Without --apply it only lists what it would remove. With --apply it removes:
#   - containers and networks left by jobs that died without cleaning up: evaluate's
#     compose projects (labrag-eval-<job>) and generation's vLLM servers
#     (labrag-<job>-<port>, serving.py)
#   - every labrag/* image (evaluate builds them; the next evaluate job rebuilds)
#   - dangling images and all build cache not used by a running build. Both are
#     node-wide, not only ours: Docker can't tell whose they are.
# Pulled images (configs/containers.yaml, eval/compose.yaml's judge) are never touched.
#
# Refuses while another labrag job runs on the node: that job could be between building
# its images and starting them, or be using the containers above.
set -euo pipefail

apply=false
case ${1:-} in
  --apply) apply=true ;;
  '') ;;
  *) echo "usage: slurm/clean-docker.sh [--apply]" >&2; exit 1 ;;
esac

node=${SLURMD_NODENAME:-$(hostname -s)}
running=$(squeue -h -w "$node" -n labrag -t RUNNING -o %i)
others=$(grep -vx "${SLURM_JOB_ID:-}" <<<"$running" || true)
if [[ -n $others ]]; then
  echo "refusing: other labrag jobs run on $node: $(echo $others)" >&2
  exit 1
fi

# id<TAB>what, one line per object.
containers=$(
  docker ps -a --filter label=com.docker.compose.project \
    --format '{{.ID}}	{{.Label "com.docker.compose.project"}}/{{.Names}} ({{.State}})' |
    awk -F'\t' '$2 ~ /^labrag-eval-/'
  docker ps -a --format '{{.ID}}	{{.Names}} ({{.State}})' |
    awk -F'\t' '$2 ~ /^labrag-([0-9]+|local)-[0-9]+ /'
)
networks=$(
  docker network ls --filter label=com.docker.compose.project \
    --format '{{.ID}}	{{.Name}}' | awk -F'\t' '$2 ~ /^labrag-eval-/'
)
images=$(docker image ls --filter reference='labrag/*' --format '{{.Repository}}:{{.Tag}}	{{.ID}} {{.Size}}')
dangling=$(docker image ls --filter dangling=true --format '{{.ID}}	{{.Size}}')

show() { # title, lines
  echo "$1: $(grep -c . <<<"$2" || true)"
  [[ -n $2 ]] && sed 's/^/  /' <<<"$2"
  return 0
}
echo "node $node ($($apply && echo apply || echo "dry run; --apply removes"))"
show "leftover containers" "$containers"
show "leftover networks" "$networks"
show "labrag images" "$images"
show "dangling images" "$dangling"
docker system df

$apply || exit 0
# rm -f also stops a container that's still running; no job owns one (checked above).
[[ -z $containers ]] || docker rm -f $(cut -f1 <<<"$containers")
[[ -z $networks ]] || docker network rm $(cut -f1 <<<"$networks")
# By tag, without -f: an image some container still uses is refused, not removed.
[[ -z $images ]] || docker image rm $(cut -f1 <<<"$images")
docker image prune --force
docker builder prune --all --force
echo "after:"
docker system df
