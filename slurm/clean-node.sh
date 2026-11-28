#!/usr/bin/env bash
# Clean one node down to what this commit uses, as a job per node:
#
#   slurm/submit.sh -w <node> --cpus-per-task=16 -- slurm/clean-node.sh [--apply]
#
# Lists by default; --apply removes:
# - run root: commits/<sha>/ of every other commit (a job rebuilds its own), and logs/
#   whose copy in the shared logs dir is identical
# - containers/networks of dead jobs (labrag-eval-<job>, labrag-<job>-<port>)
# - labrag/* images of other commits, and untagged images no name points to (node-wide:
#   Docker can't tell whose). Other images are never touched; one a container uses is
#   refused, not removed.
# - build cache the rebuild didn't use: --apply first builds every compose image of this
#   commit (before removing anything), prunes cache last used before that build, then
#   builds again and fails unless every image kept its layers (the cache it kept is the
#   cache jobs need; image ids can't tell: the containerd store's provenance
#   attestation gives every build a new id).
# Refuses while another labrag job runs on the node. Holds setup.lock and
# docker-build.lock throughout: a job starting meanwhile waits in its setup, and a run
# Slurm doesn't list (job.sh without Slurm) can't build or read an image id meanwhile.
set -euo pipefail

apply=false
case ${1:-} in
  --apply) apply=true ;;
  '') ;;
  *) echo "usage: slurm/clean-node.sh [--apply]" >&2; exit 1 ;;
esac

sha=${LABRAG_COMMIT:?run through slurm/submit.sh (job.sh sets LABRAG_COMMIT)}
cd "$(git rev-parse --show-toplevel)"
read -r root shared < <(python -c 'from labrag import paths; print(paths.RUN_ROOT, paths.SHARED_ROOT)')
node=${SLURMD_NODENAME:-$(hostname -s)}
project=labrag-clean-${SLURM_JOB_ID:-local-$$}

exec 9>"$root/setup.lock" 8>"$root/docker-build.lock"
flock 9
flock 8
running=$(squeue -h -w "$node" -n labrag -t RUNNING -o %i)
others=$(grep -vx "${SLURM_JOB_ID:-}" <<<"$running" || true)
if [[ -n $others ]]; then
  echo "refusing: other labrag jobs run on $node: $(echo $others)" >&2
  exit 1
fi

# One path per line.
commits=$(find "$root/commits" -mindepth 1 -maxdepth 1 ! -name "$sha" | sort)
logs=$(
  for f in "$root"/logs/*.out; do
    if [[ -f $f ]] && cmp -s "$f" "$shared/logs/${f##*/}"; then echo "$f"; fi
  done
)
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
# ref<TAB>id size. Dangling = no repository at all: an image pulled by digest only
# keeps its repository (and so is never listed).
image_rows=$(docker image ls --digests --no-trunc \
  --format '{{.Repository}}	{{.Tag}}	{{.Digest}}	{{.ID}}	{{.Size}}')
images=$(awk -F'\t' -v sha="$sha" '$1 ~ /^labrag\// && $2 != sha {
  print ($2 != "<none>" ? $1 ":" $2 : $1 "@" $3) "\t" $4 " " $5 }' <<<"$image_rows" | sort -u)
dangling=$(awk -F'\t' '$1 == "<none>" { print $4 "\t" $5 }' <<<"$image_rows" | sort -u)

show() { # title, lines
  echo "$1: $(grep -c . <<<"$2" || true)"
  [[ -n $2 ]] && sed 's/^/  /' <<<"$2"
  return 0
}
echo "node $node at $sha ($($apply && echo apply || echo "dry run; --apply removes"))"
show "other commits' run dirs" "$commits"
[[ -z $commits ]] || du -shc $commits | tail -1
echo "published logs: $(grep -c . <<<"$logs" || true)"
show "leftover containers" "$containers"
show "leftover networks" "$networks"
show "labrag images of other commits" "$images"
show "dangling images" "$dangling"
docker system df

$apply || exit 0

build() { # every compose file's built images at this commit: ref, digest of its layers
  local f ref
  for f in $(git ls-files '*compose.yaml'); do
    LABRAG_COMMIT=$sha docker compose -f "$f" -p "$project" build >&2
  done
  for ref in $(docker image ls --filter "reference=labrag/*:$sha" \
    --format '{{.Repository}}:{{.Tag}}' | sort); do
    echo "$ref $(docker image inspect --format '{{json .RootFS.Layers}}' "$ref" | sha256sum | cut -c1-12)"
  done
}
# Built before anything is removed: the old images' layers and cache can still serve it.
start=$(date +%s)
built=$(build)

[[ -z $commits ]] || rm -rf $commits
[[ -z $logs ]] || xargs -d '\n' rm -f <<<"$logs"
# rm -f also stops a container that's still running; no job owns one (checked above).
[[ -z $containers ]] || docker rm -f $(cut -f1 <<<"$containers")
[[ -z $networks ]] || docker network rm $(cut -f1 <<<"$networks")
# By ref, without -f: an image a container still uses is refused, not removed.
while IFS=$'\t' read -r ref _; do
  [[ -z $ref ]] || docker image rm "$ref" >/dev/null || echo "  kept $ref (in use)"
done <<<"$images"$'\n'"$dangling"

# until = last used before the build started; the build touched every record it reused.
docker builder prune --all --force --filter "until=$(($(date +%s) - start + 1))s"
echo "rebuilding: every step must come from the cache that was kept"
if ! diff <(echo "$built") <(build); then
  echo "layers changed: the prune removed cache these images' builds use" >&2
  exit 1
fi
echo "after:"
docker system df
