#!/usr/bin/env bash
# Clean one node down to what this commit uses, as a job per node:
#
#   slurm/submit.sh -w <node> --cpus-per-task=16 -- slurm/clean-node.sh [--apply]
#
# Lists by default; --apply removes:
# - run root: commits/<sha>/ of every other commit (a job rebuilds its own), and logs/
#   whose copy in the shared logs dir is identical
# - containers/networks of dead jobs (labrag-eval-<job>, labrag-<job>-<port>)
# - every image this commit can't use: kept are labrag/*:<this sha> and images pinned by
#   digest in configs/containers.yaml, compose files and Dockerfiles. Node-wide, not
#   only labrag's; an image some container uses is refused, not removed.
# - build cache the rebuild didn't use: --apply first builds every compose image of this
#   commit, then prunes cache last used before that build, then builds again and fails
#   unless every image id stayed the same (the cache it kept is the cache jobs need).
# Refuses while another labrag job runs on the node; holds setup.lock throughout, so a
# job starting meanwhile waits in its setup.
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

exec 9>"$root/setup.lock"
flock 9
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
    [[ -f $f ]] && cmp -s "$f" "$shared/logs/${f##*/}" && echo "$f"
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
pinned=$(cat configs/containers.yaml $(git ls-files '*compose.yaml' '*Dockerfile*') |
  grep -oE 'sha256:[0-9a-f]{64}' | sort -u | paste -sd' ')
# ref<TAB>id size; ref = repo:tag, repo@digest when untagged, id when dangling.
images=$(
  docker image ls --digests --no-trunc \
    --format '{{.Repository}}	{{.Tag}}	{{.Digest}}	{{.ID}}	{{.Size}}' |
    awk -F'\t' -v sha="$sha" -v pinned="$pinned" '
      BEGIN { n = split(pinned, p, " "); for (i = 1; i <= n; i++) keep[p[i]] = 1 }
      $3 in keep || ($1 ~ /^labrag\// && $2 == sha) { next }
      { ref = $2 != "<none>" ? $1 ":" $2 : $3 != "<none>" ? $1 "@" $3 : $4
        print ref "\t" $4 " " $5 }' |
    sort -u
)

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
show "images this commit can't use" "$images"
docker system df

$apply || exit 0
[[ -z $commits ]] || rm -rf $commits
[[ -z $logs ]] || xargs -d '\n' rm -f <<<"$logs"
# rm -f also stops a container that's still running; no job owns one (checked above).
[[ -z $containers ]] || docker rm -f $(cut -f1 <<<"$containers")
[[ -z $networks ]] || docker network rm $(cut -f1 <<<"$networks")
# By ref, without -f: an image a container still uses is refused, not removed.
while IFS=$'\t' read -r ref _; do
  [[ -z $ref ]] || docker image rm "$ref" >/dev/null || echo "  kept $ref (in use)"
done <<<"$images"

build() { # every compose file's built images, at this commit
  local f
  for f in $(git ls-files '*compose.yaml'); do
    LABRAG_COMMIT=$sha docker compose -f "$f" -p "$project" build >&2
  done
  docker image ls --no-trunc --filter "reference=labrag/*:$sha" \
    --format '{{.Repository}}:{{.Tag}} {{.ID}}' | sort
}
start=$(date +%s)
built=$(build)
# until = last used before the build started; the build touched every record it reused.
docker builder prune --all --force --filter "until=$(($(date +%s) - start + 1))s"
echo "rebuilding: every step must come from the cache that was kept"
if ! diff <(echo "$built") <(build); then
  echo "image ids changed: the prune removed cache this commit's builds use" >&2
  exit 1
fi
echo "after:"
docker system df
