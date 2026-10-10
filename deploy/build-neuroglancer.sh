#!/usr/bin/env bash
# Build the server image's pinned Neuroglancer for a source-based installation.
# Usage: bash deploy/build-neuroglancer.sh /absolute/path/to/neuroglancer
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
output_dir="${1:?Pass the directory to hold the built Neuroglancer client}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
# One pin for both Docker and local installs.
commit="$(sed -n 's/^ARG NEUROGLANCER_COMMIT=//p' "$repo_dir/deploy/docker/server.Dockerfile")"
if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Missing or invalid Neuroglancer commit in server.Dockerfile" >&2
    exit 1
fi
# The pinned release's OBJ decoder calls shrinkToFit(), which incorrectly
# references a global `length`. Apply the same small fix in both builds.
patch_file="$repo_dir/deploy/neuroglancer-obj.patch"
revision="$commit $(cksum < "$patch_file")"
if [ -f "$output_dir/index.html" ] && [ -f "$output_dir/.ml4paleo-build" ] \
    && [ "$(cat "$output_dir/.ml4paleo-build")" = "$revision" ]; then
    echo "Using Neuroglancer in $output_dir"
    exit 0
fi
source_dir="$(mktemp -d "${TMPDIR:-/tmp}/ml4paleo-neuroglancer.XXXXXXXX")"
trap 'rm -rf -- "$source_dir"' EXIT
git -C "$source_dir" init -q
git -C "$source_dir" remote add origin https://github.com/google/neuroglancer.git
git -C "$source_dir" fetch -q --depth 1 origin "$commit"
git -C "$source_dir" checkout -q FETCH_HEAD
git -C "$source_dir" apply "$patch_file"
(
    cd "$source_dir"
    export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
    npm ci --no-audit --no-fund
    npm run build -- --no-typecheck --no-lint
    test -f dist/client/index.html
)
# Publish the entry point last: interrupted copies must not look ready on restart.
for asset in "$source_dir"/dist/client/*; do
    [ "$(basename "$asset")" = index.html ] || cp -R "$asset" "$output_dir/"
done
cp "$source_dir/dist/client/index.html" "$output_dir/index.html"
printf '%s\n' "$revision" > "$output_dir/.ml4paleo-build"
echo "Built Neuroglancer in $output_dir"
