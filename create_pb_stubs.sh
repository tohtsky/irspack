#!/usr/bin/env bash
# Regenerate the source-tree stubs used by mypy from the wheel contents.
# The wheel itself generates these files through nanobind_add_stub() in CMake.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$(mktemp -d)"
trap 'rm -rf "$output_dir"' EXIT

cd "$repo_root"
uv build --wheel --out-dir "$output_dir"

shopt -s nullglob
wheels=("$output_dir"/*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one wheel, found ${#wheels[@]}." >&2
  exit 1
fi

stubs=(
  "irspack/recommenders/_ials_core.pyi"
  "irspack/evaluation/_core_evaluator.pyi"
  "irspack/recommenders/_knn.pyi"
  "irspack/utils/_util_cpp.pyi"
)

for stub in "${stubs[@]}"; do
  unzip -p "${wheels[0]}" "$stub" > "src/$stub"
done

# Keep the checked-in snapshots compliant with the repository's formatter.
stub_paths=("${stubs[@]/#/src/}")
uv run --group dev ruff check --fix "${stub_paths[@]}"
uv run --group dev ruff format "${stub_paths[@]}"
