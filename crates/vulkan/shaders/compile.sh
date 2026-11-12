#!/usr/bin/env bash
# Regenerates spv/<shader>.spv from every .vert, .frag and .comp here and validates each with
# spirv-val. Needs glslangValidator and spirv-val (Ubuntu: glslang-tools spirv-tools). Output is
# deterministic: no optimizer and no debug info.
#
#   ./compile.sh          rewrite spv/
#   ./compile.sh --check  compile into a scratch directory and fail if spv/ differs
set -euo pipefail

cd "$(dirname "$0")"

out=spv
if [[ "${1:-}" == "--check" ]]; then
    out=$(mktemp -d)
    trap 'rm -rf "$out"' EXIT
else
    rm -f spv/*.spv
fi
mkdir -p "$out"

for src in *.vert *.frag *.comp; do
    glslangValidator --quiet -V --target-env vulkan1.3 \
        -Iinclude -Ithird_party/fsr -Ithird_party/smaa \
        -o "$out/$src.spv" "$src"
    spirv-val --target-env vulkan1.3 "$out/$src.spv"
done

if [[ "$out" != spv ]]; then
    diff -r "$out" spv
fi
