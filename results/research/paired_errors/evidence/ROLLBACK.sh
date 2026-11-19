#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
target="${1:-$here/../../../../scripts/run_competitor_benchmark.py}"
check() { test "$(sha256sum "$1" | cut -d' ' -f1)" = "$2"; }
check "$here/run_competitor_benchmark.original" 4b3b9594c66fb53ea915039c87e0b95d84c2bdbece4496cccca01a0eba8ff733
check "$target" 554be7aeef042f0788721abe916c7ab483a32baf382f5ef536aa4285d4c5823c
cp -- "$here/run_competitor_benchmark.original" "$target"
check "$target" 4b3b9594c66fb53ea915039c87e0b95d84c2bdbece4496cccca01a0eba8ff733
echo "restored 4b3b9594c66fb53ea915039c87e0b95d84c2bdbece4496cccca01a0eba8ff733"
