#!/usr/bin/env bash
set -eu
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
original="$root/scripts/run_capdrift_recovery.py"
target="${1:-$root/scripts/run_capdrift_classifier_scope.py}"
check() { test "$(sha256sum "$1" | cut -d' ' -f1)" = "$2"; }
check "$original" f1573dd16813cfbd75f11173b2d283ef6adf9dcdaac599973b81c588259b8a2e
check "$target" 8195b47e0cae8dabefcce0644c9c9c1e88371cb42bc81840e4f12f444188d9a8
cp -- "$original" "$target"
check "$target" f1573dd16813cfbd75f11173b2d283ef6adf9dcdaac599973b81c588259b8a2e
echo "restored f1573dd16813cfbd75f11173b2d283ef6adf9dcdaac599973b81c588259b8a2e"
