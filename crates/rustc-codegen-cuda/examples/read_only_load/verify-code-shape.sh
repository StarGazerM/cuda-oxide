#!/usr/bin/env bash
set -euo pipefail
example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="$example_dir/read_only_load.ptx"
test -s "$ptx"

python3 - "$ptx" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()

def entry(needle: str) -> str:
    match = re.search(r"\.entry\s+([^\s(]*" + re.escape(needle) + r"[^\s(]*)\(", text)
    if match is None:
        raise SystemExit(f"missing PTX entry containing {needle!r}")
    begin = text.find("{", match.end())
    end = text.find("\n}", begin)
    if begin < 0 or end < 0:
        raise SystemExit(f"incomplete PTX entry containing {needle!r}")
    return text[begin : end + 2]

quad = entry("sum_quads")
tile = entry("sum_proven_tile")
checked = entry("checked_index")

if len(re.findall(r"ld\.global\.nc\.v2\.u64", quad)) != 1:
    raise SystemExit("read-only quad did not remain one 16-byte cache-qualified load")
if re.search(r"ld\.global\.nc\.(?:u32|b32)", quad):
    raise SystemExit("read-only quad was scalarized")

for forbidden in (".local", "ld.local", "st.local", "stacksave", "stackrestore"):
    if forbidden in tile:
        raise SystemExit(f"proved ThreadTile materialized local state through {forbidden!r}")
loads = re.findall(r"ld\.global\.nc\.(?:u32|b32)", tile)
if len(loads) != 16:
    raise SystemExit(f"proved ThreadTile has {len(loads)} direct scalar loads; expected 16")
if tile.count("trap;") > 1:
    raise SystemExit("proved ThreadTile retained per-access trap edges")
if "trap;" not in checked:
    raise SystemExit("ordinary unproved slice indexing lost its out-of-bounds trap")
if not re.search(r"ld\.global\.(?!nc\.)(?:u32|b32)", checked):
    raise SystemExit("ordinary checked slice indexing lost its direct global load")
PY

if command -v ptxas >/dev/null 2>&1; then
    cubin="$(mktemp)"
    log="$(mktemp)"
    sass="$(mktemp)"
    trap 'rm -f "$cubin" "$log" "$sass"' EXIT
    ptxas -arch=sm_86 -v -o "$cubin" "$ptx" 2>"$log"
    if grep -Eq '[1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill (stores|loads)' "$log"; then
        cat "$log" >&2
        printf '%s\n' 'read-only fixture has a local stack frame or register spills' >&2
        exit 1
    fi
    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump --dump-sass "$cubin" >"$sass"
        python3 - "$sass" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()

def function(name: str) -> str:
    marker = f"Function : {name}"
    begin = text.find(marker)
    if begin < 0:
        raise SystemExit(f"missing SASS function {name!r}")
    end = text.find("Function :", begin + len(marker))
    return text[begin : len(text) if end < 0 else end]

tile = function("sum_proven_tile")
checked = function("checked_index")
if len(re.findall(r"\bLDG\.E\.CONSTANT\b", tile)) != 16:
    raise SystemExit("proved ThreadTile did not lower to 16 direct read-only SASS loads")
if re.search(r"\b(?:LDL|STL)\b", tile):
    raise SystemExit("proved ThreadTile retained local-memory SASS")
if tile.count("BPT.TRAP") > 1:
    raise SystemExit("proved ThreadTile retained per-access SASS trap edges")
if checked.count("BPT.TRAP") != 1:
    raise SystemExit("ordinary checked slice indexing lost its SASS trap")
if not re.search(r"\bLDG\.E\b(?!\.CONSTANT)", checked):
    raise SystemExit("ordinary checked slice indexing lost its SASS global load")
PY
    fi
fi

printf '%s\n' 'read-only vector and proof-carrying tile code shape: PASS'
