#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="$example_dir/ordering_vote.ptx"
binary="$example_dir/target/release/ordering_vote"
test -s "$ptx"
test -x "$binary"

"$binary"
fault_log="$(mktemp)"
cubin="$(mktemp)"
ptxas_log="$(mktemp)"
sass="$(mktemp)"
trap 'rm -f "$fault_log" "$cubin" "$ptxas_log" "$sass"' EXIT
if CUDA_OXIDE_ORDER_VOTE_INJECT_FAULT=1 "$binary" >"$fault_log" 2>&1; then
    printf '%s\n' 'injected ordering fault was not detected' >&2
    exit 1
fi
python3 - "$fault_log" <<'PY'
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
if "signed Ord::cmp mismatch at 0" not in text:
    raise SystemExit("fault run failed for an unexpected reason")
PY

python3 - "$ptx" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()

def entry(name: str) -> str:
    match = re.search(r"\.entry\s+" + re.escape(name) + r"\(", text)
    if match is None:
        raise SystemExit(f"missing PTX entry {name!r}")
    begin = text.find("{", match.end())
    end = text.find("\n}", begin)
    if begin < 0 or end < 0:
        raise SystemExit(f"incomplete PTX entry {name!r}")
    return text[begin : end + 2]

ordering = entry("ordering_edges")
vote = entry("vote_edges")
for instruction in (
    "setp.lt.s32",
    "setp.gt.s32",
    "setp.lt.u32",
    "setp.gt.u32",
    "setp.lt.f32",
    "setp.le.f32",
    "setp.gt.f32",
    "setp.ge.f32",
    "setp.eq.f32",
    "setp.neu.f32",
):
    if instruction not in ordering:
        raise SystemExit(f"ordering PTX is missing {instruction}")
if ordering.count("setp.neu.f32") != 1:
    raise SystemExit("f32 != must use exactly one unordered comparison")
if vote.count("activemask.b32") != 2:
    raise SystemExit("vote PTX must preserve straight-line and divergent active-mask reads")
if vote.count("vote.sync.ballot.b32") != 3:
    raise SystemExit("vote PTX must preserve all three masked ballots")
for instruction in ("vote.sync.any.pred", "vote.sync.all.pred"):
    if vote.count(instruction) != 1:
        raise SystemExit(f"vote PTX must preserve exactly one {instruction}")
PY

ptxas -arch=sm_89 -v -o "$cubin" "$ptx" 2>"$ptxas_log"
cuobjdump --dump-sass "$cubin" >"$sass"
python3 - "$ptxas_log" "$sass" <<'PY'
import pathlib
import re
import sys

log = pathlib.Path(sys.argv[1]).read_text()
sass = pathlib.Path(sys.argv[2]).read_text()
for name in ("ordering_edges", "vote_edges"):
    marker = f"Function properties for {name}"
    begin = log.find(marker)
    if begin < 0:
        raise SystemExit(f"ptxas did not report {name}")
    section = log[begin : begin + 180]
    for wanted in ("0 bytes stack frame", "0 bytes spill stores", "0 bytes spill loads"):
        if wanted not in section:
            raise SystemExit(f"{name}: missing ptxas guarantee {wanted!r}")

if "EF_CUDA_SM89" not in sass:
    raise SystemExit("SASS is not assembled for sm_89")

def function(name: str) -> str:
    marker = f"Function : {name}"
    begin = sass.find(marker)
    if begin < 0:
        raise SystemExit(f"missing SASS function {name!r}")
    end = sass.find("Function :", begin + len(marker))
    return sass[begin : len(sass) if end < 0 else end]

ordering = function("ordering_edges")
vote = function("vote_edges")
if "ISETP.GT.U32" not in ordering:
    raise SystemExit("unsigned Ord::cmp lost its unsigned SASS predicate")
if not re.search(r"\bISETP\.GT\.AND\b", ordering):
    raise SystemExit("signed Ord::cmp lost its signed SASS predicate")
if "FSETP.NEU" not in ordering:
    raise SystemExit("f32 != lost unordered NaN-aware SASS semantics")
if "VOTE.ALL" not in vote or vote.count("VOTE.ANY") < 5:
    raise SystemExit("masked ballot/any/all did not lower to hardware vote instructions")
for body, name in ((ordering, "ordering_edges"), (vote, "vote_edges")):
    if re.search(r"\b(?:LDL|STL)\b", body):
        raise SystemExit(f"{name} unexpectedly uses local-memory SASS")
PY

printf '%s\n' 'ordering/vote semantics, injected fault, and sm_89 PTX/SASS shape: PASS'
