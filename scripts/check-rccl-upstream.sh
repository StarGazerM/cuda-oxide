#!/usr/bin/env bash
set -euo pipefail

upstream_ref="${1:-upstream/main}"
head_ref="${2:-HEAD}"
report="${RCCL_UPSTREAM_REPORT:-/tmp/rccl-upstream-report.md}"

upstream_sha="$(git rev-parse "$upstream_ref")"
head_sha="$(git rev-parse "$head_ref")"
cherry="$(git cherry "$upstream_ref" "$head_ref")"
remaining="$(printf '%s\n' "$cherry" | sed -n 's/^+ //p' | wc -l | tr -d ' ')"
absorbed="$(printf '%s\n' "$cherry" | sed -n 's/^- //p' | wc -l | tr -d ' ')"

merge_state=clean
merge_detail='The RCCL patch branch merges with upstream without textual conflicts.'
if ! git merge-tree --write-tree "$upstream_ref" "$head_ref" >/tmp/rccl-merge-tree.out 2>/tmp/rccl-merge-tree.err; then
  merge_state=conflict
  merge_detail='The RCCL patch branch conflicts with upstream and needs a manual rebase.'
fi

necessity='semantic review required'
if [[ "$remaining" == 0 ]]; then
  necessity='all RCCL patch commits have patch-equivalent changes upstream; removal should be tested'
elif [[ "$absorbed" != 0 ]]; then
  necessity='some RCCL patch commits are upstream; split/remove absorbed commits and test the remainder'
fi

cat >"$report" <<EOF
# RCCL cuda-oxide upstream check

- Upstream: \`$upstream_ref\` at \`$upstream_sha\`
- RCCL branch: \`$head_ref\` at \`$head_sha\`
- Patch commits still unique to RCCL: **$remaining**
- Patch-equivalent commits found upstream: **$absorbed**
- Textual integration: **$merge_state**
- Patch necessity: **$necessity**

$merge_detail

\`git cherry\` detects patch-equivalent commits, not semantic equivalence after a rewrite. A passing result never removes a patch automatically. Rebase in a temporary branch, run cuda-oxide tests plus RCCL's clean SM89 gates, then remove only behavior already provided upstream.
EOF

cat "$report"
[[ "$merge_state" == clean ]]
