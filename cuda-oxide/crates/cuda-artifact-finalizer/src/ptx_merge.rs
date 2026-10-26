/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver-independent PTX bundle merging shared by AOT and runtime loading.

use oxide_artifacts::{ArtifactPayloadKind, OwnedArtifactBundle};
use std::collections::HashMap;
use std::ops::Range;
use thiserror::Error;

/// Failures while merging retained PTX modules.
#[derive(Debug, Error)]
pub enum PtxMergeError {
    /// No bundle supplied a PTX payload.
    #[error("no embedded CUDA modules were found")]
    NoModules,
    /// A PTX payload was not UTF-8 text.
    #[error("embedded CUDA module '{name}' has no supported payload")]
    UnsupportedPayload { name: String },
    /// PTX could not be parsed or rewritten safely.
    #[error("embedded CUDA module '{name}' contains invalid PTX: {reason}")]
    InvalidPtx { name: String, reason: String },
}

/// Merge the PTX payloads of `bundles`, in iteration order, into one PTX
/// module string: the first PTX bundle keeps its `.version` / `.target` /
/// `.address_size` header directives, every later one contributes its body
/// with those directives stripped. Bundles without a PTX payload are skipped.
///
/// Both build-time finalization and runtime generic loading use this merger.
/// Tests and examples check order-dependent merge properties: module-scope
/// symbol uniqueness and extern alignment must hold for every bundle order,
/// not just the one the current executable happens to embed (#1277).
pub fn merge_ptx_bundles<'a>(
    bundles: impl IntoIterator<Item = &'a OwnedArtifactBundle>,
) -> Result<String, PtxMergeError> {
    let mut merged = String::new();
    let mut found_any = false;
    let mut defined = HashMap::new();

    // Keep each bundle's debug file indices distinct in the merged module.
    let mut file_index_offset = 0;

    for (bundle_index, bundle) in bundles.into_iter().enumerate() {
        if let Some(ptx_bytes) = bundle.payload(ArtifactPayloadKind::Ptx) {
            let ptx_str = std::str::from_utf8(ptx_bytes)
                .map_err(|_| PtxMergeError::UnsupportedPayload {
                    name: bundle.name.clone(),
                })?
                .trim_end_matches('\0');
            let ptx_str = strip_ptx_merge_debug(ptx_str).map_err(|reason| {
                PtxMergeError::InvalidPtx {
                    name: bundle.name.clone(),
                    reason,
                }
            })?;

            let strip_headers = found_any;
            let (body, max_file_index) =
                prepare_bundle_body(&ptx_str, strip_headers, file_index_offset).map_err(
                    |reason| PtxMergeError::InvalidPtx {
                        name: bundle.name.clone(),
                        reason,
                    },
                )?;
            file_index_offset = file_index_offset.max(max_file_index);
            let body = namespace_static_shared(&body, bundle_index).map_err(|reason| {
                PtxMergeError::InvalidPtx {
                    name: bundle.name.clone(),
                    reason,
                }
            })?;
            let body = drop_duplicate_definitions(&body, &bundle.name, &mut defined).map_err(
                |reason| PtxMergeError::InvalidPtx {
                    name: bundle.name.clone(),
                    reason,
                },
            )?;
            merged.push_str(&body);
            if !body.ends_with('\n') {
                merged.push('\n');
            }
            found_any = true;
        }
    }

    if !found_any {
        return Err(PtxMergeError::NoModules);
    }
    Ok(merged)
}

/// Strip repeated headers and shift debug indices past the previous bundles.
/// Returns the rewritten text and its highest debug file index.
fn prepare_bundle_body(
    ptx: &str,
    strip_headers: bool,
    file_index_offset: u64,
) -> Result<(String, u64), String> {
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut edits = ptx_parse::EditScript::new();
    let mut max_file_index = 0u64;

    for directive in document.directives() {
        match directive.name() {
            ".version" | ".target" | ".address_size" if strip_headers => {
                edits
                    .delete(directive.line_span())
                    .map_err(|error| error.to_string())?;
            }
            ".file" | ".loc" => {
                let span = directive.arguments_span();
                let tokens = document.tokens();
                let first = tokens.partition_point(|token| token.span().end <= span.start);
                let arguments: Vec<_> = tokens[first..]
                    .iter()
                    .take_while(|token| token.span().start < span.end)
                    .filter(|token| !token.kind().is_trivia())
                    .collect();
                let index = shift_file_index(
                    &mut edits,
                    ptx,
                    arguments.first().copied(),
                    file_index_offset,
                )?;
                max_file_index = max_file_index.max(index);
                if directive.name() == ".loc" {
                    for (position, attribute) in arguments.windows(2).enumerate() {
                        if attribute[0].text(ptx) == "," && attribute[1].text(ptx) == "inlined_at" {
                            let index = shift_file_index(
                                &mut edits,
                                ptx,
                                arguments.get(position + 2).copied(),
                                file_index_offset,
                            )?;
                            max_file_index = max_file_index.max(index);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let body = edits.apply(ptx).map_err(|error| error.to_string())?;
    Ok((body, max_file_index))
}

/// Shift one complete decimal index token, preserving surrounding PTX.
fn shift_file_index(
    edits: &mut ptx_parse::EditScript,
    ptx: &str,
    token: Option<&ptx_parse::Token>,
    offset: u64,
) -> Result<u64, String> {
    let token = token.ok_or("malformed debug directive: missing file index")?;
    let text = token.text(ptx);
    if token.kind() != ptx_parse::TokenKind::Word || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!("malformed debug file index: {text:?}"));
    }
    let index: u64 = text
        .parse()
        .map_err(|_| format!("debug file index out of range: {text:?}"))?;
    let shifted = index
        .checked_add(offset)
        .ok_or("debug file index overflow")?;
    if offset != 0 {
        edits
            .replace(token.span(), shifted.to_string())
            .map_err(|error| error.to_string())?;
    }
    Ok(shifted)
}

/// Lowering numbers anonymous static shared allocations independently in each
/// module. Their generated names are not cross-bundle linkage identities.
/// Namespace declarations and references before callable deduplication, so
/// functions using different shared storage cannot be silently conflated.
fn namespace_static_shared(ptx: &str, bundle: usize) -> Result<String, String> {
    use ptx_parse::{ScopeId, StatementKind, TokenKind};
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut names = HashMap::new();
    let mut declarations = Vec::new();
    for statement in document.statements() {
        if statement.kind() != StatementKind::Directive { continue; }
        let span = statement.span();
        let tokens = document.tokens();
        let first = tokens.partition_point(|token| token.span().start < span.start);
        let words = tokens[first..].iter()
            .take_while(|token| token.span().end <= span.end)
            .filter(|token| token.kind() == TokenKind::Word)
            .map(|token| token.text(ptx)).collect::<Vec<_>>();
        let anonymous = words.iter().find(|name| name.strip_prefix("__shared_mem_")
            .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|c| c.is_ascii_digit())));
        let Some(&name) = anonymous else { continue };
        if statement.scope() == ScopeId::ROOT {
            if words.contains(&".shared") && !words.contains(&".extern") {
                names.insert(name, format!("__cuda_oxide_bundle_{bundle}_{name}"));
            }
        } else {
            // A locally declared symbol shadows a module global of the same
            // spelling. Leave that declaration and its lexical scope alone.
            declarations.push((name, statement.scope()));
        }
    }
    let mut edits = ptx_parse::EditScript::new();
    for token in document.tokens() {
        if token.kind() != TokenKind::Word { continue; }
        let name = token.text(ptx);
        let Some(replacement) = names.get(name) else { continue };
        if declarations.iter().any(|(local, scope)| *local == name
            && document.scope(*scope).expect("parsed scope").body_span().contains(&token.span().start)) {
            continue;
        }
        edits.replace(token.span(), replacement.clone()).map_err(|error| error.to_string())?;
    }
    edits.apply(ptx).map_err(|error| error.to_string())
}

/// Delete callable definitions already contributed by an earlier bundle.
///
/// `defined` maps each callable name to its first bundle and canonical code.
/// Copies are compared without trivia and with body-local labels renumbered,
/// because label numbering depends on the function's position in its bundle.
fn drop_duplicate_definitions(
    ptx: &str,
    bundle: &str,
    defined: &mut HashMap<String, (String, String)>,
) -> Result<String, String> {
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut edits = ptx_parse::EditScript::new();
    for callable in document.callables() {
        let Some(body) = callable.body_span() else {
            continue;
        };
        let code = canonical_definition(&document, callable.span(), body);
        match defined.get(callable.name()) {
            Some((_, first)) if *first == code => edits
                .delete(callable.span())
                .map_err(|error| error.to_string())?,
            Some((owner, _)) => {
                return Err(format!(
                    "`{}` is defined differently in bundles `{owner}` and `{bundle}`",
                    callable.name()
                ));
            }
            None => {
                defined.insert(callable.name().to_string(), (bundle.to_string(), code));
            }
        }
    }
    edits.apply(ptx).map_err(|error| error.to_string())
}

/// Non-trivia, non-debug tokens of one definition, with labels defined in its
/// body replaced by their ordinal. Debug locations differ between bundles even
/// when the executable code is identical.
fn canonical_definition(
    document: &ptx_parse::Document<'_>,
    span: Range<usize>,
    body: Range<usize>,
) -> String {
    let source = document.source();
    let labels: HashMap<&str, usize> = document
        .labels()
        .iter()
        .filter(|label| body.contains(&label.span().start))
        .enumerate()
        .map(|(ordinal, label)| (label.name(), ordinal))
        .collect();
    let tokens = document.tokens();
    let first = tokens.partition_point(|token| token.span().start < span.start);
    let mut debug_spans = document
        .directives_in(span.clone())
        .filter(|directive| matches!(directive.name(), ".file" | ".loc"))
        .map(|directive| directive.span())
        .peekable();
    let mut code = String::new();
    for token in tokens[first..].iter().take_while(|token| token.span().end <= span.end) {
        if token.kind().is_trivia() {
            continue;
        }
        let start = token.span().start;
        while debug_spans.peek().is_some_and(|debug| debug.end <= start) {
            debug_spans.next();
        }
        if debug_spans.peek().is_some_and(|debug| debug.contains(&start)) {
            continue;
        }
        let text = token.text(source);
        match labels.get(text) {
            Some(ordinal) => {
                code.push_str("$L");
                code.push_str(&ordinal.to_string());
            }
            None => code.push_str(text),
        }
        code.push(' ');
    }
    code
}

/// Remove debug sections whose local string labels collide across bundles.
/// Keep `.file` and `.loc`: `prepare_bundle_body` remaps their numeric indices.
fn strip_ptx_merge_debug(ptx: &str) -> Result<String, String> {
    let mut output = String::with_capacity(ptx.len());
    let mut in_debug_section = false;
    let mut saw_section_body = false;
    let mut debug_section_depth = 0_i32;

    for line in ptx.lines() {
        let trimmed = line.trim_start();
        if !in_debug_section {
            if trimmed.starts_with(".section") && trimmed.contains(".debug_") {
                in_debug_section = true;
                saw_section_body = line.contains('{');
                debug_section_depth =
                    line.bytes().filter(|byte| *byte == b'{').count() as i32
                        - line.bytes().filter(|byte| *byte == b'}').count() as i32;
                if debug_section_depth < 0 {
                    return Err("malformed PTX debug section".to_string());
                }
                if saw_section_body && debug_section_depth == 0 {
                    in_debug_section = false;
                }
                continue;
            }
            output.push_str(line);
            output.push('\n');
            continue;
        }

        saw_section_body |= line.contains('{');
        debug_section_depth += line.bytes().filter(|byte| *byte == b'{').count() as i32;
        debug_section_depth -= line.bytes().filter(|byte| *byte == b'}').count() as i32;
        if debug_section_depth < 0 {
            return Err("malformed PTX debug section".to_string());
        }
        if saw_section_body && debug_section_depth == 0 {
            in_debug_section = false;
            saw_section_body = false;
        }
    }

    if in_debug_section {
        return Err("unterminated PTX debug section".to_string());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_artifacts::ArtifactCompileOptions;

    fn bundle_with_target(target: &str) -> OwnedArtifactBundle {
        OwnedArtifactBundle {
            name: "demo".to_string(),
            target: target.to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[test]
    fn merge_debug_strip_keeps_local_depots() {
        let ptx = ".file 1 \"a.rs\"\n.loc 1 2 3\n\t.local .align 8 .b8 __local_depot12[16];\n\tmov.u64 %rd1, __local_depot12;\n";
        let stripped = strip_ptx_merge_debug(ptx).unwrap();
        let (body, _) = prepare_bundle_body(&stripped, false, 3).unwrap();
        assert!(body.contains(".file 4 \"a.rs\""));
        assert!(body.contains(".loc 4 2 3"));
        assert!(body.contains(".local .align 8 .b8 __local_depot12[16];"));
    }

    fn ptx_bundle(name: &str, ptx: &str) -> OwnedArtifactBundle {
        use oxide_artifacts::OwnedArtifactPayload;
        OwnedArtifactBundle {
            name: name.to_string(),
            target: "sm_80".to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: vec![OwnedArtifactPayload {
                kind: ArtifactPayloadKind::Ptx,
                name: name.to_string(),
                bytes: ptx.as_bytes().to_vec(),
            }],
            entries: Vec::new(),
        }
    }

    /// The merge is order-explicit: exactly one header set (the first PTX
    /// bundle's), every body present in iteration order, non-PTX bundles
    /// skipped. #1277's collision-free-namespace example checks the same
    /// function under both bundle orders at runtime.
    #[test]
    fn merges_bundles_in_iteration_order_with_one_header_set() {
        let first = ptx_bundle(
            "first",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry a() { ret; }\n",
        );
        let second = ptx_bundle(
            "second",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry b() { ret; }\n",
        );
        let skipped = bundle_with_target("sm_80");

        let forward = merge_ptx_bundles([&first, &skipped, &second]).unwrap();
        assert_eq!(forward.matches(".version").count(), 1);
        assert_eq!(forward.matches(".target sm_80").count(), 1);
        assert!(forward.find(".entry a()").unwrap() < forward.find(".entry b()").unwrap());

        let reversed = merge_ptx_bundles([&second, &first]).unwrap();
        assert_eq!(reversed.matches(".version").count(), 1);
        assert!(reversed.find(".entry b()").unwrap() < reversed.find(".entry a()").unwrap());

        assert!(matches!(
            merge_ptx_bundles([&skipped]),
            Err(PtxMergeError::NoModules)
        ));
    }

    #[test]
    fn strips_only_structural_ptx_module_headers() {
        let ptx = "\
// .target sm_1
.version 8.9
.target sm_120a, debug
.address_size 64
.visible .entry kernel() { ret; }
";
        let (body, max_file_index) = prepare_bundle_body(ptx, true, 0).unwrap();
        assert_eq!(body, "// .target sm_1\n.visible .entry kernel() { ret; }\n");
        assert_eq!(max_file_index, 0);
    }

    #[test]
    fn debug_indices_ignore_comments_paths_and_label_substrings() {
        let ptx = ".file /* 90 */ 1 \"source12.rs\"\n.loc /* 80 */ 1 2 3, function_name $inlined_at7, inlined_at /* 70 */ 1 4 5 // inlined_at 60\n";
        let (body, max_index) = prepare_bundle_body(ptx, false, 3).unwrap();
        assert_eq!(
            body,
            ".file /* 90 */ 4 \"source12.rs\"\n.loc /* 80 */ 4 2 3, function_name $inlined_at7, inlined_at /* 70 */ 4 4 5 // inlined_at 60\n"
        );
        assert_eq!(max_index, 4);
    }

    #[test]
    fn malformed_debug_indices_and_overflow_return_errors() {
        for ptx in [
            ".file \"source12.rs\"\n",
            ".file /* 12 */\n",
            ".file -1 \"source.rs\"\n",
            ".file 1foo \"source.rs\"\n",
            ".loc 1 2 3, inlined_at\n",
            ".loc 1 2 3, inlined_at -1 4 5\n",
            ".file 18446744073709551616 \"source.rs\"\n",
        ] {
            assert!(prepare_bundle_body(ptx, false, 1).is_err(), "{ptx}");
        }
        assert!(
            prepare_bundle_body(".file 18446744073709551615 \"source.rs\"\n", false, 1)
                .unwrap_err()
                .contains("overflow")
        );
    }

    #[test]
    fn merged_bundles_renumber_debug_file_indices_per_bundle() {
        let first = ptx_bundle(
            "first",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"a/lib.rs\"
.file 2 \"shared/thread.rs\"
.visible .entry a()
{
\t.loc\t1 10 5
\tret;
}
",
        );
        let second = ptx_bundle(
            "second",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"b/main.rs\"
.visible .entry b()
{
\t.loc\t1 3 1, function_name $f, inlined_at 1 7 0
\tret;
}
",
        );
        let plain = ptx_bundle(
            "plain",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry c() { ret; }\n",
        );

        let merged = merge_ptx_bundles([&first, &plain, &second]).unwrap();

        // First bundle's table and references are untouched.
        assert!(merged.contains(".file 1 \"a/lib.rs\""));
        assert!(merged.contains(".file 2 \"shared/thread.rs\""));
        assert!(merged.contains(".loc\t1 10 5"));

        // The debug-free middle bundle does not advance the offset; the
        // second debug bundle continues after the first's maximum of 2 —
        // declaration, `.loc` reference, and `inlined_at` reference alike.
        assert!(merged.contains(".file 3 \"b/main.rs\""));
        assert!(merged.contains(".loc\t3 3 1, function_name $f, inlined_at 3 7 0"));

        // No index is declared twice across the merged module.
        let declared: Vec<&str> = merged
            .lines()
            .filter(|line| line.trim_start().starts_with(".file"))
            .map(|line| line.split_whitespace().nth(1).unwrap())
            .collect();
        let mut deduped = declared.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(
            declared.len(),
            deduped.len(),
            "duplicate .file index: {declared:?}"
        );
    }

    #[test]
    fn strips_module_scoped_debug_records_before_merge() {
        let ptx = "\
.version 7.8
.file 1 \"kernel.rs\"
.visible .entry kernel() {
    .loc 1 7 3
    ret;
}
.section .debug_str
{
$L__info_string0:
.b8 0
}
.section .debug_macinfo { }
";
        assert_eq!(
            strip_ptx_merge_debug(ptx).unwrap(),
            ".version 7.8\n.file 1 \"kernel.rs\"\n.visible .entry kernel() {\n    .loc 1 7 3\n    ret;\n}\n"
        );
    }

    #[test]
    fn merge_drops_relabelled_copies_and_rejects_conflicting_definitions() {
        let first = "\
.visible .func (.param .b32 r) pick(.param .b32 a)
{
    @%p1 bra $L__BB14_2;
$L__BB14_2:                 // %bb2
    ret;
}
.visible .entry one() { ret; }
";
        let relabelled = "\
.visible .func (.param .b32 r) pick(.param .b32 a)
{
    @%p1 bra $L__BB33_2;
$L__BB33_2:                 // %bb7
    ret;
}
.visible .entry two() { ret; }
";
        let mut defined = HashMap::new();
        assert_eq!(drop_duplicate_definitions(first, "a", &mut defined).unwrap(), first);
        let kept = drop_duplicate_definitions(relabelled, "b", &mut defined).unwrap();
        assert!(!kept.contains("pick("), "identical copy must be dropped: {kept}");
        assert!(kept.contains(".visible .entry two()"));

        let conflicting = relabelled.replace("ret;\n}", "exit;\n}");
        let error = drop_duplicate_definitions(&conflicting, "c", &mut defined).unwrap_err();
        assert!(error.contains("`pick`") && error.contains("`a`") && error.contains("`c`"), "{error}");
    }

    #[test]
    fn merge_deduplicates_functions_with_remapped_debug_locations() {
        let first = ptx_bundle(
            "first",
            ".file 1 \"a.rs\"\n.visible .func shared() {\n.loc 1 2 3\nret;\n}\n",
        );
        let second = ptx_bundle(
            "second",
            ".file 1 \"b.rs\"\n.visible .func shared() {\n.loc 1 4 5\nret;\n}\n",
        );
        let merged = merge_ptx_bundles([&first, &second]).unwrap();
        assert_eq!(merged.matches(".func shared()").count(), 1);
        assert!(merged.contains(".file 2 \"b.rs\""));
        assert!(merged.contains(".loc 1 2 3"));
    }

    #[test]
    fn merge_namespaces_anonymous_shared_declarations_and_references() {
        let source = "\
.visible .shared .align 16 .b8 __shared_mem_0[4096];
.visible .shared .align 32 .b8 __shared_mem_10[33856];
.extern .shared .align 16 .b8 dynamic[];
.visible .entry first() {
    // __shared_mem_0 is a comment, not a reference.
    mov.u64 %rd1, __shared_mem_0;
    ld.shared.u32 %r1, [__shared_mem_10+32];
    ret;
}
";
        let first = namespace_static_shared(source, 0).unwrap();
        let second = namespace_static_shared(
            &source.replace("[4096]", "[32768]").replace("first()", "second()"), 1).unwrap();
        assert!(first.contains(".b8 __cuda_oxide_bundle_0___shared_mem_0[4096]"));
        assert!(first.contains("mov.u64 %rd1, __cuda_oxide_bundle_0___shared_mem_0;"));
        assert!(first.contains("[__cuda_oxide_bundle_0___shared_mem_10+32]"));
        assert!(first.contains("// __shared_mem_0 is a comment"));
        assert!(first.contains(".b8 dynamic[];"));
        assert!(second.contains(".b8 __cuda_oxide_bundle_1___shared_mem_0[32768]"));
        let mut definitions = HashMap::new();
        drop_duplicate_definitions(&first, "a", &mut definitions).unwrap();
        drop_duplicate_definitions(&second, "b", &mut definitions).unwrap();
        // Equal callable text with different static storage is not deduplicated.
        let conflict = namespace_static_shared(source, 2).unwrap();
        assert!(drop_duplicate_definitions(&conflict, "c", &mut definitions).is_err());
    }

    #[test]
    fn merge_preserves_local_shadows_and_explicit_linkage_names() {
        let source = "\
.shared .b8 __shared_mem_0[16];
.shared .b8 named_storage[16];
.visible .entry first() {
    mov.u64 %rd1, __shared_mem_0;
    {
        .shared .b8 __shared_mem_0[8];
        mov.u64 %rd2, __shared_mem_0;
    }
    mov.u64 %rd3, __shared_mem_0;
    ret;
}
";
        let rewritten = namespace_static_shared(source, 3).unwrap();
        assert!(rewritten.contains(".shared .b8 named_storage[16];"));
        assert!(rewritten.contains(".shared .b8 __shared_mem_0[8];"));
        assert!(rewritten.contains("mov.u64 %rd2, __shared_mem_0;"));
        assert!(rewritten.contains("mov.u64 %rd1, __cuda_oxide_bundle_3___shared_mem_0;"));
        assert!(rewritten.contains("mov.u64 %rd3, __cuda_oxide_bundle_3___shared_mem_0;"));
    }
}
