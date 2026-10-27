//! Independent kernel groups for a parallel middle-end and `llc`.
//!
//! Kernels never call one another, so a module splits into groups which each
//! keep their own kernels plus private copies of everything those kernels
//! reach. Every group runs `opt` and `llc` in its own process; the module-scope
//! internal PTX symbols of later groups are renamed and the groups are
//! concatenated into one PTX module under the first group's header.
//!
//! The split never changes what a symbol denotes. Modules are not split when a
//! copy could: a defined, mutable global outside shared memory would become one
//! object per group, and a public non-function symbol (host-visible global or
//! constant) must remain a single definition. Public device functions stay
//! public only in the first group; other groups keep internal copies.

use ptx_parse::{CallableKind, Document, TokenKind};
use std::collections::BTreeSet;

/// Below this many kernels per group, process start-up and the repeated IR
/// parse outweigh the parallel middle-end.
const MIN_KERNELS_PER_PART: usize = 8;

/// Upper bound on groups; beyond it, duplicated helpers grow the PTX while the
/// remaining serial work (merge, `ptxas`) dominates.
const MAX_PARTS: usize = 16;

/// `CUDA_OXIDE_PTX_PARTITIONS` selects the group count; `0` or `1` disables
/// splitting. The default is the available parallelism, capped at 16.
pub(crate) const PARTITIONS_ENV: &str = "CUDA_OXIDE_PTX_PARTITIONS";

/// One kernel group: rewritten LLVM IR and the symbols `opt` keeps public.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Part {
    pub ir: String,
    pub public: Vec<String>,
}

pub(crate) fn requested_parts() -> usize {
    match std::env::var(PARTITIONS_ENV) {
        Ok(value) => value.trim().parse().unwrap_or(1),
        Err(_) => std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_PARTS),
    }
}

/// Split `ir` into kernel groups, or `None` when it must stay one module.
pub(crate) fn plan(ir: &str, public_symbols: &[String], requested: usize) -> Option<Vec<Part>> {
    let kernels = kernel_names(ir);
    let parts = requested
        .min(MAX_PARTS)
        .min(kernels.len() / MIN_KERNELS_PER_PART);
    if parts < 2 {
        return None;
    }
    let functions = defined_functions(ir);
    let kernel_set: BTreeSet<&str> = kernels.iter().map(String::as_str).collect();
    if public_symbols
        .iter()
        .any(|symbol| !functions.contains(symbol.as_str()))
    {
        return None;
    }
    if has_duplicable_mutable_global(ir) {
        return None;
    }
    let shared_public: Vec<String> = public_symbols
        .iter()
        .filter(|symbol| !kernel_set.contains(symbol.as_str()))
        .cloned()
        .collect();
    let mut groups = vec![Vec::new(); parts];
    for (index, kernel) in kernels.iter().enumerate() {
        groups[index % parts].push(kernel.clone());
    }
    let mut result = Vec::with_capacity(parts);
    for (index, group) in groups.into_iter().enumerate() {
        let group_set: BTreeSet<&str> = group.iter().map(String::as_str).collect();
        let keep = |name: &str| {
            if kernel_set.contains(name) {
                group_set.contains(name)
            } else {
                index == 0
            }
        };
        let ir = restrict_used_lists(ir, &keep)?;
        let mut public = group;
        if index == 0 {
            public.extend(shared_public.iter().cloned());
        }
        result.push(Part { ir, public });
    }
    Some(result)
}

/// Merge the PTX of each group into one module.
pub(crate) fn merge(parts: &[String]) -> Result<String, String> {
    let Some((first, rest)) = parts.split_first() else {
        return Err("no PTX partitions to merge".to_string());
    };
    let header = module_header(first)?;
    let mut merged = first.clone();
    let mut taken = all_words(first)?;
    for (offset, ptx) in rest.iter().enumerate() {
        let index = offset + 1;
        if module_header(ptx)? != header {
            return Err(format!(
                "PTX partition {index} declares a different .version/.target/.address_size header"
            ));
        }
        let suffix = format!("_oxpart{index}");
        let internal = internal_module_symbols(ptx)?;
        let renamed = rename_words(ptx, &internal, &suffix)?;
        for name in &internal {
            let replacement = format!("{name}{suffix}");
            if taken.contains(&replacement) {
                return Err(format!(
                    "renaming internal PTX symbol `{name}` would collide with `{replacement}`"
                ));
            }
        }
        let body = strip_header(&renamed)?;
        taken.extend(all_words(&body)?);
        merged.push('\n');
        merged.push_str(&body);
    }
    Ok(merged)
}

fn kernel_names(ir: &str) -> Vec<String> {
    ir.lines()
        .filter(|line| line.starts_with("define ") && line.contains(" ptx_kernel "))
        .filter_map(defined_name)
        .collect()
}

fn defined_functions(ir: &str) -> BTreeSet<&str> {
    ir.lines()
        .filter(|line| line.starts_with("define "))
        .filter_map(defined_name_str)
        .collect()
}

fn defined_name(line: &str) -> Option<String> {
    defined_name_str(line).map(str::to_string)
}

/// The unquoted symbol of a `define … @name(` line.
fn defined_name_str(line: &str) -> Option<&str> {
    let at = line.find('@')?;
    global_name(&line[at + 1..]).map(|(name, _)| name)
}

/// Parse an LLVM global name at the start of `text` (after `@`), returning the
/// unquoted name and the length of its spelling.
fn global_name(text: &str) -> Option<(&str, usize)> {
    if let Some(quoted) = text.strip_prefix('"') {
        let end = quoted.find('"')?;
        let name = &quoted[..end];
        // Escaped spellings (`\22`) are not decoded; refuse rather than guess.
        (!name.contains('\\')).then_some((name, end + 2))
    } else {
        let end = text
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$' | '-')))
            .unwrap_or(text.len());
        (end > 0).then_some((&text[..end], end))
    }
}

/// A defined, non-constant global outside shared memory (`addrspace(3)`)
/// would be duplicated per group.
fn has_duplicable_mutable_global(ir: &str) -> bool {
    ir.lines()
        .filter(|line| line.starts_with('@') && !line.starts_with("@llvm."))
        .any(|line| {
            let Some((_, definition)) = line.split_once(" = ") else {
                return true;
            };
            let words: Vec<&str> = definition.split_whitespace().collect();
            let Some(kind) = words
                .iter()
                .position(|word| matches!(*word, "global" | "constant" | "alias" | "ifunc"))
            else {
                return true;
            };
            let qualifiers = &words[..kind];
            if qualifiers
                .iter()
                .any(|word| *word == "external" || *word == "extern_weak")
            {
                return false;
            }
            match words[kind] {
                "constant" => false,
                "global" => !qualifiers.iter().any(|word| *word == "addrspace(3)"),
                _ => true,
            }
        })
}

/// Rewrite `@llvm.used` / `@llvm.compiler.used` to the entries `keep` accepts.
/// Returns `None` for a spelling this rewrite does not recognize.
fn restrict_used_lists(ir: &str, keep: &dyn Fn(&str) -> bool) -> Option<String> {
    let mut output = String::with_capacity(ir.len());
    for line in ir.split_inclusive('\n') {
        let trimmed = line.trim_end_matches('\n');
        let Some(list) = ["@llvm.used = ", "@llvm.compiler.used = "]
            .into_iter()
            .find(|prefix| trimmed.starts_with(prefix))
        else {
            output.push_str(line);
            continue;
        };
        let rest = &trimmed[list.len()..];
        let open = rest.find("] [")?;
        let declared = &rest[..open + 1];
        if !declared.starts_with("appending global [") || !declared.ends_with(" x ptr]") {
            return None;
        }
        let entries_and_tail = &rest[open + 3..];
        let close = entries_and_tail.find("], ")?;
        let tail = &entries_and_tail[close + 1..];
        let mut kept = Vec::new();
        for entry in entries_and_tail[..close].split(", ") {
            let name_spelling = entry.strip_prefix("ptr @")?;
            let (name, length) = global_name(name_spelling)?;
            if length != name_spelling.len() {
                return None;
            }
            if keep(name) {
                kept.push(entry);
            }
        }
        output.push_str(list);
        output.push_str(&format!(
            "appending global [{} x ptr] [{}]{tail}",
            kept.len(),
            kept.join(", ")
        ));
        if line.ends_with('\n') {
            output.push('\n');
        }
    }
    Some(output)
}

fn parse(ptx: &str) -> Result<Document<'_>, String> {
    Document::parse(ptx).map_err(|error| format!("cannot parse partition PTX: {error}"))
}

/// `.version`, `.target` and `.address_size` directive text, in order.
fn module_header(ptx: &str) -> Result<Vec<String>, String> {
    let document = parse(ptx)?;
    Ok(document
        .directives()
        .iter()
        .filter(|directive| is_header(directive.name()))
        .map(|directive| {
            directive
                .text()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect())
}

fn is_header(name: &str) -> bool {
    matches!(name, ".version" | ".target" | ".address_size")
}

fn strip_header(ptx: &str) -> Result<String, String> {
    let document = parse(ptx)?;
    let mut output = String::with_capacity(ptx.len());
    let mut last = 0;
    for directive in document.directives().iter().filter(|d| is_header(d.name())) {
        let span = directive.line_span();
        output.push_str(&ptx[last..span.start]);
        last = span.end;
    }
    output.push_str(&ptx[last..]);
    Ok(output)
}

/// Module-scope symbols without external linkage: non-`.visible` functions and
/// `.global` / `.shared` / `.const` / `.local` variables.
fn internal_module_symbols(ptx: &str) -> Result<BTreeSet<String>, String> {
    let document = parse(ptx)?;
    let Some(module_scope) = document
        .callables()
        .first()
        .map(|callable| callable.scope())
    else {
        return Ok(BTreeSet::new());
    };
    let mut names = BTreeSet::new();
    for callable in document.callables() {
        if callable.scope() != module_scope
            || callable.kind() != CallableKind::Function
            || callable.is_extern()
        {
            continue;
        }
        let header = &ptx[callable.header_span()];
        let header = header.trim_start();
        if header.starts_with(".visible") || header.starts_with(".weak") {
            continue;
        }
        names.insert(callable.name().to_string());
    }
    for directive in document.directives() {
        if directive.scope() != module_scope
            || !matches!(
                directive.name(),
                ".global" | ".shared" | ".const" | ".local"
            )
        {
            continue;
        }
        let declared = directive
            .arguments()
            .split(|c: char| c.is_whitespace() || "[];={},".contains(c))
            .find(|word| {
                !word.is_empty() && !word.starts_with('.') && !word.as_bytes()[0].is_ascii_digit()
            });
        match declared {
            Some(name) => {
                names.insert(name.to_string());
            }
            None => {
                return Err(format!(
                    "cannot find the symbol declared by `{}`",
                    directive.text().trim()
                ));
            }
        }
    }
    Ok(names)
}

fn all_words(ptx: &str) -> Result<BTreeSet<String>, String> {
    let document = parse(ptx)?;
    Ok(document
        .tokens()
        .iter()
        .filter(|token| token.kind() == TokenKind::Word)
        .map(|token| token.text(ptx).to_string())
        .collect())
}

/// Append `suffix` to every word token spelled exactly as one of `names`.
fn rename_words(ptx: &str, names: &BTreeSet<String>, suffix: &str) -> Result<String, String> {
    if names.is_empty() {
        return Ok(ptx.to_string());
    }
    let document = parse(ptx)?;
    let mut output = String::with_capacity(ptx.len() + names.len() * suffix.len());
    let mut last = 0;
    for token in document.tokens() {
        if token.kind() != TokenKind::Word || !names.contains(token.text(ptx)) {
            continue;
        }
        let span = token.span();
        output.push_str(&ptx[last..span.end]);
        output.push_str(suffix);
        last = span.end;
    }
    output.push_str(&ptx[last..]);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(kernels: usize, extra: &str) -> String {
        let mut ir = String::from("target triple = \"nvptx64-nvidia-cuda\"\n");
        ir.push_str(extra);
        let used: Vec<String> = (0..kernels).map(|i| format!("ptr @k{i}")).collect();
        ir.push_str(&format!(
            "@llvm.used = appending global [{} x ptr] [{}, ptr @helper], section \"llvm.metadata\"\n",
            kernels + 1,
            used.join(", ")
        ));
        ir.push_str("define void @helper() {\n  ret void\n}\n");
        for i in 0..kernels {
            ir.push_str(&format!(
                "define ptx_kernel void @k{i}() {{\n  call void @helper()\n  ret void\n}}\n"
            ));
        }
        ir
    }

    fn public(kernels: usize) -> Vec<String> {
        let mut symbols: Vec<String> = (0..kernels).map(|i| format!("k{i}")).collect();
        symbols.push("helper".to_string());
        symbols
    }

    #[test]
    fn groups_partition_kernels_and_keep_shared_public_symbols_once() {
        let parts = plan(&module(32, ""), &public(32), 4).expect("splittable");
        assert_eq!(parts.len(), 4);
        let mut kernels = BTreeSet::new();
        for (index, part) in parts.iter().enumerate() {
            let group: Vec<&String> = part.public.iter().filter(|s| s.starts_with('k')).collect();
            assert_eq!(group.len(), 8);
            kernels.extend(group.iter().map(|s| s.to_string()));
            assert_eq!(part.public.contains(&"helper".to_string()), index == 0);
            let used = part
                .ir
                .lines()
                .find(|l| l.starts_with("@llvm.used"))
                .unwrap();
            assert!(used.starts_with(&format!(
                "@llvm.used = appending global [{} x ptr]",
                8 + usize::from(index == 0)
            )));
            assert_eq!(used.contains("@helper"), index == 0);
        }
        assert_eq!(kernels.len(), 32);
    }

    #[test]
    fn small_modules_and_requests_stay_whole() {
        assert!(plan(&module(15, ""), &public(15), 16).is_none());
        assert!(plan(&module(64, ""), &public(64), 1).is_none());
    }

    #[test]
    fn mutable_globals_and_public_data_prevent_splitting() {
        let global = "@counter = internal addrspace(1) global i32 0, align 4\n";
        assert!(plan(&module(32, global), &public(32), 4).is_none());
        let shared = "@tile = internal addrspace(3) global [64 x i32] undef, align 4\n";
        assert!(plan(&module(32, shared), &public(32), 4).is_some());
        let constant = "@table = internal addrspace(4) constant [2 x i32] [i32 1, i32 2]\n";
        assert!(plan(&module(32, constant), &public(32), 4).is_some());
        let dynamic = "@pool = external addrspace(3) global [0 x i8], align 16\n";
        assert!(plan(&module(32, dynamic), &public(32), 4).is_some());
        let mut with_data = public(32);
        with_data.push("table".to_string());
        assert!(plan(&module(32, constant), &with_data, 4).is_none());
    }

    const PART: &str = ".version 8.7\n.target sm_89\n.address_size 64\n\n\
        .shared .align 4 .b8 tile[64];\n\
        .func (.param .b32 r) helper(.param .b32 x)\n{\n\tret;\n}\n\
        .visible .func api()\n{\n\tret;\n}\n\
        .visible .entry KERNEL()\n{\n\tcall.uni (r), helper, (x);\n\tst.shared.u32 [tile], 1;\n\tret;\n}\n";

    #[test]
    fn merge_renames_later_internal_symbols_and_keeps_one_header() {
        let second = PART
            .replace("KERNEL", "k1")
            .replace(".visible .func api()\n{\n\tret;\n}\n", "");
        let first = PART.replace("KERNEL", "k0");
        let merged = merge(&[first, second]).expect("merge");
        assert_eq!(merged.matches(".version").count(), 1);
        assert_eq!(merged.matches(".address_size").count(), 1);
        assert!(merged.contains("helper(.param"));
        assert!(merged.contains("helper_oxpart1(.param"));
        assert!(merged.contains("call.uni (r), helper_oxpart1, (x);"));
        assert!(merged.contains("[tile_oxpart1]"));
        assert!(merged.contains(".visible .func api()"));
        assert!(merged.contains(".visible .entry k1()"));
        assert!(!merged.contains("k1_oxpart1"));
    }

    #[test]
    fn merge_rejects_different_headers() {
        let other = PART.replace("sm_89", "sm_90");
        assert!(merge(&[PART.to_string(), other]).is_err());
    }

    #[test]
    fn used_list_rewrite_rejects_unknown_spellings() {
        let odd = "@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr @k0 to ptr)], section \"llvm.metadata\"\n";
        assert!(restrict_used_lists(odd, &|_| true).is_none());
    }

    #[test]
    fn quoted_names_are_unquoted() {
        assert_eq!(global_name("\"a b\"(i32"), Some(("a b", 5)));
        assert_eq!(global_name("plain_1.x(i32"), Some(("plain_1.x", 9)));
    }
}
