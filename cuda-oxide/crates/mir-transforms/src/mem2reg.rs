/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Promote memory slots to SSA values.
//!
//! This is pliron's `mem2reg` algorithm: liveness-pruned block arguments on
//! the iterated dominance frontier, then a rename over the dominator tree. It
//! drives the same [`PromotableAllocationInterface`] and
//! [`PromotableOpInterface`] implementations and makes the same decisions in
//! the same order, so block arguments and inserted operations come out
//! identical to `pliron::opts::mem2reg::mem2reg`.
//!
//! The difference is cost. pliron 0.20.0 copies a slot-to-definition map for
//! every block and asks every promotable operation about every slot of its
//! function, which is slots × (blocks + accesses). The MIR importer gives each
//! local its own slot, so that product dominated device compile time for large
//! functions. Here an access finds its slots through its operands, and one
//! definition table is rolled back when the rename leaves a dominator subtree.
//!
//! Finding slots through operands relies on the interface contract: an
//! operation that does not use a slot's pointer reports
//! [`PromotableOpKind::NonPromotableUse`] for it.
//!
//! Remove this module once the pinned pliron release finds accessed slots
//! through an index.

use pliron::{
    basic_block::BasicBlock,
    builtin::{given_names::set_block_arg_name, op_interfaces::BranchOpInterface},
    common_traits::Named,
    context::{Context, Ptr},
    graph::{
        dominance::{DomFrontierMap, DomInfo, DomTree},
        walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    },
    irbuild::{
        IRStatus,
        inserter::{IRInserter, Inserter},
        listener::{Recorder, RecorderEvent},
        rewriter::IRRewriter,
    },
    linked_list::ContainsLinkedList,
    op::{op_cast, op_impls},
    operation::Operation,
    opts::mem2reg::{
        AllocInfo, PromotableAllocationInterface, PromotableOpInterface, PromotableOpKind,
    },
    pass::AnalysisManager,
    region::Region,
    result::Result,
    utils::table::{IMap, ISet},
    value::Value,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// One promotable slot: an [`AllocInfo`] of a [`PromotableAllocationInterface`] operation.
#[derive(Clone)]
struct AllocCandidate {
    alloc_op: Ptr<Operation>,
    alloc_info: AllocInfo,
}

/// Block arguments added for promoted slots: `(candidate index, argument index)`.
type NewBlockArguments = FxHashMap<Ptr<BasicBlock>, Vec<(usize, usize)>>;

/// Collect every slot under `root`, in pre-order.
fn collect_alloc_candidates(root: Ptr<Operation>, ctx: &Context) -> Vec<AllocCandidate> {
    let mut candidates: Vec<AllocCandidate> = Vec::new();
    walk_op(
        ctx,
        &mut candidates,
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, candidates, node| {
            if let IRNode::Operation(op) = node {
                let op_obj = Operation::get_op_dyn(op, ctx);
                if let Some(alloc) = op_cast::<dyn PromotableAllocationInterface>(op_obj.as_ref()) {
                    for alloc_info in alloc.alloc_info(ctx) {
                        candidates.push(AllocCandidate {
                            alloc_op: op,
                            alloc_info,
                        });
                    }
                }
            }
        },
    );
    candidates
}

/// Keep only slots whose every use is a promotable access in the slot's own region.
fn prune_candidates(candidates: &mut Vec<AllocCandidate>, ctx: &Context) {
    candidates.retain(|cand| {
        let alloc_region = cand
            .alloc_op
            .deref(ctx)
            .get_parent_region(ctx)
            .expect("Alloc op must be in a region");
        cand.alloc_info.ptr.uses(ctx).iter().all(|r#use| {
            let user_op = r#use.user_op();
            let user_op_obj = Operation::get_op_dyn(user_op, ctx);
            op_cast::<dyn PromotableOpInterface>(user_op_obj.as_ref()).is_some_and(|promotable| {
                let user_region = user_op
                    .deref(ctx)
                    .get_parent_region(ctx)
                    .expect("Use op must be in a region");
                user_region == alloc_region
                    && !matches!(
                        promotable.promotion_kind(ctx, &cand.alloc_info),
                        PromotableOpKind::NonPromotableUse
                    )
            })
        })
    });
}

/// Position of every operation of `region` within its block.
fn operation_positions(ctx: &Context, region: Ptr<Region>) -> FxHashMap<Ptr<Operation>, usize> {
    let mut positions = FxHashMap::default();
    for block in region.deref(ctx).iter(ctx) {
        positions.extend(
            block
                .deref(ctx)
                .iter(ctx)
                .enumerate()
                .map(|(position, operation)| (operation, position)),
        );
    }
    positions
}

/// Blocks where the slot's value is live on entry, and blocks that store to it.
///
/// A block seeds liveness when it reads the slot before its first store.
/// Liveness then spreads to predecessors and stops at storing blocks.
fn live_in_and_defining_blocks(
    ctx: &Context,
    cand: &AllocCandidate,
    positions: &FxHashMap<Ptr<Operation>, usize>,
) -> (FxHashSet<Ptr<BasicBlock>>, ISet<Ptr<BasicBlock>>) {
    #[derive(Default)]
    struct BlockAccess {
        first_store: Option<usize>,
        first_read: Option<usize>,
    }
    fn earliest(seen: &mut Option<usize>, position: usize) {
        *seen = Some(seen.map_or(position, |earlier| earlier.min(position)));
    }

    // Blocks in use-list order: that order decides where block arguments land.
    let mut accesses: IMap<Ptr<BasicBlock>, BlockAccess> = IMap::default();
    for r#use in cand.alloc_info.ptr.uses(ctx) {
        let user = r#use.user_op();
        let Some(block) = user.deref(ctx).get_parent_block() else {
            continue;
        };
        let access = accesses.entry(block).or_default();
        let user_obj = Operation::get_op_dyn(user, ctx);
        let Some(promotable) = op_cast::<dyn PromotableOpInterface>(user_obj.as_ref()) else {
            continue;
        };
        let position = positions[&user];
        match promotable.promotion_kind(ctx, &cand.alloc_info) {
            PromotableOpKind::Load | PromotableOpKind::EliminatableUse => {
                earliest(&mut access.first_read, position);
            }
            PromotableOpKind::Store(_) => earliest(&mut access.first_store, position),
            // Filtered by `prune_candidates`.
            PromotableOpKind::NonPromotableUse => {}
        }
    }

    let mut defining_blocks: ISet<Ptr<BasicBlock>> = ISet::default();
    let mut live_in_worklist: Vec<Ptr<BasicBlock>> = Vec::new();
    for (block, access) in accesses {
        if access.first_store.is_some() {
            defining_blocks.insert(block);
        }
        let read_before_store = match (access.first_read, access.first_store) {
            (Some(read), Some(store)) => read < store,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if read_before_store {
            live_in_worklist.push(block);
        }
    }

    let mut live_in: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
    while let Some(live_in_block) = live_in_worklist.pop() {
        if !live_in.insert(live_in_block) {
            continue;
        }
        for pred in live_in_block.preds(ctx) {
            if !defining_blocks.contains(&pred) {
                live_in_worklist.push(pred);
            }
        }
    }

    (live_in, defining_blocks)
}

/// Blocks that need an argument for the slot: the iterated dominance frontier
/// of its storing blocks, restricted to blocks where the slot is live on entry.
fn phi_blocks(
    df_map: &DomFrontierMap<Ptr<Region>, Context>,
    live_in: &FxHashSet<Ptr<BasicBlock>>,
    defining_blocks: &ISet<Ptr<BasicBlock>>,
) -> ISet<Ptr<BasicBlock>> {
    let mut phi_blocks: ISet<Ptr<BasicBlock>> = ISet::default();
    let mut worklist: Vec<Ptr<BasicBlock>> = defining_blocks.iter().cloned().collect();
    while let Some(block) = worklist.pop() {
        for &df_block in df_map.frontier(&block) {
            if !live_in.contains(&df_block) {
                continue;
            }
            if !phi_blocks.insert(df_block) {
                continue;
            }
            if !defining_blocks.contains(&df_block) {
                worklist.push(df_block);
            }
        }
    }
    phi_blocks
}

/// Whether every predecessor of every block in `phi_blocks` ends in a branch
/// that can carry the new block argument.
fn predecessors_can_forward(ctx: &Context, phi_blocks: &ISet<Ptr<BasicBlock>>) -> bool {
    phi_blocks
        .iter()
        .flat_map(|phi_block| phi_block.preds(ctx))
        .all(|pred| {
            pred.deref(ctx).get_terminator(ctx).is_some_and(|term| {
                op_impls::<dyn BranchOpInterface>(Operation::get_op_dyn(term, ctx).as_ref())
            })
        })
}

/// Record operations a promotion callback erased.
fn note_erased_ops(recorder: &mut Recorder, erased: &mut FxHashSet<Ptr<Operation>>) {
    for event in recorder.events.drain(..) {
        match event {
            RecorderEvent::ErasedOperation(op) => {
                erased.insert(op);
            }
            RecorderEvent::ErasedBlock(_)
            | RecorderEvent::ErasedRegion(_)
            | RecorderEvent::UnlinkedBlock(_, _) => {
                panic!("mem2reg rewrite (promotion) call backs must not alter control flow");
            }
            RecorderEvent::InsertedOperation(_)
            | RecorderEvent::InsertedBlock(_)
            | RecorderEvent::ReplacedValueUses { .. }
            | RecorderEvent::ValueTypeChanged { .. }
            | RecorderEvent::UnlinkedOperation(_, _) => {}
        }
    }
}

/// State of the rename walk over one region's dominator tree.
struct Rename<'a> {
    candidates: &'a [AllocCandidate],
    /// Slot pointer to its index in `candidates`.
    index: FxHashMap<Value, usize>,
    new_block_arguments: &'a NewBlockArguments,
    /// Reaching definition of each candidate at the current point of the walk.
    reaching: Vec<Option<Value>>,
    /// Overwritten `reaching` entries, newest last. Leaving a dominator
    /// subtree replays them to restore the definitions of its parent.
    undo: Vec<(usize, Option<Value>)>,
    /// Value used where no store reaches a read, created on first need.
    defaults: Vec<Option<Value>>,
}

impl Rename<'_> {
    fn define(&mut self, candidate: usize, value: Value) {
        self.undo.push((candidate, self.reaching[candidate]));
        self.reaching[candidate] = Some(value);
    }

    fn restore(&mut self, mark: usize) {
        while self.undo.len() > mark {
            let (candidate, previous) = self.undo.pop().expect("undo log is longer than mark");
            self.reaching[candidate] = previous;
        }
    }

    /// The definition reaching this point. Without one, the slot's default
    /// value becomes the definition for the rest of the dominator subtree.
    fn reaching_or_default(&mut self, ctx: &mut Context, candidate: usize) -> Result<Value> {
        if let Some(value) = self.reaching[candidate] {
            return Ok(value);
        }
        let value = match self.defaults[candidate] {
            Some(value) => value,
            None => {
                let value = create_default_value(ctx, &self.candidates[candidate])?;
                self.defaults[candidate] = Some(value);
                value
            }
        };
        self.define(candidate, value);
        Ok(value)
    }

    fn rename_block(&mut self, ctx: &mut Context, block: Ptr<BasicBlock>) -> Result<()> {
        let candidates = self.candidates;
        let new_block_arguments = self.new_block_arguments;

        for &(candidate, argument) in new_block_arguments.get(&block).into_iter().flatten() {
            let value = block.deref(ctx).get_argument(argument);
            self.define(candidate, value);
        }

        let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
        let mut erased_ops = FxHashSet::default();
        let mut accessed: Vec<usize> = Vec::new();
        for op in ops {
            if erased_ops.contains(&op) {
                continue;
            }
            accessed.clear();
            accessed.extend(
                op.deref(ctx)
                    .operands()
                    .filter_map(|operand| self.index.get(&operand).copied()),
            );
            if accessed.is_empty() {
                continue;
            }
            // Candidate order, each candidate once.
            accessed.sort_unstable();
            accessed.dedup();

            let op_obj = Operation::get_op_dyn(op, ctx);
            let Some(promotable) = op_cast::<dyn PromotableOpInterface>(op_obj.as_ref()) else {
                continue;
            };

            let mut promote_queue = Vec::new();
            for &candidate in &accessed {
                let alloc_info = &candidates[candidate].alloc_info;
                match promotable.promotion_kind(ctx, alloc_info) {
                    PromotableOpKind::Load | PromotableOpKind::EliminatableUse => {
                        let current_def = self.reaching_or_default(ctx, candidate)?;
                        promote_queue.push((alloc_info.clone(), current_def));
                    }
                    PromotableOpKind::Store(stored_val) => {
                        self.define(candidate, stored_val);
                        promote_queue.push((alloc_info.clone(), stored_val));
                    }
                    PromotableOpKind::NonPromotableUse => {}
                }
            }
            if !promote_queue.is_empty() {
                let mut rewriter = IRRewriter::<Recorder>::default();
                rewriter.set_insertion_point_before_operation(op);
                promotable.promote(ctx, &promote_queue, &mut rewriter)?;
                note_erased_ops(rewriter.get_listener_mut(), &mut erased_ops);
            }
        }

        // Pass the definitions at the end of this block to successor arguments.
        let succs = block.deref(ctx).succs(ctx);
        for (succ_idx, succ) in succs.iter().enumerate() {
            let Some(succ_arguments) = new_block_arguments.get(succ) else {
                continue;
            };
            let term = block
                .deref(ctx)
                .get_terminator(ctx)
                .expect("Block has successors but no terminator");
            let term_obj = Operation::get_op_dyn(term, ctx);
            let branch = op_cast::<dyn BranchOpInterface>(term_obj.as_ref())
                .expect("Terminator must implement BranchOpInterface for phi blocks");
            for &(candidate, argument) in succ_arguments {
                let current_def = self.reaching_or_default(ctx, candidate)?;
                let succ_opd_idx = branch.add_successor_operand(ctx, succ_idx, current_def);
                assert!(succ_opd_idx == argument, "Mismatched phi argument index");
            }
        }

        Ok(())
    }

    /// Rename every block, parents before the blocks they dominate.
    fn rename_region(
        &mut self,
        ctx: &mut Context,
        entry_block: Ptr<BasicBlock>,
        dom_tree: &DomTree<Ptr<Region>, Context>,
    ) -> Result<()> {
        enum Step {
            Enter(Ptr<BasicBlock>),
            Leave(usize),
        }
        let mut steps = vec![Step::Enter(entry_block)];
        while let Some(step) = steps.pop() {
            match step {
                Step::Leave(mark) => self.restore(mark),
                Step::Enter(block) => {
                    let mark = self.undo.len();
                    self.rename_block(ctx, block)?;
                    steps.push(Step::Leave(mark));
                    let first_child = steps.len();
                    steps.extend(dom_tree.children(&block).map(Step::Enter));
                    // Reverse so pop() visits children in dominator-tree order.
                    steps[first_child..].reverse();
                }
            }
        }
        Ok(())
    }
}

/// Materialize the slot's default value where it dominates every use: in the
/// region's entry block, before the allocation when that is where it lives.
fn create_default_value(ctx: &mut Context, cand: &AllocCandidate) -> Result<Value> {
    let alloc_op = cand.alloc_op;
    let alloc_obj = Operation::get_op_dyn(alloc_op, ctx);
    let alloc = op_cast::<dyn PromotableAllocationInterface>(alloc_obj.as_ref())
        .expect("Alloc op must implement PromotableAllocationInterface");

    let alloc_block = alloc_op
        .deref(ctx)
        .get_parent_block()
        .expect("Alloc op must be in a block");
    let alloc_region_entry = alloc_block
        .deref(ctx)
        .get_parent_region()
        .expect("Alloc op must be in a region")
        .deref(ctx)
        .get_entry_block()
        .expect("Region must have entry block");
    let mut inserter = if alloc_region_entry == alloc_block {
        IRInserter::<Recorder>::new_before_operation(alloc_op)
    } else {
        IRInserter::<Recorder>::new_before_block_terminator(alloc_region_entry, ctx)
    };

    alloc.default_value(ctx, &mut inserter, &cand.alloc_info)
}

/// Promote the slots of one region. Returns whether any slot was promoted.
fn promote_region(
    ctx: &mut Context,
    region: Ptr<Region>,
    region_candidates: Vec<AllocCandidate>,
    dom_tree: &DomTree<Ptr<Region>, Context>,
) -> Result<bool> {
    let df_map = DomFrontierMap::new(ctx, &region, dom_tree);
    let positions = operation_positions(ctx, region);

    // Place block arguments per slot, dropping slots whose arguments a
    // predecessor could not supply.
    let mut candidates = Vec::with_capacity(region_candidates.len());
    let mut candidate_phi_blocks = Vec::with_capacity(region_candidates.len());
    for cand in region_candidates {
        let (live_in, defining_blocks) = live_in_and_defining_blocks(ctx, &cand, &positions);
        let phi_blocks = phi_blocks(&df_map, &live_in, &defining_blocks);
        if predecessors_can_forward(ctx, &phi_blocks) {
            candidates.push(cand);
            candidate_phi_blocks.push(phi_blocks);
        }
    }
    drop(positions);
    if candidates.is_empty() {
        return Ok(false);
    }

    let mut new_block_arguments = NewBlockArguments::default();
    for (candidate, (cand, phi_blocks)) in candidates.iter().zip(candidate_phi_blocks).enumerate() {
        let ptr = cand.alloc_info.ptr;
        for phi_block in phi_blocks {
            let argument = BasicBlock::push_argument(phi_block, ctx, cand.alloc_info.ty);
            set_block_arg_name(ctx, phi_block, argument, ptr.given_name(ctx));
            new_block_arguments
                .entry(phi_block)
                .or_default()
                .push((candidate, argument));
        }
    }

    let entry_block = region
        .deref(ctx)
        .get_head()
        .expect("No entry block in region");
    Rename {
        candidates: &candidates,
        index: candidates
            .iter()
            .enumerate()
            .map(|(candidate, cand)| (cand.alloc_info.ptr, candidate))
            .collect(),
        new_block_arguments: &new_block_arguments,
        reaching: vec![None; candidates.len()],
        undo: Vec::new(),
        defaults: vec![None; candidates.len()],
    }
    .rename_region(ctx, entry_block, dom_tree)?;

    // Remove the allocations themselves, one call per allocating operation.
    let mut alloc_op_to_infos: IMap<Ptr<Operation>, Vec<AllocInfo>> = IMap::default();
    for cand in &candidates {
        alloc_op_to_infos
            .entry(cand.alloc_op)
            .or_default()
            .push(cand.alloc_info.clone());
    }
    let mut rewriter = IRRewriter::<Recorder>::default();
    let mut erased_ops = FxHashSet::default();
    for (op, infos) in alloc_op_to_infos {
        assert!(
            !erased_ops.contains(&op),
            "Alloc op was already erased during promotion of another candidate"
        );
        rewriter.set_insertion_point_before_operation(op);
        let op_obj = Operation::get_op_dyn(op, ctx);
        let alloc = op_cast::<dyn PromotableAllocationInterface>(op_obj.as_ref())
            .expect("Alloc op must implement PromotableAllocationInterface");
        alloc.promote(ctx, &mut rewriter, &infos)?;
        note_erased_ops(rewriter.get_listener_mut(), &mut erased_ops);
    }

    Ok(true)
}

/// Promote memory slots to SSA values in every region under `root`.
pub fn mem2reg(
    root: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
) -> Result<IRStatus> {
    let mut candidates = collect_alloc_candidates(root, ctx);
    prune_candidates(&mut candidates, ctx);
    if candidates.is_empty() {
        return Ok(IRStatus::Unchanged);
    }

    // One dominator tree and dominance frontier per region.
    let mut by_region: IMap<Ptr<Region>, Vec<AllocCandidate>> = IMap::default();
    for cand in candidates {
        let region = cand
            .alloc_op
            .deref(ctx)
            .get_parent_region(ctx)
            .expect("Alloc op must be in a region");
        by_region.entry(region).or_default().push(cand);
    }

    let dom_info = &mut *analyses.get_analysis_mut::<DomInfo>(root, ctx)?;

    let mut status = IRStatus::Unchanged;
    for (region, region_candidates) in by_region {
        let dom_tree = dom_info.get_dom_tree(ctx, region);
        status |= IRStatus::from(promote_region(ctx, region, region_candidates, dom_tree)?);
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        ops::{
            MirAllocaOp, MirCondBranchOp, MirFuncOp, MirGotoOp, MirLoadOp, MirReturnOp, MirStoreOp,
        },
        types::MirPtrType,
    };
    use pliron::{
        builtin::{
            attributes::TypeAttr,
            op_interfaces::{
                OperandSegmentInterface, SingleBlockRegionInterface, SymbolOpInterface,
            },
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        common_traits::Verify,
        op::Op,
        printable::Printable,
        r#type::TypeHandle,
    };

    fn append<O: Op>(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        results: Vec<TypeHandle>,
        operands: Vec<Value>,
        successors: Vec<Ptr<BasicBlock>>,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            O::get_concrete_op_info(),
            results,
            operands,
            successors,
            0,
        );
        op.insert_at_back(block, ctx);
        op
    }

    /// Seven slots over a diamond followed by a loop. `observe` stores a value
    /// through the `out` argument so every promoted read stays visible.
    ///
    /// ```text
    /// entry:  a = x;                 branch flag ? then : other
    /// then:   b = x; d = x; e = x;   observe f; goto merge
    /// other:  a = y; d = y; e = y;   observe g; goto merge
    /// merge:  observe b, a; c = a                 (b is not stored through `other`)
    ///         observe d; d = x                    (read first: d needs an argument)
    ///         e = y; observe e;      goto header  (store first: e needs none)
    /// header: observe c;             branch flag ? body : exit
    /// body:   c = b;                 goto header
    /// exit:   observe c;             return
    /// ```
    ///
    /// `f` and `g` are never stored. Their default values are created in the
    /// order the sibling blocks `then` and `other` are renamed.
    fn build_fixture(ctx: &mut Context) -> Ptr<Operation> {
        dialect_mir::register(ctx);
        let value_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let flag_type: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let slot_type: TypeHandle = MirPtrType::get_generic(ctx, value_type, true).into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let arguments = vec![value_type, value_type, flag_type, slot_type];
        let function_type = FunctionType::get(ctx, arguments.clone(), vec![]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()))
            .set_symbol_name(ctx, "kernel".try_into().unwrap());
        module.append_operation(ctx, function, 0);

        let region = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, arguments);
        entry.insert_at_back(region, ctx);
        let [then, other, merge, header, body, exit] = std::array::from_fn(|_| {
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(region, ctx);
            block
        });
        let [x, y, flag, out] = std::array::from_fn(|index| entry.deref(ctx).get_argument(index));

        let slot = |ctx: &mut Context| {
            append::<MirAllocaOp>(ctx, entry, vec![slot_type], vec![], vec![])
                .deref(ctx)
                .get_result(0)
        };
        let store = |ctx: &mut Context, block, address, value| {
            append::<MirStoreOp>(ctx, block, vec![], vec![address, value], vec![]);
        };
        let load = |ctx: &mut Context, block, address| {
            append::<MirLoadOp>(ctx, block, vec![value_type], vec![address], vec![])
                .deref(ctx)
                .get_result(0)
        };
        let goto = |ctx: &mut Context, block, target| {
            append::<MirGotoOp>(ctx, block, vec![], vec![], vec![target]);
        };
        let branch = |ctx: &mut Context, block, if_true, if_false| {
            let (operands, segments) =
                MirCondBranchOp::compute_segment_sizes(vec![vec![flag], vec![], vec![]]);
            let op =
                append::<MirCondBranchOp>(ctx, block, vec![], operands, vec![if_true, if_false]);
            MirCondBranchOp::new(op).set_operand_segment_sizes(ctx, segments);
        };

        let a = slot(ctx);
        let b = slot(ctx);
        let c = slot(ctx);
        let d = slot(ctx);
        let e = slot(ctx);
        let f = slot(ctx);
        let g = slot(ctx);
        store(ctx, entry, a, x);
        branch(ctx, entry, then, other);

        store(ctx, then, b, x);
        store(ctx, then, d, x);
        store(ctx, then, e, x);
        let unset_f = load(ctx, then, f);
        store(ctx, then, out, unset_f);
        goto(ctx, then, merge);

        store(ctx, other, a, y);
        store(ctx, other, d, y);
        store(ctx, other, e, y);
        let unset_g = load(ctx, other, g);
        store(ctx, other, out, unset_g);
        goto(ctx, other, merge);

        let merged_b = load(ctx, merge, b);
        store(ctx, merge, out, merged_b);
        let merged_a = load(ctx, merge, a);
        store(ctx, merge, out, merged_a);
        store(ctx, merge, c, merged_a);
        let merged_d = load(ctx, merge, d);
        store(ctx, merge, out, merged_d);
        store(ctx, merge, d, x);
        store(ctx, merge, e, y);
        let stored_e = load(ctx, merge, e);
        store(ctx, merge, out, stored_e);
        goto(ctx, merge, header);

        let looped_c = load(ctx, header, c);
        store(ctx, header, out, looped_c);
        branch(ctx, header, body, exit);

        store(ctx, body, c, merged_b);
        goto(ctx, body, header);

        let final_c = load(ctx, exit, c);
        store(ctx, exit, out, final_c);
        append::<MirReturnOp>(ctx, exit, vec![], vec![], vec![]);

        module.get_operation()
    }

    fn promoted_fixture(
        promote: fn(Ptr<Operation>, &mut Context, &mut AnalysisManager) -> Result<IRStatus>,
    ) -> String {
        let mut ctx = Context::new();
        let module = build_fixture(&mut ctx);
        if let Err(error) = module.deref(&ctx).verify(&ctx) {
            panic!("fixture is malformed: {}", error.disp(&ctx));
        }
        let mut analyses = AnalysisManager::default();
        match promote(module, &mut ctx, &mut analyses) {
            Ok(status) => assert_eq!(status, IRStatus::Changed),
            Err(error) => panic!("promotion failed: {}", error.disp(&ctx)),
        }
        if let Err(error) = module.deref(&ctx).verify(&ctx) {
            panic!("promoted module is malformed: {}", error.disp(&ctx));
        }
        module.deref(&ctx).disp(&ctx).to_string()
    }

    #[test]
    fn promotes_merges_and_loops_exactly_like_pliron() {
        let promoted = promoted_fixture(mem2reg);
        assert!(
            !promoted.contains("mir.alloca") && !promoted.contains("mir.load"),
            "slots must be promoted:\n{promoted}"
        );
        // The reference decides block-argument order, default values, and
        // which definition reaches each read across the merge and back edge.
        assert_eq!(promoted, promoted_fixture(pliron::opts::mem2reg::mem2reg));
    }
}
