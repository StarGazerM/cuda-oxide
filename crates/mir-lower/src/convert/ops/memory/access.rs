/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conversions for `mir.store`, `mir.load`, `mir.alloca`, `mir.ref`, and `mir.ptr_offset`.

use super::common::{
    anyhow_to_pliron, copy_local_memory_provenance, fail_on_target_dependent_packed_aggregate,
    pointer_proved_alignment, value_abi_align, value_mir_type,
};
use super::debug::copy_debug_local_variable;
use crate::convert::types::{convert_type, mir_type_abi_align};
use dialect_mir::types::MirPtrType;
use llvm_export::op_interfaces::{BinArithOp, CastOpInterface};
use llvm_export::ops as llvm;
use llvm_export::ops::{AsmKind, InlineAsmOpExt};
use llvm_export::types::{ArrayType, StructLayout, StructType};
use pliron::builtin::type_interfaces::FloatTypeInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{Typed, type_cast};

/// Convert `mir.store` to `llvm.store`.
///
/// Operand order: `[ptr, value]` - stores `value` to address `ptr`.
/// No result is produced (store is a side effect).
pub(crate) fn convert_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, val) = match operands.as_slice() {
        [ptr, val] => (*ptr, *val),
        _ => {
            return pliron::input_err_noloc!("Store operation requires exactly 2 operands");
        }
    };

    // Packed whole-value stores are byte-faithful now that divergent rustc
    // layouts lower to LLVM packed structs. Keep the target-dependent AS3 case
    // fail-closed because its physical pointer width is selected only later.
    fail_on_target_dependent_packed_aggregate(
        ctx,
        value_mir_type(ctx, operands_info, val),
        "storing",
    )?;

    let llvm_store = llvm::StoreOp::new(ctx, val, ptr);
    if dialect_mir::ops::MirStoreOp::new(op).is_volatile(ctx) {
        llvm_export::ops::set_op_volatile(ctx, llvm_store.get_operation(), true);
    }
    // The stored value's own type answers first, as it did before. A scalar
    // records none, though, so fall back to whatever the address itself proved
    // when it was computed -- for a field projection that is the aggregate's
    // `abi_align` narrowed to the field's offset, which is otherwise lost here
    // and costs the pair its vectorization. This mirrors `convert_load`, which
    // consults the same record for the same reason. When both answer, the
    // weaker wins: a field of a packed aggregate can place an abi-aligned type
    // at a byte-aligned address, and the address's proved alignment is the
    // ceiling of what the store may claim.
    let abi = value_abi_align(ctx, operands_info, val);
    let proved = pointer_proved_alignment(ctx, ptr);
    let align = match (abi, proved) {
        (Some(abi), Some(proved)) => Some(abi.min(proved)),
        (abi, proved) => abi.or(proved),
    };
    if let Some(align) = align {
        llvm_export::ops::set_op_alignment(ctx, llvm_store.get_operation(), align as u32);
    }
    crate::convert::preserve_location(ctx, op, llvm_store.get_operation());
    rewriter.insert_operation(ctx, llvm_store.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.load` to `llvm.load`.
///
/// Takes a single pointer operand and returns the loaded value.
/// The result type is derived from the MIR operation's result type.
pub(crate) fn convert_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr = op.deref(ctx).get_operand(0);
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);

    // Packed whole-value loads are byte-faithful now that divergent rustc
    // layouts lower to LLVM packed structs. Keep only the target-dependent AS3
    // physical-image case fail-closed.
    fail_on_target_dependent_packed_aggregate(ctx, result_ty, "loading")?;

    let llvm_ty = convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?;

    if dialect_mir::ops::MirLoadOp::new(op).is_read_only(ctx) {
        if let Some(kind) = read_only_scalar_kind(ctx, llvm_ty) {
            return convert_read_only_scalar(ctx, rewriter, op, ptr, llvm_ty, kind);
        }
        let Some(paths) = read_only_i32x4_paths(ctx, llvm_ty) else {
            return pliron::input_err_noloc!(
                "cuda_device::read_only::load requires a primitive 8-, 16-, 32-, or 64-bit \
                 scalar or the I32x4 aggregate shape"
            );
        };
        return convert_read_only_i32x4(ctx, rewriter, op, ptr, llvm_ty, &paths);
    }

    let llvm_load = llvm::LoadOp::new(ctx, ptr, llvm_ty);
    if dialect_mir::ops::MirLoadOp::new(op).is_volatile(ctx) {
        llvm_export::ops::set_op_volatile(ctx, llvm_load.get_operation(), true);
    }
    // The loaded value's ABI alignment comes from this op's own result type,
    // which is still the MIR type: result types are only converted by the
    // op's own rewrite. A scalar records none, so fall back to whatever the
    // address itself proved when it was computed -- for a field projection
    // that is the aggregate's `abi_align` narrowed to the field's offset,
    // which is otherwise lost here and costs the pair its vectorization.
    // When both answer, the weaker wins: a field of a packed aggregate can
    // place an abi-aligned type at a byte-aligned address, and the address's
    // proved alignment is the ceiling of what the load may claim.
    let abi = mir_type_abi_align(ctx, result_ty);
    let proved = pointer_proved_alignment(ctx, ptr);
    let align = match (abi, proved) {
        (Some(abi), Some(proved)) => Some(abi.min(proved)),
        (abi, proved) => abi.or(proved),
    };
    if let Some(align) = align {
        llvm_export::ops::set_op_alignment(ctx, llvm_load.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, llvm_load.get_operation());
    rewriter.replace_operation(ctx, op, llvm_load.get_operation());

    Ok(())
}

#[derive(Clone, Copy)]
enum ReadOnlyScalarKind {
    Integer(u32),
    Float(u32),
}

/// Classify the primitive scalar forms supported by CUDA's read-only cache
/// instruction. Smaller integers are loaded into a 32-bit PTX register and
/// truncated back to their Rust value type; this preserves their bit pattern
/// without giving the compiler permission to read adjacent bytes.
fn read_only_scalar_kind(
    ctx: &Context,
    ty: pliron::r#type::TypeHandle,
) -> Option<ReadOnlyScalarKind> {
    let ty_ref = ty.deref(ctx);
    if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        return matches!(integer.width(), 8 | 16 | 32 | 64)
            .then_some(ReadOnlyScalarKind::Integer(integer.width()));
    }
    let float = type_cast::<dyn FloatTypeInterface>(&*ty_ref)?;
    let width = u32::try_from(float.get_semantics().bits).ok()?;
    matches!(width, 32 | 64).then_some(ReadOnlyScalarKind::Float(width))
}

/// Lower a safe shared borrow to one cache-qualified scalar transaction.
///
/// This is compiler-owned PTX selection: Rust device code only expresses a
/// shared `&T` read through `cuda_device::read_only::load` and never handles a
/// raw address or inline assembly itself.
fn convert_read_only_scalar(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptr: pliron::value::Value,
    result_ty: pliron::r#type::TypeHandle,
    kind: ReadOnlyScalarKind,
) -> Result<()> {
    let (load_ty, instruction, constraints, truncate) = match kind {
        ReadOnlyScalarKind::Integer(8) => (
            IntegerType::get(ctx, 32, Signedness::Signless).into(),
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            true,
        ),
        ReadOnlyScalarKind::Integer(16) => (
            IntegerType::get(ctx, 32, Signedness::Signless).into(),
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            true,
        ),
        ReadOnlyScalarKind::Integer(32) => (result_ty, "ld.global.nc.b32 $0, [$1];", "=r,l", false),
        ReadOnlyScalarKind::Integer(64) => (result_ty, "ld.global.nc.b64 $0, [$1];", "=l,l", false),
        ReadOnlyScalarKind::Float(32) => (result_ty, "ld.global.nc.f32 $0, [$1];", "=f,l", false),
        ReadOnlyScalarKind::Float(64) => (result_ty, "ld.global.nc.f64 $0, [$1];", "=d,l", false),
        ReadOnlyScalarKind::Integer(_) | ReadOnlyScalarKind::Float(_) => unreachable!(),
    };

    let load = llvm::InlineAsmOp::build(
        ctx,
        load_ty,
        vec![ptr],
        instruction,
        constraints,
        AsmKind::SideEffect,
    );
    crate::convert::preserve_location(ctx, op, load.get_operation());
    rewriter.insert_operation(ctx, load.get_operation());
    let loaded = load.get_operation().deref(ctx).get_result(0);
    if truncate {
        let truncated = llvm::TruncOp::new(ctx, loaded, result_ty);
        rewriter.insert_operation(ctx, truncated.get_operation());
        rewriter.replace_operation(ctx, op, truncated.get_operation());
    } else {
        rewriter.replace_operation(ctx, op, load.get_operation());
    }
    Ok(())
}

/// Recognize the lowered shape of cuda-device's over-aligned `I32x4`.
///
/// The public Rust type contains one `[i32; 4]` field today. Accepting the
/// equivalent flat four-field shape keeps the intrinsic coupled to the value's
/// semantic lanes instead of to one incidental aggregate nesting choice.
fn read_only_i32x4_paths(ctx: &Context, ty: pliron::r#type::TypeHandle) -> Option<Vec<Vec<u32>>> {
    fn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
        ty.deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 32)
    }

    if let Some(array) = ty.deref(ctx).downcast_ref::<ArrayType>()
        && array.size() == 4
        && is_i32(ctx, array.elem_type())
    {
        return Some((0..4).map(|lane| vec![lane]).collect());
    }

    let structure = ty.deref(ctx);
    let structure = structure.downcast_ref::<StructType>()?;
    if structure.layout() != StructLayout::Unpacked {
        return None;
    }
    if structure.num_fields() == 4 && structure.fields().all(|field| is_i32(ctx, field)) {
        return Some((0..4).map(|lane| vec![lane]).collect());
    }
    if structure.num_fields() == 1 {
        let field = structure.field_type(0);
        let field_ref = field.deref(ctx);
        let array = field_ref.downcast_ref::<ArrayType>()?;
        if array.size() == 4 && is_i32(ctx, array.elem_type()) {
            return Some((0..4).map(|lane| vec![0, lane]).collect());
        }
    }
    None
}

/// Preserve one 16-byte read-only transaction even when later code consumes
/// individual lanes. LLVM otherwise scalarizes the aggregate before NVPTX
/// instruction selection; this first-class compiler intrinsic is the same
/// boundary CUDA's `ThreadLoad<LOAD_LDG>` requires for `int4`.
fn convert_read_only_i32x4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptr: pliron::value::Value,
    result_ty: pliron::r#type::TypeHandle,
    lane_paths: &[Vec<u32>],
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let pair_ty = StructType::get_unnamed(
        ctx,
        (vec![i64_ty.into(), i64_ty.into()], StructLayout::Unpacked),
    );
    let load = llvm::InlineAsmOp::build(
        ctx,
        pair_ty.into(),
        vec![ptr],
        "ld.global.nc.v2.u64 {$0, $1}, [$2];",
        "=l,=l,l",
        AsmKind::SideEffect,
    );
    crate::convert::preserve_location(ctx, op, load.get_operation());
    rewriter.insert_operation(ctx, load.get_operation());
    let pair = load.get_operation().deref(ctx).get_result(0);

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let shift_attr = pliron::builtin::attributes::IntegerAttr::new(
        i64_ty,
        pliron::utils::apint::APInt::from_u64(
            32,
            std::num::NonZeroUsize::new(64).expect("64 is nonzero"),
        ),
    );
    let shift = llvm::ConstantOp::new(ctx, shift_attr.into());
    rewriter.insert_operation(ctx, shift.get_operation());
    let shift = shift.get_operation().deref(ctx).get_result(0);

    let mut lanes = Vec::with_capacity(4);
    for word_index in 0..2u32 {
        let extract = llvm::ExtractValueOp::new(ctx, pair, vec![word_index])?;
        rewriter.insert_operation(ctx, extract.get_operation());
        let word = extract.get_operation().deref(ctx).get_result(0);

        let low = llvm::TruncOp::new(ctx, word, i32_ty.into());
        rewriter.insert_operation(ctx, low.get_operation());
        lanes.push(low.get_operation().deref(ctx).get_result(0));

        let high = llvm::LShrOp::new(ctx, word, shift);
        rewriter.insert_operation(ctx, high.get_operation());
        let high = high.get_operation().deref(ctx).get_result(0);
        let high = llvm::TruncOp::new(ctx, high, i32_ty.into());
        rewriter.insert_operation(ctx, high.get_operation());
        lanes.push(high.get_operation().deref(ctx).get_result(0));
    }

    let undef = llvm::UndefOp::new(ctx, result_ty);
    rewriter.insert_operation(ctx, undef.get_operation());
    let mut value = undef.get_operation().deref(ctx).get_result(0);
    for (lane, path) in lanes.into_iter().zip(lane_paths) {
        let insert = llvm::InsertValueOp::new(ctx, value, lane, path.clone());
        rewriter.insert_operation(ctx, insert.get_operation());
        value = insert.get_operation().deref(ctx).get_result(0);
    }
    rewriter.replace_operation_with_values(ctx, op, vec![value]);
    Ok(())
}

/// Convert `mir.alloca` to `llvm.alloca`.
///
/// `mir.alloca` carries its element type on the result pointer's pointee, and
/// emits a single-element stack slot of that type. We therefore convert the
/// pointee to an LLVM type and emit `llvm.alloca <pointee_ty>, i32 1`.
///
/// No value is stored into the slot; that is the caller's job via subsequent
/// `mir.store` / `llvm.store` ops. This matches the mem2reg-ready translator
/// model where every local is backed by one alloca in the entry block and
/// defs/uses go through `store`/`load` rather than SSA values.
pub(crate) fn convert_alloca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let mir_pointee = {
        let ty_ref = result_ty.deref(ctx);
        let mir_ptr = ty_ref.downcast_ref::<MirPtrType>().ok_or_else(|| {
            anyhow_to_pliron(anyhow::anyhow!(
                "MirAllocaOp result must be MirPtrType (enforced by verifier)"
            ))
        })?;
        mir_ptr.pointee
    };
    let llvm_pointee = convert_type(ctx, mir_pointee).map_err(anyhow_to_pliron)?;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, llvm_pointee, one_val);
    // The allocated type's ABI alignment comes from this op's own result
    // pointee, which is still the MIR type at rewrite time.
    if let Some(align) = mir_type_abi_align(ctx, mir_pointee) {
        llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align as u32);
    }
    copy_debug_local_variable(ctx, op, alloca.get_operation());
    copy_local_memory_provenance(ctx, op, alloca.get_operation());
    rewriter.insert_operation(ctx, alloca.get_operation());
    rewriter.replace_operation(ctx, op, alloca.get_operation());

    Ok(())
}

/// Convert `mir.ref` — materialize the operand in stack memory via alloca+store.
///
/// `mir.ref` creates a pointer to an SSA value. In SSA form, values don't have
/// addresses, so we must place the value in memory to obtain a pointer.
/// This applies to all types: scalars (e.g. `&factor` where factor is `u32`),
/// aggregates (e.g. `&closure_env`), and pointers (e.g. `&&T`).
pub(crate) fn convert_ref(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operand = op.deref(ctx).get_operand(0);
    let operand_ty = operand.get_type(ctx);
    let abi_align = value_abi_align(ctx, operands_info, operand);

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, operand_ty, one_val);
    // Honour the referent's repr(align(N)) ABI alignment. Without this, the
    // synthesised alloca would be under-aligned relative to any loads/stores
    // that claim the struct's true alignment.
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, alloca.get_operation());
    let alloca_ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, operand, alloca_ptr);
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, store.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, store.get_operation());

    rewriter.replace_operation_with_values(ctx, op, vec![alloca_ptr]);

    Ok(())
}

/// Convert `mir.ptr_offset` to `llvm.getelementptr`.
///
/// Operands: `[ptr, offset]` where offset is an integer index.
/// Element sizing comes from the op's own result type, which is still the
/// MIR pointer type when this converter runs. The operand's recorded type
/// history is not usable here: a kind-only `mir.cast` lowers to a plain
/// value forwarding, and the history does not follow that replacement
/// edge, so a history miss would silently misscale the offset.
pub(crate) fn convert_ptr_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, offset) = match operands.as_slice() {
        [ptr, offset] => (*ptr, *offset),
        _ => return pliron::input_err!(loc, "PtrOffset requires exactly 2 operands"),
    };

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let pointee = result_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|mir_ptr| mir_ptr.pointee)
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                "mir.ptr_offset result must be a MIR pointer type; \
                 element sizing has no fact to derive from"
            )
        })?;
    let elem_ty = convert_type(ctx, pointee).map_err(anyhow_to_pliron)?;

    let llvm_gep = llvm::GetElementPtrOp::new(
        ctx,
        ptr,
        vec![llvm_export::ops::GepIndex::Value(offset)],
        elem_ty,
    );
    let inbounds = dialect_mir::ops::MirPtrOffsetOp::new(op).is_inbounds(ctx);
    llvm::set_gep_inbounds(ctx, llvm_gep.get_operation(), inbounds);
    rewriter.insert_operation(ctx, llvm_gep.get_operation());
    rewriter.replace_operation(ctx, op, llvm_gep.get_operation());

    Ok(())
}
