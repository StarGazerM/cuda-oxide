/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Workload-driven ordering and warp-vote semantics.
//!
//! Rust floating comparisons use partial ordering: every `<`, `<=`, `>`, `>=`,
//! and `==` comparison involving NaN is false, while `!=` is true. This fixture
//! intentionally does not invent a total order for floats. RCCL's natural
//! comparator has the same consequence as its Rust source: values unordered by
//! `<` in either direction compare as `Ordering::Equal`; callers that need a
//! total float order must supply one explicitly.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread, warp};

const ORDER_ITEMS: usize = 6;
const VOTE_THREADS: u32 = 45;
const SENTINEL: u32 = 0xdead_beef;

const LT: u32 = 1 << 0;
const LE: u32 = 1 << 1;
const GT: u32 = 1 << 2;
const GE: u32 = 1 << 3;
const EQ: u32 = 1 << 4;
const NE: u32 = 1 << 5;

#[cuda_module]
mod kernels {
    use super::*;
    use core::cmp::Ordering;

    #[inline(always)]
    fn ordering_code(value: Ordering) -> u32 {
        match value {
            Ordering::Less => 0,
            Ordering::Equal => 1,
            Ordering::Greater => 2,
        }
    }

    #[inline(always)]
    fn float_flags(left: f32, right: f32) -> u32 {
        ((left < right) as u32) * LT
            | ((left <= right) as u32) * LE
            | ((left > right) as u32) * GT
            | ((left >= right) as u32) * GE
            | ((left == right) as u32) * EQ
            | ((left != right) as u32) * NE
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn ordering_edges(
        signed_left: &[i32],
        signed_right: &[i32],
        unsigned_left: &[u32],
        unsigned_right: &[u32],
        float_left: &[f32],
        float_right: &[f32],
        mut signed_out: DisjointSlice<u32>,
        mut unsigned_out: DisjointSlice<u32>,
        mut float_out: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d(launch_context);
        let raw = index.get() as usize;
        if raw < signed_left.len()
            && let Some(output) = signed_out.get_mut(thread::index_1d(launch_context))
        {
            *output = ordering_code(signed_left[raw].cmp(&signed_right[raw]));
        }
        if raw < unsigned_left.len()
            && let Some(output) = unsigned_out.get_mut(thread::index_1d(launch_context))
        {
            *output = ordering_code(unsigned_left[raw].cmp(&unsigned_right[raw]));
        }
        if raw < float_left.len()
            && let Some(output) = float_out.get_mut(thread::index_1d(launch_context))
        {
            *output = float_flags(float_left[raw], float_right[raw]);
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(45)]
    #[launch_contract(domain = 1, coordinates = u32, block = (45, 1, 1))]
    pub fn vote_edges(
        mut active_out: DisjointSlice<u32>,
        mut ballot_out: DisjointSlice<u32>,
        mut any_out: DisjointSlice<u32>,
        mut all_out: DisjointSlice<u32>,
        mut branch_members_out: DisjointSlice<u32>,
        mut branch_active_out: DisjointSlice<u32>,
        mut branch_ballot_out: DisjointSlice<u32>,
    ) {
        let lane = warp::lane_id();
        let active = warp::active_mask();
        let ballot = warp::ballot_sync(active, lane % 2 == 0);
        let any = warp::any_sync(active, lane == 31);
        let all = warp::all_sync(active, lane < 13);
        let participates = lane % 3 != 0;
        let branch_members = warp::ballot_sync(active, participates);

        if let Some(output) = active_out.get_mut(thread::index_1d(launch_context)) {
            *output = active;
        }
        if let Some(output) = ballot_out.get_mut(thread::index_1d(launch_context)) {
            *output = ballot;
        }
        if let Some(output) = any_out.get_mut(thread::index_1d(launch_context)) {
            *output = any as u32;
        }
        if let Some(output) = all_out.get_mut(thread::index_1d(launch_context)) {
            *output = all as u32;
        }
        if let Some(output) = branch_members_out.get_mut(thread::index_1d(launch_context)) {
            *output = branch_members;
        }

        if participates {
            let branch_active = warp::active_mask();
            let branch_ballot = warp::ballot_sync(branch_members, lane % 2 == 0);
            if let Some(output) = branch_active_out.get_mut(thread::index_1d(launch_context)) {
                *output = branch_active;
            }
            if let Some(output) = branch_ballot_out.get_mut(thread::index_1d(launch_context)) {
                *output = branch_ballot;
            }
        }
    }
}

fn require_equal(name: &str, got: &[u32], expected: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    if got != expected {
        let mismatch = got
            .iter()
            .zip(expected)
            .position(|(actual, wanted)| actual != wanted)
            .unwrap_or(got.len().min(expected.len()));
        return Err(format!(
            "{name} mismatch at {mismatch}: got {:?}, expected {:?}",
            got.get(mismatch),
            expected.get(mismatch)
        )
        .into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this module was generated from `kernels`; every generated launch
    // method therefore has the exact kernel ABI used below.
    let module = unsafe { kernels::load(&ctx) }?;

    let signed_left = [i32::MIN, -1, i32::MAX, i32::MIN, 0, -17];
    let signed_right = [i32::MAX, -1, i32::MIN, -1, -1, 23];
    let unsigned_left = [0, u32::MAX, 0x8000_0000, u32::MAX, 7, 9];
    let unsigned_right = [u32::MAX, 0, 0x7fff_ffff, u32::MAX, 9, 7];
    let nan = f32::from_bits(0x7fc0_0001);
    let float_left = [f32::NEG_INFINITY, -0.0, f32::INFINITY, nan, 1.0, nan];
    let float_right = [-1.0, 0.0, f32::MAX, 1.0, nan, nan];

    let signed_left_device = DeviceBuffer::from_host(&stream, &signed_left)?;
    let signed_right_device = DeviceBuffer::from_host(&stream, &signed_right)?;
    let unsigned_left_device = DeviceBuffer::from_host(&stream, &unsigned_left)?;
    let unsigned_right_device = DeviceBuffer::from_host(&stream, &unsigned_right)?;
    let float_left_device = DeviceBuffer::from_host(&stream, &float_left)?;
    let float_right_device = DeviceBuffer::from_host(&stream, &float_right)?;
    let mut signed_out = DeviceBuffer::from_host(&stream, &[SENTINEL; ORDER_ITEMS])?;
    let mut unsigned_out = DeviceBuffer::from_host(&stream, &[SENTINEL; ORDER_ITEMS])?;
    let mut float_out = DeviceBuffer::from_host(&stream, &[SENTINEL; ORDER_ITEMS])?;

    let ordering_launch = module.prepare_ordering_edges(LaunchConfig1D::new(1, 32, 0))?;
    module.ordering_edges(
        &stream,
        &ordering_launch,
        &signed_left_device,
        &signed_right_device,
        &unsigned_left_device,
        &unsigned_right_device,
        &float_left_device,
        &float_right_device,
        &mut signed_out,
        &mut unsigned_out,
        &mut float_out,
    )?;
    stream.synchronize()?;

    let mut signed_expected = [0, 1, 2, 0, 2, 0];
    if std::env::var_os("CUDA_OXIDE_ORDER_VOTE_INJECT_FAULT").is_some() {
        signed_expected[0] ^= 1;
    }
    require_equal("signed Ord::cmp", &signed_out.to_host_vec(&stream)?, &signed_expected)?;
    require_equal(
        "unsigned Ord::cmp",
        &unsigned_out.to_host_vec(&stream)?,
        &[0, 2, 2, 1, 0, 2],
    )?;
    require_equal(
        "f32 PartialOrd",
        &float_out.to_host_vec(&stream)?,
        &[LT | LE | NE, LE | GE | EQ, GT | GE | NE, NE, NE, NE],
    )?;

    let initial = vec![SENTINEL; VOTE_THREADS as usize];
    let mut active_out = DeviceBuffer::from_host(&stream, &initial)?;
    let mut ballot_out = DeviceBuffer::from_host(&stream, &initial)?;
    let mut any_out = DeviceBuffer::from_host(&stream, &initial)?;
    let mut all_out = DeviceBuffer::from_host(&stream, &initial)?;
    let mut branch_members_out = DeviceBuffer::from_host(&stream, &initial)?;
    let mut branch_active_out = DeviceBuffer::from_host(&stream, &initial)?;
    let mut branch_ballot_out = DeviceBuffer::from_host(&stream, &initial)?;

    let vote_launch = module.prepare_vote_edges(LaunchConfig1D::new(1, VOTE_THREADS, 0))?;
    module.vote_edges(
        &stream,
        &vote_launch,
        &mut active_out,
        &mut ballot_out,
        &mut any_out,
        &mut all_out,
        &mut branch_members_out,
        &mut branch_active_out,
        &mut branch_ballot_out,
    )?;
    stream.synchronize()?;

    let mut active_expected = Vec::with_capacity(VOTE_THREADS as usize);
    let mut ballot_expected = Vec::with_capacity(VOTE_THREADS as usize);
    let mut any_expected = Vec::with_capacity(VOTE_THREADS as usize);
    let mut all_expected = Vec::with_capacity(VOTE_THREADS as usize);
    let mut branch_members_expected = Vec::with_capacity(VOTE_THREADS as usize);
    let mut branch_active_expected = Vec::with_capacity(VOTE_THREADS as usize);
    let mut branch_ballot_expected = Vec::with_capacity(VOTE_THREADS as usize);
    for gid in 0..VOTE_THREADS {
        let lane = gid % 32;
        let active = if gid < 32 { u32::MAX } else { 0x1fff };
        let even = active & 0x5555_5555;
        let members = active & 0xb6db_6db6;
        active_expected.push(active);
        ballot_expected.push(even);
        any_expected.push((gid < 32) as u32);
        all_expected.push((gid >= 32) as u32);
        branch_members_expected.push(members);
        if lane % 3 != 0 {
            branch_active_expected.push(members);
            branch_ballot_expected.push(members & 0x5555_5555);
        } else {
            branch_active_expected.push(SENTINEL);
            branch_ballot_expected.push(SENTINEL);
        }
    }

    require_equal("active mask", &active_out.to_host_vec(&stream)?, &active_expected)?;
    require_equal("partial-warp ballot", &ballot_out.to_host_vec(&stream)?, &ballot_expected)?;
    require_equal("partial-warp any", &any_out.to_host_vec(&stream)?, &any_expected)?;
    require_equal("partial-warp all", &all_out.to_host_vec(&stream)?, &all_expected)?;
    require_equal(
        "divergent members",
        &branch_members_out.to_host_vec(&stream)?,
        &branch_members_expected,
    )?;
    require_equal(
        "divergent active mask",
        &branch_active_out.to_host_vec(&stream)?,
        &branch_active_expected,
    )?;
    require_equal(
        "divergent ballot",
        &branch_ballot_out.to_host_vec(&stream)?,
        &branch_ballot_expected,
    )?;

    println!("ordering: signed/unsigned edges and explicit f32 NaN semantics PASS");
    println!("vote: full, partial, and divergent active masks/ballots PASS");
    Ok(())
}
