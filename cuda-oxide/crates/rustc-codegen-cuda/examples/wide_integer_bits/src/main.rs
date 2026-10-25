/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compare device wide-integer operations with host Rust for every valid shift.
//! Inputs and counts come from device buffers, preventing constant folding.
//! Run: cargo oxide run wide_integer_bits --arch sm_89

use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

const OPS: usize = 14;
const LANE_OPS: usize = 6;
const NAMES: [&str; OPS] = [
    "shl", "lshr", "ashr", "not", "and", "or", "not_shl", "not_lshr",
    "set_lane", "signed_not", "mask_shl", "mask_lshr", "mask_ashr", "mask_not_shl",
];

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn wide_bits(lo: &[u64], hi: &[u64], shifts: &[u32], mut out: DisjointSlice<u64>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let word = idx.get();
            let case = word / (2 * OPS);
            let op = (word / 2) % OPS;
            let a = ((hi[case] as u128) << 64) | lo[case] as u128;
            let b = ((lo[case] as u128) << 64) | hi[case] as u128;
            let s = shifts[case];
            let result = match op {
                0 => a << s,
                1 => a >> s,
                2 => ((a as i128) >> s) as u128,
                3 => !a,
                4 => a & b,
                5 => a | b,
                6 => !(a << s),
                7 => !(a >> s),
                8 => (a & !((u64::MAX as u128) << s)) | ((lo[case] as u128) << s),
                9 => (!(a as i128)) as u128,
                10 => (u64::MAX as u128) << s,
                11 => ((u64::MAX as u128) << 64) >> s,
                12 => (i128::MIN >> s) as u128,
                _ => !((u64::MAX as u128) << s),
            };
            *slot = if word % 2 == 0 { result as u64 } else { (result >> 64) as u64 };
        }
    }

    #[kernel]
    pub fn lane_bits(lo: &[u64], hi: &[u64], lanes: &[u32], mut out: DisjointSlice<u64>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let word = idx.get();
            let case = word / (2 * LANE_OPS);
            let a = ((hi[case] as u128) << 64) | lo[case] as u128;
            let shift = lanes[case] * 64;
            let result = match (word / 2) % LANE_OPS {
                0 => (a & !((u64::MAX as u128) << shift)) | ((lo[case] as u128) << shift),
                1 => (u64::MAX as u128) << shift,
                2 => ((u64::MAX as u128) << 64) >> shift,
                3 => (i128::MIN >> shift) as u128,
                4 => !((u64::MAX as u128) << shift),
                _ => !(((u64::MAX as u128) << 64) >> shift),
            };
            *slot = if word % 2 == 0 { result as u64 } else { (result >> 64) as u64 };
        }
    }

    // Keep this standalone: sharing a switch with the other operations changes
    // LLVM's known-bits combines and can hide the original miscompilation.
    #[kernel]
    pub fn set_lane(lo: &[u64], hi: &[u64], lanes: &[u32], mut out: DisjointSlice<u64>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let word = idx.get();
            let case = word / 2;
            let a = ((hi[case] as u128) << 64) | lo[case] as u128;
            let shift = lanes[case] * 64;
            let result = (a & !((u64::MAX as u128) << shift)) | ((lo[case] as u128) << shift);
            *slot = if word % 2 == 0 { result as u64 } else { (result >> 64) as u64 };
        }
    }
}

fn main() {
    let values = [
        0, 1, u64::MAX as u128, 1u128 << 64, 1u128 << 127,
        u128::MAX, (1u128 << 127) - 1,
        0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        0xfedc_ba98_7654_3210_0123_4567_89ab_cdef,
    ];
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    let mut shifts = Vec::new();
    let mut expected = Vec::new();
    for a in values {
        for s in 0..128u32 {
            let b = a.rotate_left(64);
            lo.push(a as u64);
            hi.push((a >> 64) as u64);
            shifts.push(s);
            expected.extend([
                a << s, a >> s, ((a as i128) >> s) as u128, !a, a & b, a | b,
                !(a << s), !(a >> s),
                (a & !((u64::MAX as u128) << s)) | (((a as u64) as u128) << s),
                (!(a as i128)) as u128,
                (u64::MAX as u128) << s,
                ((u64::MAX as u128) << 64) >> s,
                (i128::MIN >> s) as u128,
                !((u64::MAX as u128) << s),
            ]);
        }
    }

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load kernel");
    let lo_device = DeviceBuffer::from_host(&stream, &lo).unwrap();
    let hi_device = DeviceBuffer::from_host(&stream, &hi).unwrap();
    let shifts_device = DeviceBuffer::from_host(&stream, &shifts).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, expected.len() * 2).unwrap();
    let cfg = LaunchConfig::for_num_elems((expected.len() * 2) as u32);
    // SAFETY: buffers contain every case indexed by the output threads.
    unsafe { module.wide_bits(&stream, cfg, &lo_device, &hi_device, &shifts_device, &mut out) }
        .expect("launch kernel");
    let actual = out.to_host_vec(&stream).unwrap();
    let mut failures = [0usize; OPS];
    let mut bad_shifts = [[false; 128]; OPS];
    for (i, want) in expected.iter().enumerate() {
        let got = actual[2 * i] as u128 | ((actual[2 * i + 1] as u128) << 64);
        if got != *want {
            let op = i % OPS;
            let case = i / OPS;
            if failures[op] == 0 {
                eprintln!("{} first mismatch: input={:016x}{:016x} shift={} got={got:032x} expected={want:032x}",
                    NAMES[op], hi[case], lo[case], shifts[case]);
            }
            failures[op] += 1;
            bad_shifts[op][shifts[case] as usize] = true;
        }
    }
    for op in 0..OPS {
        let shifts: Vec<_> = (0..128).filter(|s| bad_shifts[op][*s]).collect();
        println!("{}: {}/{} failures; failing shifts={shifts:?}", NAMES[op], failures[op], lo.len());
    }
    let lanes: Vec<u32> = shifts.iter().map(|s| s % 2).collect();
    let lanes_device = DeviceBuffer::from_host(&stream, &lanes).unwrap();
    let mut lane_out = DeviceBuffer::<u64>::zeroed(&stream, lo.len() * 2 * LANE_OPS).unwrap();
    let lane_cfg = LaunchConfig::for_num_elems((lo.len() * 2 * LANE_OPS) as u32);
    // SAFETY: each output pair corresponds to one complete input case.
    unsafe { module.lane_bits(&stream, lane_cfg, &lo_device, &hi_device, &lanes_device, &mut lane_out) }
        .expect("launch lane kernel");
    let actual_lanes = lane_out.to_host_vec(&stream).unwrap();
    let mut lane_failures = [[0usize; 2]; LANE_OPS];
    for case in 0..lo.len() {
        let a = lo[case] as u128 | ((hi[case] as u128) << 64);
        let shift = lanes[case] * 64;
        let expected = [
            (a & !((u64::MAX as u128) << shift)) | ((lo[case] as u128) << shift),
            (u64::MAX as u128) << shift,
            ((u64::MAX as u128) << 64) >> shift,
            (i128::MIN >> shift) as u128,
            !((u64::MAX as u128) << shift),
            !(((u64::MAX as u128) << 64) >> shift),
        ];
        for (op, want) in expected.into_iter().enumerate() {
            let i = (case * LANE_OPS + op) * 2;
            let got = actual_lanes[i] as u128 | ((actual_lanes[i + 1] as u128) << 64);
            if got != want {
                let lane = lanes[case] as usize;
                if lane_failures[op][lane] == 0 {
                    eprintln!("lane {lane} op {op} first mismatch: input={a:032x} got={got:032x} expected={want:032x}");
                }
                lane_failures[op][lane] += 1;
            }
        }
    }
    println!("lane * 64 [set_lane, mask_shl, mask_lshr, mask_ashr, mask_not_shl, mask_not_lshr]: failures per operation/lane={lane_failures:?}");
    let mut standalone = DeviceBuffer::<u64>::zeroed(&stream, lo.len() * 2).unwrap();
    let cfg = LaunchConfig::for_num_elems((lo.len() * 2) as u32);
    // SAFETY: buffers cover the standalone kernel's input and output indices.
    unsafe { module.set_lane(&stream, cfg, &lo_device, &hi_device, &lanes_device, &mut standalone) }
        .expect("launch standalone lane kernel");
    let actual = standalone.to_host_vec(&stream).unwrap();
    let mut standalone_failures = [0usize; 2];
    for case in 0..lo.len() {
        let a = lo[case] as u128 | ((hi[case] as u128) << 64);
        let shift = lanes[case] * 64;
        let want = (a & !((u64::MAX as u128) << shift)) | ((lo[case] as u128) << shift);
        let got = actual[2 * case] as u128 | ((actual[2 * case + 1] as u128) << 64);
        if got != want {
            let lane = lanes[case] as usize;
            if standalone_failures[lane] == 0 {
                eprintln!("standalone lane {lane}: input={a:032x} got={got:032x} expected={want:032x}");
            }
            standalone_failures[lane] += 1;
        }
    }
    println!("standalone set_lane: failures per lane={standalone_failures:?}");
    assert_eq!(standalone_failures, [0; 2], "standalone lane mask mismatch");
    assert_eq!(lane_failures, [[0; 2]; LANE_OPS], "lane mask device/host mismatch");
    assert_eq!(failures, [0; OPS], "wide integer device/host mismatch");
    println!(
        "PASS: {} wide-integer comparisons, all shifts 0..127 (including 0, 1, 63, 64, 65, 127)",
        expected.len() + lo.len() * (LANE_OPS + 1),
    );
}
