/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Read-only global-memory loads.
//!
//! [`load`] is the safe Rust equivalent of CUDA's `__ldg`: its shared borrow
//! proves that the operation cannot write through the supplied reference.  The
//! cuda-oxide compiler preserves the load as one cache-qualified instruction,
//! including for the over-aligned vector values in [`crate::vector`].

/// A bounds-proved contiguous register tile for one CUDA thread.
///
/// CUDA block algorithms commonly prove a complete global-memory tile once,
/// then issue several statically unrolled register loads from that tile. A
/// Rust slice access at every unrolled coordinate would repeat the same bounds
/// failure edge and retain it in PTX. This capability keeps only the proved
/// subslice: its data pointer is the tile base and its length is the live
/// sample count. [`Self::load`] exposes only const-selected coordinates and
/// lowers those two scalar fields directly to SSA.
///
/// `FULL` preserves the same specialization as CUB's complete and partial
/// tile overloads. In the partial form, [`Self::load`] returns `None` exactly
/// for pixels beyond the live sample count; the backing slice extent was still
/// checked once by the constructor.
#[derive(Clone, Copy)]
pub struct ThreadTile<'a, T, const PIXELS: usize, const CHANNELS: usize, const FULL: bool> {
    samples: &'a [T],
}

impl<'a, T: Copy, const PIXELS: usize, const CHANNELS: usize, const FULL: bool>
    ThreadTile<'a, T, PIXELS, CHANNELS, FULL>
{
    /// Prove the complete active extent beginning at `start`.
    ///
    /// `valid_samples` is the number of active scalar samples owned by this
    /// thread, not the remaining size of the block tile. A full tile requires
    /// the complete `PIXELS * CHANNELS` extent; a partial tile accepts any
    /// prefix of it, including an empty prefix for inactive threads.
    #[inline(always)]
    pub fn new(input: &'a [T], start: usize, valid_samples: i32) -> Option<Self> {
        let capacity = PIXELS.checked_mul(CHANNELS)?;
        if capacity == 0 || capacity > i32::MAX as usize || valid_samples < 0 {
            return None;
        }
        let valid_samples = valid_samples as usize;
        if valid_samples > capacity
            || !valid_samples.is_multiple_of(CHANNELS)
            || (FULL && valid_samples != capacity)
        {
            return None;
        }
        if valid_samples == 0 {
            return Some(Self {
                samples: &input[..0],
            });
        }
        let end = start.checked_add(valid_samples)?;
        let samples = input.get(start..end)?;
        Some(Self { samples })
    }

    /// Load one statically selected channel from a statically selected pixel.
    ///
    /// `None` is the normal partial-tile inactive state. The `FULL`
    /// specialization needs no dynamic access check because construction of
    /// this private-field capability proved the complete tile. In both forms,
    /// the private compiler intrinsic performs the access without rebuilding
    /// the slice assertion.
    #[inline(always)]
    pub fn load<const PIXEL: usize, const CHANNEL: usize>(&self) -> Option<T> {
        if PIXEL >= PIXELS || CHANNEL >= CHANNELS {
            return None;
        }
        let local = PIXEL * CHANNELS;
        if !FULL && local + CHANNELS > self.samples.len() {
            return None;
        }
        Some(__load_at(self.samples, local + CHANNEL))
    }
}

/// Copy a value through CUDA's read-only global-memory path.
///
/// Device compilation recognizes this compiler stub as a memory intrinsic so
/// a value whose type carries 8- or 16-byte alignment is not scalarized when
/// its lanes are subsequently consumed.
#[must_use]
#[inline(never)]
pub fn load<T: Copy>(_value: &T) -> T {
    unreachable!("read_only::load executed outside cuda-oxide device compilation")
}

/// Return the first index whose value compares greater than `sample`.
///
/// This is CUDA's read-only-cache realization of the standard upper-bound
/// search. The loop maintains `retval <= input.len()` and
/// `retval + num_items <= input.len()`, so every private indexed load is
/// proved in bounds. cuda-oxide recognizes that private load boundary and
/// emits one cache-qualified transaction without duplicating a bounds trap at
/// every search iteration.
#[must_use]
#[inline(always)]
pub fn upper_bound<T: Copy + PartialOrd>(input: &[T], sample: T) -> i32 {
    let mut retval = 0i32;
    let mut num_items = input.len() as i32;
    while num_items > 0 {
        let half = num_items >> 1;
        let probe = retval.wrapping_add(half);
        if sample < __load_at(input, probe as usize) {
            num_items = half;
        } else {
            let consumed = half.wrapping_add(1);
            retval = retval.wrapping_add(consumed);
            num_items = num_items.wrapping_sub(consumed);
        }
    }
    retval
}

/// Compiler boundary for the one indexed read in [`upper_bound`].
///
/// This function is private so arbitrary device code cannot present an
/// unproved index. Its call sites are the interval-preserving search above and
/// [`ThreadTile::load`], whose private slice can only be produced after
/// [`ThreadTile::new`] has validated the complete active extent.
#[inline(never)]
fn __load_at<T: Copy>(input: &[T], index: usize) -> T {
    input[index]
}

#[cfg(test)]
mod tests {
    use super::ThreadTile;

    #[test]
    fn thread_tile_proves_one_extent_and_keeps_partial_loads_explicit() {
        let samples = [0i32; 12];
        let full =
            ThreadTile::<_, 2, 2, true>::new(&samples, 4, 4).expect("complete two-pixel extent");
        assert_eq!(full.load::<0, 0>(), Some(0));
        assert_eq!(full.load::<1, 1>(), Some(0));

        let partial =
            ThreadTile::<_, 2, 2, false>::new(&samples, 8, 2).expect("one complete active pixel");
        assert_eq!(partial.load::<0, 0>(), Some(0));
        assert_eq!(partial.load::<0, 1>(), Some(0));
        assert_eq!(partial.load::<1, 0>(), None);
        assert_eq!(partial.load::<0, 2>(), None);
        assert!(ThreadTile::<_, 2, 2, true>::new(&samples, 9, 4).is_none());
        assert!(ThreadTile::<_, 2, 2, false>::new(&samples, 0, 3).is_none());
        let inactive = ThreadTile::<_, 2, 2, false>::new(&samples, usize::MAX, 0)
            .expect("inactive CUDA threads never form their nominal address");
        assert_eq!(inactive.load::<0, 0>(), None);
    }
}
