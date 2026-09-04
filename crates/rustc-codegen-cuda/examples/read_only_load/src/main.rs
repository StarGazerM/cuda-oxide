use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::vector::{I32x4, as_vectors};
    use cuda_device::{DisjointSlice, debug, kernel, read_only, thread};
    #[kernel]
    pub fn sum_quads(input: &[i32], mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        let lane = index.get();
        if let Some(out) = output.get_mut(index)
            && let Some(quads) = as_vectors::<I32x4>(input)
            && let Some(quad) = quads.get(lane)
        {
            let lanes = read_only::load(quad).to_array();
            *out = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        }
    }

    #[kernel]
    pub fn sum_proven_tile(input: &[i32], mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        if index.get() != 0 {
            return;
        }
        let Some(out) = output.get_mut(index) else {
            return;
        };
        let Some(tile) = read_only::ThreadTile::<i32, 16, 1, true>::new(input, 0, 16) else {
            debug::trap();
        };
        macro_rules! value {
            ($pixel:literal) => {
                match tile.load::<$pixel, 0>() {
                    Some(value) => value,
                    None => debug::trap(),
                }
            };
        }
        *out = value!(0)
            + value!(1)
            + value!(2)
            + value!(3)
            + value!(4)
            + value!(5)
            + value!(6)
            + value!(7)
            + value!(8)
            + value!(9)
            + value!(10)
            + value!(11)
            + value!(12)
            + value!(13)
            + value!(14)
            + value!(15);
    }

    #[kernel]
    pub fn checked_index(input: &[i32], index: usize, mut output: DisjointSlice<i32>) {
        let thread = thread::index_1d();
        if let Some(out) = output.get_mut(thread) {
            *out = input[index];
        }
    }
}

fn main() {
    let fault = std::env::args().any(|arg| arg == "--fault");
    let context = CudaContext::new(0).expect("context");
    let stream = context.default_stream();
    let host_input: Vec<i32> = (1..=16).collect();
    let input = DeviceBuffer::from_host(&stream, &host_input).expect("input");
    let mut quad_output = DeviceBuffer::<i32>::zeroed(&stream, 4).expect("quad output");
    let mut tile_output = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("tile output");
    let mut checked_output = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("checked output");
    let module = kernels::load(&context).expect("module");
    unsafe {
        module.sum_quads(
            &stream,
            LaunchConfig::for_num_elems(4),
            &input,
            &mut quad_output,
        )
    }
    .expect("quad launch");
    unsafe {
        module.sum_proven_tile(
            &stream,
            LaunchConfig::for_num_elems(1),
            &input,
            &mut tile_output,
        )
    }
    .expect("tile launch");
    assert_eq!(
        quad_output.to_host_vec(&stream).expect("quad copy"),
        [10, 26, 42, 58]
    );
    assert_eq!(tile_output.to_host_vec(&stream).expect("tile copy"), [136]);
    let checked_index = if fault { host_input.len() } else { 15 };
    unsafe {
        module.checked_index(
            &stream,
            LaunchConfig::for_num_elems(1),
            &input,
            checked_index,
            &mut checked_output,
        )
    }
    .expect("checked launch");
    let checked = checked_output.to_host_vec(&stream);
    if fault {
        match checked {
            Err(error) => {
                println!("expected checked-index trap: {error}");
                return;
            }
            Ok(value) => panic!("out-of-bounds checked index completed with {value:?}"),
        }
    }
    assert_eq!(checked.expect("checked copy"), [16]);
}
