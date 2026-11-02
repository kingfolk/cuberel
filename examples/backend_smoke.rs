//! Check device initialization and buffer transfers; this does not launch a kernel.
use std::{env, error::Error};

use cubecl::Device;

fn main() -> Result<(), Box<dyn Error>> {
    let backend = env::args()
        .nth(1)
        .ok_or("select a backend: cargo run --example backend_smoke -- cpu (or metal)")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err(format!("unknown backend {backend:?}; expected cpu or metal").into()),
    };
    let client = device.client();
    let input = [1_u32, 2, 3, 4, 0x1234_5678];
    let bytes: Vec<u8> = input.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let handle = client.create_from_slice(&bytes);
    let output = client.read_one(handle)?;
    if &output[..] != bytes.as_slice() {
        return Err("device buffer round-trip returned different bytes".into());
    }
    println!(
        "backend={:?} device={} buffer_round_trip={} bytes; no kernel launched",
        device.runtime(),
        client.name(),
        output.len()
    );
    Ok(())
}
