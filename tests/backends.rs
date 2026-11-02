//! These checks exercise actual devices and fail when a requested device is absent.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

fn buffer_round_trip(device: cubecl::Device) {
    let client = device.client();
    let expected: Vec<u8> = (0..67).map(|i| (i * 3) as u8).collect();
    let handle = client.create_from_slice(&expected);
    let actual = client.read_one(handle).expect("read device buffer");
    assert_eq!(&actual[..], expected.as_slice());
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_buffer_round_trip() {
    let device = cubecl::Device::cpu().expect("CPU runtime must be available");
    assert_eq!(device.runtime(), cubecl::RuntimeId::Cpu);
    buffer_round_trip(device);
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_buffer_round_trip() {
    let device = cubecl::Device::metal(Default::default())
        .expect("native Metal runtime and device must be available");
    assert_eq!(device.runtime(), cubecl::RuntimeId::Metal);
    buffer_round_trip(device);
}
