//! CLI-only logical adapter budget configuration.
use super::support::Result;

/// Explicit logical adapter caps for a measurement; upstream/driver memory is excluded.
pub fn shared_limits(args: &[String]) -> Result<cuberel::datafusion::SharedLimits> {
    let mut limits = cuberel::datafusion::SharedLimits::default();
    let mut host = false;
    let mut device = false;
    let (pairs, remainder) = args.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err("shared budget flags require a value".into());
    }
    for pair in pairs {
        let (seen, target) = match pair[0].as_str() {
            "--shared-host-bytes" => (&mut host, &mut limits.max_host_bytes),
            "--shared-device-bytes" => (&mut device, &mut limits.max_device_bytes),
            _ => return Err(format!("unknown measurement flag: {}", pair[0]).into()),
        };
        if *seen {
            return Err(format!("duplicate measurement flag: {}", pair[0]).into());
        }
        let value = pair[1].parse::<u64>()?;
        if value == 0 {
            return Err("shared measurement budgets must be positive".into());
        }
        *seen = true;
        *target = value;
    }
    Ok(limits)
}
