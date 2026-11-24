//! CLI-only logical adapter budget configuration.
use super::support::Result;
use cuberel::datafusion::ProfilingMode;

/// Explicit logical adapter caps for a measurement; upstream/driver memory is excluded.
pub fn shared_limits(args: &[String]) -> Result<cuberel::datafusion::SharedLimits> {
    Ok(parse_measurement_flags(args)?.0)
}

/// Parse the optional profiling mode from the same measurement flag list.
pub fn profiling_mode(args: &[String]) -> Result<ProfilingMode> {
    Ok(parse_measurement_flags(args)?.1)
}

fn parse_measurement_flags(
    args: &[String],
) -> Result<(cuberel::datafusion::SharedLimits, ProfilingMode)> {
    let mut limits = cuberel::datafusion::SharedLimits::default();
    let mut host = false;
    let mut device = false;
    let mut profiling = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("measurement flag {flag} requires a value"))?;
        match flag {
            "--shared-host-bytes" | "--shared-device-bytes" => {
                let (seen, target) = if flag == "--shared-host-bytes" {
                    (&mut host, &mut limits.max_host_bytes)
                } else {
                    (&mut device, &mut limits.max_device_bytes)
                };
                if *seen {
                    return Err(format!("duplicate measurement flag: {flag}").into());
                }
                let parsed = value.parse::<u64>()?;
                if parsed == 0 {
                    return Err("shared measurement budgets must be positive".into());
                }
                *seen = true;
                *target = parsed;
            }
            "--profiling" => {
                if profiling.is_some() {
                    return Err("duplicate measurement flag: --profiling".into());
                }
                profiling = Some(match value.as_str() {
                    "off" => ProfilingMode::Off,
                    "host" => ProfilingMode::Host,
                    "device" => ProfilingMode::Device,
                    _ => {
                        return Err(
                            format!("profiling must be off, host, or device, got {value}").into(),
                        );
                    }
                });
            }
            _ => return Err(format!("unknown measurement flag: {flag}").into()),
        }
        index += 2;
    }
    Ok((limits, profiling.unwrap_or(ProfilingMode::Off)))
}
