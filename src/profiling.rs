//! Opt-in, worker-thread-local profiling. Host hooks never synchronize.
#![cfg_attr(not(feature = "datafusion"), allow(dead_code))]
use cubecl::{client::ProfileWindow, prelude::Client, profile::TimingMethod};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    time::Instant,
};

/// SQL diagnostic collection. Does not change kernel identities or SQL eligibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProfilingMode {
    /// Existing coarse metrics only; no fine-grained clocks or device windows.
    #[default]
    Off,
    /// Host wall scopes, with no additional device synchronization.
    Host,
    /// Host scopes plus one device window per nonempty SQL request. Adds backend
    /// boundary synchronization; compare its overhead with Off and Host.
    Device,
}
#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Metric {
    ValidityExpandNs,
    ValidityExpandBytes,
    ValueEncodeNs,
    ValueEncodeBytes,
    ValidityEncodeNs,
    ValidityEncodeBytes,
    RuntimeUploadNs,
    RuntimeAllocateNs,
    PipelinePrepareNs,
    PipelineSubmitNs,
    CompletionSyncNs,
    ReadbackApiNs,
    ScalarDecodeNs,
    ArrowExportNs,
    DeviceWindowNs,
    DeviceWindows,
    DeviceUnavailable,
    DeviceFailed,
    DeviceBoundaryNs,
}
pub(crate) const NAMES: [&str; 19] = [
    "profile_validity_expand_ns",
    "profile_validity_expand_bytes",
    "profile_value_encode_ns",
    "profile_value_encode_bytes",
    "profile_validity_encode_ns",
    "profile_validity_encode_bytes",
    "profile_runtime_upload_ns",
    "profile_runtime_allocate_ns",
    "profile_pipeline_prepare_ns",
    "profile_pipeline_submit_ns",
    "profile_completion_sync_ns",
    "profile_readback_api_ns",
    "profile_scalar_decode_ns",
    "profile_arrow_export_ns",
    "profile_device_window_ns",
    "profile_device_windows",
    "profile_device_unavailable",
    "profile_device_failed",
    "profile_device_boundary_ns",
];
struct State {
    mode: ProfilingMode,
    values: [Cell<u64>; NAMES.len()],
}
impl State {
    fn add(&self, metric: Metric, value: u64) {
        let cell = &self.values[metric as usize];
        cell.set(cell.get().saturating_add(value));
    }
}
thread_local! {
    static CURRENT: RefCell<Option<Rc<State>>> = const { RefCell::new(None) };
}
/// Installed only inside the blocking SQL worker; nested scopes restore their
/// predecessor and do not leak to a reused Tokio blocking thread.
pub(crate) struct Scope {
    state: Option<Rc<State>>,
    previous: Option<Rc<State>>,
}
impl Scope {
    pub(crate) fn new(mode: ProfilingMode) -> Self {
        let state = (mode != ProfilingMode::Off).then(|| {
            Rc::new(State {
                mode,
                values: std::array::from_fn(|_| Cell::new(0)),
            })
        });
        let previous = CURRENT.with(|c| c.replace(state.clone()));
        Self { state, previous }
    }
    pub(crate) fn snapshot(&self) -> Option<[u64; NAMES.len()]> {
        self.state
            .as_ref()
            .map(|s| std::array::from_fn(|i| s.values[i].get()))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| {
            c.replace(self.previous.take());
        });
    }
}
pub(crate) struct Span {
    state: Rc<State>,
    metric: Metric,
    start: Instant,
}
impl Drop for Span {
    fn drop(&mut self) {
        self.state.add(
            self.metric,
            self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
    }
}
pub(crate) fn span(metric: Metric) -> Option<Span> {
    CURRENT.with(|c| {
        c.borrow().as_ref().map(|state| Span {
            state: state.clone(),
            metric,
            start: Instant::now(),
        })
    })
}
pub(crate) fn add(metric: Metric, value: u64) {
    CURRENT.with(|c| {
        if let Some(state) = c.borrow().as_ref() {
            state.add(metric, value);
        }
    });
}
/// One coarse device window, never per-kernel synchronization. Its lifecycle is
/// contained in the same SQL worker/gate as submission and checked materialization.
pub(crate) struct DeviceWindow {
    client: Client,
    token: Option<ProfileWindow>,
    state: Rc<State>,
}
impl DeviceWindow {
    pub(crate) fn begin(client: &Client, nonempty: bool) -> Option<Self> {
        let state = CURRENT.with(|c| c.borrow().clone())?;
        if !nonempty || state.mode != ProfilingMode::Device {
            return None;
        }
        // This pinned CPU runtime advertises Device but returns system time.
        // Admit only the verified native Metal implementation, and recheck the result.
        if client.name() != "metal" || client.properties().timing_method != TimingMethod::Device {
            state.add(Metric::DeviceUnavailable, 1);
            return None;
        }
        let _timer = span(Metric::DeviceBoundaryNs);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| client.profile_start())) {
            Ok(Ok(token)) => Some(Self {
                client: client.clone(),
                token: Some(token),
                state,
            }),
            _ => {
                state.add(Metric::DeviceFailed, 1);
                None
            }
        }
    }
    pub(crate) fn finish(mut self) {
        let _timer = span(Metric::DeviceBoundaryNs);
        let token = self.token.take().expect("open profile window");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let duration = self.client.profile_end(token).ok()?;
            if duration.timing_method() != TimingMethod::Device {
                return None;
            }
            cubecl::future::block_on(duration.resolve()).map(|ticks| ticks.duration())
        }));
        match result {
            Ok(Some(duration)) if !duration.is_zero() => {
                self.state.add(
                    Metric::DeviceWindowNs,
                    duration.as_nanos().min(u64::MAX as u128) as u64,
                );
                self.state.add(Metric::DeviceWindows, 1);
            }
            _ => {
                self.state.add(Metric::DeviceFailed, 1);
                // End consumes the token even on a normal backend error. A panic
                // could precede that; abandoning is idempotent for this token.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.client.profile_abandon(token)
                }));
            }
        }
    }
}
impl Drop for DeviceWindow {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            let _timer = span(Metric::DeviceBoundaryNs);
            self.state.add(Metric::DeviceFailed, 1);
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.client.profile_abandon(token)
            }));
        }
    }
}
#[cfg(test)]
mod tests {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    use super::DeviceWindow;
    use super::{Metric, ProfilingMode, Scope, add, span};
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn device_window_abandons_failed_work_and_reopens() {
        // Other library tests deliberately fail Metal buffers. The pinned server
        // invalidates every open timing window on such a failure, even another
        // stream's window. Run this positive timing/recovery contract in a fresh
        // process so default parallel tests cannot contaminate it.
        if std::env::var_os("CUBEREL_PROFILE_TEST_CHILD").is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "profiling::tests::device_window_abandons_failed_work_and_reopens",
                    "--test-threads=1",
                ])
                .env("CUBEREL_PROFILE_TEST_CHILD", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated Metal profiling failed: {} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        use crate::{
            Batch, ColumnData, DataType, ExecutionError, Executor, Field, Parameters, Schema, col,
            param, prepare,
        };
        let device = cubecl::Device::metal(Default::default()).unwrap();
        let executor = Executor::new(device.client()).unwrap();
        let schema = Schema::from_fields(
            &[Field::new("x", DataType::I32).with_nullable(false)],
            &[Field::new("one", DataType::I32).with_nullable(false)],
        )
        .unwrap();
        let plus = prepare(&(col("x") + param("one")), &schema).unwrap();
        let params = Parameters::new().with("one", 1_i32).unwrap();
        let scope = Scope::new(ProfilingMode::Device);
        {
            let _window = DeviceWindow::begin(&executor.client, true).expect("Metal device window");
            let values = [i32::MAX];
            let batch = Batch::new(1)
                .with_column("x", ColumnData::I32(&values))
                .unwrap();
            assert!(matches!(
                executor.execute_nullable_i32(&plus, &batch, &params),
                Err(ExecutionError::IntegerOverflow { row: 0 })
            ));
            // The SQL error path likewise drops/abandons instead of publishing timing.
        }
        assert_eq!(scope.snapshot().unwrap()[Metric::DeviceFailed as usize], 1);
        assert_eq!(scope.snapshot().unwrap()[Metric::DeviceWindows as usize], 0);
        let window = DeviceWindow::begin(&executor.client, true).expect("retry window");
        let values = [2];
        let batch = Batch::new(1)
            .with_column("x", ColumnData::I32(&values))
            .unwrap();
        assert!(
            executor
                .execute_nullable_i32(&plus, &batch, &params)
                .is_ok()
        );
        window.finish();
        let snapshot = scope.snapshot().unwrap();
        assert_eq!(snapshot[Metric::DeviceWindows as usize], 1);
        assert!(snapshot[Metric::DeviceWindowNs as usize] > 0);
        assert_eq!(snapshot[Metric::DeviceFailed as usize], 1);
    }

    #[test]
    fn worker_scopes_restore_and_unwind_without_leaking() {
        assert!(span(Metric::ValueEncodeNs).is_none());
        let outer = Scope::new(ProfilingMode::Host);
        add(Metric::ValueEncodeBytes, 17);
        {
            let inner = Scope::new(ProfilingMode::Off);
            assert!(inner.snapshot().is_none());
            assert!(span(Metric::ValueEncodeNs).is_none());
            add(Metric::ValueEncodeBytes, 100);
        }
        let _ = std::panic::catch_unwind(|| {
            let inner = Scope::new(ProfilingMode::Host);
            add(Metric::ValueEncodeBytes, 33);
            assert_eq!(
                inner.snapshot().unwrap()[Metric::ValueEncodeBytes as usize],
                33
            );
            panic!("exercise unwind");
        });
        assert_eq!(
            outer.snapshot().unwrap()[Metric::ValueEncodeBytes as usize],
            17
        );
        drop(outer);
        assert!(span(Metric::ValueEncodeNs).is_none());
        std::thread::spawn(|| assert!(span(Metric::ValueEncodeNs).is_none()))
            .join()
            .unwrap();
    }
}
