//! The memory behind nx2:gpu's lazy GPU-only regions ([`lichii_gpu::lazy`]): 4 MiB blocks of the
//! graphics server's carve-out (the non-secure system pool), each mapped in the alias code region
//! like the reference's graphics server memory. A few blocks are kept ready, so the GPU thread
//! that backs a block takes one without waiting for the kernel; a host thread refills them and
//! frees the blocks of destroyed regions.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use lichii_common::{DAddr, VAddr};
use lichii_gpu::lazy::{LAZY_BLOCK_SIZE, LazyBacking, LazyBlock};
use lichii_kernel::memory::{MemoryState, Pool};
use lichii_kernel::process::Process;
use lichii_memory::{AddressSpace, DeviceAddressSpace, MemoryPermission};
use parking_lot::{Condvar, Mutex};

use crate::ServiceContext;

/// Blocks kept ready (64 MiB), as in the reference. Test kernels have a 16 MiB system pool.
pub(super) const READY_BLOCKS: usize = if cfg!(test) { 2 } else { 16 };
/// How long the refill thread sleeps without work before it checks whether the title is gone.
const IDLE: Duration = Duration::from_millis(100);

/// Gives lazy blocks their memory; installed once per GPU.
pub(crate) struct Reserve {
    device: Arc<DeviceAddressSpace>,
    space: Arc<AddressSpace>,
    state: Mutex<State>,
    work: Condvar,
    filling: Mutex<()>,
}

#[derive(Default)]
struct State {
    /// Zeroed blocks, by address in the process.
    ready: Vec<VAddr>,
    /// The block behind each device block that got memory. Its capacity covers every ready block
    /// too, so backing one never allocates.
    backed: HashMap<DAddr, VAddr>,
    /// Blocks of destroyed regions the GPU is past, to free.
    released: Vec<VAddr>,
    /// The pool had no block left; cleared when blocks are freed.
    exhausted: bool,
}

impl State {
    fn has_work(&self) -> bool {
        !self.released.is_empty() || (self.ready.len() < READY_BLOCKS && !self.exhausted)
    }
}

impl Reserve {
    /// Installs a reserve as the backing of the GPU's lazy ranges and starts its refill thread.
    pub(crate) fn start(context: &ServiceContext) -> Arc<Self> {
        let host1x = context.gpu.host1x();
        let reserve = Arc::new(Self {
            device: host1x.memory.clone(),
            space: context.process.memory().clone(),
            state: Mutex::default(),
            work: Condvar::new(),
            filling: Mutex::new(()),
        });
        host1x.lazy.set_backing(reserve.clone());
        let (weak, process): (Weak<Self>, Weak<Process>) =
            (Arc::downgrade(&reserve), Arc::downgrade(&context.process));
        let spawned = std::thread::Builder::new()
            .name("nx2:gpu reserve".into())
            .spawn(move || {
                loop {
                    let Some(reserve) = weak.upgrade() else {
                        return;
                    };
                    let Some(process) = process.upgrade() else {
                        return;
                    };
                    reserve.refill(&process);
                    drop(process);
                    let mut state = reserve.state.lock();
                    if !state.has_work() {
                        reserve.work.wait_for(&mut state, IDLE);
                    }
                }
            });
        if let Err(err) = spawned {
            tracing::error!("spawning the nx2:gpu reserve thread: {err}");
        }
        reserve
    }

    /// Frees the released blocks, then maps blocks until [`READY_BLOCKS`] are ready or the pool
    /// runs out. Makes kernel calls, so it runs on a service or host thread, never a GPU one.
    pub(crate) fn refill(&self, process: &Process) {
        let _filling = self.filling.lock();
        let page_table = process.page_table();
        let released = std::mem::take(&mut self.state.lock().released);
        for &va in &released {
            if let Err(code) =
                page_table.unmap_and_free(va, LAZY_BLOCK_SIZE, MemoryState::IO_MEMORY)
            {
                tracing::warn!("nx2:gpu: freeing GPU-only block at {va:#x}: {code:?}");
            }
        }
        if !released.is_empty() {
            self.state.lock().exhausted = false;
        }
        loop {
            let state = self.state.lock();
            if state.ready.len() >= READY_BLOCKS || state.exhausted {
                return;
            }
            drop(state);
            let mapped = page_table.map_new_from(
                Pool::SystemNonSecure,
                page_table.regions().alias_code,
                LAZY_BLOCK_SIZE,
                MemoryState::IO_MEMORY,
                MemoryPermission::READ_WRITE,
            );
            let mut state = self.state.lock();
            match mapped {
                Ok(va) => {
                    state.ready.push(va);
                    let ready = state.ready.len();
                    state.backed.reserve(ready);
                }
                Err(code) => {
                    tracing::warn!(
                        "nx2:gpu: the graphics carve-out has no GPU-only block left ({code:?}); \
                         writes to unbacked blocks are dropped until regions are destroyed"
                    );
                    state.exhausted = true;
                }
            }
        }
    }

    /// Takes back the memory of `blocks`, which the device no longer maps and the GPU is past:
    /// the refill thread frees it.
    pub(crate) fn release(&self, blocks: &[LazyBlock]) {
        if blocks.is_empty() {
            return;
        }
        let mut state = self.state.lock();
        for block in blocks {
            match state.backed.remove(&block.daddr) {
                Some(va) => state.released.push(va),
                None => tracing::warn!(
                    "nx2:gpu: device block {:#x} was not backed by the reserve",
                    block.daddr
                ),
            }
        }
        self.work.notify_one();
    }
}

impl LazyBacking for Reserve {
    fn back(&self, daddr: DAddr, size: u64) -> bool {
        let mut state = self.state.lock();
        let Some(va) = state.ready.pop() else {
            return false;
        };
        self.work.notify_one();
        if let Err(fault) = self.device.map_process(daddr, &self.space, va, size) {
            tracing::error!("nx2:gpu: GPU-only block at {va:#x} is not mapped: {fault:?}");
            state.released.push(va);
            return false;
        }
        state.backed.insert(daddr, va);
        true
    }
}
