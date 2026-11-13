//! `jit:u`: the JIT service titles use to run a code-generation plugin. `CreateJitEnvironment`
//! maps the title's two code memory objects into its own alias code region, as the reference
//! does, and hands out `IJitEnvironment`.
//!
//! Loading the plugin NRO and calling into it (`LoadPlugin`, `Control`, `GenerateCode`) needs a
//! second guest CPU context with its own memory, which the CPU crate does not expose to
//! services, so those commands are left unimplemented.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode, VAddr};
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory::{MemoryState, Regions};
use lichii_kernel::memory_objects::{CodeMemory, CodeMemoryOperation};
use lichii_kernel::process::Process;
use lichii_kernel::result;
use lichii_memory::MemoryPermission;

use crate::{Service, ServiceContext};

const UNKNOWN: ResultCode = ResultCode(u32::MAX);

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("jit:u", "jit", |context| JitU {
        context: context.clone(),
    })
}

struct JitU {
    context: Arc<ServiceContext>,
}

impl Service for JitU {
    fn name(&self) -> &str {
        "jit:u"
    }

    /// `CreateJitEnvironment`: rx and ro sizes, then copy handles to the process and its rx and
    /// ro code memory.
    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if ctx.command() != 0 {
            return None;
        }
        let rx_size: u64 = ctx.pop();
        let ro_size: u64 = ctx.pop();
        let objects = (
            ctx.copy_handle::<Process>(0),
            ctx.copy_handle::<CodeMemory>(1),
            ctx.copy_handle::<CodeMemory>(2),
        );
        let (Ok(process), Ok(rx), Ok(ro)) = objects else {
            return Some(UNKNOWN);
        };
        let rx = match CodeMapping::new(&process, rx, rx_size, MemoryPermission::READ_EXECUTE) {
            Ok(rx) => rx,
            Err(code) => return Some(code),
        };
        let ro = match CodeMapping::new(&process, ro, ro_size, MemoryPermission::READ) {
            Ok(ro) => ro,
            Err(code) => return Some(code),
        };
        ctx.push_interface(self.context.handler(JitEnvironment { rx, ro }));
        Some(ResultCode::SUCCESS)
    }
}

/// A code memory object mapped into its owner's alias code region, unmapped on drop.
struct CodeMapping {
    memory: Arc<CodeMemory>,
    address: VAddr,
    size: u64,
}

impl CodeMapping {
    /// Maps `memory` at the first free range of `process`'s alias code region that takes it,
    /// clear of the heap and alias regions (`CanContain(AliasCode)`).
    fn new(
        process: &Process,
        memory: Arc<CodeMemory>,
        size: u64,
        permission: MemoryPermission,
    ) -> HleResult<Self> {
        let table = process.page_table();
        let regions = table.regions();
        let region = regions.alias_code;
        let mut address = region.start;
        while address < region.end && region.end - address >= size {
            let clear = skip_reserved(regions, address, size);
            if clear != address {
                address = clear;
                continue;
            }
            let info = table.query(address);
            let end = (info.base_address + info.size).min(region.end);
            if info.state == MemoryState::FREE.0 && end - address >= size {
                match memory.control(CodeMemoryOperation::MapSlave, address, size, permission) {
                    Ok(()) => {
                        return Ok(Self {
                            memory,
                            address,
                            size,
                        });
                    }
                    Err(code)
                        if code != result::INVALID_CURRENT_MEMORY
                            && code != result::INVALID_MEMORY_REGION =>
                    {
                        return Err(code);
                    }
                    Err(_) => {}
                }
            }
            address = end;
        }
        Err(UNKNOWN)
    }
}

/// The first address from `address` where `size` bytes stay clear of the heap and alias
/// regions, which the alias code region contains but code may not use.
fn skip_reserved(regions: &Regions, mut address: VAddr, size: u64) -> VAddr {
    while let Some(reserved) = [regions.heap, regions.alias]
        .into_iter()
        .find(|reserved| reserved.overlaps(address, size))
    {
        address = reserved.end;
    }
    address
}

impl Drop for CodeMapping {
    fn drop(&mut self) {
        let _ = self.memory.control(
            CodeMemoryOperation::UnmapSlave,
            self.address,
            self.size,
            MemoryPermission::NONE,
        );
    }
}

/// `IJitEnvironment`.
struct JitEnvironment {
    rx: CodeMapping,
    ro: CodeMapping,
}

impl Service for JitEnvironment {
    fn name(&self) -> &str {
        "IJitEnvironment"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            1001 => {
                ctx.push(self.rx.address);
                ctx.push(self.ro.address);
                Some(ResultCode::SUCCESS)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
