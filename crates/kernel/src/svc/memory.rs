use std::sync::Arc;

use lichii_common::{HleResult, PAGE_SIZE};
use lichii_memory::MemoryPermission;

use super::{Flow, SvcContext};
use crate::memory::attribute;
use crate::memory_objects::{CodeMemory, CodeMemoryOperation, SharedMemory, TransferMemory};
use crate::object::KObject;
use crate::result;

fn permission(raw: u32) -> HleResult<MemoryPermission> {
    match raw {
        0 | 1 | 3 | 5 => Ok(MemoryPermission(raw as u8)),
        _ => Err(result::INVALID_NEW_MEMORY_PERMISSION),
    }
}

fn check_range(addr: u64, size: u64) -> HleResult<()> {
    if !addr.is_multiple_of(PAGE_SIZE) {
        return Err(result::INVALID_ADDRESS);
    }
    if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
        return Err(result::INVALID_SIZE);
    }
    if addr.checked_add(size).is_none() {
        return Err(result::INVALID_CURRENT_MEMORY);
    }
    Ok(())
}

/// `svcSetHeapSize(size x1)` -> `x1` heap address.
pub(super) fn set_heap_size(svc: &mut SvcContext) -> Flow {
    match svc.process().page_table().set_heap_size(svc.arg(1)) {
        Ok(addr) => {
            svc.set(1, addr);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcSetMemoryPermission(addr x0, size x1, perm w2)`.
pub(super) fn set_memory_permission(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(0), svc.arg(1));
    let result = check_range(addr, size)
        .and_then(|()| permission(svc.arg32(2)))
        .and_then(|perm| {
            svc.process()
                .page_table()
                .set_memory_permission(addr, size, perm)
        });
    svc.finish(result)
}

fn set_attribute(svc: &mut SvcContext, supported: u32) -> Flow {
    let (addr, size, mask, attr) = (svc.arg(0), svc.arg(1), svc.arg32(2), svc.arg32(3));
    let result = check_range(addr, size).and_then(|()| {
        if attr & !mask != 0 || (mask | attr) & !supported != 0 {
            return Err(result::INVALID_COMBINATION);
        }
        svc.process()
            .page_table()
            .set_memory_attribute(addr, size, mask, attr)
    });
    svc.finish(result)
}

/// `svcSetMemoryAttribute(addr x0, size x1, mask w2, attr w3)`.
pub(super) fn set_memory_attribute(svc: &mut SvcContext) -> Flow {
    set_attribute(svc, attribute::UNCACHED | attribute::PERMISSION_LOCKED)
}

/// Switch 2 `svcSetGpuMemoryAttribute(addr x0, size x1, mask w2, attr w3)` (0x80): the same
/// checks as `SetMemoryAttribute`, restricted to the GPU-accessible bit (NX2_NOTES rows 27/47).
pub(super) fn set_gpu_memory_attribute(svc: &mut SvcContext) -> Flow {
    set_attribute(svc, attribute::GPU_ACCESSIBLE)
}

/// `svcMapMemory(dst x0, src x1, size x2)`.
pub(super) fn map_memory(svc: &mut SvcContext) -> Flow {
    let result = svc
        .process()
        .page_table()
        .map_memory(svc.arg(0), svc.arg(1), svc.arg(2));
    svc.finish(result)
}

/// `svcUnmapMemory(dst x0, src x1, size x2)`.
pub(super) fn unmap_memory(svc: &mut SvcContext) -> Flow {
    let result = svc
        .process()
        .page_table()
        .unmap_memory(svc.arg(0), svc.arg(1), svc.arg(2));
    svc.finish(result)
}

/// `svcQueryMemory(info x0, _, addr x2)` -> `w1` page info.
pub(super) fn query_memory(svc: &mut SvcContext) -> Flow {
    let (out, addr) = (svc.arg(0), svc.arg(2));
    let info = svc.process().page_table().query(addr);
    let result = svc
        .process()
        .memory()
        .write_bytes(out, bytemuck::bytes_of(&info))
        .map_err(|_| result::INVALID_POINTER);
    svc.set(1, 0);
    svc.finish(result)
}

/// `svcMapPhysicalMemory(addr x0, size x1)`.
pub(super) fn map_physical_memory(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(0), svc.arg(1));
    let result = check_range(addr, size)
        .and_then(|()| svc.process().page_table().map_physical_memory(addr, size));
    svc.finish(result)
}

/// `svcUnmapPhysicalMemory(addr x0, size x1)`.
pub(super) fn unmap_physical_memory(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(0), svc.arg(1));
    let result = check_range(addr, size)
        .and_then(|()| svc.process().page_table().unmap_physical_memory(addr, size));
    svc.finish(result)
}

/// `svcMapSharedMemory(handle w0, addr x1, size x2, perm w3)`.
pub(super) fn map_shared_memory(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(1), svc.arg(2));
    let result = check_range(addr, size).and_then(|()| {
        let perm = permission(svc.arg32(3))?;
        let shared = svc.typed::<SharedMemory>(svc.arg32(0))?;
        shared.map(svc.process(), addr, size, perm)
    });
    svc.finish(result)
}

/// `svcUnmapSharedMemory(handle w0, addr x1, size x2)`.
pub(super) fn unmap_shared_memory(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(1), svc.arg(2));
    let result = check_range(addr, size).and_then(|()| {
        let shared = svc.typed::<SharedMemory>(svc.arg32(0))?;
        shared.unmap(svc.process(), addr, size)
    });
    svc.finish(result)
}

/// `svcCreateTransferMemory(addr x1, size x2, perm w3)` -> `w1` handle.
pub(super) fn create_transfer_memory(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(1), svc.arg(2));
    let result = check_range(addr, size).and_then(|()| {
        let perm = permission(svc.arg32(3))?;
        let process = svc.process();
        let memory = TransferMemory::new(process.clone(), addr, size, perm)?;
        process
            .handles()
            .write()
            .add(KObject::TransferMemory(Arc::new(memory)))
    });
    match result {
        Ok(handle) => {
            svc.set(1, handle as u64);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcCreateCodeMemory(addr x1, size x2)` -> `w1` handle.
pub(super) fn create_code_memory(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(1), svc.arg(2));
    let result = check_range(addr, size).and_then(|()| {
        let memory = CodeMemory::new(svc.process().clone(), addr, size)?;
        svc.process()
            .handles()
            .write()
            .add(KObject::CodeMemory(Arc::new(memory)))
    });
    match result {
        Ok(handle) => {
            svc.set(1, handle as u64);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcControlCodeMemory(handle w0, op w1, addr x2, size x3, perm w4)`.
pub(super) fn control_code_memory(svc: &mut SvcContext) -> Flow {
    let (addr, size) = (svc.arg(2), svc.arg(3));
    let result = check_range(addr, size).and_then(|()| {
        let op = CodeMemoryOperation::from_raw(svc.arg32(1))?;
        let perm = permission(svc.arg32(4))?;
        let memory = svc.typed::<CodeMemory>(svc.arg32(0))?;
        memory.control(op, addr, size, perm)
    });
    svc.finish(result)
}
