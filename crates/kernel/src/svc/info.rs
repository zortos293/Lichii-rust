use lichii_common::HleResult;

use super::{Flow, SvcContext};
use crate::handle_table::CURRENT_PROCESS;
use crate::memory::Pool;
use crate::object::KObject;
use crate::process::Process;
use crate::resource_limit::{LimitableResource, ResourceLimit};
use crate::result;

const INVALID_HANDLE: u32 = 0;
/// `GetInfo` 33: a page granule; the Switch 2 SDK checks module ranges against `value - 1`
/// and VAMM maps one granule up front (NX2_NOTES row 8).
const NX2_VAMM_GRANULE: u64 = 0x1000;

fn process_info(svc: &SvcContext, process: &Process, id: u32) -> HleResult<u64> {
    let regions = process.page_table().regions();
    Ok(match id {
        0 => {
            let mask = process.capabilities().core_mask;
            if mask == 0 {
                svc.kernel.scheduler.all_cores_mask()
            } else {
                mask
            }
        }
        1 => process.capabilities().priority_mask,
        2 => regions.alias.start,
        3 => regions.alias.size(),
        4 => regions.heap.start,
        5 => regions.heap.size(),
        6 | 21 => process.total_memory(),
        7 | 22 => process.used_memory(),
        12 => regions.alias_code.start,
        13 => regions.alias_code.size(),
        14 => regions.stack.start,
        15 => regions.stack.size(),
        16 => process.params().system_resource_size,
        17 => 0,
        18 => process.program_id(),
        20 => process.local_region(),
        23 => process.is_application() as u64,
        24 => {
            let limit = process.resource_limit();
            (limit.limit(LimitableResource::Threads) - limit.current(LimitableResource::Threads))
                as u64
        }
        28 => regions.alias_extra_size,
        // Switch 2: the region VAMM reserves address space in. VAMM backs it with
        // MapPhysicalMemory, which only works in the alias region, so it is the alias region's
        // upper half (NX2_NOTES row 7).
        29 => regions.alias.start + regions.alias.size() / 2,
        30 => regions.alias.size() / 2,
        // Switch 2 remote memory for gsv/nvsrv (row 16): none.
        31 | 32 => 0,
        33 => NX2_VAMM_GRANULE,
        _ => return Err(result::INVALID_ENUM_VALUE),
    })
}

fn info(svc: &mut SvcContext, id: u32, handle: u32, sub: u64) -> HleResult<u64> {
    match id {
        0..=7 | 12..=18 | 20..=24 | 28..=33 => {
            if sub != 0 {
                return Err(result::INVALID_ENUM_VALUE);
            }
            let process = match svc.object(handle) {
                Some(KObject::Process(process)) => process,
                _ => return Err(result::INVALID_HANDLE),
            };
            process_info(svc, &process, id)
        }
        8 => Ok(0),
        9 => {
            if handle != INVALID_HANDLE {
                return Err(result::INVALID_HANDLE);
            }
            if sub != 0 {
                return Err(result::INVALID_COMBINATION);
            }
            let limit = svc.process().resource_limit().clone();
            let handle = svc
                .process()
                .handles()
                .write()
                .add(KObject::ResourceLimit(limit))?;
            Ok(handle as u64)
        }
        10 => {
            if handle != INVALID_HANDLE {
                return Err(result::INVALID_HANDLE);
            }
            if sub != u64::MAX && sub != svc.core as u64 {
                return Err(result::INVALID_COMBINATION);
            }
            Ok(0)
        }
        11 => {
            if handle != INVALID_HANDLE {
                return Err(result::INVALID_HANDLE);
            }
            if sub >= 4 {
                return Err(result::INVALID_COMBINATION);
            }
            Ok(svc.process().random_entropy(sub as usize))
        }
        19 => match sub {
            0 => Ok(0),
            1 => Ok(0x50),
            _ => Err(result::INVALID_COMBINATION),
        },
        25 => {
            if sub != u64::MAX && sub >= svc.kernel.scheduler.core_count() as u64 {
                return Err(result::INVALID_COMBINATION);
            }
            svc.typed::<crate::thread::Thread>(handle)?;
            Ok(0)
        }
        26 => Ok(svc.process().capabilities().permits_svc(sub as u32) as u64),
        0xf000_0000 => match sub {
            0 => Ok(0),
            1 | 2 => Ok(0),
            _ => Err(result::INVALID_COMBINATION),
        },
        0xf000_0001 => {
            if handle != INVALID_HANDLE {
                return Err(result::INVALID_HANDLE);
            }
            let process = svc.process().clone();
            Ok(svc
                .process()
                .handles()
                .write()
                .add(KObject::Process(process))? as u64)
        }
        _ => {
            tracing::warn!(id, sub, "unknown GetInfo id");
            Err(result::INVALID_ENUM_VALUE)
        }
    }
}

/// `svcGetInfo(id w1, handle w2, sub x3)` -> `x1`.
pub(super) fn get_info(svc: &mut SvcContext) -> Flow {
    let (id, handle, sub) = (svc.arg32(1), svc.arg32(2), svc.arg(3));
    let handle = if handle == INVALID_HANDLE && matches!(id, 0..=7 | 12..=18 | 20..=24 | 28..=33) {
        CURRENT_PROCESS
    } else {
        handle
    };
    match info(svc, id, handle, sub) {
        Ok(value) => {
            svc.set(1, value);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcGetSystemInfo(id w1, handle w2, sub x3)` -> `x1`.
pub(super) fn get_system_info(svc: &mut SvcContext) -> Flow {
    let (id, handle, sub) = (svc.arg32(1), svc.arg32(2), svc.arg(3));
    let result = (|| {
        if handle != INVALID_HANDLE {
            return Err(result::INVALID_HANDLE);
        }
        let memory = &svc.kernel.memory;
        match id {
            0 | 1 => {
                let pool = Pool::from_raw(sub).ok_or(result::INVALID_COMBINATION)?;
                let size = memory.size(pool);
                Ok(if id == 0 {
                    size
                } else {
                    size - memory.free_size(pool)
                })
            }
            2 => match sub {
                0 => Ok(1),
                1 => Ok(8),
                _ => Err(result::INVALID_COMBINATION),
            },
            _ => Err(result::INVALID_ENUM_VALUE),
        }
    })();
    match result {
        Ok(value) => {
            svc.set(1, value);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

pub(super) enum LimitQuery {
    Limit,
    Current,
    Peak,
}

/// `svcGetResourceLimit{Limit,Current,Peak}Value(handle w1, which w2)` -> `x1`.
pub(super) fn get_resource_limit_value(svc: &mut SvcContext, query: LimitQuery) -> Flow {
    let result = (|| {
        let which = LimitableResource::from_raw(svc.arg32(2))?;
        let limit = svc.typed::<ResourceLimit>(svc.arg32(1))?;
        Ok(match query {
            LimitQuery::Limit => limit.limit(which),
            LimitQuery::Current => limit.current(which),
            LimitQuery::Peak => limit.peak(which),
        })
    })();
    match result {
        Ok(value) => {
            svc.set(1, value as u64);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}
