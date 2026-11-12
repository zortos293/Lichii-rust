//! Resource limits (`KResourceLimit`).

use std::sync::atomic::{AtomicI64, Ordering};

use lichii_common::{HleResult, ResultCode};

use crate::result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitableResource {
    PhysicalMemory = 0,
    Threads = 1,
    Events = 2,
    TransferMemories = 3,
    Sessions = 4,
}

impl LimitableResource {
    pub fn from_raw(raw: u32) -> HleResult<Self> {
        Ok(match raw {
            0 => Self::PhysicalMemory,
            1 => Self::Threads,
            2 => Self::Events,
            3 => Self::TransferMemories,
            4 => Self::Sessions,
            _ => return Err(result::INVALID_ENUM_VALUE),
        })
    }
}

struct Counter {
    limit: AtomicI64,
    current: AtomicI64,
    peak: AtomicI64,
}

/// Limits and live counts for the five limitable resources.
pub struct ResourceLimit {
    counters: [Counter; 5],
}

impl ResourceLimit {
    pub fn new(limits: [i64; 5]) -> Self {
        Self {
            counters: limits.map(|limit| Counter {
                limit: AtomicI64::new(limit),
                current: AtomicI64::new(0),
                peak: AtomicI64::new(0),
            }),
        }
    }

    /// The application limits Horizon sets: the pool size, 608 threads, 700 events, 128
    /// transfer memories and 894 sessions.
    pub fn application(memory: u64) -> Self {
        Self::new([memory as i64, 608, 700, 128, 894])
    }

    pub fn limit(&self, which: LimitableResource) -> i64 {
        self.counters[which as usize].limit.load(Ordering::Acquire)
    }

    pub fn current(&self, which: LimitableResource) -> i64 {
        self.counters[which as usize]
            .current
            .load(Ordering::Acquire)
    }

    pub fn peak(&self, which: LimitableResource) -> i64 {
        self.counters[which as usize].peak.load(Ordering::Acquire)
    }

    pub fn set_limit(&self, which: LimitableResource, value: i64) -> HleResult<()> {
        let c = &self.counters[which as usize];
        if value < c.current.load(Ordering::Acquire) {
            return Err(result::INVALID_STATE);
        }
        c.limit.store(value, Ordering::Release);
        Ok(())
    }

    /// Reserves `amount` or fails with `LimitReached`.
    pub fn reserve(&self, which: LimitableResource, amount: i64) -> Result<(), ResultCode> {
        let c = &self.counters[which as usize];
        let limit = c.limit.load(Ordering::Acquire);
        let mut current = c.current.load(Ordering::Acquire);
        loop {
            if current + amount > limit {
                return Err(result::LIMIT_REACHED);
            }
            match c.current.compare_exchange_weak(
                current,
                current + amount,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        c.peak.fetch_max(current + amount, Ordering::AcqRel);
        Ok(())
    }

    pub fn release(&self, which: LimitableResource, amount: i64) {
        self.counters[which as usize]
            .current
            .fetch_sub(amount, Ordering::AcqRel);
    }
}
