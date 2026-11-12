//! Address arbitration, mutexes and condition variables, without a global lock.
//!
//! Waiters live in 64 shards keyed by `(process, address)`. Every operation on one address takes
//! only that address's shard lock, so `WaitForAddress`/`SignalToAddress` pairs on different
//! addresses (the Switch 2 job system issues ~1,400 a frame, NX2_NOTES row 109) run on all cores
//! in parallel. Guest words are read and updated with host atomics on the mapped DRAM, which is
//! what the guest's own exclusive loads and stores observe.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicI64, AtomicU32, Ordering};

use lichii_common::{HleResult, ResultCode, VAddr};
use lichii_memory::{AddressSpace, MemoryPermission};
use parking_lot::{Mutex, MutexGuard};

use crate::process::ProcessId;
use crate::result;
use crate::thread::{Thread, WakeResult};

const SHARDS: usize = 64;
/// Set in a mutex word when threads wait for it.
pub const HANDLE_WAIT_MASK: u32 = 0x4000_0000;

struct Waiter {
    pid: ProcessId,
    key: VAddr,
    thread: Arc<Thread>,
    token: u64,
    priority: u8,
    /// Condition variables: the mutex to reacquire and the tag to write into it.
    mutex: VAddr,
    tag: u32,
}

impl Waiter {
    fn is_live(&self) -> bool {
        self.thread.is_waiting_on(self.token)
    }

    fn matches(&self, pid: ProcessId, key: VAddr) -> bool {
        self.pid == pid && self.key == key
    }
}

#[derive(Default)]
struct Shard {
    address: Vec<Waiter>,
    mutex: Vec<Waiter>,
    condvar: Vec<Waiter>,
}

/// Removes and returns the highest-priority live waiter on `(pid, key)` (first come first served
/// within a priority), dropping stale entries on the way.
fn take_best(list: &mut Vec<Waiter>, pid: ProcessId, key: VAddr) -> Option<Waiter> {
    list.retain(Waiter::is_live);
    let index = list
        .iter()
        .enumerate()
        .filter(|(_, w)| w.matches(pid, key))
        .min_by_key(|(i, w)| (w.priority, *i))
        .map(|(i, _)| i)?;
    Some(list.remove(index))
}

fn count_live(list: &[Waiter], pid: ProcessId, key: VAddr) -> usize {
    list.iter()
        .filter(|w| w.matches(pid, key) && w.is_live())
        .count()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArbitrationType {
    WaitIfLessThan,
    DecrementAndWaitIfLessThan,
    WaitIfEqual,
    WaitIfEqual64,
}

impl ArbitrationType {
    pub fn from_raw(raw: u32) -> HleResult<Self> {
        Ok(match raw {
            0 => Self::WaitIfLessThan,
            1 => Self::DecrementAndWaitIfLessThan,
            2 => Self::WaitIfEqual,
            3 => Self::WaitIfEqual64,
            _ => return Err(result::INVALID_ENUM_VALUE),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalType {
    Signal,
    SignalAndIncrementIfEqual,
    SignalAndModifyByWaitingCountIfEqual,
}

impl SignalType {
    pub fn from_raw(raw: u32) -> HleResult<Self> {
        Ok(match raw {
            0 => Self::Signal,
            1 => Self::SignalAndIncrementIfEqual,
            2 => Self::SignalAndModifyByWaitingCountIfEqual,
            _ => return Err(result::INVALID_ENUM_VALUE),
        })
    }
}

/// A guest word as a host atomic. The address must be aligned and mapped readable (and
/// writable when `write` is set).
fn guest_word<T>(memory: &AddressSpace, addr: VAddr, write: bool) -> HleResult<&T> {
    let (ptr, perm) = memory
        .translate(addr)
        .ok_or(result::INVALID_CURRENT_MEMORY)?;
    if !perm.contains(MemoryPermission::READ) || (write && !perm.contains(MemoryPermission::WRITE))
    {
        return Err(result::INVALID_CURRENT_MEMORY);
    }
    debug_assert_eq!(ptr as usize % std::mem::align_of::<T>(), 0);
    // SAFETY: the address is aligned for T (checked by every caller) and lies inside one mapped
    // DRAM page, which stays valid for the life of the address space.
    Ok(unsafe { &*(ptr as *const T) })
}

/// What a wait operation decided: either it is done (no block), or the thread was registered
/// with `token` and must now commit its wait.
pub(crate) enum WaitOutcome {
    Done(ResultCode),
    Block(u64),
}

pub(crate) struct Arbiter {
    shards: [Mutex<Shard>; SHARDS],
}

impl Default for Arbiter {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(Shard::default())),
        }
    }
}

impl Arbiter {
    fn index(pid: ProcessId, addr: VAddr) -> usize {
        let h = (addr ^ (pid << 40)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (h >> 58) as usize
    }

    fn shard(&self, pid: ProcessId, addr: VAddr) -> MutexGuard<'_, Shard> {
        self.shards[Self::index(pid, addr)].lock()
    }

    /// `svcWaitForAddress`.
    pub fn wait_for_address(
        &self,
        thread: &Arc<Thread>,
        addr: VAddr,
        kind: ArbitrationType,
        value: i64,
        timeout_zero: bool,
    ) -> HleResult<WaitOutcome> {
        let wide = kind == ArbitrationType::WaitIfEqual64;
        if !addr.is_multiple_of(if wide { 8 } else { 4 }) {
            return Err(result::INVALID_ADDRESS);
        }
        let process = thread.process();
        let memory = process.memory();
        let pid = process.id();
        let mut shard = self.shard(pid, addr);
        match kind {
            ArbitrationType::WaitIfLessThan => {
                let word: &AtomicI32 = guest_word(memory, addr, false)?;
                if word.load(Ordering::SeqCst) as i64 >= value {
                    return Ok(WaitOutcome::Done(result::INVALID_STATE));
                }
            }
            ArbitrationType::DecrementAndWaitIfLessThan => {
                let word: &AtomicI32 = guest_word(memory, addr, true)?;
                let mut current = word.load(Ordering::SeqCst);
                loop {
                    if current as i64 >= value {
                        return Ok(WaitOutcome::Done(result::INVALID_STATE));
                    }
                    match word.compare_exchange_weak(
                        current,
                        current.wrapping_sub(1),
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(actual) => current = actual,
                    }
                }
            }
            ArbitrationType::WaitIfEqual => {
                let word: &AtomicI32 = guest_word(memory, addr, false)?;
                if word.load(Ordering::SeqCst) != value as i32 {
                    return Ok(WaitOutcome::Done(result::INVALID_STATE));
                }
            }
            ArbitrationType::WaitIfEqual64 => {
                let word: &AtomicI64 = guest_word(memory, addr, false)?;
                if word.load(Ordering::SeqCst) != value {
                    return Ok(WaitOutcome::Done(result::INVALID_STATE));
                }
            }
        }
        if timeout_zero {
            return Ok(WaitOutcome::Done(result::TIMED_OUT));
        }
        let token = match thread.begin_wait(crate::thread::Resume::Result, true) {
            Ok(token) => token,
            Err(code) => return Ok(WaitOutcome::Done(code)),
        };
        shard.address.retain(Waiter::is_live);
        shard.address.push(Waiter {
            pid,
            key: addr,
            thread: thread.clone(),
            token,
            priority: thread.priority(),
            mutex: 0,
            tag: 0,
        });
        Ok(WaitOutcome::Block(token))
    }

    /// `svcSignalToAddress`.
    pub fn signal_to_address(
        &self,
        process: &crate::process::Process,
        addr: VAddr,
        kind: SignalType,
        value: i32,
        count: i32,
    ) -> HleResult<()> {
        if !addr.is_multiple_of(4) {
            return Err(result::INVALID_ADDRESS);
        }
        let pid = process.id();
        let mut shard = self.shard(pid, addr);
        match kind {
            SignalType::Signal => {}
            SignalType::SignalAndIncrementIfEqual => {
                let word: &AtomicI32 = guest_word(process.memory(), addr, true)?;
                if word
                    .compare_exchange(
                        value,
                        value.wrapping_add(1),
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                    .is_err()
                {
                    return Err(result::INVALID_STATE);
                }
            }
            SignalType::SignalAndModifyByWaitingCountIfEqual => {
                let waiting = count_live(&shard.address, pid, addr);
                let new_value = match (count <= 0, waiting) {
                    (_, 0) => value.wrapping_add(1),
                    (true, _) => value.wrapping_sub(2),
                    (false, n) if ((n - 1) as i64) < count as i64 => value.wrapping_sub(1),
                    (false, _) => value,
                };
                let word: &AtomicI32 = guest_word(process.memory(), addr, new_value != value)?;
                let ok = if new_value != value {
                    word.compare_exchange(value, new_value, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                } else {
                    word.load(Ordering::SeqCst) == value
                };
                if !ok {
                    return Err(result::INVALID_STATE);
                }
            }
        }
        let mut woken = 0;
        while count <= 0 || woken < count {
            let Some(waiter) = take_best(&mut shard.address, pid, addr) else {
                break;
            };
            if waiter.thread.wake(waiter.token, WakeResult::SUCCESS) {
                woken += 1;
            }
        }
        Ok(())
    }

    /// `svcArbitrateLock`: waits for the mutex at `addr` owned by `owner_handle`.
    pub fn arbitrate_lock(
        &self,
        thread: &Arc<Thread>,
        owner_handle: u32,
        addr: VAddr,
        tag: u32,
    ) -> HleResult<WaitOutcome> {
        if !addr.is_multiple_of(4) {
            return Err(result::INVALID_ADDRESS);
        }
        let process = thread.process();
        let pid = process.id();
        let mut shard = self.shard(pid, addr);
        let word: &AtomicU32 = guest_word(process.memory(), addr, false)?;
        if word.load(Ordering::SeqCst) != owner_handle | HANDLE_WAIT_MASK {
            return Ok(WaitOutcome::Done(ResultCode::SUCCESS));
        }
        process
            .handles()
            .read()
            .get_typed::<Thread>(owner_handle)
            .map_err(|_| result::INVALID_HANDLE)?;
        let token = match thread.begin_wait(crate::thread::Resume::Result, false) {
            Ok(token) => token,
            Err(code) => return Ok(WaitOutcome::Done(code)),
        };
        shard.mutex.retain(Waiter::is_live);
        shard.mutex.push(Waiter {
            pid,
            key: addr,
            thread: thread.clone(),
            token,
            priority: thread.priority(),
            mutex: addr,
            tag,
        });
        Ok(WaitOutcome::Block(token))
    }

    /// Hands the mutex at `addr` to its best waiter, or clears it.
    fn release_mutex(
        shard: &mut Shard,
        memory: &AddressSpace,
        pid: ProcessId,
        addr: VAddr,
    ) -> HleResult<()> {
        let word: &AtomicU32 = guest_word(memory, addr, true)?;
        let Some(next) = take_best(&mut shard.mutex, pid, addr) else {
            word.store(0, Ordering::SeqCst);
            return Ok(());
        };
        let more = count_live(&shard.mutex, pid, addr) > 0;
        word.store(
            next.tag | if more { HANDLE_WAIT_MASK } else { 0 },
            Ordering::SeqCst,
        );
        next.thread.wake(next.token, WakeResult::SUCCESS);
        Ok(())
    }

    /// `svcArbitrateUnlock`.
    pub fn arbitrate_unlock(
        &self,
        process: &crate::process::Process,
        addr: VAddr,
    ) -> HleResult<()> {
        if !addr.is_multiple_of(4) {
            return Err(result::INVALID_ADDRESS);
        }
        let pid = process.id();
        let mut shard = self.shard(pid, addr);
        Self::release_mutex(&mut shard, process.memory(), pid, addr)
    }

    /// `svcWaitProcessWideKeyAtomic`: releases the mutex at `addr` and waits on `key`.
    pub fn wait_condvar(
        &self,
        thread: &Arc<Thread>,
        addr: VAddr,
        key: VAddr,
        tag: u32,
        timeout_zero: bool,
    ) -> HleResult<WaitOutcome> {
        if !addr.is_multiple_of(4) || !key.is_multiple_of(4) {
            return Err(result::INVALID_ADDRESS);
        }
        let process = thread.process();
        let pid = process.id();
        let memory = process.memory();
        let token = match thread.begin_wait(crate::thread::Resume::Result, false) {
            Ok(token) => token,
            Err(code) => return Ok(WaitOutcome::Done(code)),
        };
        let (a, b) = (Self::index(pid, addr), Self::index(pid, key));
        let (first, second) = (a.min(b), a.max(b));
        let mut lock_first = self.shards[first].lock();
        let mut lock_second = (first != second).then(|| self.shards[second].lock());
        let (mutex_shard, key_shard) = if a == b {
            (&mut *lock_first, None)
        } else if a < b {
            (&mut *lock_first, lock_second.as_deref_mut())
        } else {
            (
                lock_second.as_deref_mut().expect("second shard"),
                Some(&mut *lock_first),
            )
        };
        if let Err(code) = Self::release_mutex(mutex_shard, memory, pid, addr) {
            thread.cancel_wait(token);
            return Err(code);
        }
        if timeout_zero {
            thread.cancel_wait(token);
            return Ok(WaitOutcome::Done(result::TIMED_OUT));
        }
        let key_word: &AtomicU32 = match guest_word(memory, key, true) {
            Ok(word) => word,
            Err(code) => {
                thread.cancel_wait(token);
                return Err(code);
            }
        };
        key_word.store(1, Ordering::SeqCst);
        let shard = key_shard.unwrap_or(mutex_shard);
        shard.condvar.retain(Waiter::is_live);
        shard.condvar.push(Waiter {
            pid,
            key,
            thread: thread.clone(),
            token,
            priority: thread.priority(),
            mutex: addr,
            tag,
        });
        Ok(WaitOutcome::Block(token))
    }

    /// `svcSignalProcessWideKey`: moves up to `count` waiters of `key` (all when `count <= 0`)
    /// to their mutexes, waking those that get it at once.
    pub fn signal_condvar(&self, process: &crate::process::Process, key: VAddr, count: i32) {
        let pid = process.id();
        let memory = process.memory();
        let mut moved = Vec::new();
        {
            let mut shard = self.shard(pid, key);
            while count <= 0 || (moved.len() as i32) < count {
                let Some(waiter) = take_best(&mut shard.condvar, pid, key) else {
                    break;
                };
                moved.push(waiter);
            }
            if count_live(&shard.condvar, pid, key) == 0
                && let Ok(word) = guest_word::<AtomicU32>(memory, key, true)
            {
                word.store(0, Ordering::SeqCst);
            }
        }
        for waiter in moved {
            let mut shard = self.shard(pid, waiter.mutex);
            let mut s = waiter.thread.sched.lock();
            if s.wait != waiter.token {
                continue;
            }
            let word: &AtomicU32 = match guest_word(memory, waiter.mutex, true) {
                Ok(word) => word,
                Err(code) => {
                    waiter.thread.wake_locked(&mut s, WakeResult::error(code));
                    continue;
                }
            };
            let mut current = word.load(Ordering::SeqCst);
            let acquired = loop {
                let (new, acquired) = if current == 0 {
                    (waiter.tag, true)
                } else {
                    (current | HANDLE_WAIT_MASK, false)
                };
                match word.compare_exchange_weak(current, new, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break acquired,
                    Err(actual) => current = actual,
                }
            };
            if acquired {
                waiter.thread.wake_locked(&mut s, WakeResult::SUCCESS);
            } else {
                drop(s);
                shard.mutex.retain(Waiter::is_live);
                shard.mutex.push(Waiter {
                    key: waiter.mutex,
                    ..waiter
                });
            }
        }
    }
}
