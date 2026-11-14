//! Translated code, shared by every core that runs one process, and the fault-site tables the
//! fastmem fault handler searches.
//!
//! A process's cores share one [`Shared`]: code memory, the blocks in it, their fastmem sites and
//! the links between them, so a block translated on one core runs on every other. Each core keeps
//! its own [`JitCache`] in front of it: a lookup table, the indirect-branch table translated code
//! probes, and the blocks it has seen, all written only by that core's thread. Readers of the
//! shared state take no lock: blocks are published with atomics. Translating, linking and
//! invalidating take the shared writer lock, which the dispatch hot path (a hit in the core's own
//! table, or a linked jump) never touches.
//!
//! Code memory is never reused while a core may run it. When it fills, or on a full clear, the
//! current [`Generation`] retires and a fresh one replaces it; the old one is freed once every
//! core has left the dispatcher since then.

use std::cell::UnsafeCell;
use std::collections::{BTreeMap, HashMap};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use lichii_common::{PAGE_SIZE, VAddr};
use lichii_memory::{AddressSpace, HostFault};

use super::x64::Compiled;
use super::{JitState, x64};
use crate::ThreadContext;
use crate::decoder::Inst;

const CODE_SIZE: usize = 128 << 20;
const FAST_BITS: u32 = 14;
const EMPTY: (u64, u32) = (u64::MAX, u32::MAX);
const REGIONS: usize = 256;
/// Blocks one generation holds; its lookup table has twice as many slots.
const BLOCKS: usize = 1 << 18;
const LOOKUP_BITS: u32 = 19;
/// Fastmem sites one generation holds.
const SITES: usize = 1 << 21;

pub(super) type Trampoline =
    unsafe extern "sysv64" fn(*mut ThreadContext, *mut JitState, *mut u8, *const u8) -> u64;

/// How the dispatcher runs the guest code at a block's start.
#[derive(Clone, Copy)]
pub(super) enum Code {
    /// Compiled host code.
    Native(*const u8),
    /// A lone instruction the translator rejects, for the interpreter.
    Interpret(Inst),
}

// SAFETY: `Native` points into code memory that stays mapped while any core can reach the entry.
unsafe impl Send for Code {}
// SAFETY: as above; the pointer is only read.
unsafe impl Sync for Code {}

/// One translated guest block.
#[derive(Clone, Copy)]
pub(super) struct Entry {
    pub code: Code,
    /// Guest bytes covered.
    len: u64,
}

/// Code memory for one generation: one shared-memory object mapped twice, so blocks run from a
/// read-execute view and are written and patched through a read-write one, never through
/// `mprotect`.
struct CodeMemory {
    exec: *mut u8,
    write: *mut u8,
}

impl CodeMemory {
    fn new() -> Option<Self> {
        let (write, exec) = sys::map(CODE_SIZE)?;
        Some(Self { exec, write })
    }

    /// Copies `code` in at the first 16-byte boundary at or after `*used`.
    ///
    /// # Safety
    /// `*used` must be past every byte any core may run, so nothing reads what this writes.
    unsafe fn push(&self, used: &mut usize, code: &[u8]) -> Option<*const u8> {
        let start = used.next_multiple_of(16);
        if start + code.len() > CODE_SIZE {
            return None;
        }
        // SAFETY: `[start, start + len)` is inside the mapping and, per the caller, unused.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), self.write.add(start), code.len()) };
        *used = start + code.len();
        // SAFETY: as above.
        Some(unsafe { self.exec.add(start) })
    }

    /// Points the `jmp rel32` at executable address `site` to `target`. The offset is 4-byte
    /// aligned, so the store is atomic and a core running the code sees the old jump or the new.
    ///
    /// # Safety
    /// `site` must be a link site of a block in this memory.
    unsafe fn patch(&self, site: usize, target: usize) {
        let rel = target.wrapping_sub(site + 5) as i32 as u32;
        // SAFETY: guaranteed by the caller; the operand follows the one-byte opcode, at the same
        // offset in the writable view, and `x64` aligns it to 4 bytes.
        unsafe {
            let operand = self.write.add(site + 1 - self.exec as usize);
            debug_assert_eq!(operand as usize % 4, 0);
            (*operand.cast::<AtomicU32>()).store(rel, Ordering::Release);
        }
    }
}

impl Drop for CodeMemory {
    fn drop(&mut self) {
        let exec = self.exec as usize;
        super::map::clear(exec, exec + CODE_SIZE);
        // SAFETY: both views came from one `sys::map(CODE_SIZE)`, and no core runs this
        // generation's code any more.
        unsafe {
            sys::unmap(self.exec, CODE_SIZE);
            sys::unmap(self.write, CODE_SIZE);
        }
    }
}

/// Host addresses of a generation's fastmem accesses and their slow paths, sorted by access
/// because code is only ever appended. Entries below `len` are final.
struct SiteTable {
    sites: Table<(AtomicUsize, AtomicUsize)>,
    len: AtomicUsize,
}

impl SiteTable {
    fn find(&self, pc: usize) -> Option<usize> {
        let sites = &self.sites[..self.len.load(Ordering::Acquire)];
        let i = sites
            .binary_search_by_key(&pc, |(site, _)| site.load(Ordering::Relaxed))
            .ok()?;
        Some(sites[i].1.load(Ordering::Relaxed))
    }
}

/// A process-wide slot naming one generation's code range and site table.
struct Region {
    start: AtomicUsize,
    end: AtomicUsize,
    sites: AtomicPtr<SiteTable>,
}

static REGION_TABLE: [Region; REGIONS] = [const {
    Region {
        start: AtomicUsize::new(0),
        end: AtomicUsize::new(0),
        sites: AtomicPtr::new(std::ptr::null_mut()),
    }
}; REGIONS];
static REGION_LOCK: Mutex<()> = Mutex::new(());

/// The fastmem fault handler shared by every core: moves a faulting access to its slow path.
pub(super) fn on_fault(_data: usize, _guest_addr: u64, fault: &mut HostFault) -> bool {
    let pc = fault.pc() as usize;
    for region in &REGION_TABLE {
        let start = region.start.load(Ordering::Acquire);
        if start == 0 || pc < start || pc >= region.end.load(Ordering::Acquire) {
            continue;
        }
        // SAFETY: a fault at `pc` comes from code of this region's generation, which is not
        // freed while a core can run it, and its slot is cleared before the table is freed.
        let sites = unsafe { &*region.sites.load(Ordering::Acquire) };
        return match sites.find(pc) {
            Some(stub) => {
                fault.set_pc(stub as u64);
                true
            }
            None => false,
        };
    }
    false
}

/// One code memory and everything that points into it. Readers find blocks through `lookup`
/// without a lock; only the writer lock's holder adds to it.
struct Generation {
    id: u64,
    memory: CodeMemory,
    trampoline: Trampoline,
    /// The code empty indirect-table slots point at.
    miss: usize,
    materialize: usize,
    /// Blocks by index; each is written once, before its index is published.
    blocks: Table<UnsafeCell<MaybeUninit<Entry>>>,
    /// Open addressing on the guest start: the key (`pc | 1`, so 0 is empty), then the block
    /// index plus one, or 0 when the block was invalidated. Keys are never removed, so a probe
    /// stops at the first empty key.
    lookup: Table<(AtomicU64, AtomicU32)>,
    sites: Box<SiteTable>,
    region: Option<usize>,
}

/// Types for which all zero bytes are a valid value.
///
/// # Safety
/// Implementors must be valid when zeroed.
unsafe trait Zeroable {}
// SAFETY: atomics of integers are plain integers.
unsafe impl Zeroable for (AtomicU64, AtomicU32) {}
// SAFETY: as above.
unsafe impl Zeroable for (AtomicUsize, AtomicUsize) {}
// SAFETY: uninitialised memory is a valid `MaybeUninit`.
unsafe impl Zeroable for UnsafeCell<MaybeUninit<Entry>> {}

/// A fixed-size array in anonymous memory the OS zeroes and maps on first touch, so a big table
/// costs only the pages it uses.
struct Table<T: Zeroable> {
    ptr: *mut T,
    len: usize,
}

impl<T: Zeroable> Table<T> {
    fn new(len: usize) -> Option<Self> {
        let ptr = sys::anonymous(len * size_of::<T>())?.cast();
        Some(Self { ptr, len })
    }
}

impl<T: Zeroable> std::ops::Deref for Table<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` holds `len` zero-initialised, and so valid, values for this table's life.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Zeroable> Drop for Table<T> {
    fn drop(&mut self) {
        // SAFETY: the mapping came from `sys::anonymous` with this size, and is unused now.
        unsafe { sys::release(self.ptr.cast(), self.len * size_of::<T>()) };
    }
}

fn lookup_slot(pc: u64) -> usize {
    ((pc >> 2).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - LOOKUP_BITS)) as usize
}

impl Generation {
    /// A fresh generation and the code bytes its stubs use. `None` when the host cannot map
    /// more code or every fault-table slot is taken, so all code can use fastmem.
    fn new(id: u64, profile: bool) -> Option<(Box<Self>, usize)> {
        let memory = CodeMemory::new()?;
        let mut used = 0;
        // SAFETY: the memory is new, so no core runs any of it.
        let (trampoline, miss, materialize) = unsafe {
            (
                memory.push(&mut used, &x64::trampoline())?,
                memory.push(&mut used, &x64::miss_stub())? as usize,
                memory.push(&mut used, &x64::materialize_stub(profile))? as usize,
            )
        };
        // SAFETY: the bytes at `trampoline` are the trampoline, which has this signature.
        let trampoline = unsafe { std::mem::transmute::<*const u8, Trampoline>(trampoline) };
        let mut generation = Box::new(Self {
            id,
            trampoline,
            miss,
            materialize,
            blocks: Table::new(BLOCKS)?,
            lookup: Table::new(1 << LOOKUP_BITS)?,
            sites: Box::new(SiteTable {
                sites: Table::new(SITES)?,
                len: AtomicUsize::new(0),
            }),
            region: None,
            memory,
        });
        let _guard = REGION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let slot = REGION_TABLE
            .iter()
            .position(|r| r.start.load(Ordering::Acquire) == 0)?;
        let region = &REGION_TABLE[slot];
        let exec = generation.memory.exec as usize;
        region
            .sites
            .store(&mut *generation.sites, Ordering::Release);
        region.end.store(exec + CODE_SIZE, Ordering::Release);
        region.start.store(exec, Ordering::Release);
        generation.region = Some(slot);
        Some((generation, used))
    }

    /// The lookup slot holding `pc`'s key, or the empty slot where it would go.
    fn slot(&self, pc: u64) -> usize {
        let mask = (1 << LOOKUP_BITS) - 1;
        let mut slot = lookup_slot(pc);
        loop {
            match self.lookup[slot].0.load(Ordering::Acquire) {
                0 => return slot,
                k if k == pc | 1 => return slot,
                _ => slot = (slot + 1) & mask,
            }
        }
    }

    fn find(&self, pc: u64) -> Option<Entry> {
        let (key, index) = &self.lookup[self.slot(pc)];
        if key.load(Ordering::Acquire) != pc | 1 {
            return None;
        }
        match index.load(Ordering::Acquire) {
            0 => None,
            index => Some(self.block(index - 1)),
        }
    }

    fn block(&self, index: u32) -> Entry {
        // SAFETY: an index is only published, or kept by the writer, after its block is written,
        // and a block is never written again.
        unsafe { (*self.blocks[index as usize].get()).assume_init() }
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        if let Some(slot) = self.region {
            let _guard = REGION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            REGION_TABLE[slot].start.store(0, Ordering::Release);
        }
    }
}

/// What only the writer lock's holder touches.
struct Writer {
    used: usize,
    blocks: usize,
    /// Distinct keys in the current generation's lookup table.
    keys: usize,
    by_start: BTreeMap<u64, u32>,
    /// Host addresses of the linked `jmp`s into each guest block.
    links: HashMap<u64, Vec<usize>>,
    /// Generations no core may enter again, with the epoch they retired at.
    retired: Vec<(Box<Generation>, u64)>,
    /// Each core's dispatcher state: 0 outside it, else one more than the epoch it entered at.
    cores: Vec<Weak<AtomicU64>>,
}

/// The translated code every core of one process shares.
pub(super) struct Shared {
    writer: Mutex<Writer>,
    /// The generation cores enter; owned by the writer, freed only after it retires.
    current: AtomicPtr<Generation>,
    /// Bumped by every retirement.
    epoch: AtomicU64,
    profile: bool,
}

// SAFETY: the raw pointers name code memory and generations whose lifetime the epoch scheme
// guards; everything else is atomics or behind the writer lock.
unsafe impl Send for Shared {}
// SAFETY: as above.
unsafe impl Sync for Shared {}

type Registry = Vec<(Weak<AddressSpace>, u64, Weak<Shared>)>;
static SHARED: Mutex<Registry> = Mutex::new(Vec::new());

impl Shared {
    /// The code shared by every core of `memory` whose blocks compile like `flavour`.
    pub fn of(memory: &Arc<AddressSpace>, flavour: u64, profile: bool) -> Option<Arc<Self>> {
        let mut registry = SHARED.lock().unwrap_or_else(|e| e.into_inner());
        registry
            .retain(|(memory, _, shared)| memory.strong_count() > 0 && shared.strong_count() > 0);
        let found = registry
            .iter()
            .find(|(m, f, _)| *f == flavour && std::ptr::eq(m.as_ptr(), Arc::as_ptr(memory)))
            .and_then(|(_, _, shared)| shared.upgrade());
        if found.is_some() {
            return found;
        }
        let (generation, used) = Generation::new(1, profile)?;
        let shared = Arc::new(Self {
            writer: Mutex::new(Writer {
                used,
                blocks: 0,
                keys: 0,
                by_start: BTreeMap::new(),
                links: HashMap::new(),
                retired: Vec::new(),
                cores: Vec::new(),
            }),
            current: AtomicPtr::new(Box::into_raw(generation)),
            epoch: AtomicU64::new(0),
            profile,
        });
        registry.push((Arc::downgrade(memory), flavour, Arc::downgrade(&shared)));
        Some(shared)
    }

    fn lock(&self) -> MutexGuard<'_, Writer> {
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The generation a core in the dispatcher may use.
    fn current(&self) -> &Generation {
        // SAFETY: a generation is freed only once every core has left the dispatcher since it
        // retired, and callers are in the dispatcher or hold the writer lock.
        unsafe { &*self.current.load(Ordering::SeqCst) }
    }

    /// Counts `core` among the cores retired code waits for, and returns the current
    /// generation's id, trampoline, materialize stub and miss stub.
    fn register(&self, core: &Arc<AtomicU64>) -> (u64, Trampoline, usize, usize) {
        let mut w = self.lock();
        w.cores.retain(|c| c.strong_count() > 0);
        w.cores.push(Arc::downgrade(core));
        let g = self.current();
        (g.id, g.trampoline, g.materialize, g.miss)
    }

    /// Adds the block at `start` to the current generation, unless another core added it first,
    /// and returns the block's entry and the generation it lives in.
    fn insert(&self, start: u64, len: u64, body: Result<Compiled, Inst>) -> (Entry, u64) {
        let mut w = self.lock();
        let generation = self.current();
        if let Some(&index) = w.by_start.get(&start) {
            return (generation.block(index), generation.id);
        }
        let (bytes, sites) = body
            .as_ref()
            .map_or((0, 0), |c| (c.code.len(), c.sites.len()));
        let full = w.blocks == BLOCKS
            || 2 * (w.keys + 1) > 1 << LOOKUP_BITS
            || generation.sites.len.load(Ordering::Relaxed) + sites > SITES
            || w.used.next_multiple_of(16) + bytes > CODE_SIZE;
        if full {
            self.retire(&mut w);
        }
        let generation = self.current();
        let code = match body {
            Err(inst) => Code::Interpret(inst),
            Ok(compiled) => {
                // SAFETY: `used` is past every block of this generation.
                let code = unsafe { generation.memory.push(&mut w.used, &compiled.code) }
                    .expect("one block fits in an empty code cache");
                let base = code as usize;
                let table = &generation.sites;
                let len = table.len.load(Ordering::Relaxed);
                for (i, &(site, stub)) in compiled.sites.iter().enumerate() {
                    let (s, t) = &table.sites[len + i];
                    s.store(base + site as usize, Ordering::Relaxed);
                    t.store(base + stub as usize, Ordering::Relaxed);
                }
                table
                    .len
                    .store(len + compiled.sites.len(), Ordering::Release);
                Code::Native(code)
            }
        };
        let entry = Entry { code, len };
        let index = w.blocks as u32;
        w.blocks += 1;
        // SAFETY: `index` is new, so no reader can see this block yet.
        unsafe { (*generation.blocks[index as usize].get()).write(entry) };
        let (key, value) = &generation.lookup[generation.slot(start)];
        value.store(index + 1, Ordering::Release);
        if key.load(Ordering::Relaxed) != start | 1 {
            key.store(start | 1, Ordering::Release);
            w.keys += 1;
        }
        w.by_start.insert(start, index);
        (entry, generation.id)
    }

    /// Replaces the current generation with a fresh one and frees the retired ones no core can
    /// still be running.
    fn retire(&self, w: &mut Writer) {
        let id = self.current().id + 1;
        let (fresh, used) =
            Generation::new(id, self.profile).expect("the host cannot map more JIT code");
        let old = self.current.swap(Box::into_raw(fresh), Ordering::SeqCst);
        let epoch = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        // SAFETY: `old` came from `Box::into_raw` and is no longer current.
        w.retired.push((unsafe { Box::from_raw(old) }, epoch));
        w.used = used;
        w.blocks = 0;
        w.keys = 0;
        w.by_start.clear();
        w.links.clear();
        self.collect(w);
    }

    /// Frees the retired generations every core has left the dispatcher since.
    fn collect(&self, w: &mut Writer) {
        if w.retired.is_empty() {
            return;
        }
        w.cores.retain(|c| c.strong_count() > 0);
        let states: Vec<u64> = w
            .cores
            .iter()
            .filter_map(Weak::upgrade)
            .map(|c| c.load(Ordering::SeqCst))
            .collect();
        w.retired
            .retain(|&(_, epoch)| states.iter().any(|&s| s != 0 && s <= epoch));
    }

    /// Drops every block that overlaps `[addr, addr + size)` and unlinks the jumps into them.
    /// Their code stays in memory, unused, until the generation is freed.
    fn invalidate(&self, addr: VAddr, size: u64) {
        let end = addr.saturating_add(size);
        let first = addr.saturating_sub(PAGE_SIZE);
        let mut w = self.lock();
        let generation = self.current();
        let doomed: Vec<u64> = w
            .by_start
            .range(first..end)
            .filter(|&(&start, &index)| start + generation.block(index).len > addr)
            .map(|(&start, _)| start)
            .collect();
        for start in doomed {
            for site in w.links.remove(&start).unwrap_or_default() {
                // SAFETY: link sites stay in code memory for the generation's lifetime.
                unsafe { generation.memory.patch(site, site + 5) };
            }
            generation.lookup[generation.slot(start)]
                .1
                .store(0, Ordering::Release);
            w.by_start.remove(&start);
        }
        self.collect(&mut w);
    }

    /// Points the link `site`, recorded by a block exit of generation `id`, at `code`, the
    /// block at guest `target`, if both are still current.
    fn link(&self, id: u64, site: usize, target: u64, code: *const u8) {
        let mut w = self.lock();
        let generation = self.current();
        if generation.id != id {
            return;
        }
        let Some(&index) = w.by_start.get(&target) else {
            return;
        };
        if !matches!(generation.block(index).code, Code::Native(c) if c == code) {
            return;
        }
        // SAFETY: the site belongs to code of the current generation.
        unsafe { generation.memory.patch(site, code as usize) };
        w.links.entry(target).or_default().push(site);
    }

    fn clear(&self) {
        let mut w = self.lock();
        self.retire(&mut w);
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: no core is left, so nothing runs or reads the current generation.
        drop(unsafe { Box::from_raw(*self.current.get_mut()) });
    }
}

/// One core's view of the shared code: the blocks it has used, the indirect-branch table its
/// code probes, and the generation they belong to. Only the core's thread touches it.
pub(super) struct JitCache {
    shared: Arc<Shared>,
    /// 0 outside the dispatcher, else one more than the epoch the core entered at.
    state: Arc<AtomicU64>,
    generation: u64,
    trampoline: Trampoline,
    materialize: usize,
    entries: Vec<Entry>,
    free: Vec<u32>,
    by_start: BTreeMap<u64, u32>,
    fast: Box<[(u64, u32)]>,
    /// The table indirect branches probe, indexed like `fast` but with `TABLE_BITS` bits.
    table: Box<[(u64, usize)]>,
    /// The value of an empty table slot: a key and the miss stub.
    empty: (u64, usize),
    /// Blocks this core took from the shared code instead of translating them.
    adopted: u64,
}

fn table_slot(pc: u64) -> usize {
    ((pc >> 2) as usize) & ((1 << x64::TABLE_BITS) - 1)
}

#[inline]
fn fast_slot(pc: u64) -> usize {
    ((pc >> 2) as usize) & ((1 << FAST_BITS) - 1)
}

impl JitCache {
    pub fn new(shared: Arc<Shared>) -> Self {
        let state = Arc::new(AtomicU64::new(0));
        let (generation, trampoline, materialize, miss) = shared.register(&state);
        let empty = (u64::MAX, miss);
        Self {
            generation,
            trampoline,
            materialize,
            empty,
            table: vec![empty; 1 << x64::TABLE_BITS].into_boxed_slice(),
            entries: Vec::new(),
            free: Vec::new(),
            by_start: BTreeMap::new(),
            fast: vec![EMPTY; 1 << FAST_BITS].into_boxed_slice(),
            adopted: 0,
            state,
            shared,
        }
    }

    /// Marks the core as in the dispatcher and moves it to the current generation if that
    /// changed.
    pub fn enter(&mut self) {
        let epoch = self.shared.epoch.load(Ordering::SeqCst);
        self.state.store(epoch + 1, Ordering::SeqCst);
        self.sync();
    }

    /// Marks the core as out of the dispatcher, so retired code may be freed.
    pub fn leave(&mut self) {
        self.state.store(0, Ordering::SeqCst);
    }

    /// Forgets every block of an older generation. Returns whether the generation changed.
    fn sync(&mut self) -> bool {
        let generation = self.shared.current();
        if generation.id == self.generation {
            return false;
        }
        self.generation = generation.id;
        self.trampoline = generation.trampoline;
        self.materialize = generation.materialize;
        self.empty = (u64::MAX, generation.miss);
        self.entries.clear();
        self.free.clear();
        self.by_start.clear();
        self.fast.fill(EMPTY);
        self.table.fill(self.empty);
        true
    }

    pub fn trampoline(&self) -> Trampoline {
        self.trampoline
    }

    /// The address of `x64::materialize_stub` in the current generation.
    pub fn materialize(&self) -> usize {
        self.materialize
    }

    pub fn table(&self) -> *const (u64, usize) {
        self.table.as_ptr()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Blocks this core took from the shared code instead of translating them.
    pub fn adopted(&self) -> u64 {
        self.adopted
    }

    /// Blocks in the shared code's current generation.
    #[cfg(test)]
    pub fn shared_blocks(&self) -> usize {
        self.shared.lock().blocks
    }

    /// Makes `code`, the block at guest `pc`, reachable from this core's indirect branches.
    #[inline]
    pub fn publish(&mut self, pc: u64, code: *const u8) {
        self.table[table_slot(pc)] = (pc, code as usize);
    }

    /// Points the link `site`, recorded by a block exit in this generation, at `code`, the block
    /// at guest `target`.
    pub fn link(&mut self, site: usize, target: u64, code: *const u8) {
        self.shared.link(self.generation, site, target, code);
    }

    /// The block at `pc`: from this core's table, else from the shared code. Only in the
    /// dispatcher.
    #[inline]
    pub fn lookup(&mut self, pc: u64) -> Option<Entry> {
        let slot = fast_slot(pc);
        let (start, index) = self.fast[slot];
        if start == pc && index != EMPTY.1 {
            return Some(self.entries[index as usize]);
        }
        if let Some(&index) = self.by_start.get(&pc) {
            self.fast[slot] = (pc, index);
            return Some(self.entries[index as usize]);
        }
        if self.sync() {
            return None;
        }
        let entry = self.shared.current().find(pc)?;
        self.adopted += 1;
        self.remember(pc, entry);
        Some(entry)
    }

    /// Adds a block covering guest `[start, start + len)` to the shared code, or takes the one
    /// another core added first.
    pub fn insert(&mut self, start: u64, len: u64, body: Result<Compiled, Inst>) -> Entry {
        let (entry, generation) = self.shared.insert(start, len, body);
        if generation != self.generation {
            self.sync();
        }
        self.remember(start, entry);
        entry
    }

    fn remember(&mut self, start: u64, entry: Entry) {
        let index = match self.free.pop() {
            Some(index) => {
                self.entries[index as usize] = entry;
                index
            }
            None => {
                self.entries.push(entry);
                (self.entries.len() - 1) as u32
            }
        };
        if let Some(old) = self.by_start.insert(start, index) {
            self.free.push(old);
        }
        self.fast[fast_slot(start)] = (start, index);
    }

    /// Drops every block that overlaps `[addr, addr + size)`, here and in the shared code.
    pub fn invalidate(&mut self, addr: VAddr, size: u64) {
        if size == 0 {
            return;
        }
        let end = addr.saturating_add(size);
        let first = addr.saturating_sub(PAGE_SIZE);
        let doomed: Vec<(u64, u32)> = self
            .by_start
            .range(first..end)
            .filter(|&(&start, &index)| start + self.entries[index as usize].len > addr)
            .map(|(&start, &index)| (start, index))
            .collect();
        for (start, index) in doomed {
            let slot = table_slot(start);
            if self.table[slot].0 == start {
                self.table[slot] = self.empty;
            }
            self.by_start.remove(&start);
            let slot = fast_slot(start);
            if self.fast[slot].0 == start {
                self.fast[slot] = EMPTY;
            }
            self.free.push(index);
        }
        self.shared.invalidate(addr, size);
    }

    /// Discards all translated code of every core: the shared code moves to a fresh generation,
    /// which each core picks up the next time it enters the dispatcher or misses.
    pub fn clear(&mut self) {
        self.shared.clear();
    }
}

#[cfg(unix)]
mod sys {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn create_fd() -> libc::c_int {
        // SAFETY: plain system call with a NUL-terminated name.
        unsafe { libc::memfd_create(c"lichii-jit".as_ptr(), libc::MFD_CLOEXEC) }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn create_fd() -> libc::c_int {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = format!("/lichii-jit-{}-{id}\0", std::process::id());
        // SAFETY: the name is NUL-terminated; the object is unlinked right away so it lives only
        // as long as the descriptor.
        unsafe {
            let fd = libc::shm_open(
                name.as_ptr().cast(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600,
            );
            if fd >= 0 {
                libc::shm_unlink(name.as_ptr().cast());
            }
            fd
        }
    }

    /// Maps one shared-memory object of `size` bytes twice: read-write, then read-execute.
    pub fn map(size: usize) -> Option<(*mut u8, *mut u8)> {
        let fd = create_fd();
        if fd < 0 {
            return None;
        }
        let view = |prot| {
            // SAFETY: a fresh shared mapping of a descriptor this function owns.
            let ptr =
                unsafe { libc::mmap(std::ptr::null_mut(), size, prot, libc::MAP_SHARED, fd, 0) };
            (ptr != libc::MAP_FAILED).then_some(ptr.cast::<u8>())
        };
        // SAFETY: `fd` is open and owned here.
        let sized = unsafe { libc::ftruncate(fd, size as libc::off_t) } == 0;
        let write = sized
            .then(|| view(libc::PROT_READ | libc::PROT_WRITE))
            .flatten();
        let exec = write.and_then(|_| view(libc::PROT_READ | libc::PROT_EXEC));
        // SAFETY: the mappings keep the object alive without the descriptor.
        unsafe { libc::close(fd) };
        match (write, exec) {
            (Some(write), Some(exec)) => Some((write, exec)),
            (Some(write), None) => {
                // SAFETY: `write` is the mapping made above.
                unsafe { unmap(write, size) };
                None
            }
            _ => None,
        }
    }

    /// # Safety
    /// `ptr` and `size` must name one view from `map`, and no code in it may run again.
    pub unsafe fn unmap(ptr: *mut u8, size: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe { libc::munmap(ptr.cast(), size) };
    }

    /// `size` bytes of zeroed read-write memory, backed only once touched.
    pub fn anonymous(size: usize) -> Option<*mut u8> {
        // SAFETY: a fresh private anonymous mapping has no preconditions.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        (ptr != libc::MAP_FAILED).then_some(ptr.cast())
    }

    /// # Safety
    /// `ptr` and `size` must name one mapping from `anonymous` that nothing uses any more.
    pub unsafe fn release(ptr: *mut u8, size: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe { libc::munmap(ptr.cast(), size) };
    }
}

#[cfg(windows)]
mod sys {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP, FILE_MAP_EXECUTE, FILE_MAP_READ, FILE_MAP_WRITE, MEM_COMMIT,
        MEM_RELEASE, MEM_RESERVE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        PAGE_EXECUTE_READWRITE, PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc, VirtualFree,
    };

    /// Maps one pagefile-backed section of `size` bytes twice: read-write, then read-execute.
    pub fn map(size: usize) -> Option<(*mut u8, *mut u8)> {
        // SAFETY: a fresh anonymous section has no preconditions.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_EXECUTE_READWRITE,
                (size as u64 >> 32) as u32,
                size as u32,
                std::ptr::null(),
            )
        };
        if section.is_null() {
            return None;
        }
        let view = |access: FILE_MAP| {
            // SAFETY: `section` is a live section of `size` bytes.
            let ptr = unsafe { MapViewOfFile(section, access, 0, 0, size) }.Value;
            (!ptr.is_null()).then_some(ptr.cast::<u8>())
        };
        let write = view(FILE_MAP_WRITE);
        let exec = write.and_then(|_| view(FILE_MAP_READ | FILE_MAP_EXECUTE));
        // SAFETY: the views keep the section alive without the handle.
        unsafe { CloseHandle(section) };
        match (write, exec) {
            (Some(write), Some(exec)) => Some((write, exec)),
            (Some(write), None) => {
                // SAFETY: `write` is the view made above.
                unsafe { unmap(write, size) };
                None
            }
            _ => None,
        }
    }

    /// # Safety
    /// `ptr` must be one view from `map`, and no code in it may run again.
    pub unsafe fn unmap(ptr: *mut u8, _size: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: ptr.cast() }) };
    }

    /// `size` bytes of zeroed read-write memory, backed only once touched.
    pub fn anonymous(size: usize) -> Option<*mut u8> {
        // SAFETY: a fresh allocation has no preconditions.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                size,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        (!ptr.is_null()).then_some(ptr.cast())
    }

    /// # Safety
    /// `ptr` must be one allocation from `anonymous` that nothing uses any more.
    pub unsafe fn release(ptr: *mut u8, _size: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }
}
