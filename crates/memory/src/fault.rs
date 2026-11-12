//! Host fault dispatch for fastmem arenas.
//!
//! The JIT emits plain host loads and stores at `arena base + guest address`. When the guest page
//! is unmapped or protected, the access raises a host fault (SIGSEGV/SIGBUS on Unix, an access
//! violation on Windows). The process-wide handler installed here finds the arena that contains
//! the faulting address and calls the handler registered for it, which can rewrite the host
//! context (for example, move the program counter to a slow-path stub). Faults outside every
//! arena go to the previously installed handler.

use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

/// Called on a fault inside a fastmem arena. `guest_addr` is the faulting offset into the arena.
/// Returning `true` resumes the host thread with the (possibly modified) context; `false`
/// passes the fault on.
///
/// The handler runs inside a signal handler or a vectored exception handler: it must not
/// allocate, lock, or unwind.
pub type FaultHandler = fn(data: usize, guest_addr: u64, fault: &mut HostFault) -> bool;

/// The interrupted host thread's state at a fastmem fault.
pub struct HostFault {
    ctx: *mut std::ffi::c_void,
    host_addr: usize,
}

const SLOTS: usize = 64;

struct Slot {
    base: AtomicUsize,
    size: AtomicUsize,
    handler: AtomicUsize,
    data: AtomicUsize,
}

static ARENAS: [Slot; SLOTS] = [const {
    Slot {
        base: AtomicUsize::new(0),
        size: AtomicUsize::new(0),
        handler: AtomicUsize::new(0),
        data: AtomicUsize::new(0),
    }
}; SLOTS];
static REGISTRY: Mutex<()> = Mutex::new(());
static INSTALL: Once = Once::new();

/// A registered arena. Dropping it unregisters the range.
pub(crate) struct Registration {
    slot: usize,
}

impl Registration {
    /// Registers `[base, base + size)`. Returns `None` when the fault handler is unsupported on
    /// this host or every slot is taken.
    pub(crate) fn new(base: usize, size: usize) -> Option<Self> {
        if !sys::SUPPORTED {
            return None;
        }
        INSTALL.call_once(sys::install);
        let _guard = REGISTRY.lock();
        let slot = ARENAS
            .iter()
            .position(|s| s.base.load(Ordering::Acquire) == 0)?;
        let entry = &ARENAS[slot];
        entry.handler.store(0, Ordering::Release);
        entry.size.store(size, Ordering::Release);
        entry.base.store(base, Ordering::Release);
        Some(Self { slot })
    }

    pub(crate) fn set_handler(&self, handler: Option<FaultHandler>, data: usize) {
        let entry = &ARENAS[self.slot];
        entry.handler.store(0, Ordering::Release);
        entry.data.store(data, Ordering::Release);
        entry
            .handler
            .store(handler.map_or(0, |h| h as usize), Ordering::Release);
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let _guard = REGISTRY.lock();
        let entry = &ARENAS[self.slot];
        entry.base.store(0, Ordering::Release);
        entry.handler.store(0, Ordering::Release);
    }
}

/// Looks up the arena for `addr` and runs its handler. Async-signal-safe.
fn dispatch(addr: usize, ctx: *mut std::ffi::c_void) -> bool {
    for entry in &ARENAS {
        let base = entry.base.load(Ordering::Acquire);
        if base == 0 || addr < base || addr - base >= entry.size.load(Ordering::Acquire) {
            continue;
        }
        let handler = entry.handler.load(Ordering::Acquire);
        if handler == 0 {
            return false;
        }
        // SAFETY: only FaultHandler pointers are stored in the slot.
        let handler: FaultHandler = unsafe { std::mem::transmute::<usize, FaultHandler>(handler) };
        let data = entry.data.load(Ordering::Acquire);
        let mut fault = HostFault {
            ctx,
            host_addr: addr,
        };
        return handler(data, (addr - base) as u64, &mut fault);
    }
    false
}

impl HostFault {
    /// Host address of the faulting access.
    pub fn host_addr(&self) -> usize {
        self.host_addr
    }

    /// Host program counter of the faulting instruction.
    pub fn pc(&self) -> u64 {
        // SAFETY: ctx is the live context passed to the fault handler.
        unsafe { sys::pc(self.ctx) }
    }

    pub fn set_pc(&mut self, pc: u64) {
        // SAFETY: as above.
        unsafe { sys::set_pc(self.ctx, pc) }
    }

    /// Host general-purpose register by its encoding number (x86-64: 0 = RAX ... 15 = R15;
    /// AArch64: 0 = X0 ... 30 = X30, 31 = SP).
    pub fn gpr(&self, index: usize) -> u64 {
        // SAFETY: as above.
        unsafe { sys::gpr(self.ctx, index) }
    }

    pub fn set_gpr(&mut self, index: usize, value: u64) {
        // SAFETY: as above.
        unsafe { sys::set_gpr(self.ctx, index, value) }
    }
}

#[cfg(all(
    unix,
    any(target_arch = "x86_64", target_arch = "aarch64"),
    target_os = "linux"
))]
mod sys {
    use std::ffi::c_void;
    use std::sync::OnceLock;

    pub const SUPPORTED: bool = true;

    struct Previous([libc::sigaction; 2]);
    // SAFETY: sigaction is plain data; it is written once before any fault can read it.
    unsafe impl Sync for Previous {}
    unsafe impl Send for Previous {}
    static PREVIOUS: OnceLock<Previous> = OnceLock::new();

    pub fn install() {
        // SAFETY: installs a SA_SIGINFO handler; the old actions are kept for chaining.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = handler as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
            libc::sigemptyset(&mut action.sa_mask);
            let mut old: [libc::sigaction; 2] = std::mem::zeroed();
            libc::sigaction(libc::SIGSEGV, &action, &mut old[0]);
            libc::sigaction(libc::SIGBUS, &action, &mut old[1]);
            let _ = PREVIOUS.set(Previous(old));
        }
    }

    unsafe extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut c_void) {
        // SAFETY: the kernel passes a valid siginfo for SIGSEGV/SIGBUS.
        let addr = unsafe { (*info).si_addr() } as usize;
        if super::dispatch(addr, ctx) {
            return;
        }
        let Some(previous) = PREVIOUS.get() else {
            return;
        };
        let old = &previous.0[usize::from(sig == libc::SIGBUS)];
        // SAFETY: chains to the handler that was installed before ours, or restores the default
        // action so the faulting instruction re-raises the signal and terminates the process.
        unsafe {
            if old.sa_sigaction == libc::SIG_DFL || old.sa_sigaction == libc::SIG_IGN {
                libc::sigaction(sig, old, std::ptr::null_mut());
            } else if old.sa_flags & libc::SA_SIGINFO != 0 {
                let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut c_void) =
                    std::mem::transmute(old.sa_sigaction);
                f(sig, info, ctx);
            } else {
                let f: extern "C" fn(libc::c_int) = std::mem::transmute(old.sa_sigaction);
                f(sig);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    const GREGS: [libc::c_int; 16] = [
        libc::REG_RAX,
        libc::REG_RCX,
        libc::REG_RDX,
        libc::REG_RBX,
        libc::REG_RSP,
        libc::REG_RBP,
        libc::REG_RSI,
        libc::REG_RDI,
        libc::REG_R8,
        libc::REG_R9,
        libc::REG_R10,
        libc::REG_R11,
        libc::REG_R12,
        libc::REG_R13,
        libc::REG_R14,
        libc::REG_R15,
    ];

    #[cfg(target_arch = "x86_64")]
    unsafe fn gregs(ctx: *mut c_void) -> *mut [libc::greg_t; 23] {
        // SAFETY: ctx is the ucontext_t of the signal.
        unsafe { &raw mut (*ctx.cast::<libc::ucontext_t>()).uc_mcontext.gregs }
    }

    #[cfg(target_arch = "x86_64")]
    pub unsafe fn pc(ctx: *mut c_void) -> u64 {
        // SAFETY: see gregs.
        unsafe { (*gregs(ctx))[libc::REG_RIP as usize] as u64 }
    }

    #[cfg(target_arch = "x86_64")]
    pub unsafe fn set_pc(ctx: *mut c_void, pc: u64) {
        // SAFETY: see gregs.
        unsafe { (*gregs(ctx))[libc::REG_RIP as usize] = pc as libc::greg_t }
    }

    #[cfg(target_arch = "x86_64")]
    pub unsafe fn gpr(ctx: *mut c_void, index: usize) -> u64 {
        // SAFETY: see gregs.
        unsafe { (*gregs(ctx))[GREGS[index] as usize] as u64 }
    }

    #[cfg(target_arch = "x86_64")]
    pub unsafe fn set_gpr(ctx: *mut c_void, index: usize, value: u64) {
        // SAFETY: see gregs.
        unsafe { (*gregs(ctx))[GREGS[index] as usize] = value as libc::greg_t }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn mcontext(ctx: *mut c_void) -> *mut libc::mcontext_t {
        // SAFETY: ctx is the ucontext_t of the signal.
        unsafe { &raw mut (*ctx.cast::<libc::ucontext_t>()).uc_mcontext }
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn pc(ctx: *mut c_void) -> u64 {
        // SAFETY: see mcontext.
        unsafe { (*mcontext(ctx)).pc }
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn set_pc(ctx: *mut c_void, pc: u64) {
        // SAFETY: see mcontext.
        unsafe { (*mcontext(ctx)).pc = pc }
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn gpr(ctx: *mut c_void, index: usize) -> u64 {
        // SAFETY: see mcontext.
        unsafe {
            let m = &*mcontext(ctx);
            if index == 31 { m.sp } else { m.regs[index] }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn set_gpr(ctx: *mut c_void, index: usize, value: u64) {
        // SAFETY: see mcontext.
        unsafe {
            let m = &mut *mcontext(ctx);
            if index == 31 {
                m.sp = value;
            } else {
                m.regs[index] = value;
            }
        }
    }
}

#[cfg(all(windows, target_arch = "x86_64"))]
mod sys {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS,
    };

    pub const SUPPORTED: bool = true;

    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    pub fn install() {
        // SAFETY: registers a first-in-line vectored handler for the life of the process.
        unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
    }

    unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: Windows passes valid exception pointers to vectored handlers.
        unsafe {
            let record = &*(*info).ExceptionRecord;
            if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2 {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let addr = record.ExceptionInformation[1];
            if super::dispatch(addr, (*info).ContextRecord.cast()) {
                EXCEPTION_CONTINUE_EXECUTION
            } else {
                EXCEPTION_CONTINUE_SEARCH
            }
        }
    }

    unsafe fn context<'a>(ctx: *mut c_void) -> &'a mut CONTEXT {
        // SAFETY: ctx is the CONTEXT record of the exception.
        unsafe { &mut *ctx.cast::<CONTEXT>() }
    }

    pub unsafe fn pc(ctx: *mut c_void) -> u64 {
        // SAFETY: see context.
        unsafe { context(ctx).Rip }
    }

    pub unsafe fn set_pc(ctx: *mut c_void, pc: u64) {
        // SAFETY: see context.
        unsafe { context(ctx).Rip = pc }
    }

    pub unsafe fn gpr(ctx: *mut c_void, index: usize) -> u64 {
        // SAFETY: see context.
        let c = unsafe { context(ctx) };
        *reg(c, index)
    }

    pub unsafe fn set_gpr(ctx: *mut c_void, index: usize, value: u64) {
        // SAFETY: see context.
        let c = unsafe { context(ctx) };
        *reg(c, index) = value;
    }

    fn reg(c: &mut CONTEXT, index: usize) -> &mut u64 {
        match index {
            0 => &mut c.Rax,
            1 => &mut c.Rcx,
            2 => &mut c.Rdx,
            3 => &mut c.Rbx,
            4 => &mut c.Rsp,
            5 => &mut c.Rbp,
            6 => &mut c.Rsi,
            7 => &mut c.Rdi,
            8 => &mut c.R8,
            9 => &mut c.R9,
            10 => &mut c.R10,
            11 => &mut c.R11,
            12 => &mut c.R12,
            13 => &mut c.R13,
            14 => &mut c.R14,
            15 => &mut c.R15,
            _ => panic!("x86-64 has no register {index}"),
        }
    }
}

#[cfg(not(any(
    all(
        unix,
        any(target_arch = "x86_64", target_arch = "aarch64"),
        target_os = "linux"
    ),
    all(windows, target_arch = "x86_64")
)))]
mod sys {
    use std::ffi::c_void;

    pub const SUPPORTED: bool = false;

    pub fn install() {}

    pub unsafe fn pc(_ctx: *mut c_void) -> u64 {
        unreachable!("fastmem faults are unsupported on this host")
    }

    pub unsafe fn set_pc(_ctx: *mut c_void, _pc: u64) {
        unreachable!("fastmem faults are unsupported on this host")
    }

    pub unsafe fn gpr(_ctx: *mut c_void, _index: usize) -> u64 {
        unreachable!("fastmem faults are unsupported on this host")
    }

    pub unsafe fn set_gpr(_ctx: *mut c_void, _index: usize, _value: u64) {
        unreachable!("fastmem faults are unsupported on this host")
    }
}
