//! Host class (puller) methods: subchannel binds, semaphores, syncpoints, reference counts and
//! wait-for-idle. Methods below 0x40 of every subchannel go here.
//!
//! Maxwell channels use the B06F host class (semaphores through SEMAPHOREA..D, and the legacy
//! SEMAPHORE_ACQUIRE/RELEASE at 0x1A/0x1B); Switch 2 channels use the Ampere C56F class, which
//! puts SEM_ADDR/SEM_PAYLOAD/SEM_EXECUTE at 0x17..0x1B instead (NX2_NOTES rows 50, 52).

use lichii_common::GpuVAddr;

/// First method that belongs to the bound engine instead of the host class.
pub const FIRST_ENGINE_METHOD: u32 = 0x40;

mod method {
    pub const SET_OBJECT: u32 = 0x00;
    pub const SEMAPHORE_A: u32 = 0x04;
    pub const SEMAPHORE_B: u32 = 0x05;
    pub const SEMAPHORE_C: u32 = 0x06;
    pub const SEMAPHORE_D: u32 = 0x07;
    pub const MEM_OP_B: u32 = 0x0b;
    pub const SET_REFERENCE: u32 = 0x14;
    pub const SEM_ADDR_LO: u32 = 0x17;
    pub const SEM_ADDR_HI: u32 = 0x18;
    pub const SEM_PAYLOAD_LO: u32 = 0x19;
    pub const SEM_PAYLOAD_HI: u32 = 0x1a;
    pub const SEM_EXECUTE: u32 = 0x1b;
    pub const SEMAPHORE_ACQUIRE: u32 = 0x1a;
    pub const SEMAPHORE_RELEASE: u32 = 0x1b;
    pub const SYNCPOINT_A: u32 = 0x1c;
    pub const SYNCPOINT_B: u32 = 0x1d;
    pub const WFI: u32 = 0x1e;
}

/// Which host class a channel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostClass {
    /// B06F (Switch 1).
    Maxwell,
    /// C56F and later (Switch 2).
    Ampere,
}

/// Condition of a semaphore acquire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquireOp {
    Equal,
    /// Unsigned greater or equal.
    GreaterEqual,
    /// Greater or equal on a wrapping counter (signed difference).
    CircularGreaterEqual,
    /// Any of the payload's bits set.
    And,
    /// None of the payload's bits set.
    Nor,
}

impl AcquireOp {
    #[inline]
    pub fn is_met(self, current: u64, payload: u64, wide: bool) -> bool {
        let (current, payload) = if wide {
            (current, payload)
        } else {
            (current & 0xffff_ffff, payload & 0xffff_ffff)
        };
        match self {
            Self::Equal => current == payload,
            Self::GreaterEqual => current >= payload,
            Self::CircularGreaterEqual if wide => current.wrapping_sub(payload) as i64 >= 0,
            Self::CircularGreaterEqual => (current as u32).wrapping_sub(payload as u32) as i32 >= 0,
            Self::And => current & payload != 0,
            Self::Nor => current & payload == 0,
        }
    }
}

/// An atomic operation a reduction release applies to memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Min,
    Max,
    Xor,
    And,
    Or,
    Add,
    Inc,
    Dec,
}

impl Reduction {
    fn from_bits(bits: u32) -> Option<Self> {
        Some(match bits {
            0 => Self::Min,
            1 => Self::Max,
            2 => Self::Xor,
            3 => Self::And,
            4 => Self::Or,
            5 => Self::Add,
            6 => Self::Inc,
            7 => Self::Dec,
            _ => return None,
        })
    }

    /// The new memory value. `signed` selects signed min/max.
    pub fn apply(self, old: u32, operand: u32, signed: bool) -> u32 {
        match self {
            Self::Min if signed => (old as i32).min(operand as i32) as u32,
            Self::Max if signed => (old as i32).max(operand as i32) as u32,
            Self::Min => old.min(operand),
            Self::Max => old.max(operand),
            Self::Xor => old ^ operand,
            Self::And => old & operand,
            Self::Or => old | operand,
            Self::Add => old.wrapping_add(operand),
            Self::Inc => {
                if old >= operand {
                    0
                } else {
                    old + 1
                }
            }
            Self::Dec => {
                if old == 0 || old > operand {
                    operand
                } else {
                    old - 1
                }
            }
        }
    }
}

/// Size of a semaphore release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseKind {
    /// 32-bit payload.
    Four,
    /// 64-bit payload.
    Eight,
    /// Payload and a 64-bit timestamp, 16 bytes.
    WithTimestamp,
}

/// What a host method asks the GPU to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAction {
    /// Only stored a register.
    None,
    /// Binds an engine class to a subchannel.
    Bind {
        subchannel: u32,
        class: u32,
    },
    /// Wait until the semaphore condition holds. Checked by the front thread.
    Acquire {
        address: GpuVAddr,
        payload: u64,
        op: AcquireOp,
        wide: bool,
    },
    /// Write a semaphore once earlier work is done. Executed by the back thread.
    Release {
        address: GpuVAddr,
        payload: u64,
        kind: ReleaseKind,
    },
    /// Atomically combine a 32-bit semaphore with the payload.
    Reduce {
        address: GpuVAddr,
        payload: u32,
        op: Reduction,
        signed: bool,
    },
    /// Wait until syncpoint `id` reaches `value`.
    SyncpointWait {
        id: u32,
        value: u32,
    },
    /// Increment syncpoint `id` once earlier work is done.
    SyncpointIncrement {
        id: u32,
    },
    SetReference(u32),
    WaitForIdle,
    /// MEM_OP_B: invalidate GPU caches.
    InvalidateCaches,
}

/// Host class registers of one channel.
#[derive(Clone)]
pub struct HostState {
    class: HostClass,
    regs: [u32; FIRST_ENGINE_METHOD as usize],
}

impl HostState {
    pub fn new(class: HostClass) -> Self {
        Self {
            class,
            regs: [0; FIRST_ENGINE_METHOD as usize],
        }
    }

    pub fn class(&self) -> HostClass {
        self.class
    }

    /// Writes host method `method` (below [`FIRST_ENGINE_METHOD`]) on `subchannel`.
    pub fn call(&mut self, subchannel: u32, method: u32, argument: u32) -> HostAction {
        self.regs[method as usize] = argument;
        match method {
            method::SET_OBJECT => HostAction::Bind {
                subchannel,
                class: argument & 0xffff,
            },
            method::SEMAPHORE_D => self.semaphore_d(argument),
            method::SEM_EXECUTE if self.class == HostClass::Ampere => self.sem_execute(argument),
            method::SEMAPHORE_ACQUIRE if self.class == HostClass::Maxwell => HostAction::Acquire {
                address: self.semaphore_address(),
                payload: u64::from(argument),
                op: AcquireOp::Equal,
                wide: false,
            },
            method::SEMAPHORE_RELEASE if self.class == HostClass::Maxwell => HostAction::Release {
                address: self.semaphore_address(),
                payload: u64::from(argument),
                kind: ReleaseKind::Four,
            },
            method::SYNCPOINT_B => {
                let id = (argument >> 8) & 0xfff;
                if argument & 1 != 0 {
                    HostAction::SyncpointIncrement { id }
                } else {
                    HostAction::SyncpointWait {
                        id,
                        value: self.regs[method::SYNCPOINT_A as usize],
                    }
                }
            }
            method::SET_REFERENCE => HostAction::SetReference(argument),
            method::WFI => HostAction::WaitForIdle,
            method::MEM_OP_B => HostAction::InvalidateCaches,
            _ => HostAction::None,
        }
    }

    fn reg(&self, method: u32) -> u32 {
        self.regs[method as usize]
    }

    fn semaphore_address(&self) -> GpuVAddr {
        (u64::from(self.reg(method::SEMAPHORE_A) & 0xff) << 32)
            | u64::from(self.reg(method::SEMAPHORE_B) & !3)
    }

    fn semaphore_d(&self, argument: u32) -> HostAction {
        let address = self.semaphore_address();
        let payload = self.reg(method::SEMAPHORE_C);
        let acquire = |op| HostAction::Acquire {
            address,
            payload: u64::from(payload),
            op,
            wide: false,
        };
        match argument & 0x1f {
            1 => acquire(AcquireOp::Equal),
            2 => HostAction::Release {
                address,
                payload: u64::from(payload),
                kind: if argument & (1 << 24) != 0 {
                    ReleaseKind::Four
                } else {
                    ReleaseKind::WithTimestamp
                },
            },
            4 => acquire(AcquireOp::GreaterEqual),
            8 => acquire(AcquireOp::And),
            16 => match Reduction::from_bits((argument >> 27) & 0xf) {
                Some(op) => HostAction::Reduce {
                    address,
                    payload,
                    op,
                    signed: argument & (1 << 31) == 0,
                },
                None => HostAction::None,
            },
            _ => HostAction::None,
        }
    }

    fn sem_execute(&self, argument: u32) -> HostAction {
        let address = (u64::from(self.reg(method::SEM_ADDR_HI) & 0x1ff_ffff) << 32)
            | u64::from(self.reg(method::SEM_ADDR_LO) & !3);
        let wide = argument & (1 << 24) != 0;
        let low = u64::from(self.reg(method::SEM_PAYLOAD_LO));
        let payload = if wide {
            low | (u64::from(self.reg(method::SEM_PAYLOAD_HI)) << 32)
        } else {
            low
        };
        let acquire = |op| HostAction::Acquire {
            address,
            payload,
            op,
            wide,
        };
        match argument & 7 {
            0 => acquire(AcquireOp::Equal),
            1 => HostAction::Release {
                address,
                payload,
                kind: match (argument & (1 << 25) != 0, wide) {
                    (true, _) => ReleaseKind::WithTimestamp,
                    (false, true) => ReleaseKind::Eight,
                    (false, false) => ReleaseKind::Four,
                },
            },
            2 => acquire(AcquireOp::GreaterEqual),
            3 => acquire(AcquireOp::CircularGreaterEqual),
            4 => acquire(AcquireOp::And),
            5 => acquire(AcquireOp::Nor),
            6 => match Reduction::from_bits((argument >> 27) & 0xf) {
                Some(op) => HostAction::Reduce {
                    address,
                    payload: payload as u32,
                    op,
                    signed: argument & (1 << 31) == 0,
                },
                None => HostAction::None,
            },
            _ => HostAction::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxwell_legacy_semaphore_acquire_and_release() {
        let mut host = HostState::new(HostClass::Maxwell);
        host.call(0, method::SEMAPHORE_A, 0x12);
        host.call(0, method::SEMAPHORE_B, 0x3456_7893);
        assert_eq!(
            host.call(0, method::SEMAPHORE_ACQUIRE, 5),
            HostAction::Acquire {
                address: 0x12_3456_7890,
                payload: 5,
                op: AcquireOp::Equal,
                wide: false
            }
        );
        assert_eq!(
            host.call(0, method::SEMAPHORE_RELEASE, 6),
            HostAction::Release {
                address: 0x12_3456_7890,
                payload: 6,
                kind: ReleaseKind::Four
            }
        );
        let mut ampere = HostState::new(HostClass::Ampere);
        assert_eq!(ampere.call(0, method::SEM_PAYLOAD_HI, 5), HostAction::None);
    }

    #[test]
    fn maxwell_semaphores() {
        let mut host = HostState::new(HostClass::Maxwell);
        host.call(0, method::SEMAPHORE_A, 0x12);
        host.call(0, method::SEMAPHORE_B, 0x3456_7890);
        host.call(0, method::SEMAPHORE_C, 7);
        assert_eq!(
            host.call(0, method::SEMAPHORE_D, 2),
            HostAction::Release {
                address: 0x12_3456_7890,
                payload: 7,
                kind: ReleaseKind::WithTimestamp
            }
        );
        assert_eq!(
            host.call(0, method::SEMAPHORE_D, 2 | 1 << 24),
            HostAction::Release {
                address: 0x12_3456_7890,
                payload: 7,
                kind: ReleaseKind::Four
            }
        );
        assert_eq!(
            host.call(0, method::SEMAPHORE_D, 4),
            HostAction::Acquire {
                address: 0x12_3456_7890,
                payload: 7,
                op: AcquireOp::GreaterEqual,
                wide: false
            }
        );
    }

    #[test]
    fn ampere_sem_execute() {
        let mut host = HostState::new(HostClass::Ampere);
        host.call(0, method::SEM_ADDR_LO, 0x2106_0280);
        host.call(0, method::SEM_ADDR_HI, 0x21);
        host.call(0, method::SEM_PAYLOAD_LO, 3);
        host.call(0, method::SEM_PAYLOAD_HI, 1);
        assert_eq!(
            host.call(0, method::SEM_EXECUTE, 2),
            HostAction::Acquire {
                address: 0x21_2106_0280,
                payload: 3,
                op: AcquireOp::GreaterEqual,
                wide: false
            }
        );
        assert_eq!(
            host.call(0, method::SEM_EXECUTE, 1 | 1 << 24),
            HostAction::Release {
                address: 0x21_2106_0280,
                payload: 0x1_0000_0003,
                kind: ReleaseKind::Eight
            }
        );
        assert_eq!(
            host.call(0, method::SEM_EXECUTE, 6 | 5 << 27),
            HostAction::Reduce {
                address: 0x21_2106_0280,
                payload: 3,
                op: Reduction::Add,
                signed: true
            }
        );
    }

    #[test]
    fn syncpoints_and_binds() {
        let mut host = HostState::new(HostClass::Ampere);
        assert_eq!(
            host.call(2, method::SET_OBJECT, 0xc797),
            HostAction::Bind {
                subchannel: 2,
                class: 0xc797
            }
        );
        host.call(0, method::SYNCPOINT_A, 5);
        assert_eq!(
            host.call(0, method::SYNCPOINT_B, 1 << 8 | 1),
            HostAction::SyncpointIncrement { id: 1 }
        );
        assert_eq!(
            host.call(0, method::SYNCPOINT_B, 3 << 8),
            HostAction::SyncpointWait { id: 3, value: 5 }
        );
    }

    #[test]
    fn acquire_conditions() {
        assert!(AcquireOp::CircularGreaterEqual.is_met(1, 0xffff_fff0, false));
        assert!(!AcquireOp::GreaterEqual.is_met(1, 0xffff_fff0, false));
        assert!(AcquireOp::Nor.is_met(0b100, 0b011, false));
        assert!(AcquireOp::Equal.is_met(0x1_0000_0005, 5, false));
        assert!(!AcquireOp::Equal.is_met(0x1_0000_0005, 5, true));
        assert_eq!(Reduction::Inc.apply(3, 3, false), 0);
        assert_eq!(Reduction::Dec.apply(0, 9, false), 9);
        assert_eq!(Reduction::Max.apply(u32::MAX, 1, true), 1);
    }
}
