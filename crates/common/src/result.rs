use std::fmt;

/// A Horizon result code: module in bits 0..9, description in bits 9..22.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct ResultCode(pub u32);

pub type HleResult<T> = Result<T, ResultCode>;

impl ResultCode {
    pub const SUCCESS: Self = Self(0);

    pub const fn new(module: u32, description: u32) -> Self {
        Self((module & 0x1ff) | ((description & 0x1fff) << 9))
    }

    pub const fn module(self) -> u32 {
        self.0 & 0x1ff
    }

    pub const fn description(self) -> u32 {
        (self.0 >> 9) & 0x1fff
    }

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }

    pub const fn is_error(self) -> bool {
        self.0 != 0
    }

    /// `Ok(())` for success, `Err(self)` otherwise.
    pub const fn into_result(self) -> HleResult<()> {
        if self.is_success() { Ok(()) } else { Err(self) }
    }
}

impl From<HleResult<()>> for ResultCode {
    fn from(value: HleResult<()>) -> Self {
        match value {
            Ok(()) => Self::SUCCESS,
            Err(code) => code,
        }
    }
}

impl fmt::Debug for ResultCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:04} (0x{:x})",
            2000 + self.module(),
            self.description(),
            self.0
        )
    }
}

impl fmt::Display for ResultCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:04}", 2000 + self.module(), self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_like_horizon() {
        let r = ResultCode::new(2, 1002);
        assert_eq!(r.module(), 2);
        assert_eq!(r.description(), 1002);
        assert_eq!(r.to_string(), "2002-1002");
        assert!(ResultCode::SUCCESS.into_result().is_ok());
    }
}
