//! `fatal:u`: a title reporting an unrecoverable error. The report is logged, as the reference
//! does, and the call succeeds.

use std::fmt::Write;
use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

/// `FatalInfo`: x0-x30, sp, pc, pstate, afsr0, afsr1, esr, far, 32 backtrace entries, the program
/// entry point, set flags, then the backtrace size and the architecture.
const INFO_SIZE: usize = 0x250;
const BACKTRACE: usize = 38 * 8;
const BACKTRACE_SIZE: usize = 0x240;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("fatal:u", "services", |context| Fatal {
        program_id: context.program.program_id,
    })
}

struct Fatal {
    program_id: u64,
}

impl Service for Fatal {
    fn name(&self) -> &str {
        "fatal:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command = ctx.command();
        if command > 2 {
            return None;
        }
        let result = ResultCode(ctx.pop());
        let policy: u32 = if command == 0 { 2 } else { ctx.pop() };
        let info = (command == 2)
            .then(|| ctx.read_buffer(0).ok())
            .flatten()
            .filter(|info| info.len() >= INFO_SIZE);
        tracing::error!(
            "fatal:u {command}: title {:016x} threw {result} (policy {policy}){}",
            self.program_id,
            info.as_deref().map(report).unwrap_or_default()
        );
        Some(ResultCode::SUCCESS)
    }
}

/// The registers and backtrace of a `FatalInfo`.
fn report(info: &[u8]) -> String {
    let word = |i: usize| u64::from_le_bytes(info[i * 8..i * 8 + 8].try_into().unwrap());
    let mut text = String::new();
    for (i, name) in (31..38).zip(["sp", "pc", "pstate", "afsr0", "afsr1", "esr", "far"]) {
        let _ = write!(text, " {name}={:#x}", word(i));
    }
    for i in 0..31 {
        let _ = write!(text, " x{i}={:#x}", word(i));
    }
    let count = u32::from_le_bytes(info[BACKTRACE_SIZE..BACKTRACE_SIZE + 4].try_into().unwrap());
    text.push_str(" backtrace:");
    for i in 0..(count as usize).min(32) {
        let _ = write!(text, " {:#x}", word(BACKTRACE / 8 + i));
    }
    text
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;

    use crate::testing::{Request, SWITCH1_TITLE, booted};

    #[test]
    fn fatal_reports_are_accepted() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let fatal = client.connect("fatal:u").unwrap();
        let code = ResultCode::new(2, 1).0;
        let mut info = vec![0u8; super::INFO_SIZE];
        info[32 * 8..33 * 8].copy_from_slice(&0x8000_1234u64.to_le_bytes());
        info[super::BACKTRACE_SIZE] = 1;
        info[super::BACKTRACE..super::BACKTRACE + 8].copy_from_slice(&0x8000_0040u64.to_le_bytes());
        let report = super::report(&info);
        assert!(report.contains("pc=0x80001234"), "{report}");
        assert!(report.ends_with("backtrace: 0x80000040"), "{report}");

        let buffer = client.alloc(&info);
        for request in [
            Request::new(0).arg(code).send_pid().arg(0u64),
            Request::new(1).arg(code).arg(2u32).send_pid().arg(0u64),
            Request::new(2)
                .arg(code)
                .arg(2u32)
                .send_pid()
                .arg(0u64)
                .buffer_a(buffer, super::INFO_SIZE as u64),
        ] {
            assert_eq!(
                client.send(fatal, &request).unwrap().result,
                ResultCode::SUCCESS
            );
        }
    }
}
