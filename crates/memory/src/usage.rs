//! What the OS counts against this process, for the kernel's periodic memory line.

/// This process's memory as the OS counts it, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcessMemory {
    /// Memory only this process can use: private bytes (the commit charge, `PagefileUsage`) on
    /// Windows, anonymous resident plus swapped pages on Linux. Guest DRAM is shared memory, so
    /// it is not in here.
    pub private: u64,
    /// Every resident page, guest DRAM included: the working set on Windows, the RSS on Linux.
    pub resident: u64,
}

/// The current process's memory, or `None` when the OS does not report it.
pub fn process_memory() -> Option<ProcessMemory> {
    sys::process_memory()
}

#[cfg(target_os = "linux")]
mod sys {
    use super::ProcessMemory;

    pub fn process_memory() -> Option<ProcessMemory> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let field = |name: &str| -> Option<u64> {
            let line = status.lines().find(|l| l.starts_with(name))?;
            let kib: u64 = line[name.len()..]
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .ok()?;
            Some(kib << 10)
        };
        Some(ProcessMemory {
            private: field("RssAnon:")? + field("VmSwap:").unwrap_or(0),
            resident: field("VmRSS:")?,
        })
    }
}

#[cfg(windows)]
mod sys {
    use super::ProcessMemory;

    pub fn process_memory() -> Option<ProcessMemory> {
        let stats = memory_stats::memory_stats()?;
        Some(ProcessMemory {
            private: stats.virtual_mem as u64,
            resident: stats.physical_mem as u64,
        })
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod sys {
    pub fn process_memory() -> Option<super::ProcessMemory> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(target_os = "linux", windows))]
    fn reports_this_process() {
        let usage = super::process_memory().unwrap();
        assert!(
            usage.private > 0 && usage.resident >= usage.private / 2,
            "{usage:?}"
        );
    }
}
