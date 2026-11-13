//! Guest memory as the audio DSP sees it.

/// Byte access to the memory the guest shares with the audio DSP: wave buffers, ADPCM
/// coefficients and contexts, aux and capture ring buffers, circular buffer sinks.
///
/// Reads of unmapped bytes yield zeroes and writes to unmapped bytes are dropped; the
/// implementation decides whether to log them. The services bind this to the guest process's
/// address space; tests use [`VecMemory`].
pub trait DspMemory {
    /// Fills `out` with the bytes at `address`.
    fn read(&self, address: u64, out: &mut [u8]);

    /// Stores `data` at `address`.
    fn write(&mut self, address: u64, data: &[u8]);

    fn read_u32(&self, address: u64) -> u32 {
        let mut bytes = [0; 4];
        self.read(address, &mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn write_u32(&mut self, address: u64, value: u32) {
        self.write(address, &value.to_le_bytes());
    }
}

/// A flat byte buffer mapped at `base`, for tests and tools.
#[derive(Clone, Debug, Default)]
pub struct VecMemory {
    pub base: u64,
    pub bytes: Vec<u8>,
}

impl VecMemory {
    pub fn new(base: u64, size: usize) -> Self {
        Self {
            base,
            bytes: vec![0; size],
        }
    }

    fn mapped(&self, address: u64, len: usize) -> Option<(usize, usize)> {
        let start = usize::try_from(address.checked_sub(self.base)?).ok()?;
        let available = self.bytes.len().checked_sub(start)?;
        Some((start, len.min(available)))
    }
}

impl DspMemory for VecMemory {
    fn read(&self, address: u64, out: &mut [u8]) {
        out.fill(0);
        if let Some((start, len)) = self.mapped(address, out.len()) {
            out[..len].copy_from_slice(&self.bytes[start..start + len]);
        }
    }

    fn write(&mut self, address: u64, data: &[u8]) {
        if let Some((start, len)) = self.mapped(address, data.len()) {
            self.bytes[start..start + len].copy_from_slice(&data[..len]);
        }
    }
}
