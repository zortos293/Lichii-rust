//! Upsampler slots. Device sinks fed at 32 kHz take one to reach the output rate; the DSP keeps
//! each slot's resampler state.

/// Which upsampler slots are taken.
#[derive(Clone, Debug, Default)]
pub struct UpsamplerManager {
    enabled: Vec<bool>,
}

impl UpsamplerManager {
    pub fn new(count: u32) -> Self {
        Self {
            enabled: vec![false; count as usize],
        }
    }

    pub fn count(&self) -> u32 {
        self.enabled.len() as u32
    }

    /// Takes the first free slot.
    pub fn allocate(&mut self) -> Option<u32> {
        let index = self.enabled.iter().position(|enabled| !enabled)?;
        self.enabled[index] = true;
        Some(index as u32)
    }

    pub fn free(&mut self, index: u32) {
        if let Some(enabled) = self.enabled.get_mut(index as usize) {
            *enabled = false;
        }
    }

    pub fn is_enabled(&self, index: u32) -> bool {
        self.enabled.get(index as usize).copied().unwrap_or(false)
    }
}
