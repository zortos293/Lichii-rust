use ash::vk;

/// Errors from the renderer's Vulkan layer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the Vulkan loader is unavailable: {0}")]
    Loader(#[from] ash::LoadingError),
    #[error("{call} failed: {result}")]
    Vulkan {
        call: &'static str,
        result: vk::Result,
    },
    #[error("no Vulkan 1.3 device with graphics and the required features")]
    NoDevice,
    #[error("the surface supports no usable format")]
    NoSurfaceFormat,
    #[error("allocating device memory: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),
    #[error("accessing the disk shader cache: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) trait VkResultExt<T> {
    fn call(self, call: &'static str) -> Result<T>;
}

impl<T> VkResultExt<T> for Result<T, vk::Result> {
    fn call(self, call: &'static str) -> Result<T> {
        self.map_err(|result| Error::Vulkan { call, result })
    }
}
