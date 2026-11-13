//! `irs` and `irs:sys`: the Joy-Con IR camera. There is no camera, so every processor a title
//! runs reports an unconnected, stopped camera in the IR shared memory and never produces a
//! sample, as the reference answers when no IR data arrives.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Pool;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::{SharedMemory, TransferMemory};
use lichii_kernel::object::KObject;
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use crate::dispatch::Commandless;
use crate::{Service, ServiceContext};

const INVALID_PROCESSOR_STATE: ResultCode = ResultCode::new(205, 78);
const INVALID_IR_CAMERA_HANDLE: ResultCode = ResultCode::new(205, 204);
const INVALID_NPAD_ID: ResultCode = ResultCode::new(202, 709);

/// `nn::irsensor::detail::StatusManager`: nine `DeviceFormat`s, then five aruid entries.
const SHARED_MEMORY_SIZE: u64 = 0x8000;
const DEVICE_COUNT: usize = 9;
const DEVICE_SIZE: u64 = 0xe30;
const PROCESSOR_STATE_OFFSET: u64 = 0x10;
const PROCESSOR_STATE_SIZE: usize = 0xe20;

/// `IrCameraStatus::Unconnected`.
const UNCONNECTED: u32 = 2;
/// `IrCameraInternalStatus::Stopped` and `Ready`.
const STOPPED: u32 = 0;
const READY: u32 = 7;
/// `IrSensorMode`.
const MOMENT: u64 = 1;
const CLUSTERING: u64 = 2;
const IMAGE_TRANSFER: u64 = 3;
const POINTING: u64 = 4;
const TERA_PLUGIN: u64 = 5;
const IR_LED: u64 = 6;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let memory = Arc::new(SharedMemory::new(
        context.kernel.memory_manager().clone(),
        None,
        Pool::System,
        SHARED_MEMORY_SIZE,
        MemoryPermission::NONE,
        MemoryPermission::READ,
    )?);
    for index in 0..DEVICE_COUNT as u64 {
        let device = index * DEVICE_SIZE;
        memory.write(device, &UNCONNECTED.to_le_bytes());
        memory.write(device + 4, &READY.to_le_bytes());
    }
    let irs = Arc::new(IrsState {
        memory,
        image_transfer: Mutex::new(Default::default()),
    });
    context.register("irs", "hid", move |_| Irs { irs: irs.clone() })?;
    context.register("irs:sys", "hid", |_| Commandless("irs:sys"))
}

struct IrsState {
    memory: Arc<SharedMemory>,
    image_transfer: Mutex<[Option<ImageTransfer>; DEVICE_COUNT]>,
}

/// A camera's image transfer processor: where the image goes and its trimmed format.
#[derive(Clone)]
struct ImageTransfer {
    memory: Arc<TransferMemory>,
    trimming_format: u8,
}

impl IrsState {
    /// Starts processor `mode` on camera `index`: the camera stays unconnected and stopped.
    /// The moment and clustering processors also clear their sample LIFO.
    fn start_processor(&self, index: usize, mode: u64) {
        let device = index as u64 * DEVICE_SIZE;
        self.memory.write(device, &UNCONNECTED.to_le_bytes());
        self.memory.write(device + 4, &STOPPED.to_le_bytes());
        self.memory.write(device + 8, &mode.to_le_bytes());
        if matches!(mode, MOMENT | CLUSTERING) {
            self.memory
                .write(device + PROCESSOR_STATE_OFFSET, &[0; PROCESSOR_STATE_SIZE]);
        }
        if mode != IMAGE_TRANSFER {
            self.image_transfer.lock()[index] = None;
        }
    }

    fn mode(&self, index: usize) -> u64 {
        let mut mode = [0; 8];
        self.memory.read(index as u64 * DEVICE_SIZE + 8, &mut mode);
        u64::from_le_bytes(mode)
    }
}

/// The device index of an `nn::irsensor::IrCameraHandle` (npad index, then npad style, which
/// must be `None`). Handheld (index 8) is the last camera.
fn camera_index(handle: u32) -> Result<usize, ResultCode> {
    let [npad, style, ..] = handle.to_le_bytes();
    if usize::from(npad) >= DEVICE_COUNT || style != 0 {
        return Err(INVALID_IR_CAMERA_HANDLE);
    }
    Ok(usize::from(npad))
}

/// Bytes of one image in `ImageTransferProcessorFormat` `format`.
fn image_size(format: u8) -> usize {
    match format {
        0 => 320 * 240,
        1 => 160 * 120,
        2 => 80 * 60,
        3 => 40 * 30,
        4 => 20 * 15,
        _ => 0,
    }
}

struct Irs {
    irs: Arc<IrsState>,
}

impl Irs {
    fn command(&self, ctx: &mut HleRequestContext) -> Option<HleResult<()>> {
        let command = ctx.command();
        Some(match command {
            302 | 303 | 319 => Ok(()),
            304 => {
                ctx.push_copy_object(KObject::SharedMemory(self.irs.memory.clone()));
                Ok(())
            }
            305 | 313 | 314 | 315 | 318 => camera_index(ctx.pop()).map(drop),
            306 | 307 | 310 | 312 | 317 => camera_index(ctx.pop()).map(|index| {
                let mode = match command {
                    306 => MOMENT,
                    307 => CLUSTERING,
                    310 => TERA_PLUGIN,
                    312 => POINTING,
                    _ => IR_LED,
                };
                self.irs.start_processor(index, mode);
            }),
            308 | 316 => self.run_image_transfer(ctx),
            309 => self.image_transfer_state(ctx),
            311 => {
                let npad_id: u32 = ctx.pop();
                let index = match npad_id {
                    0..=7 => npad_id,
                    0x20 => 8,
                    0x10 => 9,
                    _ => return Some(Err(INVALID_NPAD_ID)),
                };
                ctx.push(index);
                Ok(())
            }
            _ => return None,
        })
    }

    /// `RunImageTransferProcessor` (308) and `RunImageTransferExProcessor` (316): camera
    /// handle, aruid, packed config, then the transfer memory size and handle.
    fn run_image_transfer(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let index = camera_index(ctx.pop())?;
        let input = ctx.input();
        let trimming_format = match ctx.command() {
            308 => input.get(0x24),
            _ => input.get(0x25),
        };
        let trimming_format = trimming_format.copied().unwrap_or_default();
        let memory = ctx.copy_handle::<TransferMemory>(0)?;
        self.irs.start_processor(index, IMAGE_TRANSFER);
        self.irs.image_transfer.lock()[index] = Some(ImageTransfer {
            memory,
            trimming_format,
        });
        Ok(())
    }

    /// `GetImageTransferProcessorState`: the camera never sampled, so the state is zero and the
    /// image is whatever the title's transfer memory holds.
    fn image_transfer_state(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let index = camera_index(ctx.pop())?;
        if self.irs.mode(index) != IMAGE_TRANSFER {
            return Err(INVALID_PROCESSOR_STATE);
        }
        let processor = self.irs.image_transfer.lock()[index].clone();
        if let Some(processor) = processor {
            let capacity = ctx
                .writable_buffer(0)
                .map_or(0, |buffer| buffer.size as usize);
            let mut image = vec![0; image_size(processor.trimming_format).min(capacity)];
            processor.memory.read(0, &mut image)?;
            ctx.write_buffer(0, &image)?;
        }
        ctx.push([0u64; 2]);
        Ok(())
    }
}

impl Service for Irs {
    fn name(&self) -> &str {
        "irs"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(self.command(ctx)?.err().unwrap_or(ResultCode::SUCCESS))
    }
}

#[cfg(test)]
mod tests;
