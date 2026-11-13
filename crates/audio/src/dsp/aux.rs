//! Aux sends and returns, and capture, through guest ring buffers (`aux_.cpp`, `capture.cpp`).

use std::mem::{offset_of, size_of};

use super::abi::{AuxBufferInfo, AuxInfoDsp};
use super::command::{AuxCommand, CaptureCommand};
use super::memory::DspMemory;
use super::processor::Frame;

fn read_pod<T: bytemuck::Pod>(memory: &dyn DspMemory, address: u64) -> T {
    let mut value = T::zeroed();
    memory.read(address, bytemuck::bytes_of_mut(&mut value));
    value
}

fn reset_aux_info(memory: &mut dyn DspMemory, address: u64) {
    if address == 0 {
        tracing::error!("aux info address is 0");
        return;
    }
    let mut info: AuxInfoDsp = read_pod(memory, address);
    info.read_offset = 0;
    info.write_offset = 0;
    info.total_sample_count = 0;
    memory.write(address, bytemuck::bytes_of(&info));
}

/// Writes `input` into the ring at `buffer` starting `offset` samples past `start`, wrapping at
/// `count_max`.
fn write_ring(memory: &mut dyn DspMemory, buffer: u64, count_max: u32, start: u32, input: &[i32]) {
    let mut target = start;
    let mut position = 0usize;
    while position < input.len() {
        let chunk = ((count_max - target) as usize).min(input.len() - position);
        if chunk > 0 {
            let address = buffer.wrapping_add(target as u64 * 4);
            memory.write(
                address,
                bytemuck::cast_slice(&input[position..position + chunk]),
            );
        }
        target = (target + chunk as u32) % count_max;
        position += chunk;
    }
}

fn write_aux_buffer(
    memory: &mut dyn DspMemory,
    send_info: u64,
    send_buffer: u64,
    count_max: u32,
    input: &[i32],
    write_offset: u32,
    update_count: u32,
) -> u32 {
    let write_count = input.len() as u32;
    if write_count > count_max {
        tracing::error!(
            write_count,
            count_max,
            "aux write count exceeds the ring size"
        );
        return 0;
    }
    if send_info == 0 || input.is_empty() || send_buffer == 0 || count_max == 0 {
        return 0;
    }
    let mut info: AuxInfoDsp = read_pod(memory, send_info);
    let target = info.write_offset.wrapping_add(write_offset);
    if target > count_max {
        return 0;
    }
    write_ring(memory, send_buffer, count_max, target, input);
    if update_count != 0 {
        info.write_offset = (info.write_offset.wrapping_add(update_count)) % count_max;
    }
    memory.write(send_info, bytemuck::bytes_of(&info));
    write_count
}

fn read_aux_buffer(
    memory: &mut dyn DspMemory,
    return_info: u64,
    return_buffer: u64,
    count_max: u32,
    output: &mut [i32],
    read_offset: u32,
    update_count: u32,
) -> u32 {
    let read_count = output.len() as u32;
    if count_max == 0 {
        return 0;
    }
    if read_count > count_max {
        tracing::error!(
            read_count,
            count_max,
            "aux read count exceeds the ring size"
        );
        return 0;
    }
    if return_info == 0 || output.is_empty() || return_buffer == 0 {
        return 0;
    }
    let mut info: AuxInfoDsp = read_pod(memory, return_info);
    let mut target = info.read_offset.wrapping_add(read_offset);
    if target > count_max {
        return 0;
    }
    let mut position = 0usize;
    while position < output.len() {
        let chunk = ((count_max - target) as usize).min(output.len() - position);
        if chunk > 0 {
            let address = return_buffer.wrapping_add(target as u64 * 4);
            memory.read(
                address,
                bytemuck::cast_slice_mut(&mut output[position..position + chunk]),
            );
        }
        target = (target + chunk as u32) % count_max;
        position += chunk;
    }
    if update_count != 0 {
        info.read_offset = (info.read_offset.wrapping_add(update_count)) % count_max;
    }
    memory.write(return_info, bytemuck::bytes_of(&info));
    read_count
}

pub(super) fn aux(command: &AuxCommand, frame: &mut Frame<'_>, memory: &mut dyn DspMemory) {
    let (Some(input), Some(output)) = (frame.range(command.input), frame.range(command.output))
    else {
        return;
    };
    if !command.effect_enabled {
        reset_aux_info(memory, command.send_buffer_info);
        reset_aux_info(memory, command.return_buffer_info);
        frame.mix.copy_within(input, output.start);
        return;
    }
    write_aux_buffer(
        memory,
        command.send_buffer_info,
        command.send_buffer,
        command.count_max,
        &frame.mix[input],
        command.write_offset,
        command.update_count,
    );
    let output = &mut frame.mix[output];
    let read = read_aux_buffer(
        memory,
        command.return_buffer_info,
        command.return_buffer,
        command.count_max,
        output,
        command.write_offset,
        command.update_count,
    );
    output[read as usize..].fill(0);
}

fn reset_capture_info(memory: &mut dyn DspMemory, address: u64) {
    if address == 0 {
        tracing::error!("capture info address is 0");
        return;
    }
    memory.write_u32(address + offset_of!(AuxInfoDsp, read_offset) as u64, 0);
    memory.write_u32(address + offset_of!(AuxInfoDsp, write_offset) as u64, 0);
    memory.write_u32(
        address + offset_of!(AuxInfoDsp, total_sample_count) as u64,
        0,
    );
}

fn write_capture_buffer(
    memory: &mut dyn DspMemory,
    command: &CaptureCommand,
    input: &[i32],
) -> u32 {
    let (count_max, update_count) = (command.count_max, command.update_count);
    let write_count = input.len() as u32;
    if write_count > count_max {
        tracing::error!(
            write_count,
            count_max,
            "capture write count exceeds the ring size"
        );
        return 0;
    }
    if command.send_buffer_info == 0
        || input.is_empty()
        || command.send_buffer == 0
        || count_max == 0
    {
        return 0;
    }
    let mut info: AuxBufferInfo = read_pod(memory, command.send_buffer_info);
    let target = info
        .dsp_info
        .write_offset
        .wrapping_add(command.write_offset);
    if target > count_max || write_count == 0 {
        return 0;
    }
    write_ring(memory, command.send_buffer, count_max, target, input);
    if update_count != 0 {
        let (dsp, cpu) = (&mut info.dsp_info, &info.cpu_info);
        let count_diff = dsp.total_sample_count.wrapping_sub(cpu.total_sample_count);
        if count_diff >= count_max {
            let mut lost = dsp.lost_sample_count.wrapping_add(update_count);
            if lost.wrapping_sub(cpu.lost_sample_count)
                < dsp.lost_sample_count.wrapping_sub(cpu.lost_sample_count)
            {
                lost = cpu.lost_sample_count.wrapping_sub(1);
            }
            dsp.lost_sample_count = lost;
        }
        dsp.write_offset = dsp
            .write_offset
            .wrapping_add(update_count)
            .wrapping_add(count_max)
            % count_max;
        let mut total = dsp.total_sample_count.wrapping_add(update_count);
        if total.wrapping_sub(cpu.total_sample_count) < count_diff {
            total = cpu.total_sample_count.wrapping_sub(1);
        }
        dsp.total_sample_count = total;
    }
    memory.write(command.send_buffer_info, bytemuck::bytes_of(&info));
    write_count
}

pub(super) fn capture(command: &CaptureCommand, frame: &Frame<'_>, memory: &mut dyn DspMemory) {
    if !command.effect_enabled {
        reset_capture_info(memory, command.send_buffer_info);
        return;
    }
    if let Some(input) = frame.range(command.input) {
        write_capture_buffer(memory, command, &frame.mix[input]);
    }
}

const _: () = assert!(size_of::<AuxBufferInfo>() == 2 * size_of::<AuxInfoDsp>());
