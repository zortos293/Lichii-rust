use std::sync::Arc;

use lichii_common::{DAddr, GpuVAddr};
use lichii_memory::DeviceAddressSpace;

use super::*;
use crate::backend::{
    ChannelId, ComputeContext, ConstBufferBinding, DrawTexture, FramebufferConfig, Query,
    QueryKind, Rasterizer, ReleaseSize, RootTables,
};
use crate::engines::maxwell_3d::DirtyState;
use crate::engines::maxwell_3d::regs::{
    LaunchDmaLayout, PrimitiveTopology, RenderEnableMode, RenderEnableOverride,
    ReportSemaphoreCompare,
};
use crate::engines::testing::Harness;
use crate::gmmu::Gmmu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Drawn {
    topology: PrimitiveTopology,
    indexed: bool,
    first: u32,
    count: u32,
    base_vertex: i32,
    base_instance: u32,
    instance_count: u32,
}

#[derive(Debug, PartialEq)]
enum Event {
    Draw {
        draw: Drawn,
        inline: Option<Vec<u32>>,
        dirty: Vec<u8>,
        const_buffers: Box<GraphicsConstBuffers>,
        root_tables: Box<RootTables>,
    },
    Clear {
        layers: u32,
        surface: u32,
    },
    Indirect(IndirectDraw),
    DrawTexture(DrawTexture),
    Query(Query),
    Fence(FenceAction),
    InlineWrite(GpuVAddr, u64),
}

/// A condition result the renderer holds until a fence passes or its range is flushed.
struct Staged {
    daddr: DAddr,
    value: ReportSemaphoreCompare,
    until_fence: bool,
}

#[derive(Default)]
struct Renderer {
    events: Vec<Event>,
    device: Option<Arc<DeviceAddressSpace>>,
    staged: Vec<Staged>,
}

impl Renderer {
    fn write_back(&mut self, release: impl Fn(&Staged) -> bool) {
        let device = self.device.as_ref().unwrap();
        self.staged.retain(|staged| {
            if !release(staged) {
                return true;
            }
            device.write(staged.daddr, staged.value).unwrap();
            false
        });
    }
}

fn consume(dirty: &mut DirtyState) -> Vec<u8> {
    (0..=u8::MAX).filter(|&flag| dirty.take(flag)).collect()
}

impl Rasterizer for Renderer {
    fn draw(&mut self, ctx: &mut GraphicsContext<'_>, draw: &DrawParams<'_>) {
        let inline = draw.inline_indices.map(|indices| match indices {
            InlineIndices::U8(i) => i.iter().map(|&x| u32::from(x)).collect(),
            InlineIndices::U16(i) => i.iter().map(|&x| u32::from(x)).collect(),
            InlineIndices::U32(i) => i.to_vec(),
        });
        self.events.push(Event::Draw {
            draw: Drawn {
                topology: draw.topology,
                indexed: draw.indexed,
                first: draw.first,
                count: draw.count,
                base_vertex: draw.base_vertex,
                base_instance: draw.base_instance,
                instance_count: draw.instance_count,
            },
            inline,
            dirty: consume(ctx.dirty),
            const_buffers: Box::new(*ctx.const_buffers),
            root_tables: Box::new(*ctx.root_tables),
        });
    }
    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, draw: &IndirectDraw) {
        self.events.push(Event::Indirect(*draw));
    }
    fn clear(&mut self, ctx: &mut GraphicsContext<'_>, layer_count: u32) {
        self.events.push(Event::Clear {
            layers: layer_count,
            surface: ctx.regs.clear_surface.0,
        });
    }
    fn draw_texture(&mut self, _ctx: &mut GraphicsContext<'_>, draw: &DrawTexture) {
        self.events.push(Event::DrawTexture(*draw));
    }
    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}
    fn inline_written(&mut self, _: ChannelId, _: &Gmmu, address: GpuVAddr, size: u64) {
        self.events.push(Event::InlineWrite(address, size));
    }
    fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, query: &Query) {
        self.events.push(Event::Query(*query));
    }
    fn signal_fence(&mut self, action: FenceAction) {
        self.events.push(Event::Fence(action));
    }
    fn release_fences(&mut self, wait: bool) {
        if wait {
            self.write_back(|staged| staged.until_fence);
        }
    }
    fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}
    fn flush_region(&mut self, addr: DAddr, size: u64) {
        self.write_back(|staged| {
            !staged.until_fence && (addr..addr + size).contains(&staged.daddr)
        });
    }
    fn wait_for_idle(&mut self) {}
    fn present(&mut self, _framebuffer: &FramebufferConfig) {}
}

struct Bench {
    h: Harness,
    renderer: Renderer,
    engine: Maxwell3d,
}

impl Bench {
    fn new() -> Self {
        let h = Harness::new();
        let renderer = Renderer {
            device: Some(h.memory.device().clone()),
            ..Renderer::default()
        };
        Self {
            h,
            renderer,
            engine: Maxwell3d::new(0xc797),
        }
    }

    fn run(&mut self, f: impl FnOnce(&mut Maxwell3d, &mut EngineContext<'_>)) {
        let mut ctx = EngineContext {
            channel: ChannelId(0),
            memory: &self.h.memory,
            rasterizer: &mut self.renderer,
            syncpoints: &self.h.syncpoints,
            gob_layout: self.h.gob_layout,
            qmd_scheduler: &mut self.h.qmd_scheduler,
            written_pages: &self.h.written_pages,
            activity: &crate::engines::testing::ACTIVITY,
        };
        f(&mut self.engine, &mut ctx);
    }

    fn call(&mut self, method: u32, argument: u32) {
        self.run(|engine, ctx| engine.call_method(ctx, method, argument));
    }

    fn multi(&mut self, method: u32, arguments: &[u32]) {
        self.run(|engine, ctx| engine.call_multi(ctx, method, arguments));
    }

    fn flush(&mut self) {
        self.run(|engine, ctx| engine.flush(ctx));
    }

    fn begin(&mut self, topology: PrimitiveTopology, subsequent: bool) {
        self.call(draw::DRAW_BEGIN, topology.0 | u32::from(subsequent) << 26);
    }

    fn end(&mut self) {
        self.call(draw::DRAW_END, 0);
    }

    fn take(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.renderer.events)
    }

    fn draws(&mut self) -> Vec<Drawn> {
        self.take()
            .into_iter()
            .filter_map(|event| match event {
                Event::Draw { draw, .. } => Some(draw),
                _ => None,
            })
            .collect()
    }
}

fn arrays(first: u32, count: u32, base_instance: u32, instance_count: u32) -> Drawn {
    Drawn {
        topology: PrimitiveTopology::TRIANGLES,
        indexed: false,
        first,
        count,
        base_vertex: 0,
        base_instance,
        instance_count,
    }
}

const DEPTH_TEST_ENABLE: u32 = reg_index!(Regs, depth_test_enable);
const CB_SIZE: u32 = reg_index!(Regs, const_buffer.size);
const GLOBAL_BASE_VERTEX: u32 = reg_index!(Regs, global_base_vertex_index);
const GLOBAL_BASE_INSTANCE: u32 = reg_index!(Regs, global_base_instance_index);

fn select_const_buffer(bench: &mut Bench, address: u64, size: u32, offset: u32) {
    bench.call(CB_SIZE, size);
    bench.call(CB_SIZE + 1, (address >> 32) as u32);
    bench.call(CB_SIZE + 2, address as u32);
    bench.call(CB_SIZE + 3, offset);
}

#[test]
fn non_indexed_draw() {
    let mut b = Bench::new();
    b.call(draw::VERTEX_BUFFER_FIRST, 3);
    b.call(draw::VERTEX_BUFFER_COUNT, 6);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    assert!(b.take().is_empty(), "held until it cannot gain instances");
    b.flush();
    assert_eq!(b.draws(), [arrays(3, 6, 0, 1)]);
}

#[test]
fn indexed_draw_uses_the_index_range_and_global_bases() {
    let mut b = Bench::new();
    b.call(GLOBAL_BASE_VERTEX, 5);
    b.call(GLOBAL_BASE_INSTANCE, 2);
    b.call(draw::INDEX_BUFFER_FIRST, 1);
    b.call(draw::INDEX_BUFFER_COUNT, 9);
    b.call(draw::VERTEX_BUFFER_COUNT, 4);
    b.begin(PrimitiveTopology::TRIANGLE_STRIP, false);
    b.end();
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.call(DEPTH_TEST_ENABLE, 1);
    let indexed = Drawn {
        topology: PrimitiveTopology::TRIANGLE_STRIP,
        indexed: true,
        first: 1,
        count: 9,
        base_vertex: 5,
        base_instance: 2,
        instance_count: 1,
    };
    assert_eq!(b.draws(), [indexed, arrays(0, 4, 2, 1)]);
}

#[test]
fn instance_loop_reaches_the_renderer_as_one_draw() {
    let mut b = Bench::new();
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.call(DEPTH_TEST_ENABLE, 1);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    for _ in 0..3 {
        b.begin(PrimitiveTopology::TRIANGLES, true);
        b.call(DEPTH_TEST_ENABLE, 1);
        b.call(draw::VERTEX_BUFFER_COUNT, 3);
        b.end();
    }
    assert!(b.take().is_empty());
    b.call(DEPTH_TEST_ENABLE, 0);
    assert_eq!(b.draws(), [arrays(0, 3, 0, 4)]);

    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.flush();
    assert_eq!(b.draws(), [arrays(0, 3, 0, 1), arrays(0, 3, 0, 1)]);
}

#[test]
fn instances_with_another_vertex_range_draw_separately() {
    let mut b = Bench::new();
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.begin(PrimitiveTopology::TRIANGLES, true);
    b.call(draw::VERTEX_BUFFER_FIRST, 3);
    b.end();
    b.begin(PrimitiveTopology::TRIANGLES, true);
    b.end();
    b.flush();
    assert_eq!(b.draws(), [arrays(0, 3, 0, 1), arrays(3, 3, 1, 2)]);
}

#[test]
fn indirect_draw_follows_the_pending_draw() {
    let mut b = Bench::new();
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    let indirect = IndirectDraw {
        topology: PrimitiveTopology::TRIANGLES,
        indexed: false,
        buffer: 0x7000,
        stride: 16,
        max_draws: 4,
        count_buffer: Some(0x7100),
    };
    b.run(|engine, ctx| engine.draw_indirect(ctx, &indirect));
    let events = b.take();
    assert!(matches!(events[0], Event::Draw { .. }));
    assert_eq!(events[1], Event::Indirect(indirect));
}

#[test]
fn small_index_and_vertex_array_methods_count_instances() {
    let mut b = Bench::new();
    let small = 6 << 16 | PrimitiveTopology::TRIANGLES.0 << 28;
    b.call(draw::INDEX_BUFFER16_FIRST, small);
    b.call(draw::INDEX_BUFFER16_SUBSEQUENT, small);
    b.call(draw::INDEX_BUFFER16_SUBSEQUENT, small);
    let array = 2 | 4 << 16 | PrimitiveTopology::TRIANGLES.0 << 28;
    b.call(draw::VERTEX_ARRAY_INSTANCE_FIRST, array);
    b.call(draw::VERTEX_ARRAY_INSTANCE_SUBSEQUENT, array);
    b.flush();
    let indexed = Drawn {
        indexed: true,
        ..arrays(0, 6, 0, 3)
    };
    assert_eq!(b.draws(), [indexed, arrays(2, 4, 0, 2)]);
}

#[test]
fn inline_16_bit_indices_draw_from_the_reused_buffer() {
    let mut b = Bench::new();
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.call(draw::INLINE_INDEX_2X16_COUNT, 5);
    b.multi(draw::INLINE_INDEX_2X16, &[0x0001_0000, 0x0003_0002]);
    b.call(draw::INLINE_INDEX_2X16, 0x0000_0004);
    b.end();
    let events = b.take();
    let [Event::Draw { draw, inline, .. }] = events.as_slice() else {
        panic!("one draw expected, got {events:?}");
    };
    assert_eq!(
        *draw,
        Drawn {
            indexed: true,
            ..arrays(0, 6, 0, 1)
        }
    );
    assert_eq!(inline.as_deref(), Some(&[0, 1, 2, 3, 4, 0][..]));

    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.multi(draw::INLINE_INDEX_4X8, &[0x0302_0100]);
    b.end();
    let events = b.take();
    let [Event::Draw { inline, dirty, .. }] = events.as_slice() else {
        panic!("one draw expected, got {events:?}");
    };
    assert_eq!(inline.as_deref(), Some(&[0, 1, 2, 3][..]));
    assert!(dirty.contains(&dirty::INDEX_BUFFER));
}

#[test]
fn const_buffer_data_lands_in_memory_and_advances_the_offset() {
    let mut b = Bench::new();
    select_const_buffer(&mut b, 0x2_0000, 0x100, 0x10);
    b.take();
    b.multi(CB_DATA_FIRST, &[1, 2, 3]);
    b.call(CB_DATA_FIRST + 5, 4);
    let mut words = [0u32; 4];
    b.h.memory
        .read_block(0x2_0010, bytemuck::cast_slice_mut(&mut words))
        .unwrap();
    assert_eq!(words, [1, 2, 3, 4]);
    assert_eq!(b.engine.state.regs.const_buffer.offset, 0x20);
    assert_eq!(
        b.take(),
        [
            Event::InlineWrite(0x2_0010, 12),
            Event::InlineWrite(0x2_001c, 4)
        ]
    );

    b.call(CB_SIZE + 3, 0xf8);
    b.multi(CB_DATA_FIRST, &[7, 8, 9]);
    assert_eq!(b.engine.state.regs.const_buffer.offset, 0x104);
    assert_eq!(b.take(), [Event::InlineWrite(0x2_00f8, 8)]);
    assert_eq!(b.h.memory.read::<u32>(0x2_0100).unwrap(), 0);
}

#[test]
fn const_buffer_bind_is_visible_to_the_next_draw_only() {
    let mut b = Bench::new();
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    select_const_buffer(&mut b, 0x4_0000, 0x400, 0);
    let fragment = 4;
    let bind = BIND_GROUP_FIRST + fragment as u32 * BIND_GROUP_STRIDE;
    b.call(bind, 1 | 3 << 4);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.flush();
    let events = b.take();
    let [
        Event::Draw {
            const_buffers: before,
            ..
        },
        Event::Draw {
            const_buffers: after,
            dirty,
            ..
        },
    ] = events.as_slice()
    else {
        panic!("two draws expected, got {events:?}");
    };
    assert!(!before[fragment][3].enabled);
    assert_eq!(
        after[fragment][3],
        ConstBufferBinding {
            address: 0x4_0000,
            size: 0x400,
            enabled: true
        }
    );
    assert!(dirty.contains(&(dirty::CONST_BUFFER0 + fragment as u8)));
}

#[test]
fn clear_runs_after_the_pending_draw_unless_rendering_is_disabled() {
    let mut b = Bench::new();
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    let surface = 0x3f | 2 << 10;
    b.call(CLEAR_SURFACE, surface);
    let events = b.take();
    assert!(matches!(events[0], Event::Draw { .. }));
    assert_eq!(events[1], Event::Clear { layers: 1, surface });

    b.call(
        reg_index!(Regs, render_enable_override),
        RenderEnableOverride::NEVER_RENDER.0,
    );
    b.call(CLEAR_SURFACE, surface);
    assert!(b.take().is_empty());
}

#[test]
fn draw_texture_decodes_fixed_point_after_the_pending_draw() {
    let mut b = Bench::new();
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    let base = reg_index!(Regs, draw_texture);
    let words = [
        16 << 12,
        8 << 12,
        64 << 12,
        32 << 12,
        1 << 31,
        0,
        0,
        2,
        5,
        3,
        1 << 12,
        2 << 12,
    ];
    for (offset, word) in words.into_iter().enumerate() {
        b.call(base + offset as u32, word);
    }
    let events = b.take();
    assert!(matches!(events[0], Event::Draw { .. }));
    assert_eq!(
        events[1],
        Event::DrawTexture(DrawTexture {
            dst_x0: 16.0,
            dst_y0: 8.0,
            dst_x1: 80.0,
            dst_y1: 40.0,
            src_x0: 1.0,
            src_y0: 2.0,
            src_x1: 33.0,
            src_y1: 66.0,
            texture: 3,
            sampler: 5,
        })
    );
    b.call(base + 11, 2 << 12);
    assert!(matches!(b.take()[..], [Event::DrawTexture(_)]));

    b.call(reg_index!(Regs, surface_clip) + 1, 100 << 16);
    b.call(reg_index!(Regs, window_origin), 1);
    b.call(base + 11, 2 << 12);
    let [Event::DrawTexture(flipped)] = b.take()[..] else {
        panic!("one textured draw expected");
    };
    assert_eq!((flipped.dst_y0, flipped.dst_y1), (92.0, 124.0));
}

#[test]
fn render_condition_skips_draws_until_the_sequences_match() {
    let mut b = Bench::new();
    let compare = ReportSemaphoreCompare {
        initial_sequence: 1,
        initial_mode: 2,
        unknown1: 0,
        unknown2: 0,
        current_sequence: 5,
        current_mode: 2,
    };
    b.h.memory.write(0x6000, compare).unwrap();
    let render_enable = reg_index!(Regs, render_enable);
    b.call(render_enable + 1, 0x6000);
    b.call(RENDER_ENABLE_MODE, RenderEnableMode::IF_EQUAL.0);
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.flush();
    assert!(b.draws().is_empty());
    b.call(RENDER_ENABLE_MODE, RenderEnableMode::IF_NOT_EQUAL.0);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.flush();
    assert_eq!(b.draws(), [arrays(0, 3, 0, 1)]);
}

fn condition(current_sequence: u32) -> ReportSemaphoreCompare {
    ReportSemaphoreCompare {
        initial_sequence: 1,
        initial_mode: 2,
        unknown1: 0,
        unknown2: 0,
        current_sequence,
        current_mode: 2,
    }
}

fn staged_condition_draws(until_fence: bool) -> Vec<Drawn> {
    let mut b = Bench::new();
    b.h.memory.write(0x6000, condition(5)).unwrap();
    b.renderer.staged.push(Staged {
        daddr: b.h.memory.translate(0x6000).unwrap(),
        value: condition(1),
        until_fence,
    });
    b.call(reg_index!(Regs, render_enable) + 1, 0x6000);
    b.call(RENDER_ENABLE_MODE, RenderEnableMode::IF_EQUAL.0);
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.flush();
    assert!(b.renderer.staged.is_empty());
    b.draws()
}

#[test]
fn render_condition_waits_for_results_the_renderer_still_holds() {
    assert_eq!(staged_condition_draws(true), [arrays(0, 3, 0, 1)]);
}

#[test]
fn render_condition_flushes_results_cached_by_the_renderer() {
    assert_eq!(staged_condition_draws(false), [arrays(0, 3, 0, 1)]);
}

#[test]
fn indirect_draw_applies_the_topology_override() {
    let mut b = Bench::new();
    let indirect = IndirectDraw {
        topology: PrimitiveTopology::TRIANGLES,
        indexed: true,
        buffer: 0x7000,
        stride: 20,
        max_draws: 2,
        count_buffer: None,
    };
    let topology_of = |b: &mut Bench| {
        b.run(|engine, ctx| engine.draw_indirect(ctx, &indirect));
        match b.take().as_slice() {
            [Event::Indirect(draw)] => draw.topology,
            events => panic!("one indirect draw expected, got {events:?}"),
        }
    };
    b.call(reg_index!(Regs, topology_override), 0xe);
    assert_eq!(topology_of(&mut b), PrimitiveTopology::TRIANGLES);
    b.call(reg_index!(Regs, primitive_topology_control), 1);
    assert_eq!(topology_of(&mut b), PrimitiveTopology::PATCHES);
    b.call(reg_index!(Regs, topology_override), 0);
    assert_eq!(topology_of(&mut b), PrimitiveTopology::TRIANGLES);
}

#[test]
fn inline_indices_keep_their_buffer_and_draw_every_index() {
    let mut b = Bench::new();
    let capacity = b.engine.draw.inline_indices.capacity();
    assert!(capacity >= draw::INLINE_INDEX_CAPACITY);

    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.multi(draw::INLINE_INDEX_4X8, &vec![0x0302_0100; 1025]);
    b.end();
    let events = b.take();
    let [Event::Draw { draw, .. }] = events.as_slice() else {
        panic!("one draw expected, got {events:?}");
    };
    assert_eq!(draw.count, 4100);
    assert_eq!(b.engine.draw.inline_indices.capacity(), capacity);

    b.begin(PrimitiveTopology::TRIANGLES, false);
    let words = vec![0x0001_0000; draw::INLINE_INDEX_CAPACITY / 2];
    let large = |b: &mut Bench| {
        b.begin(PrimitiveTopology::TRIANGLES, false);
        b.multi(draw::INLINE_INDEX_2X16, &words);
        b.multi(draw::INLINE_INDEX_2X16, &words);
        b.multi(draw::DRAW_INLINE_INDEX, &[7; 16]);
        b.end();
        let events = b.take();
        let [Event::Draw { draw, .. }] = events.as_slice() else {
            panic!("one draw expected, got {events:?}");
        };
        assert_eq!(draw.count as usize, 2 * draw::INLINE_INDEX_CAPACITY + 16);
        b.engine.draw.inline_indices.capacity()
    };
    let grown = large(&mut b);
    assert!(grown > capacity);
    assert_eq!(large(&mut b), grown);
}

#[test]
fn query_release_and_report() {
    let mut b = Bench::new();
    let semaphore = reg_index!(Regs, report_semaphore);
    b.call(semaphore + 1, 0x3000);
    b.call(semaphore + 2, 0x1234);
    b.call(REPORT_SEMAPHORE_QUERY, 1 << 28);
    b.call(REPORT_SEMAPHORE_QUERY, 2 | 2 << 23);
    b.call(REPORT_SEMAPHORE_QUERY, 1);
    assert_eq!(
        b.take(),
        [
            Event::Query(Query {
                address: 0x3000,
                kind: QueryKind::Payload,
                payload: 0x1234,
                size: ReleaseSize::Four,
                after_work: true,
            }),
            Event::Query(Query {
                address: 0x3000,
                kind: QueryKind::SamplesPassed,
                payload: 0x1234,
                size: ReleaseSize::Sixteen,
                after_work: false,
            }),
        ]
    );
}

#[test]
fn same_value_writes_set_no_dirty_flags_and_keep_the_pending_draw() {
    let mut b = Bench::new();
    consume(&mut b.engine.state.dirty);
    let stream2_low = reg_index!(Regs, vertex_streams) + 2 * 4 + 2;
    b.call(stream2_low, 0);
    assert!(consume(&mut b.engine.state.dirty).is_empty());
    b.call(stream2_low, 0x1000);
    assert_eq!(
        consume(&mut b.engine.state.dirty),
        [dirty::VERTEX_BUFFERS, dirty::VERTEX_BUFFER0 + 2]
    );
    b.call(draw::VERTEX_BUFFER_COUNT, 3);
    b.begin(PrimitiveTopology::TRIANGLES, false);
    b.end();
    b.call(stream2_low, 0x1000);
    assert!(b.take().is_empty());
    assert!(!b.engine.state.dirty.is_set(dirty::VERTEX_BUFFER0 + 2));
}

#[test]
fn root_tables_load_at_the_selected_offset() {
    let mut b = Bench::new();
    b.call(ROOT_TABLE_SELECT, 2 | 8 << 8);
    b.multi(ROOT_TABLE_LOAD, &[0xa, 0xb]);
    b.call(ROOT_TABLE_LOAD, 0xc);
    assert_eq!(b.engine.root_tables.tables[2][1..6], [0, 0xa, 0xb, 0xc, 0]);
    assert!(b.engine.state.dirty.take(dirty::ROOT_TABLES));
    b.call(ROOT_TABLE_SELECT, 2 | 8 << 8);
    b.call(ROOT_TABLE_LOAD, 0xa);
    assert!(!b.engine.state.dirty.is_set(dirty::ROOT_TABLES));
}

#[test]
fn sync_point_and_falcon_call() {
    let mut b = Bench::new();
    b.call(SYNC_INFO, 7);
    assert_eq!(b.h.syncpoints.host().value(7), Ok(1));
    assert_eq!(b.take(), [Event::Fence(FenceAction::IncrementSyncpoint(7))]);
    b.call(FALCON4, 0);
    assert_eq!(b.engine.state.regs.shadow_scratch[0], 1);
}

#[test]
fn inline_to_memory_upload() {
    let mut b = Bench::new();
    let upload = reg_index!(Regs, upload);
    b.call(upload, 8);
    b.call(upload + 1, 1);
    b.call(upload + 3, 0x5000);
    b.call(LAUNCH_DMA, LaunchDmaLayout::PITCH.0);
    b.multi(INLINE_DATA, &[0x1111_1111, 0x2222_2222]);
    assert_eq!(
        b.h.memory.read::<[u32; 2]>(0x5000).unwrap(),
        [0x1111_1111, 0x2222_2222]
    );
    assert_eq!(b.take(), [Event::InlineWrite(0x5000, 8)]);
}
