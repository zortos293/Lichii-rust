// Golden RequestUpdate fixtures for lichii-audio, produced by the C++ Lichii reference.
//
// Builds synthetic RequestUpdate input buffers with the reference's own parameter structs, runs
// them through a copy of System::Initialize/System::Update that skips the kernel, and writes
// one fixture per case: the open parameters, then every update's input, output, performance
// buffer, result code and a digest of the renderer state it leaves behind. The Rust test in ../request_update.rs replays them.
//
// Build against the reference (https://github.com/zortos293/Lichii, branch nx2), with a
// logging.h that defines the LOG_* macros as no-ops and the reference renderer sources that
// info_updater.cpp pulls in (PoolMapper with the kernel calls stubbed out):
//   g++ -std=c++20 -I shim -I <lichii>/src golden.cpp <renderer objects> -o golden
//   ./golden <output dir>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// The state digest reads renderer members the reference keeps private.
#define private public
#define protected public

#include "audio_core/common/audio_renderer_parameter.h"
#include "audio_core/common/common.h"
#include "audio_core/common/feature_support.h"
#include "audio_core/common/workbuffer_allocator.h"
#include "audio_core/renderer/behavior/behavior_info.h"
#include "audio_core/renderer/behavior/info_updater.h"
#include "audio_core/renderer/effect/aux_.h"
#include "audio_core/renderer/effect/biquad_filter.h"
#include "audio_core/renderer/effect/buffer_mixer.h"
#include "audio_core/renderer/effect/delay.h"
#include "audio_core/renderer/effect/effect_context.h"
#include "audio_core/renderer/effect/effect_result_state.h"
#include "audio_core/renderer/effect/light_limiter.h"
#include "audio_core/renderer/memory/memory_pool_info.h"
#include "audio_core/renderer/memory/pool_mapper.h"
#include "audio_core/renderer/mix/mix_context.h"
#include "audio_core/renderer/nodes/edge_matrix.h"
#include "audio_core/renderer/nodes/node_states.h"
#include "audio_core/renderer/performance/performance_manager.h"
#include "audio_core/renderer/sink/sink_context.h"
#include "audio_core/renderer/splitter/splitter_context.h"
#include "audio_core/renderer/upsampler/upsampler_info.h"
#include "audio_core/renderer/upsampler/upsampler_manager.h"
#include "audio_core/renderer/voice/voice_context.h"
#include "audio_core/renderer/command/commands.h"
#include "common/alignment.h"

using namespace AudioCore;
using namespace AudioCore::Renderer;

static u64 SplitterSize(const BehaviorInfo& behavior, const AudioRendererParameterInternal& params) {
    u64 size{0};
    if (!behavior.IsSplitterSupported()) return size;
    size += params.splitter_destinations * sizeof(SplitterDestinationData) +
            params.splitter_infos * sizeof(SplitterInfo);
    if (behavior.IsSplitterBugFixed())
        size += Common::AlignUp(params.splitter_destinations * sizeof(u32), 0x10);
    return size;
}

static u64 CmdSize(const BehaviorInfo& behavior, const AudioRendererParameterInternal& params) {
    u64 size{0};
    size += params.effects * sizeof(EffectInfoBase);
    u64 voice_size{0};
    if (behavior.IsWaveBufferVer2Supported()) {
        voice_size = (std::max)((std::max)(sizeof(AdpcmDataSourceVersion2Command),
                                           sizeof(PcmInt16DataSourceVersion2Command)),
                                sizeof(PcmFloatDataSourceVersion2Command));
    } else {
        voice_size = (std::max)((std::max)(sizeof(AdpcmDataSourceVersion1Command),
                                           sizeof(PcmInt16DataSourceVersion1Command)),
                                sizeof(PcmFloatDataSourceVersion1Command));
    }
    voice_size += sizeof(BiquadFilterCommand) * MaxBiquadFilters;
    voice_size += sizeof(VolumeRampCommand);
    voice_size += sizeof(MixRampGroupedCommand);
    size += params.voices * (params.splitter_infos * sizeof(DepopPrepareCommand) + voice_size);
    size += sizeof(DepopForMixBuffersCommand) + (sizeof(MixCommand) * MaxMixBuffers) * MaxMixBuffers;
    size += sizeof(DepopForMixBuffersCommand) + sizeof(VolumeCommand) * MaxMixBuffers;
    size += params.splitter_destinations * sizeof(MixRampCommand) * MaxMixBuffers;
    size += params.sinks * (std::max)(sizeof(DeviceSinkCommand), sizeof(CircularBufferSinkCommand));
    size += (params.effects + params.voices + params.sinks + params.sub_mixes + 1 +
             PerformanceManager::MaxDetailEntries) *
            sizeof(PerformanceCommand);
    return size;
}

static u64 GetWorkBufferSize(const AudioRendererParameterInternal& params) {
    BehaviorInfo behavior;
    behavior.SetUserLibRevision(params.revision);
    u64 size{0};
    size += Common::AlignUp(params.mixes * sizeof(s32), 0x40);
    size += params.sub_mixes * MaxEffects * sizeof(s32);
    size += (params.sub_mixes + 1) * sizeof(MixInfo);
    size += params.voices * (sizeof(VoiceInfo) + sizeof(VoiceChannelResource) + sizeof(VoiceState));
    size += Common::AlignUp((params.sub_mixes + 1) * sizeof(MixInfo*), 0x10);
    size += Common::AlignUp(params.voices * sizeof(VoiceInfo*), 0x10);
    size += Common::AlignUp(((params.sinks + params.sub_mixes) * TargetSampleCount * sizeof(s32) +
                             params.sample_count * sizeof(s32)) *
                                (params.mixes + MaxChannels),
                            0x40);
    if (behavior.IsSplitterSupported()) {
        const auto node_size{NodeStates::GetWorkBufferSize(params.sub_mixes + 1)};
        const auto edge_size{EdgeMatrix::GetWorkBufferSize(params.sub_mixes + 1)};
        size += Common::AlignUp(node_size + edge_size, 0x10);
    }
    size += SplitterSize(behavior, params);
    size += (params.effects + params.voices * MaxWaveBuffers) * sizeof(MemoryPoolInfo);
    if (behavior.IsEffectInfoVersion2Supported()) size += params.effects * sizeof(EffectResultState);
    size += 0x50;
    size = Common::AlignUp(size, 0x40);
    size += (params.sinks + params.sub_mixes) * sizeof(UpsamplerInfo);
    size += params.effects * sizeof(EffectInfoBase);
    size += Common::AlignUp(params.voices * sizeof(VoiceState), 0x40);
    size += params.sinks * sizeof(SinkInfoBase);
    if (behavior.IsEffectInfoVersion2Supported()) size += params.effects * sizeof(EffectResultState);
    if (params.perf_frames > 0) {
        auto perf_size{
            PerformanceManager::GetRequiredBufferSizeForPerformanceMetricsPerFrame(behavior, params)};
        size += Common::AlignUp(perf_size * (params.perf_frames + 1) + 0xC0, 0x100);
    }
    if (behavior.IsVariadicCommandBufferSizeSupported())
        size += CmdSize(behavior, params) + (0x40 - 1) * 2;
    else
        size += 0x18000 + (0x40 - 1) * 2;
    return Common::AlignUp(size, 0x1000);
}

// System::Initialize and System::Update without the kernel, timing and DSP pieces.
struct Golden {
    bool active{};
    u32 sample_count{};
    s16 mix_buffer_count{};
    std::span<s32> samples_workbuffer{};
    std::span<s32> depop_buffer{};
    u32 memory_pool_count{};
    std::span<MemoryPoolInfo> memory_pool_workbuffer{};
    MemoryPoolInfo memory_pool_info{};
    std::span<u8> command_workbuffer{};
    UpsamplerManager* upsampler_manager{};
    std::span<UpsamplerInfo> upsampler_infos{};
    u32 upsampler_count{};
    VoiceContext voice_context{};
    MixContext mix_context{};
    EffectContext effect_context{};
    SinkContext sink_context{};
    SplitterContext splitter_context{};
    s32 voice_channels{};
    Kernel::KProcess* process_handle{};
    PerformanceManager performance_manager{};
    std::span<u8> performance_workbuffer{};
    std::unique_ptr<u8[]> workbuffer{};
    u64 workbuffer_size{};
    BehaviorInfo behavior{};
    u64 frames_elapsed{};

    Result Initialize(const AudioRendererParameterInternal& params, u64 transfer_memory_size) {
        if (!CheckValidRevision(params.revision)) return Service::Audio::ResultInvalidRevision;
        if (GetWorkBufferSize(params) > transfer_memory_size)
            return Service::Audio::ResultInsufficientBuffer;
        behavior.SetUserLibRevision(params.revision);
        sample_count = params.sample_count;
        mix_buffer_count = static_cast<s16>(params.mixes);
        voice_channels = MaxChannels;
        upsampler_count = params.sinks + params.sub_mixes;
        memory_pool_count = params.effects + params.voices * MaxWaveBuffers;
        workbuffer = std::make_unique<u8[]>(transfer_memory_size);
        std::memset(workbuffer.get(), 0, transfer_memory_size);
        workbuffer_size = transfer_memory_size;
        WorkbufferAllocator allocator({workbuffer.get(), workbuffer_size}, workbuffer_size);
        samples_workbuffer =
            allocator.Allocate<s32>((voice_channels + mix_buffer_count) * sample_count, 0x10);
        if (samples_workbuffer.empty()) return Service::Audio::ResultInsufficientBuffer;
        auto upsampler_workbuffer{allocator.Allocate<s32>(
            (voice_channels + mix_buffer_count) * TargetSampleCount * upsampler_count, 0x10)};
        if (upsampler_workbuffer.empty()) return Service::Audio::ResultInsufficientBuffer;
        depop_buffer = allocator.Allocate<s32>(
            Common::AlignUp(static_cast<u32>(mix_buffer_count), 0x40), 0x40);
        if (depop_buffer.empty()) return Service::Audio::ResultInsufficientBuffer;
        auto voice_infos{allocator.Allocate<VoiceInfo>(params.voices, 0x10)};
        for (auto& voice_info : voice_infos) std::construct_at<VoiceInfo>(&voice_info);
        if (voice_infos.empty()) return Service::Audio::ResultInsufficientBuffer;
        auto sorted_voice_infos{allocator.Allocate<VoiceInfo*>(params.voices, 0x10)};
        if (sorted_voice_infos.empty()) return Service::Audio::ResultInsufficientBuffer;
        std::memset(sorted_voice_infos.data(), 0, sorted_voice_infos.size_bytes());
        auto voice_channel_resources{allocator.Allocate<VoiceChannelResource>(params.voices, 0x10)};
        u32 i{0};
        for (auto& r : voice_channel_resources) std::construct_at<VoiceChannelResource>(&r, i++);
        if (voice_channel_resources.empty()) return Service::Audio::ResultInsufficientBuffer;
        auto voice_cpu_states{allocator.Allocate<VoiceState>(params.voices, 0x10)};
        if (voice_cpu_states.empty()) return Service::Audio::ResultInsufficientBuffer;
        for (auto& s : voice_cpu_states) s = {};
        auto mix_infos{allocator.Allocate<MixInfo>(params.sub_mixes + 1, 0x10)};
        if (mix_infos.empty()) return Service::Audio::ResultInsufficientBuffer;
        u32 effect_process_order_count{0};
        std::span<s32> effect_process_order_buffer{};
        if (params.effects > 0) {
            effect_process_order_count = params.effects * (params.sub_mixes + 1);
            effect_process_order_buffer = allocator.Allocate<s32>(effect_process_order_count, 0x10);
            if (effect_process_order_buffer.empty()) return Service::Audio::ResultInsufficientBuffer;
        }
        i = 0;
        for (auto& mix_info : mix_infos) {
            std::construct_at<MixInfo>(
                &mix_info, effect_process_order_buffer.subspan(i * params.effects, params.effects),
                params.effects, this->behavior);
            i++;
        }
        auto sorted_mix_infos{allocator.Allocate<MixInfo*>(params.sub_mixes + 1, 0x10)};
        if (sorted_mix_infos.empty()) return Service::Audio::ResultInsufficientBuffer;
        std::memset(sorted_mix_infos.data(), 0, sorted_mix_infos.size_bytes());
        if (behavior.IsSplitterSupported()) {
            u64 node_state_size{NodeStates::GetWorkBufferSize(params.sub_mixes + 1)};
            u64 edge_matrix_size{EdgeMatrix::GetWorkBufferSize(params.sub_mixes + 1)};
            auto node_states_workbuffer{allocator.Allocate<u8>(node_state_size, 1)};
            auto edge_matrix_workbuffer{allocator.Allocate<u8>(edge_matrix_size, 1)};
            if (node_states_workbuffer.empty() || edge_matrix_workbuffer.size() == 0)
                return Service::Audio::ResultInsufficientBuffer;
            mix_context.Initialize(sorted_mix_infos, mix_infos, params.sub_mixes + 1,
                                   effect_process_order_buffer, effect_process_order_count,
                                   node_states_workbuffer, node_state_size, edge_matrix_workbuffer,
                                   edge_matrix_size);
        } else {
            mix_context.Initialize(sorted_mix_infos, mix_infos, params.sub_mixes + 1,
                                   effect_process_order_buffer, effect_process_order_count, {}, 0,
                                   {}, 0);
        }
        upsampler_manager = allocator.Allocate<UpsamplerManager>(1, 0x10).data();
        if (upsampler_manager == nullptr) return Service::Audio::ResultInsufficientBuffer;
        memory_pool_workbuffer = allocator.Allocate<MemoryPoolInfo>(memory_pool_count, 0x10);
        for (auto& memory_pool : memory_pool_workbuffer)
            std::construct_at<MemoryPoolInfo>(&memory_pool, MemoryPoolInfo::Location::DSP);
        if (memory_pool_workbuffer.empty() && memory_pool_count > 0)
            return Service::Audio::ResultInsufficientBuffer;
        if (!splitter_context.Initialize(behavior, params, allocator))
            return Service::Audio::ResultInsufficientBuffer;
        std::span<EffectResultState> effect_result_states_cpu{};
        if (behavior.IsEffectInfoVersion2Supported() && params.effects > 0) {
            effect_result_states_cpu = allocator.Allocate<EffectResultState>(params.effects, 0x10);
            if (effect_result_states_cpu.empty()) return Service::Audio::ResultInsufficientBuffer;
            std::memset(effect_result_states_cpu.data(), 0, effect_result_states_cpu.size_bytes());
        }
        allocator.Align(0x40);
        upsampler_infos = allocator.Allocate<UpsamplerInfo>(upsampler_count, 0x40);
        for (auto& u : upsampler_infos) std::construct_at<UpsamplerInfo>(&u);
        std::construct_at<UpsamplerManager>(upsampler_manager, upsampler_count, upsampler_infos,
                                            upsampler_workbuffer);
        if (upsampler_infos.empty()) return Service::Audio::ResultInsufficientBuffer;
        auto effect_infos{allocator.Allocate<EffectInfoBase>(params.effects, 0x40)};
        for (auto& e : effect_infos) std::construct_at<EffectInfoBase>(&e);
        if (effect_infos.empty() && params.effects > 0)
            return Service::Audio::ResultInsufficientBuffer;
        std::span<EffectResultState> effect_result_states_dsp{};
        if (behavior.IsEffectInfoVersion2Supported() && params.effects > 0) {
            effect_result_states_dsp = allocator.Allocate<EffectResultState>(params.effects, 0x40);
            if (effect_result_states_dsp.empty()) return Service::Audio::ResultInsufficientBuffer;
            std::memset(effect_result_states_dsp.data(), 0, effect_result_states_dsp.size_bytes());
        }
        effect_context.Initialize(effect_infos, params.effects, effect_result_states_cpu,
                                  effect_result_states_dsp, effect_result_states_dsp.size());
        auto sinks{allocator.Allocate<SinkInfoBase>(params.sinks, 0x10)};
        for (auto& sink : sinks) std::construct_at<SinkInfoBase>(&sink);
        if (sinks.empty()) return Service::Audio::ResultInsufficientBuffer;
        sink_context.Initialize(sinks, params.sinks);
        auto voice_dsp_states{allocator.Allocate<VoiceState>(params.voices, 0x40)};
        if (voice_dsp_states.empty()) return Service::Audio::ResultInsufficientBuffer;
        for (auto& s : voice_dsp_states) s = {};
        voice_context.Initialize(sorted_voice_infos, voice_infos, voice_channel_resources,
                                 voice_cpu_states, voice_dsp_states, params.voices);
        if (params.perf_frames > 0) {
            const auto perf_workbuffer_size{
                PerformanceManager::GetRequiredBufferSizeForPerformanceMetricsPerFrame(behavior,
                                                                                       params) *
                    (params.perf_frames + 1) +
                0xC};
            performance_workbuffer = allocator.Allocate<u8>(perf_workbuffer_size, 0x40);
            if (performance_workbuffer.empty()) return Service::Audio::ResultInsufficientBuffer;
            std::memset(performance_workbuffer.data(), 0, performance_workbuffer.size_bytes());
            performance_manager.Initialize(performance_workbuffer,
                                           performance_workbuffer.size_bytes(), params, behavior,
                                           memory_pool_info);
        }
        allocator.Align(0x40);
        command_workbuffer = allocator.Allocate<u8>(allocator.GetRemainingSize(), 0x40);
        if (command_workbuffer.empty()) return Service::Audio::ResultInsufficientBuffer;
        return ResultSuccess;
    }

    Result Update(std::span<const u8> input, std::span<u8> performance, std::span<u8> output) {
        std::memset(output.data(), 0, output.size());
        InfoUpdater info_updater(input, output, process_handle, behavior);
        auto result{info_updater.UpdateBehaviorInfo(behavior)};
        if (result.IsError()) return result;
        result = info_updater.UpdateMemoryPools(memory_pool_workbuffer, memory_pool_count);
        if (result.IsError()) return result;
        result = info_updater.UpdateVoiceChannelResources(voice_context);
        if (result.IsError()) return result;
        result = info_updater.UpdateVoices(voice_context, memory_pool_workbuffer, memory_pool_count);
        if (result.IsError()) return result;
        result = info_updater.UpdateEffects(effect_context, active, memory_pool_workbuffer,
                                            memory_pool_count);
        if (result.IsError()) return result;
        if (behavior.IsSplitterSupported()) {
            result = info_updater.UpdateSplitterInfo(splitter_context);
            if (result.IsError()) return result;
        }
        result = info_updater.UpdateMixes(mix_context, mix_buffer_count, effect_context,
                                          splitter_context);
        if (result.IsError()) return result;
        result = info_updater.UpdateSinks(sink_context, memory_pool_workbuffer, memory_pool_count);
        if (result.IsError()) return result;
        PerformanceManager* perf_manager{nullptr};
        if (performance_manager.IsInitialized()) perf_manager = &performance_manager;
        result = info_updater.UpdatePerformanceBuffer(performance, performance.size_bytes(),
                                                      perf_manager);
        if (result.IsError()) return result;
        result = info_updater.UpdateErrorInfo(behavior);
        if (result.IsError()) return result;
        if (behavior.IsElapsedFrameCountSupported()) {
            result = info_updater.UpdateRendererInfo(frames_elapsed);
            if (result.IsError()) return result;
        }
        return info_updater.CheckConsumedSize();
    }
};

// State digest: the renderer state an update leaves behind, as little-endian words. The Rust
// test builds the same words from its own state.
struct Digest {
    std::vector<u32> words;
    void U(u32 v) { words.push_back(v); }
    void S(s32 v) { words.push_back(static_cast<u32>(v)); }
    void F(f32 v) { words.push_back(std::bit_cast<u32>(v)); }
    void Q(u64 v) {
        U(static_cast<u32>(v));
        U(static_cast<u32>(v >> 32));
    }
};

static void DigestState(Digest& d, Golden& g, const AudioRendererParameterInternal& params) {
    auto pool_index = [&](const MemoryPoolInfo* pool) -> u32 {
        if (pool == nullptr) return 0xFFFFFFFF;
        return static_cast<u32>(pool - g.memory_pool_workbuffer.data());
    };
    auto address = [&](const AddressInfo& a) {
        d.Q(a.cpu_address);
        d.Q(a.size);
        d.U(pool_index(a.memory_pool));
        d.Q(a.dsp_address);
    };

    for (auto& pool : g.memory_pool_workbuffer) {
        d.Q(pool.cpu_address);
        d.Q(pool.dsp_address);
        d.Q(pool.size);
        d.U(static_cast<u32>(pool.location));
        d.U(pool.in_use);
    }

    auto& voices = g.voice_context;
    for (u32 i = 0; i < voices.GetCount(); i++) {
        auto& r = voices.GetChannelResource(i);
        d.U(r.in_use);
        for (auto v : r.mix_volumes) d.F(v);
        for (auto v : r.prev_mix_volumes) d.F(v);
    }
    for (u32 i = 0; i < voices.GetCount(); i++) {
        auto& v = voices.GetInfo(i);
        d.U(v.in_use);
        d.U(v.is_new);
        d.U(v.was_playing);
        d.U(static_cast<u32>(v.sample_format));
        d.U(v.sample_rate);
        d.S(v.channel_count);
        d.U(v.id);
        d.U(v.node_id);
        d.U(v.mix_id);
        d.U(static_cast<u32>(v.current_play_state));
        d.U(static_cast<u32>(v.last_play_state));
        d.S(v.priority);
        d.S(v.sort_order);
        d.F(v.pitch);
        d.F(v.volume);
        d.F(v.prev_volume);
        for (auto& b : v.biquads) {
            d.U(b.enabled);
            for (auto x : b.b) d.S(x);
            for (auto x : b.a) d.S(x);
        }
        for (auto& b : v.biquads_float) {
            d.U(b.enabled);
            for (auto x : b.numerator) d.F(x);
            for (auto x : b.denominator) d.F(x);
        }
        d.U(v.use_float_biquads);
        d.U(v.wave_buffer_count);
        d.U(v.wave_buffer_index);
        d.U(v.flags);
        address(v.data_address);
        for (auto& w : v.wavebuffers) {
            address(w.buffer_address);
            address(w.context_address);
            d.U(w.start_offset);
            d.U(w.end_offset);
            d.U(w.loop);
            d.U(w.stream_ended);
            d.U(w.sent_to_DSP);
            d.U(w.loop_start_offset);
            d.U(w.loop_end_offset);
            d.S(w.loop_count);
        }
        for (auto id : v.channel_resource_ids) d.U(id);
        d.S(v.splitter_id);
        d.U(static_cast<u32>(v.src_quality));
        d.U(v.voice_dropped);
        d.U(v.data_unmapped);
        d.U(v.buffer_unmapped);
    }
    for (u32 i = 0; i < voices.GetCount(); i++) {
        auto& s = voices.GetState(i);
        d.Q(s.played_sample_count);
        d.U(s.offset);
        d.U(s.wave_buffer_index);
        for (auto valid : s.wave_buffer_valid) d.U(valid);
        d.U(s.wave_buffers_consumed);
    }

    for (u32 i = 0; i < g.effect_context.GetCount(); i++) {
        auto& e = g.effect_context.GetInfo(i);
        d.U(static_cast<u32>(e.type));
        d.U(e.enabled);
        d.U(e.buffer_unmapped);
        d.U(static_cast<u32>(e.usage_state));
        d.S(e.mix_id);
        d.S(e.process_order);
        address(e.workbuffers[0]);
        address(e.workbuffers[1]);
        d.Q(e.send_buffer_info);
        d.Q(e.send_buffer);
        d.Q(e.return_buffer_info);
        d.Q(e.return_buffer);
        for (size_t j = 0; j < 0xA0; j += 4) {
            u32 word;
            std::memcpy(&word, e.parameter.data() + j, 4);
            d.U(word);
        }
    }

    auto& mixes = g.mix_context;
    const s32 mix_count = mixes.GetCount();
    for (s32 i = 0; i < mix_count; i++) d.S(mixes.GetSortedInfo(i)->mix_id);
    for (s32 i = 0; i < mix_count; i++) {
        auto& m = *mixes.GetInfo(i);
        d.U(m.in_use);
        d.S(m.mix_id);
        d.S(m.node_id);
        d.F(m.volume);
        d.U(m.sample_rate);
        d.S(m.buffer_count);
        d.S(m.buffer_offset);
        d.S(m.distance_from_final_mix);
        d.S(m.dst_mix_id);
        d.S(m.dst_splitter_id);
        d.F(m.mix_volumes[0][0]);
        d.F(m.mix_volumes[1][1]);
        d.F(m.mix_volumes[5][5]);
        for (s32 j = 0; j < m.effect_count; j++) d.S(m.effect_order_buffer[j]);
    }

    if (g.behavior.IsSplitterSupported()) {
        for (s32 i = 0; i < mix_count; i++)
            for (s32 j = 0; j < mix_count; j++) d.U(mixes.GetEdgeMatrix().Connected(i, j));
        auto& sc = g.splitter_context;
        auto dest_index = [&](const SplitterDestinationData* p) -> u32 {
            if (p == nullptr) return 0xFFFFFFFF;
            return static_cast<u32>(p - sc.splitter_destinations);
        };
        for (s32 i = 0; i < sc.info_count; i++) {
            auto& info = sc.splitter_infos[i];
            d.S(info.id);
            d.U(info.sample_rate);
            d.U(info.destination_count);
            d.U(info.has_new_connection);
            d.U(dest_index(info.destinations));
        }
        for (s32 i = 0; i < sc.destinations_count; i++) {
            auto& dest = sc.splitter_destinations[i];
            d.S(dest.id);
            d.S(dest.destination_id);
            d.U(dest.IsConfigured());
            for (u32 j = 0; j < 2; j++) d.F(dest.mix_volumes[j]);
            for (u32 j = 0; j < 2; j++) d.F(dest.prev_mix_volumes[j]);
            d.U(dest.biquad_filters[0].enabled);
            d.F(dest.biquad_filters[0].numerator[0]);
            d.F(dest.biquad_filters[0].denominator[1]);
            d.U(dest_index(dest.next));
        }
    }

    for (u32 i = 0; i < g.sink_context.GetCount(); i++) {
        auto& s = *g.sink_context.GetInfo(i);
        d.U(static_cast<u32>(s.type));
        d.U(s.in_use);
        d.U(s.buffer_unmapped);
        d.U(s.node_id);
        if (s.type == SinkInfoBase::Type::DeviceSink) {
            auto& st = *reinterpret_cast<SinkInfoBase::DeviceState*>(s.state.data());
            for (auto c : st.downmix_coeff) d.S(c.to_raw());
        } else if (s.type == SinkInfoBase::Type::CircularBufferSink) {
            auto& st = *reinterpret_cast<SinkInfoBase::CircularBufferState*>(s.state.data());
            d.U(st.last_pos2);
            d.S(st.current_pos);
            d.U(st.last_pos);
            address(st.address_info);
        }
    }
}

// Synthetic scenario.

// Copies of private reference layouts.
struct UpdateDataHeader {
    u32 revision;
    u32 behaviour_size;
    u32 memory_pool_size;
    u32 voices_size;
    u32 voice_resources_size;
    u32 effects_size;
    u32 mix_size;
    u32 sinks_size;
    u32 performance_buffer_size;
    u32 splitter_size;
    u32 render_info_size;
    char unk2C[0x10];
    u32 size;
};
static_assert(sizeof(UpdateDataHeader) == 0x40);

struct SplitterHeader {
    u32 magic;
    s32 info_count;
    s32 destination_count;
    char unk0C[0x14];
};
static_assert(sizeof(SplitterHeader) == 0x20);

constexpr u32 kVoices = 4;
constexpr u32 kEffects = 5;
constexpr u32 kSubMixes = 2;
constexpr u32 kSinks = 2;
constexpr u32 kPools = kEffects + kVoices * MaxWaveBuffers;
constexpr u64 kPool0 = 0x1000000;
constexpr u64 kPool1 = 0x2000000;
constexpr u32 kUnusedMix = 0x7FFFFFFF;
constexpr u32 kUnusedSplitter = 0xFFFFFFFF;

enum class Fault { None, BehaviorSize, VoicesSize, Crossfade };

struct Step {
    u32 revision;
    int phase;
    u64 flags;
    Fault fault;
};

struct Builder {
    std::vector<u8> buf;
    UpdateDataHeader header{};
    Builder() { buf.resize(sizeof(header)); }
    template <typename T>
    void Push(const T& value) {
        const auto* p = reinterpret_cast<const u8*>(&value);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    void Zeros(size_t n) { buf.insert(buf.end(), n, 0); }
    size_t Mark() const { return buf.size(); }
    u32 Since(size_t mark) const { return static_cast<u32>(buf.size() - mark); }
    std::vector<u8> Finish() {
        header.size = static_cast<u32>(buf.size());
        std::memcpy(buf.data(), &header, sizeof(header));
        return buf;
    }
};

template <typename T>
static T& At(std::array<u8, 0xA0>& specific) {
    static_assert(sizeof(T) <= 0xA0);
    return *reinterpret_cast<T*>(specific.data());
}

static VoiceInfo::InParameter Voice(const BehaviorInfo& b, u32 index, int phase) {
    VoiceInfo::InParameter v{};
    v.id = index;
    v.node_id = 0x10000 + index;
    v.splitter_id = kUnusedSplitter;
    v.pitch = 1.0f;
    v.volume = 0.8f - 0.1f * static_cast<f32>(index);
    v.priority = 10 + static_cast<s32>(index);
    v.sort_order = static_cast<s32>(index);
    switch (index) {
    case 0:
        v.is_new = phase == 1;
        v.in_use = true;
        v.play_state = phase == 1 ? PlayState::Started : PlayState::Stopped;
        v.sample_format = SampleFormat::PcmInt16;
        v.sample_rate = 32000;
        v.channel_count = 1;
        v.biquads[0].enabled = true;
        v.biquads[0].b = {0x3000, 0x1000, 0x800};
        v.biquads[0].a = {-0x2000, 0x400};
        v.wave_buffer_count = 2;
        v.wave_buffer_index = 0;
        v.mix_id = 0;
        v.wave_buffer_internal[0] = {kPool0 + 0x100, 0x400, 0, 0x200, false, false, false, 0,
                                     0, 0, 0, 0};
        v.wave_buffer_internal[1] = {0x5000000, 0x200, 0, 0x100, false, false, false, 0,
                                     0, 0, 0, 0};
        v.channel_resource_ids[0] = 0;
        break;
    case 1:
        v.is_new = phase == 1;
        v.in_use = true;
        v.play_state = phase == 1 ? PlayState::Started : PlayState::Paused;
        v.sample_format = SampleFormat::Adpcm;
        v.sample_rate = 48000;
        v.channel_count = 2;
        v.src_data_address = kPool0 + 0x1000;
        v.src_data_size = 0x20;
        v.wave_buffer_count = 1;
        v.wave_buffer_index = 0;
        v.wave_buffer_internal[0] = {kPool0 + 0x3000, 0x800, 0, 0x700, true, false, false, 2,
                                     kPool0 + 0x2000, 0x10, 0x10, 0x600};
        v.wave_buffer_internal[1] = {kPool1 + 0x100, 0x100, 0, 0x80, false, true, false, 0,
                                     0, 0, 0, 0};
        if (b.IsSplitterSupported()) {
            v.mix_id = kUnusedMix;
            v.splitter_id = 0;
        } else {
            v.mix_id = 1;
        }
        v.channel_resource_ids[0] = 1;
        v.channel_resource_ids[1] = 2;
        v.flush_buffer_count = phase == 1 ? 0 : 1;
        v.flags.IsVoicePlayedSampleCountResetAtLoopPointSupported = phase != 1;
        v.flags.IsVoicePitchAndSrcSkippedSupported = phase != 1;
        v.src_quality = SrcQuality::High;
        break;
    case 2:
        break;
    case 3:
        v.is_new = true;
        v.in_use = true;
        v.play_state = PlayState::Started;
        v.sample_format = phase == 1 ? SampleFormat::PcmFloat : SampleFormat::PcmInt8;
        v.sample_rate = 22050;
        v.channel_count = 1;
        v.wave_buffer_count = 1;
        v.wave_buffer_index = 3;
        v.mix_id = phase == 1 ? 0 : 2;
        v.wave_buffer_internal[3] = {kPool1 + 0x8000, 0x1000, 0x10, 0x3F0, false, false, false,
                                     0, 0, 0, 0, 0};
        v.channel_resource_ids[0] = 3;
        v.clear_voice_drop = phase != 1;
        v.src_quality = SrcQuality::Low;
        break;
    }
    return v;
}

static void PushVoice(Builder& out, const BehaviorInfo& b, const VoiceInfo::InParameter& v,
                      u8 crossfade) {
    if (!b.IsVoiceInParameterV2Supported()) {
        out.Push(v);
        return;
    }
    std::array<u8, 0x188> raw{};
    auto put = [&](size_t offset, const auto& value) {
        std::memcpy(raw.data() + offset, &value, sizeof(value));
    };
    std::memcpy(raw.data(), &v, 0x24);
    for (size_t i = 0; i < MaxBiquadFilters; i++) {
        VoiceInfo::BiquadFilterParameter2 f{};
        f.enabled = v.biquads[i].enabled;
        for (size_t j = 0; j < 3; j++) f.numerator[j] = v.biquads[i].b[j] / 16384.0f;
        for (size_t j = 0; j < 2; j++) f.denominator[j] = v.biquads[i].a[j] / 16384.0f;
        put(0x24 + i * sizeof(f), f);
    }
    put(0x54, v.wave_buffer_count);
    put(0x58, static_cast<u32>(v.wave_buffer_index));
    put(0x60, v.src_data_address);
    put(0x68, v.src_data_size);
    put(0x70, v.mix_id);
    put(0x74, v.splitter_id);
    put(0x78, v.wave_buffer_internal);
    put(0x158, v.channel_resource_ids);
    put(0x170, v.clear_voice_drop);
    put(0x171, v.flush_buffer_count);
    put(0x174, v.flags);
    put(0x176, v.src_quality);
    put(0x177, crossfade);
    out.Push(raw);
}

static EffectInfoBase::InParameterVersion1 Effect(const BehaviorInfo& b, u32 index, int phase) {
    EffectInfoBase::InParameterVersion1 e{};
    const bool v2 = b.IsEffectInfoVersion2Supported();
    e.is_new = phase == 1;
    e.enabled = true;
    switch (index) {
    case 0: {
        e.type = phase == 1 ? EffectInfoBase::Type::Mix : EffectInfoBase::Type::Invalid;
        e.mix_id = 0;
        e.process_order = 0;
        auto& p = At<BufferMixerInfo::ParameterVersion1>(e.specific);
        p.inputs[0] = 0;
        p.inputs[1] = 1;
        p.outputs[0] = 2;
        p.outputs[1] = 3;
        p.volumes[0] = 0.5f;
        p.volumes[1] = 0.25f;
        p.mix_count = 2;
        break;
    }
    case 1: {
        e.type = EffectInfoBase::Type::Aux;
        e.mix_id = 1;
        e.process_order = 0;
        auto& p = At<AuxInfo::ParameterVersion1>(e.specific);
        p.inputs[0] = 0;
        p.inputs[1] = 1;
        p.outputs[0] = 0;
        p.outputs[1] = 1;
        p.mix_buffer_count = 2;
        p.sample_rate = 48000;
        p.count_max = 0x400;
        p.mix_buffer_count_max = 2;
        p.send_buffer_info_address = kPool0 + 0x10000;
        p.send_buffer_address = kPool0 + 0x10040;
        p.return_buffer_info_address = phase == 1 ? kPool0 + 0x18000 : 0x7000000;
        p.return_buffer_address = kPool0 + 0x18040;
        p.mix_buffer_sample_size = 4;
        p.sample_count = 240;
        p.mix_buffer_sample_count = 0x400;
        break;
    }
    case 2: {
        e.type = EffectInfoBase::Type::Delay;
        e.mix_id = 1;
        e.process_order = 1;
        e.enabled = phase == 1;
        e.workbuffer = kPool0 + 0x20000;
        e.workbuffer_size = 0x10000;
        auto& p = At<DelayInfo::ParameterVersion1>(e.specific);
        p.inputs[0] = 0;
        p.inputs[1] = 1;
        p.outputs[0] = 0;
        p.outputs[1] = 1;
        p.channel_count_max = 2;
        p.channel_count = phase == 1 ? 2 : 3;
        p.delay_time_max = 100;
        p.delay_time = 50;
        break;
    }
    case 3: {
        e.type = EffectInfoBase::Type::BiquadFilter;
        e.mix_id = 0;
        e.process_order = 1;
        if (v2) {
            auto& p = At<BiquadFilterInfo::ParameterVersion2>(e.specific);
            p.inputs[0] = 4;
            p.outputs[0] = 4;
            p.b = {0.5f, 0.25f, 0.125f};
            p.a = {-0.5f, 0.0625f};
            p.channel_count = 1;
        } else {
            auto& p = At<BiquadFilterInfo::ParameterVersion1>(e.specific);
            p.inputs[0] = 4;
            p.outputs[0] = 4;
            p.b = {0x2000, 0x1000, 0x800};
            p.a = {-0x2000, 0x400};
            p.channel_count = 1;
        }
        break;
    }
    case 4: {
        e.type = EffectInfoBase::Type::LightLimiter;
        e.mix_id = 0;
        e.process_order = 2;
        e.workbuffer = phase == 1 ? kPool0 + 0x40000 : kPool1 + 0x9000;
        e.workbuffer_size = 0x8000;
        auto& p = At<LightLimiterInfo::ParameterVersion1>(e.specific);
        p.inputs[0] = 0;
        p.inputs[1] = 1;
        p.outputs[0] = 0;
        p.outputs[1] = 1;
        p.channel_count_max = 2;
        p.channel_count = 2;
        p.sample_rate = 48000;
        p.statistics_enabled = true;
        p.statistics_reset_required = phase != 1;
        break;
    }
    }
    return e;
}

static MixInfo::InParameter Mix(const BehaviorInfo& b, u32 index, int phase) {
    MixInfo::InParameter m{};
    m.mix_id = static_cast<s32>(index);
    m.node_id = static_cast<s32>(0x20000 + index);
    m.sample_rate = 48000;
    m.dest_splitter_id = static_cast<s32>(kUnusedSplitter);
    m.dest_mix_id = static_cast<s32>(kUnusedMix);
    m.is_dirty = true;
    for (u32 i = 0; i < MaxMixBuffers; i++) m.mix_volumes[i][i] = 1.0f - 0.03125f * i;
    switch (index) {
    case 0:
        m.in_use = true;
        m.volume = 1.0f;
        m.buffer_count = 6;
        m.effect_count = 3;
        break;
    case 1:
        m.in_use = true;
        m.volume = phase == 1 ? 0.5f : 0.75f;
        m.buffer_count = 2;
        m.effect_count = 2;
        if (b.IsSplitterSupported()) {
            m.dest_splitter_id = 1;
        } else {
            m.dest_mix_id = 0;
        }
        break;
    case 2:
        m.in_use = phase != 1;
        m.volume = 0.25f;
        m.buffer_count = 2;
        m.dest_mix_id = 1;
        break;
    }
    return m;
}

static SinkInfoBase::InParameter Sink(u32 index, int phase) {
    SinkInfoBase::InParameter s{};
    s.node_id = 0x30000 + index;
    s.in_use = true;
    if (index == 0) {
        s.type = SinkInfoBase::Type::DeviceSink;
        std::strcpy(s.device.name, "MainAudioOut");
        s.device.input_count = phase == 1 ? 2 : 6;
        for (u32 i = 0; i < MaxChannels; i++) s.device.inputs[i] = static_cast<s8>(i);
        s.device.downmix_enabled = phase == 1;
        s.device.downmix_coeff = {1.0f, 0.707f, 0.5f, 0.25f};
    } else {
        s.type = phase == 1 ? SinkInfoBase::Type::CircularBufferSink : SinkInfoBase::Type::Invalid;
        s.circular_buffer.cpu_address = kPool1 + 0x4000;
        s.circular_buffer.size = 0x2000;
        s.circular_buffer.input_count = 2;
        s.circular_buffer.sample_count = 240;
        s.circular_buffer.format = SampleFormat::PcmInt16;
        s.circular_buffer.inputs = {2, 3, 0, 0, 0, 0};
        s.circular_buffer.in_use = true;
    }
    return s;
}

static void PushSplitters(Builder& out, const BehaviorInfo& b, int phase) {
    SplitterHeader header{};
    header.magic = GetSplitterInParamHeaderMagic();
    header.info_count = phase == 1 ? 2 : 1;
    header.destination_count = phase == 1 ? 4 : 2;
    out.Push(header);
    auto info = [&](s32 id, std::vector<u32> ids) {
        SplitterInfo::InParameter p{};
        p.magic = GetSplitterInfoMagic();
        p.id = id;
        p.sample_rate = 48000;
        p.destination_count = static_cast<u32>(ids.size());
        out.Push(p);
        for (auto d : ids) out.Push(d);
        out.Zeros(3 * sizeof(s32));
    };
    if (phase == 1) {
        info(0, {0, 1});
        info(1, {2, 3});
    } else {
        info(0, {1});
    }
    auto destination = [&](s32 id, u32 mix_id, f32 scale, bool reset) {
        std::array<f32, MaxMixBuffers> volumes{};
        for (u32 i = 0; i < MaxMixBuffers; i++) volumes[i] = scale / static_cast<f32>(i + 1);
        const u32 magic = GetSplitterSendDataMagic();
        if (!b.IsBiquadFilterParameterForSplitterEnabled()) {
            SplitterDestinationData::InParameter p{};
            p.magic = magic;
            p.id = id;
            p.mix_volumes = volumes;
            p.mix_id = mix_id;
            p.in_use = true;
            p.reset_prev_volume = reset;
            out.Push(p);
        } else if (!b.IsSplitterDestinationV2bSupported()) {
            SplitterDestinationData::InParameterVersion2a p{};
            p.magic = magic;
            p.id = id;
            p.mix_volumes = volumes;
            p.mix_id = mix_id;
            p.biquad_filters[0].enabled = id == 0;
            p.biquad_filters[0].b = {0x2000, -0x1000, 0x800};
            p.biquad_filters[0].a = {0x1000, -0x400};
            p.in_use = true;
            p.reset_prev_volume = reset;
            out.Push(p);
        } else {
            SplitterDestinationData::InParameterVersion2b p{};
            p.magic = magic;
            p.id = id;
            p.mix_volumes = volumes;
            p.mix_id = mix_id;
            p.biquad_filters[0].enabled = id == 0;
            p.biquad_filters[0].numerator = {0.5f, -0.25f, 0.125f};
            p.biquad_filters[0].denominator = {0.25f, -0.0625f};
            p.in_use = true;
            p.reset_prev_volume = reset;
            out.Push(p);
        }
    };
    if (phase == 1) {
        destination(0, 1, 1.0f, false);
        destination(1, 2, 0.5f, false);
        destination(2, 0, 0.75f, false);
        destination(3, 0, 0.25f, false);
    } else {
        destination(1, 2, 0.125f, true);
        destination(3, 0, 0.5f, false);
    }
}

static std::vector<u8> BuildInput(const BehaviorInfo& b, const Step& step) {
    Builder out;
    auto& h = out.header;
    h.revision = step.revision;
    const int phase = step.phase;

    auto mark = out.Mark();
    BehaviorInfo::InParameter behavior{};
    behavior.revision = step.revision;
    std::memcpy(&behavior.flags, &step.flags, sizeof(step.flags));
    out.Push(behavior);
    h.behaviour_size = out.Since(mark) + (step.fault == Fault::BehaviorSize ? 0x10 : 0);

    mark = out.Mark();
    for (u32 i = 0; i < kPools; i++) {
        MemoryPoolInfo::InParameter p{};
        if (i == 0) {
            p = {kPool0, 0x100000,
                 phase == 1 ? MemoryPoolInfo::State::RequestAttach : MemoryPoolInfo::State::Attached,
                 true};
        } else if (i == 1) {
            p = {kPool1, 0x10000,
                 phase == 1 ? MemoryPoolInfo::State::RequestAttach
                            : MemoryPoolInfo::State::RequestDetach,
                 true};
        } else if (i == 2 && phase != 1) {
            p = {0x3000800, 0x1000, MemoryPoolInfo::State::RequestAttach, true};
        } else if (i == 3 && phase != 1) {
            p = {0x4000000, 0x2000, MemoryPoolInfo::State::RequestDetach, true};
        }
        out.Push(p);
    }
    h.memory_pool_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kVoices; i++) {
        VoiceChannelResource::InParameter r{};
        r.id = i;
        r.in_use = i < 3 || phase != 1;
        for (u32 j = 0; j < MaxMixBuffers; j++)
            r.mix_volumes[j] = 0.25f * static_cast<f32>(i + 1) + 0.01f * static_cast<f32>(j) +
                               (phase == 1 ? 0.0f : 0.5f);
        out.Push(r);
    }
    h.voice_resources_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kVoices; i++) {
        const u8 crossfade = step.fault == Fault::Crossfade && i == 0 ? 1 : 0;
        PushVoice(out, b, Voice(b, i, phase), crossfade);
    }
    h.voices_size = out.Since(mark) - (step.fault == Fault::VoicesSize ? 0x10 : 0);

    mark = out.Mark();
    for (u32 i = 0; i < kEffects; i++) out.Push(Effect(b, i, phase));
    h.effects_size = out.Since(mark);

    if (b.IsSplitterSupported()) {
        mark = out.Mark();
        PushSplitters(out, b, phase);
        h.splitter_size = out.Since(mark);
    }

    mark = out.Mark();
    if (b.IsMixInParameterDirtyOnlyUpdateSupported()) {
        MixInfo::InDirtyParameter dirty{};
        dirty.magic = Common::MakeMagic('M', 'I', 'X', 'D');
        dirty.count = phase == 1 ? 3 : 2;
        out.Push(dirty);
        for (u32 i = phase == 1 ? 0 : 1; i < kSubMixes + 1; i++) out.Push(Mix(b, i, phase));
    } else {
        for (u32 i = 0; i < kSubMixes + 1; i++) out.Push(Mix(b, i, phase));
    }
    h.mix_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kSinks; i++) out.Push(Sink(i, phase));
    h.sinks_size = out.Since(mark);

    mark = out.Mark();
    PerformanceManager::InParameter perf{};
    perf.target_node_id = phase == 1 ? 0x10000 : 0;
    out.Push(perf);
    h.performance_buffer_size = out.Since(mark);
    return out.Finish();
}

static size_t OutputSize(const BehaviorInfo& b) {
    size_t size = sizeof(UpdateDataHeader);
    size += kPools * sizeof(MemoryPoolInfo::OutStatus);
    size += kVoices * sizeof(VoiceInfo::OutStatus);
    size += kEffects * (b.IsEffectInfoVersion2Supported()
                            ? sizeof(EffectInfoBase::OutStatusVersion2)
                            : sizeof(EffectInfoBase::OutStatusVersion1));
    size += kSinks * sizeof(SinkInfoBase::OutStatus);
    size += sizeof(PerformanceManager::OutStatus);
    size += sizeof(BehaviorInfo::OutStatus);
    if (b.IsElapsedFrameCountSupported()) size += 0x10;
    return size;
}

struct Case {
    const char* name;
    u32 open_revision;
    u32 perf_frames;
    bool active;
    std::vector<Step> steps;
};

static void Append(std::vector<u8>& out, const void* data, size_t size) {
    const auto* p = static_cast<const u8*>(data);
    out.insert(out.end(), p, p + size);
}

static void Append32(std::vector<u8>& out, u32 value) {
    Append(out, &value, sizeof(value));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: golden <output dir>\n");
        return 1;
    }
    const u32 rev1 = EncodeRevision(1, false);
    const u32 rev5 = EncodeRevision(5, false);
    const u32 rev9 = EncodeRevision(9, false);
    const u32 rev13 = EncodeRevision(13, false);
    const u32 rev15 = EncodeRevision(15, false);
    const u32 nx15 = EncodeRevision(15, true);
    const u32 nx16 = EncodeRevision(16, true);
    auto steps = [](u32 revision) {
        return std::vector<Step>{{revision, 1, 0, Fault::None},
                                 {revision, 2, 0, Fault::None},
                                 {revision, 2, 1, Fault::None}};
    };
    std::vector<Case> cases{
        {"rev1", rev1, 0, false, steps(rev1)},
        {"rev5_perf", rev5, 2, true, steps(rev5)},
        {"rev9_effects_v2", rev9, 1, true, steps(rev9)},
        {"rev13_splitter_v2a", rev13, 0, false, steps(rev13)},
        {"rev15_current", rev15, 1, true, steps(rev15)},
        {"rev15_opened_rev_tagged_rev_lower", rev15, 0, true, steps(nx15)},
        {"rev15_native", nx15, 0, false, steps(nx15)},
        {"rev16_native", nx16, 1, true, steps(nx16)},
        {"fault_behavior_size", rev15, 0, false,
         {{rev15, 1, 0, Fault::None}, {rev15, 2, 0, Fault::BehaviorSize}}},
        {"fault_voices_size", rev9, 0, false, {{rev9, 1, 0, Fault::VoicesSize}}},
        {"fault_revision_mismatch", rev15, 0, false, {{rev5, 1, 0, Fault::None}}},
        {"fault_crossfade", nx16, 0, false,
         {{nx16, 1, 0, Fault::None}, {nx16, 2, 0, Fault::Crossfade}}},
    };

    for (const auto& c : cases) {
        AudioRendererParameterInternal params{};
        params.sample_rate = 48000;
        params.sample_count = 240;
        params.mixes = 24;
        params.sub_mixes = kSubMixes;
        params.voices = kVoices;
        params.sinks = kSinks;
        params.effects = kEffects;
        params.perf_frames = c.perf_frames;
        params.splitter_infos = 2;
        params.splitter_destinations = 4;
        params.revision = c.open_revision;
        static_assert(sizeof(params) == 0x34);

        Golden golden;
        const u64 work_buffer_size = GetWorkBufferSize(params);
        auto init = golden.Initialize(params, work_buffer_size);
        if (init.IsError()) {
            std::fprintf(stderr, "%s: initialize failed %08x\n", c.name, init.raw);
            return 1;
        }
        golden.active = c.active;
        BehaviorInfo layout;
        layout.SetUserLibRevision(c.open_revision);

        std::vector<u8> fixture;
        Append(fixture, "LRUG", 4);
        Append32(fixture, 2);
        Append(fixture, &params, sizeof(params));
        Append32(fixture, c.active);
        Append32(fixture, static_cast<u32>(c.steps.size()));
        for (const auto& step : c.steps) {
            const auto input = BuildInput(layout, step);
            std::vector<u8> output(OutputSize(layout));
            const size_t perf_size =
                c.perf_frames > 0
                    ? PerformanceManager::GetRequiredBufferSizeForPerformanceMetricsPerFrame(
                          layout, params) * c.perf_frames
                    : 0;
            std::vector<u8> perf(perf_size);
            const auto result = golden.Update(input, perf, output);
            Append32(fixture, result.raw);
            Append32(fixture, static_cast<u32>(input.size()));
            Append32(fixture, static_cast<u32>(output.size()));
            Append32(fixture, static_cast<u32>(perf.size()));
            Append(fixture, input.data(), input.size());
            Append(fixture, output.data(), output.size());
            Append(fixture, perf.data(), perf.size());
            Digest digest;
            DigestState(digest, golden, params);
            Append32(fixture, static_cast<u32>(digest.words.size()));
            Append(fixture, digest.words.data(), digest.words.size() * 4);
            std::printf("%s phase %d: result %08x in %zu out %zu perf %zu\n", c.name, step.phase,
                        result.raw, input.size(), output.size(), perf.size());
        }
        const std::string path = std::string(argv[1]) + "/" + c.name + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return 1;
        std::fwrite(fixture.data(), 1, fixture.size(), f);
        std::fclose(f);
    }
    return 0;
}
