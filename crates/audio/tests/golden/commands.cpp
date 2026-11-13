// Golden command list fixtures for lichii-audio, produced by the C++ Lichii reference.
//
// Reuses the synthetic RequestUpdate scenarios of golden.cpp, then runs a copy of
// System::GenerateCommand (voice drop included) after every update and records each command
// the reference generates: its id, node id, enabled flag, estimated processing time and the
// fields the DSP reads. DSP state pointers are recorded as the index of the voice, effect or
// upsampler they point into, which is how the Rust command list names them. The Rust test in
// ../command_generation.rs replays them.
//
// Build like golden.cpp, adding the reference's command_generator.cpp, command_buffer.cpp and
// command_processing_time_estimator.cpp:
//   g++ -std=c++20 -include algorithm -I shim -I <lichii>/src commands.cpp <renderer objects> -o commands
//   ./commands <output dir>

#define main request_update_main
#include "golden.cpp"
#undef main

#include "audio_core/renderer/command/command_buffer.h"
#include "audio_core/renderer/command/command_generator.h"
#include "audio_core/renderer/command/command_list_header.h"
#include "audio_core/renderer/command/command_processing_time_estimator.h"
#include "audio_core/renderer/effect/capture.h"
#include "audio_core/renderer/effect/compressor.h"
#include "audio_core/renderer/effect/i3dl2.h"
#include "audio_core/renderer/effect/reverb.h"

// The commands' DSP entry points live with the ADSP; generation only needs their vtables.
#define STUB_COMMAND(T)                                                                            \
    void T::Dump(const AudioRenderer::CommandListProcessor&, std::string&) {}                      \
    void T::Process(const AudioRenderer::CommandListProcessor&) {}                                 \
    bool T::Verify(const AudioRenderer::CommandListProcessor&) {                                   \
        return true;                                                                               \
    }
namespace AudioCore::Renderer {
STUB_COMMAND(AdpcmDataSourceVersion1Command)
STUB_COMMAND(AdpcmDataSourceVersion2Command)
STUB_COMMAND(AuxCommand)
STUB_COMMAND(BiquadFilterCommand)
STUB_COMMAND(CaptureCommand)
STUB_COMMAND(CircularBufferSinkCommand)
STUB_COMMAND(ClearMixBufferCommand)
STUB_COMMAND(CompressorCommand)
STUB_COMMAND(CopyMixBufferCommand)
STUB_COMMAND(DelayCommand)
STUB_COMMAND(DepopForMixBuffersCommand)
STUB_COMMAND(DepopPrepareCommand)
STUB_COMMAND(DeviceSinkCommand)
STUB_COMMAND(DownMix6chTo2chCommand)
STUB_COMMAND(I3dl2ReverbCommand)
STUB_COMMAND(LightLimiterVersion1Command)
STUB_COMMAND(LightLimiterVersion2Command)
STUB_COMMAND(MixCommand)
STUB_COMMAND(MixRampCommand)
STUB_COMMAND(MixRampGroupedCommand)
STUB_COMMAND(MultiTapBiquadFilterCommand)
STUB_COMMAND(PcmFloatDataSourceVersion1Command)
STUB_COMMAND(PcmFloatDataSourceVersion2Command)
STUB_COMMAND(PcmInt16DataSourceVersion1Command)
STUB_COMMAND(PcmInt16DataSourceVersion2Command)
STUB_COMMAND(PerformanceCommand)
STUB_COMMAND(ReverbCommand)
STUB_COMMAND(UpsampleCommand)
STUB_COMMAND(VolumeCommand)
STUB_COMMAND(VolumeRampCommand)
} // namespace AudioCore::Renderer

void AssertFatalImpl() {
    std::fprintf(stderr, "reference assertion failed\n");
    std::abort();
}

struct Session {
    s32 session_id{};
    u32 sample_rate{};
    s8 channels{2};
    bool drop_voice{};
    u32 render_time_limit_percent{100};
    f32 drop_voice_param{1.0f};
    u32 num_voices_dropped{};
    std::unique_ptr<ICommandProcessingTimeEstimator> estimator{};
};

static std::unique_ptr<ICommandProcessingTimeEstimator> MakeEstimator(const BehaviorInfo& b,
                                                                      u32 sample_count,
                                                                      s16 mix_buffer_count) {
    if (b.IsCommandProcessingTimeEstimatorVersion5Supported())
        return std::make_unique<CommandProcessingTimeEstimatorVersion5>(sample_count,
                                                                        mix_buffer_count);
    if (b.IsCommandProcessingTimeEstimatorVersion4Supported())
        return std::make_unique<CommandProcessingTimeEstimatorVersion4>(sample_count,
                                                                        mix_buffer_count);
    if (b.IsCommandProcessingTimeEstimatorVersion3Supported())
        return std::make_unique<CommandProcessingTimeEstimatorVersion3>(sample_count,
                                                                        mix_buffer_count);
    if (b.IsCommandProcessingTimeEstimatorVersion2Supported())
        return std::make_unique<CommandProcessingTimeEstimatorVersion2>(sample_count,
                                                                        mix_buffer_count);
    return std::make_unique<CommandProcessingTimeEstimatorVersion1>(sample_count,
                                                                    mix_buffer_count);
}

// System::DropVoices.
static u32 DropVoices(Golden& g, Session& s, CommandBuffer& command_buffer,
                      u32 estimated_process_time, u32 time_limit) {
    u32 i{0};
    auto command_list{command_buffer.command_list.data() + sizeof(CommandListHeader)};
    ICommand* cmd{nullptr};
    while (i < command_buffer.count) {
        cmd = reinterpret_cast<ICommand*>(command_list);
        if (cmd->type == CommandId::Performance ||
            cmd->type == CommandId::DataSourcePcmInt16Version1 ||
            cmd->type == CommandId::DataSourcePcmInt16Version2 ||
            cmd->type == CommandId::DataSourcePcmFloatVersion1 ||
            cmd->type == CommandId::DataSourcePcmFloatVersion2 ||
            cmd->type == CommandId::DataSourceAdpcmVersion1 ||
            cmd->type == CommandId::DataSourceAdpcmVersion2) {
            break;
        }
        command_list += cmd->size;
        i++;
    }
    if (cmd == nullptr || command_buffer.count == 0 || i >= command_buffer.count) return 0;

    u32 voices_dropped{0};
    while (i < command_buffer.count) {
        const auto node_id{cmd->node_id};
        if (estimated_process_time <= time_limit) break;
        if ((node_id >> 28) != 1) break;
        auto& voice_info{g.voice_context.GetInfo((node_id >> 16) & 0xFFF)};
        if (voice_info.priority == HighestVoicePriority) break;
        voices_dropped++;
        voice_info.voice_dropped = true;
        while (i < command_buffer.count && cmd->node_id == node_id) {
            if (cmd->type == CommandId::DepopPrepare) {
                cmd->enabled = true;
            } else if (cmd->enabled && cmd->type != CommandId::Performance) {
                cmd->enabled = false;
                estimated_process_time -= static_cast<u32>(
                    s.drop_voice_param * static_cast<f32>(cmd->estimated_process_time));
            }
            command_list += cmd->size;
            cmd = reinterpret_cast<ICommand*>(command_list);
            i++;
        }
        i++;
    }
    return voices_dropped;
}

// System::GenerateCommand without the kernel and timing pieces. Returns the command count.
static u32 GenerateCommand(Golden& g, Session& s) {
    PoolMapper::ClearUseState(g.memory_pool_workbuffer, g.memory_pool_count);
    auto* header{reinterpret_cast<CommandListHeader*>(g.command_workbuffer.data())};
    header->buffer_count = static_cast<s16>(g.voice_channels + g.mix_buffer_count);
    header->sample_count = g.sample_count;
    header->sample_rate = s.sample_rate;
    header->samples_buffer = g.samples_workbuffer;

    const auto performance_initialized{g.performance_manager.IsInitialized()};
    if (performance_initialized) {
        g.performance_manager.TapFrame(false, s.num_voices_dropped, 0);
        s.num_voices_dropped = 0;
    }

    AudioRendererSystemContext render_context{
        .session_id{s.session_id},
        .channels{s.channels},
        .mix_buffer_count{g.mix_buffer_count},
        .behavior{&g.behavior},
        .depop_buffer{g.depop_buffer},
        .upsampler_manager{g.upsampler_manager},
        .memory_pool_info{&g.memory_pool_info},
    };
    CommandBuffer command_buffer{
        .command_list{g.command_workbuffer},
        .sample_count{g.sample_count},
        .sample_rate{s.sample_rate},
        .size{sizeof(CommandListHeader)},
        .count{0},
        .estimated_process_time{0},
        .memory_pool{&g.memory_pool_info},
        .time_estimator{s.estimator.get()},
        .behavior{&g.behavior},
    };
    CommandGenerator generator{command_buffer,    *header,
                               render_context,    g.voice_context,
                               g.mix_context,     g.effect_context,
                               g.sink_context,    g.splitter_context,
                               performance_initialized ? &g.performance_manager : nullptr};

    g.voice_context.SortInfo();
    generator.GenerateVoiceCommands();
    const auto start_estimated_time{s.drop_voice_param *
                                    static_cast<f32>(command_buffer.estimated_process_time)};
    generator.GenerateSubMixCommands();
    generator.GenerateFinalMixCommands();
    generator.GenerateSinkCommands();

    if (s.drop_voice) {
        f32 time_limit_percent{70.0f};
        if (g.behavior.IsAudioRendererProcessingTimeLimit80PercentSupported()) {
            time_limit_percent = 80.0f;
        } else if (g.behavior.IsAudioRendererProcessingTimeLimit75PercentSupported()) {
            time_limit_percent = 75.0f;
        }
        const auto end_estimated_time{s.drop_voice_param *
                                      static_cast<f32>(command_buffer.estimated_process_time)};
        const auto dsp_time_limit{((time_limit_percent / 100.0f) * 2'880'000.0f) *
                                  (static_cast<f32>(s.render_time_limit_percent) / 100.0f)};
        const auto estimated_time{start_estimated_time - end_estimated_time};
        const auto time_limit{
            static_cast<u32>((std::max)(dsp_time_limit + estimated_time, 0.0f))};
        s.num_voices_dropped = DropVoices(g, s, command_buffer,
                                          static_cast<u32>(start_estimated_time), time_limit);
    }

    header->buffer_size = command_buffer.size;
    header->command_count = command_buffer.count;
    g.voice_context.UpdateStateByDspShared();
    if (g.behavior.IsEffectInfoVersion2Supported()) g.effect_context.UpdateStateByDspShared();
    g.frames_elapsed++;
    return command_buffer.count;
}

// Command dump: per command [id, node id, enabled, estimate, field count, fields...]. The
// Rust test builds the same words from its own command list.
struct Dump {
    Golden& g;
    std::vector<u32> words{};
    std::vector<u32> fields{};

    void U(u32 v) { fields.push_back(v); }
    void S(s32 v) { fields.push_back(static_cast<u32>(v)); }
    void F(f32 v) { fields.push_back(std::bit_cast<u32>(v)); }
    void Q(u64 v) {
        U(static_cast<u32>(v));
        U(static_cast<u32>(v >> 32));
    }
    template <typename T>
    void Bytes(const T& value, size_t size = sizeof(T)) {
        static_assert(sizeof(T) % 4 == 0);
        for (size_t i = 0; i < size; i += 4) {
            u32 word;
            std::memcpy(&word, reinterpret_cast<const u8*>(&value) + i, 4);
            U(word);
        }
    }
    template <size_t N>
    void Indices(const std::array<s16, N>& values) {
        for (auto v : values) S(v);
    }

    // A DSP pointer inside span, as (element index, byte offset in the element).
    template <typename T>
    std::pair<u32, u32> Locate(std::span<T> span, CpuAddr address) {
        const auto base{reinterpret_cast<CpuAddr>(span.data())};
        if (address < base || address >= base + span.size_bytes()) {
            std::fprintf(stderr, "pointer %llx outside its context\n",
                         static_cast<unsigned long long>(address));
            std::abort();
        }
        const auto offset{address - base};
        return {static_cast<u32>(offset / sizeof(T)), static_cast<u32>(offset % sizeof(T))};
    }
    u32 VoiceState(CpuAddr address) { return Locate(g.voice_context.dsp_states, address).first; }
    void VoiceBiquadState(CpuAddr address) {
        const auto [voice, offset] = Locate(g.voice_context.dsp_states, address);
        U(0);
        U(voice);
        U(static_cast<u32>((offset - offsetof(::AudioCore::Renderer::VoiceState, biquad_states)) /
                           sizeof(::AudioCore::Renderer::VoiceState::biquad_states[0])));
    }
    void PreviousSample(CpuAddr address) {
        const auto [voice, offset] = Locate(g.voice_context.dsp_states, address);
        U(voice);
        U(static_cast<u32>(
            (offset - offsetof(::AudioCore::Renderer::VoiceState, previous_samples)) /
            sizeof(s32)));
    }
    u32 EffectState(CpuAddr address) {
        if (address == 0) return 0;
        return Locate(g.effect_context.effect_infos, address).first;
    }
    void EffectBiquadState(CpuAddr address) {
        const auto [effect, offset] = Locate(g.effect_context.effect_infos, address);
        const auto state{reinterpret_cast<CpuAddr>(
                             g.effect_context.effect_infos[effect].GetStateBuffer()) -
                         reinterpret_cast<CpuAddr>(&g.effect_context.effect_infos[effect])};
        U(1);
        U(effect);
        U(static_cast<u32>((offset - state) /
                           sizeof(::AudioCore::Renderer::VoiceState::BiquadFilterState)));
    }
    u32 EffectResult(CpuAddr address) {
        if (address == 0) return 0xFFFFFFFF;
        return Locate(g.effect_context.result_states_dsp, address).first;
    }
    u32 Upsampler(CpuAddr address) { return Locate(g.upsampler_infos, address).first; }

    void Biquad(const VoiceInfo::BiquadFilterParameter& b) {
        U(b.enabled);
        for (auto x : b.b) S(x);
        for (auto x : b.a) S(x);
    }
    void BiquadFloat(const VoiceInfo::BiquadFilterParameter2& b) {
        U(b.enabled);
        for (auto x : b.numerator) F(x);
        for (auto x : b.denominator) F(x);
    }
    template <typename W>
    void WaveBuffers(const std::array<W, MaxWaveBuffers>& wave_buffers) {
        for (const auto& w : wave_buffers) {
            Q(w.buffer);
            Q(w.buffer_size);
            U(w.start_offset);
            U(w.end_offset);
            U(w.loop);
            U(w.stream_ended);
            Q(w.context);
            Q(w.context_size);
            if constexpr (requires { w.loop_count; }) {
                U(w.loop_start_offset);
                U(w.loop_end_offset);
                S(w.loop_count);
            }
        }
    }
    template <typename C>
    void DataSource(const C& c) {
        U(static_cast<u32>(c.src_quality));
        S(c.output_index);
        U(c.flags);
        U(c.sample_rate);
        F(c.pitch);
        if constexpr (requires { c.channel_index; }) {
            S(c.channel_index);
            S(c.channel_count);
        } else {
            S(0);
            S(0);
        }
        U(VoiceState(c.voice_state));
        WaveBuffers(c.wave_buffers);
        if constexpr (requires { c.data_address; }) {
            Q(c.data_address);
            Q(c.data_size);
        }
    }
    template <typename C>
    void Effect(const C& c, size_t parameter_size = sizeof(C::parameter)) {
        Indices(c.inputs);
        Indices(c.outputs);
        Bytes(c.parameter, parameter_size);
        U(EffectState(c.state));
        U(c.effect_enabled);
    }

    void Command(const ICommand& base) {
        fields.clear();
        switch (base.type) {
        case CommandId::DataSourcePcmInt16Version1:
            DataSource(static_cast<const PcmInt16DataSourceVersion1Command&>(base));
            break;
        case CommandId::DataSourcePcmInt16Version2:
            DataSource(static_cast<const PcmInt16DataSourceVersion2Command&>(base));
            break;
        case CommandId::DataSourcePcmFloatVersion1:
            DataSource(static_cast<const PcmFloatDataSourceVersion1Command&>(base));
            break;
        case CommandId::DataSourcePcmFloatVersion2:
            DataSource(static_cast<const PcmFloatDataSourceVersion2Command&>(base));
            break;
        case CommandId::DataSourceAdpcmVersion1:
            DataSource(static_cast<const AdpcmDataSourceVersion1Command&>(base));
            break;
        case CommandId::DataSourceAdpcmVersion2:
            DataSource(static_cast<const AdpcmDataSourceVersion2Command&>(base));
            break;
        case CommandId::Volume: {
            const auto& c{static_cast<const VolumeCommand&>(base)};
            U(c.precision);
            S(c.input_index);
            S(c.output_index);
            F(c.volume);
        } break;
        case CommandId::VolumeRamp: {
            const auto& c{static_cast<const VolumeRampCommand&>(base)};
            U(c.precision);
            S(c.input_index);
            S(c.output_index);
            F(c.prev_volume);
            F(c.volume);
        } break;
        case CommandId::BiquadFilter: {
            const auto& c{static_cast<const BiquadFilterCommand&>(base)};
            S(c.input);
            S(c.output);
            Biquad(c.biquad);
            BiquadFloat(c.biquad_float);
            const auto voice_states{g.voice_context.dsp_states};
            const auto address{static_cast<CpuAddr>(c.state)};
            const auto base_address{reinterpret_cast<CpuAddr>(voice_states.data())};
            if (address >= base_address && address < base_address + voice_states.size_bytes())
                VoiceBiquadState(address);
            else
                EffectBiquadState(address);
            U(c.needs_init);
            U(c.use_float_processing);
            U(c.use_float_coefficients);
        } break;
        case CommandId::MultiTapBiquadFilter: {
            const auto& c{static_cast<const MultiTapBiquadFilterCommand&>(base)};
            S(c.input);
            S(c.output);
            for (u32 i = 0; i < MaxBiquadFilters; i++) {
                Biquad(c.biquads[i]);
                BiquadFloat(c.biquads_float[i]);
                VoiceBiquadState(c.states[i]);
                U(c.needs_init[i]);
            }
            U(c.filter_tap_count);
            U(c.use_float_coefficients);
        } break;
        case CommandId::Mix: {
            const auto& c{static_cast<const MixCommand&>(base)};
            U(c.precision);
            S(c.input_index);
            S(c.output_index);
            F(c.volume);
        } break;
        case CommandId::MixRamp: {
            const auto& c{static_cast<const MixRampCommand&>(base)};
            U(c.precision);
            S(c.input_index);
            S(c.output_index);
            F(c.prev_volume);
            F(c.volume);
            PreviousSample(c.previous_sample);
        } break;
        case CommandId::MixRampGrouped: {
            const auto& c{static_cast<const MixRampGroupedCommand&>(base)};
            U(c.precision);
            U(c.buffer_count);
            U(VoiceState(c.previous_samples));
            for (u32 i = 0; i < c.buffer_count; i++) {
                S(c.inputs[i]);
                S(c.outputs[i]);
                F(c.prev_volumes[i]);
                F(c.volumes[i]);
            }
        } break;
        case CommandId::DepopPrepare: {
            const auto& c{static_cast<const DepopPrepareCommand&>(base)};
            Indices(c.inputs);
            U(VoiceState(c.previous_samples));
            U(c.buffer_count);
        } break;
        case CommandId::DepopForMixBuffers: {
            const auto& c{static_cast<const DepopForMixBuffersCommand&>(base)};
            U(c.input);
            U(c.count);
            Q(static_cast<u64>(c.decay.to_raw()));
        } break;
        case CommandId::Delay:
            Effect(static_cast<const DelayCommand&>(base));
            break;
        case CommandId::Reverb: {
            const auto& c{static_cast<const ReverbCommand&>(base)};
            Effect(c);
            U(c.long_size_pre_delay_supported);
        } break;
        case CommandId::I3dl2Reverb:
            Effect(static_cast<const I3dl2ReverbCommand&>(base));
            break;
        // The documented limiter parameter is 0x44 bytes with a byte processing mode at 0x43;
        // the reference declares the mode as an int-sized enum, padding its struct to 0x48.
        case CommandId::LightLimiterVersion1:
            Effect(static_cast<const LightLimiterVersion1Command&>(base), 0x44);
            break;
        case CommandId::LightLimiterVersion2: {
            const auto& c{static_cast<const LightLimiterVersion2Command&>(base)};
            Effect(c, 0x44);
            U(EffectResult(c.result_state));
        } break;
        case CommandId::Compressor: {
            // The reference leaves state and effect_enabled unset and passes the state buffer
            // as the work buffer.
            const auto& c{static_cast<const CompressorCommand&>(base)};
            Indices(c.inputs);
            Indices(c.outputs);
            Bytes(c.parameter);
            U(EffectState(c.workbuffer));
        } break;
        case CommandId::Upsample: {
            const auto& c{static_cast<const UpsampleCommand&>(base)};
            U(c.buffer_count);
            U(c.unk_20);
            U(c.source_sample_count);
            U(c.source_sample_rate);
            U(Upsampler(c.upsampler_info));
        } break;
        case CommandId::DownMix6chTo2ch: {
            const auto& c{static_cast<const DownMix6chTo2chCommand&>(base)};
            Indices(c.inputs);
            Indices(c.outputs);
            for (auto coeff : c.down_mix_coeff) Q(static_cast<u64>(coeff.to_raw()));
        } break;
        case CommandId::Aux: {
            const auto& c{static_cast<const AuxCommand&>(base)};
            S(c.input);
            S(c.output);
            Q(c.send_buffer_info);
            Q(c.return_buffer_info);
            Q(c.send_buffer);
            Q(c.return_buffer);
            U(c.count_max);
            U(c.write_offset);
            U(c.update_count);
            U(c.effect_enabled);
        } break;
        case CommandId::Capture: {
            const auto& c{static_cast<const CaptureCommand&>(base)};
            S(c.input);
            S(c.output);
            Q(c.send_buffer_info);
            Q(c.send_buffer);
            U(c.count_max);
            U(c.write_offset);
            U(c.update_count);
            U(c.effect_enabled);
        } break;
        case CommandId::DeviceSink: {
            const auto& c{static_cast<const DeviceSinkCommand&>(base)};
            S(c.session_id);
            U(c.input_count);
            for (u32 i = 0; i < c.input_count; i++) S(c.inputs[i]);
            if (c.sample_buffer.data() == g.samples_workbuffer.data()) {
                U(0xFFFFFFFF);
            } else {
                u32 found{0xFFFFFFFE};
                for (u32 i = 0; i < g.upsampler_infos.size(); i++) {
                    if (g.upsampler_infos[i].samples_pos ==
                        reinterpret_cast<CpuAddr>(c.sample_buffer.data()))
                        found = i;
                }
                U(found);
            }
        } break;
        case CommandId::CircularBufferSink: {
            const auto& c{static_cast<const CircularBufferSinkCommand&>(base)};
            U(c.input_count);
            for (u32 i = 0; i < c.input_count; i++) S(c.inputs[i]);
            Q(c.address);
            U(c.size);
            U(c.pos);
        } break;
        case CommandId::Performance: {
            const auto& c{static_cast<const PerformanceCommand&>(base)};
            U(static_cast<u32>(c.state));
            U(static_cast<u32>(c.entry_address.entry_start_time_offset));
            U(static_cast<u32>(c.entry_address.header_entry_count_offset));
            U(static_cast<u32>(c.entry_address.entry_processed_time_offset));
        } break;
        case CommandId::ClearMixBuffer:
            break;
        case CommandId::CopyMixBuffer: {
            const auto& c{static_cast<const CopyMixBufferCommand&>(base)};
            S(c.input_index);
            S(c.output_index);
        } break;
        default:
            std::fprintf(stderr, "unexpected command %u\n", static_cast<u32>(base.type));
            std::abort();
        }
        words.push_back(static_cast<u32>(base.type));
        words.push_back(base.node_id);
        words.push_back(base.enabled);
        words.push_back(base.estimated_process_time);
        words.push_back(static_cast<u32>(fields.size()));
        words.insert(words.end(), fields.begin(), fields.end());
    }

    void List(u32 count) {
        auto* p{g.command_workbuffer.data() + sizeof(CommandListHeader)};
        for (u32 i = 0; i < count; i++) {
            const auto& cmd{*reinterpret_cast<const ICommand*>(p)};
            Command(cmd);
            // CircularBufferSinkCommand declares its own size member, so GenerateStart's
            // cmd.size = sizeof(T) lands there and leaves ICommand::size at 0.
            p += cmd.type == CommandId::CircularBufferSink ? sizeof(CircularBufferSinkCommand)
                                                           : cmd.size;
        }
    }
};

// Synthetic scenario for generation. golden.cpp's scenario exercises the update edge cases, so
// most of its voices never reach generation; this one keeps every voice playable: PCM16 with
// both biquads to a sub mix, stereo ADPCM through a splitter, PCM float to a 12 buffer sub mix
// (a grouped ramp) that stops in phase 2, and a highest priority PCM16 voice on the final mix.
// Every effect type sits on some mix, and both sink types read the final mix.

constexpr u32 kGenVoices = 5;
constexpr u32 kGenEffects = 9;
constexpr u32 kGenPools = kGenEffects + kGenVoices * MaxWaveBuffers;

static VoiceInfo::InParameter GenVoice(const BehaviorInfo& b, u32 index, int phase) {
    VoiceInfo::InParameter v{};
    v.id = index;
    v.node_id = (1u << 28) | (index << 16);
    v.splitter_id = kUnusedSplitter;
    v.in_use = true;
    v.is_new = phase == 1;
    v.play_state = PlayState::Started;
    v.pitch = 1.0f + 0.125f * static_cast<f32>(index);
    v.volume = (phase == 1 ? 0.9f : 0.6f) - 0.1f * static_cast<f32>(index);
    v.priority = 40 - 10 * static_cast<s32>(index);
    v.sort_order = static_cast<s32>(index);
    v.wave_buffer_count = 1;
    v.channel_count = 1;
    v.sample_rate = 48000;
    v.src_quality = SrcQuality::Medium;
    switch (index) {
    case 0:
        v.sample_format = SampleFormat::PcmInt16;
        v.sample_rate = 32000;
        v.mix_id = 1;
        v.biquads[0].enabled = true;
        v.biquads[0].b = {0x3000, 0x1000, 0x800};
        v.biquads[0].a = {-0x2000, 0x400};
        v.biquads[1].enabled = true;
        v.biquads[1].b = {0x2000, -0x800, 0x400};
        v.biquads[1].a = {0x1000, -0x200};
        v.wave_buffer_count = 2;
        v.wave_buffer_internal[0] = {kPool0 + 0x1000, 0x400, 0, 0x200, false, false, false, 0,
                                     0, 0, 0, 0};
        v.wave_buffer_internal[1] = {kPool0 + 0x1400, 0x400, 0, 0x200, true, false, false, 3,
                                     0, 0, 0x20, 0x180};
        v.channel_resource_ids[0] = 0;
        v.flags.IsVoicePlayedSampleCountResetAtLoopPointSupported = true;
        break;
    case 1:
        v.sample_format = SampleFormat::Adpcm;
        v.channel_count = 2;
        v.src_data_address = kPool0 + 0x2000;
        v.src_data_size = 0x20;
        v.wave_buffer_internal[0] = {kPool0 + 0x3000, 0x800, 0, 0x700, true, false, false, 2,
                                     kPool0 + 0x2800, 0x10, 0x10, 0x600};
        if (b.IsSplitterSupported()) {
            v.mix_id = kUnusedMix;
            v.splitter_id = 0;
        } else {
            v.mix_id = 1;
        }
        v.channel_resource_ids[0] = 1;
        v.channel_resource_ids[1] = 2;
        v.src_quality = SrcQuality::High;
        break;
    case 2:
        v.sample_format = SampleFormat::PcmFloat;
        v.mix_id = 2;
        v.play_state = phase == 1 ? PlayState::Started : PlayState::Stopped;
        v.wave_buffer_internal[0] = {kPool0 + 0x4000, 0x800, 0x10, 0x1F0, false, true, false, 0,
                                     0, 0, 0, 0};
        v.channel_resource_ids[0] = 3;
        v.src_quality = SrcQuality::Low;
        break;
    case 3:
        v.sample_format = SampleFormat::PcmInt16;
        v.mix_id = 0;
        v.priority = HighestVoicePriority;
        v.wave_buffer_internal[0] = {kPool0 + 0x5000, 0x400, 0, 0x200, false, false, false, 0,
                                     0, 0, 0, 0};
        v.channel_resource_ids[0] = 4;
        v.flags.IsVoicePitchAndSrcSkippedSupported = phase != 1;
        break;
    case 4:
        v.in_use = false;
        break;
    }
    for (u32 i = 0; i < MaxWaveBuffers; i++) {
        if (i >= v.wave_buffer_count || phase != 1) v.wave_buffer_internal[i].sent_to_DSP = true;
    }
    return v;
}

static EffectInfoBase::InParameterVersion1 GenEffect(const BehaviorInfo& b, u32 index, int phase) {
    EffectInfoBase::InParameterVersion1 e{};
    e.is_new = phase == 1;
    e.enabled = true;
    auto routes = [&](auto& p, u32 channels, s8 first) {
        for (u32 i = 0; i < channels; i++) {
            p.inputs[i] = static_cast<s8>(first + i);
            p.outputs[i] = static_cast<s8>(first + i);
        }
    };
    switch (index) {
    case 0: {
        e.type = EffectInfoBase::Type::Mix;
        e.mix_id = 0;
        e.process_order = 0;
        auto& p = At<BufferMixerInfo::ParameterVersion1>(e.specific);
        p.inputs[0] = 0;
        p.inputs[1] = 1;
        p.inputs[2] = 2;
        p.outputs[0] = 2;
        p.outputs[1] = 3;
        p.outputs[2] = 4;
        p.volumes[0] = 0.5f;
        p.volumes[1] = 0.0f;
        p.volumes[2] = 0.25f;
        p.mix_count = 3;
        break;
    }
    case 1: {
        e.type = EffectInfoBase::Type::Aux;
        e.mix_id = 1;
        e.process_order = 0;
        auto& p = At<AuxInfo::ParameterVersion1>(e.specific);
        routes(p, 2, 0);
        p.mix_buffer_count = 2;
        p.sample_rate = 48000;
        p.count_max = 0x400;
        p.mix_buffer_count_max = 2;
        p.send_buffer_info_address = kPool0 + 0x10000;
        p.send_buffer_address = kPool0 + 0x10040;
        p.return_buffer_info_address = kPool0 + 0x18000;
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
        e.workbuffer = kPool0 + 0x20000;
        e.workbuffer_size = 0x10000;
        auto& p = At<DelayInfo::ParameterVersion1>(e.specific);
        routes(p, 2, 0);
        p.channel_count_max = 2;
        p.channel_count = 2;
        p.delay_time_max = 100;
        p.delay_time = 50;
        p.sample_rate = 48000;
        p.in_gain = 0.5f;
        p.feedback_gain = 0.25f;
        p.wet_gain = 0.5f;
        p.dry_gain = 0.5f;
        p.channel_spread = 0.25f;
        p.lowpass_amount = 0.5f;
        break;
    }
    case 3: {
        e.type = EffectInfoBase::Type::BiquadFilter;
        e.mix_id = 0;
        e.process_order = 1;
        e.enabled = phase == 1;
        if (b.IsEffectInfoVersion2Supported()) {
            auto& p = At<BiquadFilterInfo::ParameterVersion2>(e.specific);
            routes(p, 2, 4);
            p.b = {0.5f, 0.25f, 3.0f};
            p.a = {-0.5f, 0.0625f};
            p.channel_count = 2;
        } else {
            auto& p = At<BiquadFilterInfo::ParameterVersion1>(e.specific);
            routes(p, 2, 4);
            p.b = {0x2000, 0x1000, 0x800};
            p.a = {-0x2000, 0x400};
            p.channel_count = 2;
        }
        break;
    }
    case 4: {
        e.type = EffectInfoBase::Type::LightLimiter;
        e.mix_id = 0;
        e.process_order = 2;
        e.workbuffer = kPool0 + 0x30000;
        e.workbuffer_size = 0x8000;
        auto& p = At<LightLimiterInfo::ParameterVersion1>(e.specific);
        routes(p, 2, 0);
        p.channel_count_max = 2;
        p.channel_count = 2;
        p.sample_rate = 48000;
        p.attack_time = 1000;
        p.release_time = 2000;
        p.look_ahead_time_max = 1000;
        p.look_ahead_time = 500;
        p.threshold = 0.5f;
        p.input_gain = 1.0f;
        p.output_gain = 1.0f;
        p.statistics_enabled = true;
        p.statistics_reset_required = phase != 1;
        break;
    }
    case 5: {
        e.type = EffectInfoBase::Type::Reverb;
        e.mix_id = 0;
        e.process_order = 3;
        e.workbuffer = kPool0 + 0x40000;
        e.workbuffer_size = 0x20000;
        auto& p = At<ReverbInfo::ParameterVersion1>(e.specific);
        routes(p, 6, 0);
        p.channel_count_max = 6;
        p.channel_count = 6;
        p.sample_rate = 48000;
        p.early_mode = 1;
        p.early_gain = 0x4000;
        p.pre_delay = 0x2000;
        p.late_mode = 2;
        p.late_gain = 0x4000;
        p.decay_time = 0x8000;
        p.high_freq_Decay_ratio = 0x4000;
        p.colouration = 0x2000;
        p.base_gain = 0x4000;
        p.wet_gain = 0x4000;
        p.dry_gain = 0x4000;
        break;
    }
    case 6: {
        e.type = EffectInfoBase::Type::I3dl2Reverb;
        e.mix_id = 2;
        e.process_order = 0;
        e.workbuffer = kPool0 + 0x60000;
        e.workbuffer_size = 0x20000;
        auto& p = At<I3dl2ReverbInfo::ParameterVersion1>(e.specific);
        routes(p, 6, 2);
        p.channel_count_max = 6;
        p.channel_count = 6;
        p.sample_rate = 48000;
        p.room_HF_gain = -100.0f;
        p.reference_HF = 5000.0f;
        p.late_reverb_decay_time = 1.5f;
        p.late_reverb_HF_decay_ratio = 0.8f;
        p.room_gain = -1000.0f;
        p.reflection_gain = -500.0f;
        p.reverb_gain = -200.0f;
        p.late_reverb_diffusion = 100.0f;
        p.reflection_delay = 0.02f;
        p.late_reverb_delay_time = 0.04f;
        p.late_reverb_density = 100.0f;
        p.dry_gain = 1.0f;
        break;
    }
    case 7: {
        e.type = EffectInfoBase::Type::Capture;
        e.mix_id = 2;
        e.process_order = 1;
        auto& p = At<AuxInfo::ParameterVersion1>(e.specific);
        routes(p, 2, 0);
        p.mix_buffer_count = 2;
        p.sample_rate = 48000;
        p.count_max = 0x800;
        p.mix_buffer_count_max = 2;
        p.send_buffer_info_address = kPool0 + 0x80000;
        p.send_buffer_address = kPool0 + 0x80040;
        p.mix_buffer_sample_size = 4;
        p.sample_count = 240;
        p.mix_buffer_sample_count = 0x800;
        break;
    }
    case 8: {
        e.type = EffectInfoBase::Type::Compressor;
        e.mix_id = 1;
        e.process_order = 2;
        e.enabled = phase != 3;
        auto& p = At<CompressorInfo::ParameterVersion1>(e.specific);
        routes(p, 2, 0);
        p.channel_count_max = 2;
        p.channel_count = 2;
        p.sample_rate = 48000;
        p.threshold = -12.0f;
        p.compressor_ratio = 4.0f;
        p.attack_time = 10;
        p.release_time = 100;
        p.out_gain = 1.0f;
        p.makeup_gain_enabled = true;
        break;
    }
    }
    return e;
}

static MixInfo::InParameter GenMix(const BehaviorInfo& b, u32 index, int phase) {
    MixInfo::InParameter m{};
    m.mix_id = static_cast<s32>(index);
    m.node_id = static_cast<s32>((2u << 28) | (index << 16));
    m.sample_rate = 48000;
    m.dest_splitter_id = static_cast<s32>(kUnusedSplitter);
    m.dest_mix_id = static_cast<s32>(kUnusedMix);
    m.is_dirty = true;
    m.in_use = true;
    m.effect_count = kGenEffects;
    for (u32 i = 0; i < MaxMixBuffers; i++)
        for (u32 j = 0; j < MaxMixBuffers; j++)
            if ((i + j) % 3 == 0) m.mix_volumes[i][j] = 0.5f / static_cast<f32>(1 + i + j);
    switch (index) {
    case 0:
        m.volume = phase == 1 ? 1.0f : 0.875f;
        m.buffer_count = 6;
        break;
    case 1:
        m.volume = 0.75f;
        m.buffer_count = 2;
        if (b.IsSplitterSupported()) {
            m.dest_splitter_id = 1;
        } else {
            m.dest_mix_id = 0;
        }
        break;
    case 2:
        m.volume = phase == 1 ? 0.5f : 0.625f;
        m.sample_rate = 32000;
        m.buffer_count = 12;
        m.dest_mix_id = 1;
        break;
    }
    return m;
}

static SinkInfoBase::InParameter GenSink(u32 index, int phase) {
    SinkInfoBase::InParameter s{};
    s.node_id = (3u << 28) | (index << 16);
    s.in_use = true;
    if (index == 0) {
        s.type = SinkInfoBase::Type::DeviceSink;
        std::strcpy(s.device.name, "MainAudioOut");
        s.device.input_count = phase == 1 ? 2 : 6;
        for (u32 i = 0; i < MaxChannels; i++) s.device.inputs[i] = static_cast<s8>(i);
        s.device.downmix_enabled = phase != 3;
        s.device.downmix_coeff = {1.0f, 0.707f, 0.5f, 0.25f};
    } else {
        s.type = SinkInfoBase::Type::CircularBufferSink;
        s.circular_buffer.cpu_address = kPool0 + 0x90000;
        s.circular_buffer.size = 0x2000;
        s.circular_buffer.input_count = 2;
        s.circular_buffer.sample_count = 240;
        s.circular_buffer.format = SampleFormat::PcmInt16;
        s.circular_buffer.inputs = {2, 3, 0, 0, 0, 0};
        s.circular_buffer.in_use = true;
    }
    return s;
}

static std::vector<u8> BuildGenInput(const BehaviorInfo& b, u32 revision, int phase) {
    Builder out;
    auto& h = out.header;
    h.revision = revision;

    auto mark = out.Mark();
    BehaviorInfo::InParameter behavior{};
    behavior.revision = revision;
    out.Push(behavior);
    h.behaviour_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kGenPools; i++) {
        MemoryPoolInfo::InParameter p{};
        if (i == 0)
            p = {kPool0, 0x100000,
                 phase == 1 ? MemoryPoolInfo::State::RequestAttach : MemoryPoolInfo::State::Attached,
                 true};
        out.Push(p);
    }
    h.memory_pool_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kGenVoices; i++) {
        VoiceChannelResource::InParameter r{};
        r.id = i;
        r.in_use = true;
        for (u32 j = 0; j < MaxMixBuffers; j++)
            if ((i + j) % 4 != 3)
                r.mix_volumes[j] = 0.125f * static_cast<f32>(i + 1) +
                                   0.0625f * static_cast<f32>(j) +
                                   (phase == 1 ? 0.0f : 0.25f);
        out.Push(r);
    }
    h.voice_resources_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kGenVoices; i++) PushVoice(out, b, GenVoice(b, i, phase), 0);
    h.voices_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kGenEffects; i++) out.Push(GenEffect(b, i, phase));
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
        dirty.count = kSubMixes + 1;
        out.Push(dirty);
    }
    for (u32 i = 0; i < kSubMixes + 1; i++) out.Push(GenMix(b, i, phase));
    h.mix_size = out.Since(mark);

    mark = out.Mark();
    for (u32 i = 0; i < kSinks; i++) out.Push(GenSink(i, phase));
    h.sinks_size = out.Since(mark);

    mark = out.Mark();
    PerformanceManager::InParameter perf{};
    perf.target_node_id = (1u << 28) | (1u << 16);
    out.Push(perf);
    h.performance_buffer_size = out.Since(mark);
    return out.Finish();
}

static size_t GenOutputSize(const BehaviorInfo& b) {
    size_t size = sizeof(UpdateDataHeader);
    size += kGenPools * sizeof(MemoryPoolInfo::OutStatus);
    size += kGenVoices * sizeof(VoiceInfo::OutStatus);
    size += kGenEffects * (b.IsEffectInfoVersion2Supported()
                               ? sizeof(EffectInfoBase::OutStatusVersion2)
                               : sizeof(EffectInfoBase::OutStatusVersion1));
    size += kSinks * sizeof(SinkInfoBase::OutStatus);
    size += sizeof(PerformanceManager::OutStatus);
    size += sizeof(BehaviorInfo::OutStatus);
    if (b.IsElapsedFrameCountSupported()) size += 0x10;
    return size;
}

struct CommandCase {
    const char* name;
    u32 open_revision;
    u32 perf_frames;
    bool active;
    u32 revision;
    u32 sample_rate{48000};
    u32 sample_count{240};
    s8 channels{2};
    bool drop_voice{};
    u32 render_time_limit_percent{100};
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: commands <output dir>\n");
        return 1;
    }
    const u32 rev1 = EncodeRevision(1, false);
    const u32 rev5 = EncodeRevision(5, false);
    const u32 rev9 = EncodeRevision(9, false);
    const u32 rev10 = EncodeRevision(10, false);
    const u32 rev13 = EncodeRevision(13, false);
    const u32 rev15 = EncodeRevision(15, false);
    const u32 nx15 = EncodeRevision(15, true);
    const u32 nx16 = EncodeRevision(16, true);
    std::vector<CommandCase> cases{
        {"commands_rev1", rev1, 0, false, rev1},
        {"commands_rev5_perf", rev5, 2, true, rev5},
        {"commands_rev9_effects_v2", rev9, 1, true, rev9},
        {"commands_rev10_6ch", rev10, 0, true, rev10, 48000, 240, 6},
        {"commands_rev13_splitter_v2a", rev13, 0, false, rev13},
        {"commands_rev15_current", rev15, 1, true, rev15},
        {"commands_rev15_32k", rev15, 1, true, rev15, 32000, 160},
        {"commands_rev15_native", nx15, 0, false, nx15},
        {"commands_rev16_native", nx16, 1, true, nx16},
        {"commands_rev16_voice_drop", nx16, 1, true, nx16, 48000, 240, 2, true, 3},
        {"commands_rev5_voice_drop", rev5, 0, true, rev5, 48000, 240, 2, true, 5},
    };
    constexpr int kPhases = 3;

    for (const auto& c : cases) {
        AudioRendererParameterInternal params{};
        params.sample_rate = c.sample_rate;
        params.sample_count = c.sample_count;
        params.mixes = 24;
        params.sub_mixes = kSubMixes;
        params.voices = kGenVoices;
        params.sinks = kSinks;
        params.effects = kGenEffects;
        params.perf_frames = c.perf_frames;
        params.splitter_infos = 2;
        params.splitter_destinations = 4;
        params.revision = c.open_revision;
        params.voice_drop_enabled = c.drop_voice;

        Golden golden;
        auto init = golden.Initialize(params, GetWorkBufferSize(params));
        if (init.IsError()) {
            std::fprintf(stderr, "%s: initialize failed %08x\n", c.name, init.raw);
            return 1;
        }
        golden.active = c.active;
        // PoolMapper::InitializeSystemPool maps the work buffer at its own address.
        golden.memory_pool_info.cpu_address = reinterpret_cast<CpuAddr>(golden.workbuffer.get());
        golden.memory_pool_info.dsp_address = golden.memory_pool_info.cpu_address;
        golden.memory_pool_info.size = golden.workbuffer_size;
        golden.memory_pool_info.location = MemoryPoolInfo::Location::DSP;

        Session session;
        session.sample_rate = c.sample_rate;
        session.channels = c.channels;
        session.drop_voice = c.drop_voice;
        session.render_time_limit_percent = c.render_time_limit_percent;
        session.estimator = MakeEstimator(golden.behavior, golden.sample_count,
                                          golden.mix_buffer_count);
        BehaviorInfo layout;
        layout.SetUserLibRevision(c.open_revision);

        std::vector<u8> fixture;
        Append(fixture, "LRCG", 4);
        Append32(fixture, 1);
        Append(fixture, &params, sizeof(params));
        Append32(fixture, c.active);
        Append32(fixture, static_cast<u32>(c.channels));
        Append32(fixture, c.drop_voice);
        Append32(fixture, c.render_time_limit_percent);
        Append32(fixture, kPhases);
        for (int phase = 1; phase <= kPhases; phase++) {
            const auto input = BuildGenInput(layout, c.revision, phase);
            std::vector<u8> output(GenOutputSize(layout));
            const size_t perf_size =
                c.perf_frames > 0
                    ? PerformanceManager::GetRequiredBufferSizeForPerformanceMetricsPerFrame(
                          layout, params) * c.perf_frames
                    : 0;
            std::vector<u8> perf(perf_size);
            const auto result = golden.Update(input, perf, output);
            if (result.IsError()) {
                std::fprintf(stderr, "%s: update failed %08x\n", c.name, result.raw);
                return 1;
            }
            const auto count = GenerateCommand(golden, session);
            Dump dump{golden};
            dump.List(count);
            Digest digest;
            DigestState(digest, golden, params);

            Append32(fixture, static_cast<u32>(input.size()));
            Append32(fixture, static_cast<u32>(output.size()));
            Append32(fixture, static_cast<u32>(perf.size()));
            Append(fixture, input.data(), input.size());
            Append32(fixture, count);
            Append32(fixture, session.num_voices_dropped);
            Append32(fixture, static_cast<u32>(dump.words.size()));
            Append(fixture, dump.words.data(), dump.words.size() * 4);
            Append32(fixture, static_cast<u32>(digest.words.size()));
            Append(fixture, digest.words.data(), digest.words.size() * 4);
            std::printf("%s phase %d: %u commands, %u voices dropped\n", c.name, phase,
                        count, session.num_voices_dropped);
        }
        const std::string path = std::string(argv[1]) + "/" + c.name + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return 1;
        std::fwrite(fixture.data(), 1, fixture.size(), f);
        std::fclose(f);
    }
    return 0;
}
