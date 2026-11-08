// Native sound for Gitaroo Man: the game's IOP sound drivers answered on the host.
//
// The game reaches sound only through two IOP services over SIF RPC (docs/native-audio.md):
// MSTREAM (SID 0x1563, iNiS's streaming driver: music, effects, the song clock) and SDRDRV
// (SID 0x80000701, Sony's remote LIBSD: master volume and the movies' PCM ring). On the PS2 they
// drive SPU2 voices from ring buffers refilled off the disc; here each MSTREAM stream channel
// becomes a host voice that reads its PS-ADPCM data straight from the IMC library, and the mixer
// produces the 48 kHz output.
//
// What the game can observe is kept as MSTREAM has it: the reply words of every command (the
// reply buffer is never cleared, so stale words stay), stream slot and voice allocation (a voice
// keeps its stored volume and pan across streams), the status word, which MSTREAM changes on its
// buffer-refill schedule rather than when the sound is heard, and the song clock in 28-sample
// blocks. Function addresses in comments are MSTREAM's (base 0x4AA00) and LIBSD's.
#include "ps2_native_sound.h"

#include "ps2_host_audio.h"
#include "ps2_runtime.h"
#include "ps2x/iop/iop_audio.h"
#include "ps2x/iop/ps2_path.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace ps2_native_sound
{
    namespace
    {
        constexpr uint32_t kMstreamSid = 0x1563u;
        constexpr uint32_t kSdrSid = 0x80000701u;
        constexpr uint64_t kEeCyclesPerFrame = 6144u; // 294.912 MHz / 48 kHz, the SPU2's sample period
        constexpr int kSlots = 256;
        constexpr int kVoices = 24;
        constexpr int kQueueEntries = 256;
        constexpr int kLibraryEntries = 128;

        bool envOn(const char *name)
        {
            const char *value = std::getenv(name);
            return value && *value && std::strcmp(value, "0") != 0;
        }

        bool traceOn()
        {
            static const bool on = envOn("PS2X_TRACE_NATIVE_AUDIO");
            return on;
        }

#define NATIVE_AUDIO_TRACE(...)                        \
    do                                                 \
    {                                                  \
        if (traceOn())                                 \
        {                                              \
            std::fprintf(stderr, "[native-audio] " __VA_ARGS__); \
        }                                              \
    } while (0)

        int32_t clamp16(int32_t value)
        {
            return std::clamp(value, -32768, 32767);
        }

        int32_t mulVol(int32_t sample, int32_t level)
        {
            return (sample * level) >> 15;
        }

        // A fixed-volume register (bit 15 clear) as a level, like the SPU2.
        int32_t volumeLevel(uint16_t reg)
        {
            return static_cast<int16_t>(static_cast<uint16_t>(reg << 1u));
        }

        int32_t catmullRom(const std::array<int32_t, 4> &w, uint32_t frac)
        {
            const int64_t t = frac;
            const int64_t a = -w[0] + 3 * w[1] - 3 * w[2] + w[3];
            const int64_t b = 2 * w[0] - 5 * w[1] + 4 * w[2] - w[3];
            const int64_t c = -w[0] + w[2];
            int64_t v = (a * t) >> 12;
            v = ((v + b) * t) >> 12;
            v = ((v + c) * t) >> 12;
            return clamp16(static_cast<int32_t>(w[1] + v / 2));
        }

        // The SPU pitch for a stream's sample rate (0x4EB40); 0 (silent) for any other rate.
        uint16_t pitchForRate(int32_t rate)
        {
            switch (rate)
            {
            case 11025: return 0x3AD;
            case 16000: return 0x555;
            case 22050: return 0x75A;
            case 32000: return 0xAAB;
            case 44100: return 0xEB3;
            case 48000: return 0x1000;
            default: return 0;
            }
        }

        uint32_t readU32(const uint8_t *p)
        {
            return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                   (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
        }

        // PS-ADPCM data of one sound as its voices read it: channel c's block b is at 16-byte
        // index ((b / span) * channels + c) * span + b % span, which is the IMC layout of a stream
        // (span = interleave) and channel-major for sounds loaded whole (span = all its blocks).
        struct SoundData
        {
            uint32_t channels = 1;
            uint32_t span = 1;
            uint32_t blocksPerChannel = 0;
            std::vector<uint8_t> bytes;
            std::atomic<bool> ready{false};
            std::atomic<bool> failed{false};

            const uint8_t *block(uint32_t channel, uint32_t index) const
            {
                const size_t at = (static_cast<size_t>(index / span) * channels + channel) * span + index % span;
                return at * 16u + 16u <= bytes.size() ? bytes.data() + at * 16u : nullptr;
            }
        };

        // Reads stream data off the disc ahead of play, so neither the game thread nor the audio
        // thread waits on file reads.
        class Loader
        {
        public:
            struct Job
            {
                std::shared_ptr<SoundData> data;
                std::string hostPath;
                uint64_t offset = 0;
                uint64_t size = 0;
            };

            Loader()
                : m_thread([this] { run(); })
            {
                m_thread.detach();
            }

            void enqueue(Job job)
            {
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_jobs.push_back(std::move(job));
                }
                m_wake.notify_one();
            }

            void wait(const std::shared_ptr<SoundData> &data)
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_done.wait(lock, [&] { return data->ready.load() || data->failed.load(); });
            }

            static bool readRange(const std::string &path, uint64_t offset, uint64_t size, std::vector<uint8_t> &out)
            {
                std::ifstream file(path, std::ios::binary);
                if (!file)
                    return false;
                file.seekg(static_cast<std::streamoff>(offset));
                out.resize(static_cast<size_t>(size));
                file.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(size));
                out.resize(static_cast<size_t>(file.gcount()));
                return file.gcount() > 0;
            }

        private:
            void run()
            {
                for (;;)
                {
                    Job job;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_wake.wait(lock, [&] { return !m_jobs.empty(); });
                        job = std::move(m_jobs.front());
                        m_jobs.pop_front();
                    }
                    // Data nobody holds any more (closed before it loaded) is skipped.
                    if (job.data.use_count() > 1 && readRange(job.hostPath, job.offset, job.size, job.data->bytes))
                        job.data->ready.store(true, std::memory_order_release);
                    else
                        job.data->failed.store(true);
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                    }
                    m_done.notify_all();
                }
            }

            std::mutex m_mutex;
            std::condition_variable m_wake;
            std::condition_variable m_done;
            std::deque<Job> m_jobs;
            std::thread m_thread;
        };

        struct Voice
        {
            // What the drivers set through LIBSD: fixed volumes and pitch.
            uint16_t volL = 0;
            uint16_t volR = 0;
            uint16_t pitch = 0;

            // Keyed on and not off: the voice's timeline runs (the refill schedule and the song
            // clock follow it), whether or not it still has data to play.
            bool keyed = false;
            bool sounding = false;
            std::shared_ptr<const SoundData> data;
            uint32_t channel = 0;
            uint32_t block = 0;
            uint32_t loopBlock = 0;
            bool endAfterBlock = false;
            int16_t decoded[28]{};
            uint32_t decodedPos = 28;
            int32_t hist1 = 0;
            int32_t hist2 = 0;
            std::array<int32_t, 4> window{};
            uint32_t phase = 0;
            uint64_t samples = 0; // consumed since key-on

            // A seamless section change: from timeline block `switchAt` on, play `next` from its start.
            std::shared_ptr<const SoundData> next;
            uint32_t nextChannel = 0;
            uint64_t switchAt = UINT64_MAX;
        };

        // MSTREAM's per-voice record (0x54B20 + core * 0x120 + voice * 12).
        struct VoiceRecord
        {
            bool used = false;
            uint16_t volume = 0;
            uint16_t pan = 0;
            int32_t mute = 0;
        };

        // MSTREAM's 0xF8-byte stream record (0x55FB0 + id * 0xF8), the fields that matter.
        struct Slot
        {
            int32_t kind = 0; // +0x00: 0 free, 1 library stream, 3 loaded whole, 4 from EE memory
            int32_t channels = 0, rate = 0, interleave = 0, length = 0; // +0x04..+0x10, the IMC header
            int32_t shareBase = 0;   // +0x14: -1, or the stream whose voices this one plays on
            int32_t fromLibrary = 0; // +0x18
            int32_t entry = 0;       // +0x1C
            int32_t voices[8]{};     // +0x20
            int32_t core = 0;        // +0x60
            uint16_t pitch = 0;      // +0x64
            int32_t rows = 0;        // +0x68: length / channels, chunks of `interleave` blocks per channel
            int32_t consumed = 0;    // +0x7C: blocks of the IOP buffer uploaded, all channels
            int32_t status = 0;      // +0x84
            int32_t preloading = 0;  // +0xEC
            std::shared_ptr<SoundData> data;
        };

        // A stream being refilled (MSTREAM's per-core refill list at 0x522C0, one entry per channel;
        // all channels of a stream move together, so one record per stream). Rounds fall on the
        // current stream's chunk boundaries: MSTREAM points the SPU IRQ at the start of the next
        // chunk, and after a section change at the new section's chunks, which the split shifted.
        struct Run
        {
            int32_t slot = -1;
            uint32_t span = 0;        // SPU buffer size in blocks (+0x74, the interleave)
            uint64_t streamStart = 0; // timeline block where the current stream began
            uint32_t nextRound = 1;   // the next round is at streamStart + nextRound * span
            int32_t counter = 0;      // entry +0x18
            int32_t limit = 0;        // entry +0x14
        };

        struct LibraryEntry
        {
            char name[9]{};
            uint32_t offset = 0;
            uint32_t kind = 0;
        };

        struct CoreRegs
        {
            uint16_t mvolL = 0, mvolR = 0, evolL = 0, evolR = 0;
            int16_t avolL = 0, avolR = 0, bvolL = 0, bvolR = 0;
        };

        // The movies' sound: SPU2 AutoDMA from a PCM ring in IOP memory the EE writes over SIF DMA
        // (blocks of 256 left then 256 right samples). The DMA moves each block into the SPU2 two
        // blocks before it plays, and the EE may then overwrite it in the ring: `ring` is IOP
        // memory, `fetched` what the DMA took from it.
        struct AutoDma
        {
            bool active = false;
            uint32_t base = 0;
            uint32_t size = 0;
            std::vector<uint8_t> ring;
            std::vector<uint8_t> fetched;
            uint64_t frames = 0;

            uint32_t blocks() const { return size / 0x400u; }

            void fetch(uint64_t block)
            {
                const size_t at = static_cast<size_t>(block % blocks()) * 0x400u;
                std::memcpy(fetched.data() + at, ring.data() + at, 0x400u);
            }
        };

        class Engine
        {
        public:
            Engine()
            {
                // The emulated SPU2 (still stepped while other IOP modules run) must not feed the device.
                ps2x::iop::setAudioSink(nullptr, nullptr);
                sdInit();
                const char *clock = std::getenv("PS2X_NATIVE_AUDIO_CLOCK");
                m_eeClock = clock && std::strcmp(clock, "ee") == 0;
                if (!m_eeClock && !ps2_host_audio::setPullSource(&Engine::pull, this))
                {
                    std::fprintf(stderr, "[native-audio] no audio device; mixing on emulated time\n");
                    m_eeClock = true;
                }
                if (const char *offset = std::getenv("PS2X_NATIVE_AUDIO_CLOCK_OFFSET_MS"))
                    m_clockOffsetFrames = static_cast<int64_t>(std::strtod(offset, nullptr) * 48.0);
                else if (!m_eeClock)
                    m_clockOffsetFrames = ps2_host_audio::deviceBufferedFrames();
                std::printf("[native-audio] native sound drivers, %s clock, song clock %lld frames behind the mix\n",
                            m_eeClock ? "emulated" : "device", static_cast<long long>(m_clockOffsetFrames));
            }

            ps2x::iop::RpcResult handle(PS2Runtime &runtime, uint8_t *rdram, const ps2x::iop::RpcRequest &request)
            {
                std::array<uint32_t, 16> args{};
                const uint8_t *send = request.send.address ? getConstMemPtr(rdram, request.send.address) : nullptr;
                if (send)
                    std::memcpy(args.data(), send, std::min<size_t>(request.send.size, sizeof(args)));
                std::array<uint8_t, 64> raw{};
                if (send)
                    std::memcpy(raw.data(), send, std::min<size_t>(request.send.size, raw.size()));

                std::array<uint32_t, 4> reply{};
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_runtime = &runtime;
                    if (request.sid == kMstreamSid)
                    {
                        mstream(runtime, request.function, args, raw);
                        reply = m_reply;
                    }
                    else
                    {
                        sdr(request.function, args);
                        reply = {m_sdrReply, 0u, 0u, 0u};
                    }
                }
                if (request.receive.address && request.receive.size)
                {
                    if (uint8_t *out = getMemPtr(rdram, request.receive.address))
                    {
                        std::array<uint8_t, 16> bytes{};
                        std::memcpy(bytes.data(), reply.data(), bytes.size());
                        const size_t n = std::min<size_t>(request.receive.size, bytes.size());
                        std::memcpy(out, bytes.data(), n);
                        if (request.receive.size > n)
                            std::memset(out + n, 0, request.receive.size - n);
                    }
                }
                ps2x::iop::RpcResult result;
                result.handled = true;
                result.resultAddress = request.receive.address;
                return result;
            }

            void onSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
            {
                if (transfer.kind != ps2x::iop::SifTransferKind::SetDma ||
                    transfer.phase != ps2x::iop::SifTransferPhase::AfterCopy)
                    return;
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_adma.ring.empty())
                    return;
                const uint32_t dest = transfer.destinationAddress & 0x1FFFFFu;
                const uint32_t begin = std::max(dest, m_adma.base);
                const uint32_t end = std::min(dest + transfer.size, m_adma.base + m_adma.size);
                if (begin >= end)
                    return;
                if (const uint8_t *src = getConstMemPtr(rdram, transfer.sourceAddress + (begin - dest)))
                    std::memcpy(m_adma.ring.data() + (begin - m_adma.base), src, end - begin);
            }

            void advanceEeCycles(uint64_t eeCycles)
            {
                if (!m_eeClock)
                    return;
                m_eeCycles += eeCycles;
                const uint64_t frames = m_eeCycles / kEeCyclesPerFrame;
                if (frames == 0)
                    return;
                m_eeCycles -= frames * kEeCyclesPerFrame;
                m_scratch.resize(static_cast<size_t>(frames) * 2u);
                render(m_scratch.data(), static_cast<uint32_t>(frames));
                ps2_host_audio::push(m_scratch.data(), static_cast<size_t>(frames));
            }

        private:
            static void pull(void *user, int16_t *out, uint32_t frames)
            {
                static_cast<Engine *>(user)->render(out, frames);
            }

            // ---------------------------------------------------------------------------------
            // Voices
            // ---------------------------------------------------------------------------------

            Voice &voice(int32_t core, int32_t index)
            {
                return m_voices[core & 1][static_cast<uint32_t>(index) % kVoices];
            }

            void keyOn(int32_t core, uint32_t mask)
            {
                for (int32_t i = 0; i < kVoices; ++i)
                {
                    if ((mask & (1u << i)) == 0u)
                        continue;
                    Voice &v = voice(core, i);
                    v.keyed = true;
                    v.sounding = v.data != nullptr;
                    v.block = 0;
                    v.loopBlock = 0;
                    v.endAfterBlock = false;
                    v.decodedPos = 28;
                    v.hist1 = v.hist2 = 0;
                    v.window = {};
                    v.phase = 0;
                    v.samples = 0;
                    v.next.reset();
                    v.switchAt = UINT64_MAX;
                }
            }

            void keyOff(int32_t core, uint32_t mask)
            {
                for (int32_t i = 0; i < kVoices; ++i)
                {
                    if ((mask & (1u << i)) == 0u)
                        continue;
                    Voice &v = voice(core, i);
                    v.keyed = false;
                    v.sounding = false;
                    v.next.reset();
                    v.switchAt = UINT64_MAX;
                }
            }

            void fetchBlock(Voice &v)
            {
                v.decodedPos = 0;
                if (v.endAfterBlock)
                {
                    v.sounding = false;
                    v.endAfterBlock = false;
                }
                if (v.next && v.samples / 28u == v.switchAt)
                {
                    v.data = std::move(v.next);
                    v.channel = v.nextChannel;
                    v.next.reset();
                    v.switchAt = UINT64_MAX;
                    v.block = 0;
                    v.loopBlock = 0;
                    v.sounding = true;
                }
                const SoundData *data = v.data.get();
                const uint8_t *b = nullptr;
                if (v.sounding && data)
                {
                    if (!data->ready.load(std::memory_order_acquire))
                    {
                        if (!m_warnedUnloaded)
                            NATIVE_AUDIO_TRACE("voice reached data still loading\n");
                        m_warnedUnloaded = true;
                        ++v.block;
                    }
                    else if (v.block >= data->blocksPerChannel || !(b = data->block(v.channel, v.block)))
                    {
                        v.sounding = false;
                    }
                }
                if (!b)
                {
                    std::fill(std::begin(v.decoded), std::end(v.decoded), int16_t{0});
                    return;
                }

                static constexpr int32_t kPos[5] = {0, 60, 115, 98, 122};
                static constexpr int32_t kNeg[5] = {0, 0, -52, -55, -60};
                uint32_t shift = b[0] & 0x0Fu;
                const uint32_t filter = std::min<uint32_t>((b[0] >> 4) & 7u, 4u);
                const uint8_t flags = b[1];
                if (shift > 12u)
                    shift = 9u;
                for (uint32_t i = 0; i < 28u; ++i)
                {
                    const int32_t nibble = (b[2 + i / 2] >> ((i & 1u) * 4u)) & 0xF;
                    int32_t sample = static_cast<int16_t>(static_cast<uint16_t>(nibble << 12)) >> shift;
                    sample += (v.hist1 * kPos[filter] + v.hist2 * kNeg[filter] + 32) >> 6;
                    sample = clamp16(sample);
                    v.hist2 = v.hist1;
                    v.hist1 = sample;
                    v.decoded[i] = static_cast<int16_t>(sample);
                }
                // Loop flags, as the data and MSTREAM leave them: 4 marks the loop start; 1 ends the
                // block, then jumps back with 2 (repeat) or stops the voice without.
                if (flags & 4u)
                    v.loopBlock = v.block;
                if (flags & 1u)
                {
                    if (flags & 2u)
                        v.block = v.loopBlock;
                    else
                    {
                        v.endAfterBlock = true;
                        ++v.block;
                    }
                }
                else
                {
                    ++v.block;
                }
            }

            int32_t voiceSample(Voice &v)
            {
                const int32_t out = catmullRom(v.window, v.phase & 0xFFFu);
                v.phase += std::min<uint32_t>(v.pitch, 0x3FFFu);
                while (v.phase >= 0x1000u)
                {
                    v.phase -= 0x1000u;
                    if (v.decodedPos >= 28u)
                        fetchBlock(v);
                    v.window[0] = v.window[1];
                    v.window[1] = v.window[2];
                    v.window[2] = v.window[3];
                    v.window[3] = v.decoded[v.decodedPos++];
                    ++v.samples;
                }
                return mulVol(out, 0x7FFF);
            }

            // ---------------------------------------------------------------------------------
            // Mixer
            // ---------------------------------------------------------------------------------

            void render(int16_t *out, uint32_t frames)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                for (uint32_t f = 0; f < frames; ++f)
                {
                    int32_t dry[2][2] = {};
                    for (int32_t c = 0; c < 2; ++c)
                    {
                        for (Voice &v : m_voices[c])
                        {
                            if (!v.keyed)
                                continue;
                            const int32_t s = voiceSample(v);
                            dry[c][0] += mulVol(s, volumeLevel(v.volL));
                            dry[c][1] += mulVol(s, volumeLevel(v.volR));
                        }
                    }

                    int32_t inL = 0, inR = 0;
                    if (m_adma.active && m_adma.size >= 0x400u)
                    {
                        if (m_adma.frames % 256u == 0u)
                            m_adma.fetch(m_adma.frames / 256u + 2u);
                        const uint32_t block = static_cast<uint32_t>((m_adma.frames / 256u) % m_adma.blocks());
                        const uint32_t at = block * 0x400u + static_cast<uint32_t>(m_adma.frames % 256u) * 2u;
                        const std::vector<uint8_t> &f = m_adma.fetched;
                        inL = static_cast<int16_t>(f[at] | (f[at + 1] << 8));
                        inR = static_cast<int16_t>(f[at + 0x200] | (f[at + 0x201] << 8));
                        ++m_adma.frames;
                    }
                    const CoreRegs &c0 = m_cores[0];
                    const CoreRegs &c1 = m_cores[1];
                    const int32_t core0L = clamp16(mulVol(clamp16(clamp16(dry[0][0]) + mulVol(inL, c0.bvolL)), volumeLevel(c0.mvolL)));
                    const int32_t core0R = clamp16(mulVol(clamp16(clamp16(dry[0][1]) + mulVol(inR, c0.bvolR)), volumeLevel(c0.mvolR)));
                    const int32_t mixL = clamp16(clamp16(dry[1][0]) + mulVol(core0L, c1.avolL));
                    const int32_t mixR = clamp16(clamp16(dry[1][1]) + mulVol(core0R, c1.avolR));
                    out[f * 2u] = static_cast<int16_t>(clamp16(mulVol(mixL, volumeLevel(c1.mvolL))));
                    out[f * 2u + 1u] = static_cast<int16_t>(clamp16(mulVol(mixR, volumeLevel(c1.mvolR))));

                    if (m_threads)
                        refillRounds();
                }
                m_framesRendered += frames;
            }

            // ---------------------------------------------------------------------------------
            // MSTREAM's refill schedule (the refill threads 0x4D020 and FUN_0004cc4c), kept for
            // what it makes visible: when a stream's status changes. On the PS2 a refill round runs
            // each time the lead voice enters the next SPU buffer (its IRQ); the round uploads the
            // buffer after it, so a section change and a stream's end show in the status a buffer
            // or two before they are heard.
            // ---------------------------------------------------------------------------------

            void refillRounds()
            {
                for (size_t i = 0; i < m_runs.size();)
                {
                    Run &run = m_runs[i];
                    Slot &s = m_slots[run.slot];
                    const Voice &lead = voice(s.core, s.voices[0]);
                    if (!lead.keyed || run.span == 0u ||
                        lead.samples / 28u < run.streamStart + static_cast<uint64_t>(run.nextRound) * run.span)
                    {
                        ++i;
                        continue;
                    }
                    if (!refillRound(run))
                        m_runs.erase(m_runs.begin() + static_cast<std::ptrdiff_t>(i));
                    else
                        ++i;
                }
            }

            // One round: the uploads (FUN_0004cc4c per channel), which may switch to the queued
            // section, then the end check. Returns false once the stream has left the refill list.
            bool refillRound(Run &run)
            {
                const uint32_t span = run.span;
                // Blocks of the current stream in the voice's buffers before this round's upload.
                const uint32_t uploaded = (run.nextRound + 1u) * span;
                const uint32_t pos = m_eventPos;
                ++run.nextRound;
                if (pos != 0xFFFFFFFFu && m_eventSlot >= 0 && pos - 2u * span <= uploaded && uploaded >= pos - span)
                {
                    Slot &from = m_slots[run.slot];
                    Slot &to = m_slots[m_eventSlot];
                    const uint64_t switchAt = run.streamStart + pos;
                    for (int32_t c = 0; c < from.channels && c < 8; ++c)
                    {
                        Voice &v = voice(from.core, from.voices[c]);
                        v.next = to.data;
                        v.nextChannel = static_cast<uint32_t>(c);
                        v.switchAt = switchAt;
                    }
                    NATIVE_AUDIO_TRACE("section change: stream %d -> %d at block %u (timeline %llu), status flips now\n", run.slot,
                                       m_eventSlot, pos, static_cast<unsigned long long>(switchAt));
                    from.status = 0;
                    to.status = 1;
                    run.slot = m_eventSlot;
                    run.streamStart = switchAt;
                    run.nextRound = 0;
                    run.counter = 0;
                    run.limit = to.rows;
                    m_eventPos = 0xFFFFFFFFu;
                }
                if (run.counter < run.limit - 1)
                {
                    ++run.counter;
                    return true;
                }
                NATIVE_AUDIO_TRACE("stream %d leaves the refill list, status 0\n", run.slot);
                m_slots[run.slot].status = 0;
                return false;
            }

            void removeRun(int32_t slot)
            {
                m_runs.erase(std::remove_if(m_runs.begin(), m_runs.end(), [&](const Run &r) { return r.slot == slot; }),
                             m_runs.end());
            }

            // ---------------------------------------------------------------------------------
            // LIBSD (the parts of the SPU2 the drivers set)
            // ---------------------------------------------------------------------------------

            void sdInit()
            {
                // sceSdInit(0) (LIBSD 0x4576C): master and effect volumes off, core 1 takes core 0's
                // output at full volume, AutoDMA input off.
                m_cores[0] = CoreRegs{};
                m_cores[1] = CoreRegs{};
                m_cores[1].avolL = m_cores[1].avolR = 0x7FFF;
                m_adma.active = false;
            }

            void sdSetParam(uint16_t param, uint16_t value)
            {
                const int32_t core = param & 1;
                if (param & 0x80u)
                {
                    CoreRegs &c = m_cores[core];
                    switch (param & 0xFF00u)
                    {
                    case 0x0900: c.mvolL = value; break;
                    case 0x0A00: c.mvolR = value; break;
                    case 0x0B00: c.evolL = value; break;
                    case 0x0C00: c.evolR = value; break;
                    case 0x0D00: c.avolL = static_cast<int16_t>(value); break;
                    case 0x0E00: c.avolR = static_cast<int16_t>(value); break;
                    case 0x0F00: c.bvolL = static_cast<int16_t>(value); break;
                    case 0x1000: c.bvolR = static_cast<int16_t>(value); break;
                    default: break;
                    }
                    if ((c.evolL || c.evolR) && !m_warnedReverb)
                    {
                        m_warnedReverb = true;
                        std::fprintf(stderr, "[native-audio] effect (reverb) volume set; reverb is not mixed\n");
                    }
                    return;
                }
                Voice &v = voice(core, (param >> 1) & 0x1F);
                switch (param & 0xFF00u)
                {
                case 0x0000: v.volL = value; break;
                case 0x0100: v.volR = value; break;
                case 0x0200: v.pitch = value; break;
                default: break;
                }
            }

            uint16_t sdGetParam(uint16_t param)
            {
                const int32_t core = param & 1;
                if (param & 0x80u)
                {
                    const CoreRegs &c = m_cores[core];
                    switch (param & 0xFF00u)
                    {
                    case 0x0900: return c.mvolL;
                    case 0x0A00: return c.mvolR;
                    case 0x0B00: return c.evolL;
                    case 0x0C00: return c.evolR;
                    case 0x0D00: return static_cast<uint16_t>(c.avolL);
                    case 0x0E00: return static_cast<uint16_t>(c.avolR);
                    case 0x0F00: return static_cast<uint16_t>(c.bvolL);
                    case 0x1000: return static_cast<uint16_t>(c.bvolR);
                    default: return 0;
                    }
                }
                const Voice &v = voice(core, (param >> 1) & 0x1F);
                switch (param & 0xFF00u)
                {
                case 0x0000: return v.volL;
                case 0x0100: return v.volR;
                case 0x0200: return v.pitch;
                default: return 0;
                }
            }

            void sdSetSwitch(uint16_t entry, uint32_t value)
            {
                const int32_t core = entry & 1;
                if ((entry & 0xFF00u) == 0x1500u)
                    keyOn(core, value);
                else if ((entry & 0xFF00u) == 0x1600u)
                    keyOff(core, value);
            }

            // sceSdBlockTransStatus: which half of the ring the DMA is in, and its address, which runs
            // two 0x400-byte blocks ahead of the sample being played (the SPU2's input buffer).
            uint32_t admaStatus() const
            {
                if (!m_adma.active || m_adma.size < 0x400u)
                    return 0u;
                const uint32_t blocks = m_adma.size / 0x400u;
                const uint32_t offset = static_cast<uint32_t>(((m_adma.frames / 256u) + 2u) % blocks) * 0x400u;
                const uint32_t half = offset >= m_adma.size / 2u ? 1u : 0u;
                return (half << 24) | ((m_adma.base + offset) & 0xFFFFFFu);
            }

            // ---------------------------------------------------------------------------------
            // SDRDRV (0x497EC): fno & 0xFFF0 picks the LIBSD call, arguments from word 1.
            // ---------------------------------------------------------------------------------

            void sdr(uint32_t fno, const std::array<uint32_t, 16> &a)
            {
                const uint32_t call = fno & 0xFFF0u;
                switch (call)
                {
                case 0x8000:
                    sdInit();
                    m_sdrReply = 0;
                    break;
                case 0x8010:
                    sdSetParam(static_cast<uint16_t>(a[1]), static_cast<uint16_t>(a[2]));
                    break;
                case 0x8020:
                    m_sdrReply = sdGetParam(static_cast<uint16_t>(a[1]));
                    break;
                case 0x8030:
                    sdSetSwitch(static_cast<uint16_t>(a[1]), a[2]);
                    break;
                case 0x80D0: // sceSdVoiceTrans: uploads into SPU memory, which nothing here reads
                    m_sdrReply = a[5];
                    break;
                case 0x80E0:
                {
                    const uint32_t core = a[1] & 1u;
                    const uint32_t mode = a[2] & 0xFFFFu;
                    if ((mode & 3u) == 2u)
                    {
                        m_sdrReply = core == 0u ? admaStatus() : 0u;
                        if (core == 0u)
                            m_adma.active = false;
                        NATIVE_AUDIO_TRACE("movie PCM stop\n");
                    }
                    else if (core == 0u)
                    {
                        m_adma.active = true;
                        m_adma.base = a[3] & 0x1FFFFFu;
                        m_adma.size = a[4] & ~0x3FFu;
                        m_adma.ring.assign(m_adma.size, 0u);
                        if (m_runtime)
                            (void)m_runtime->readIopMemory(m_adma.base, m_adma.ring.data(), m_adma.ring.size());
                        const uint32_t start = a[5] ? (a[5] & 0x1FFFFFu) - m_adma.base : 0u;
                        m_adma.frames = start < m_adma.size ? (start / 0x400u) * 256u : 0u;
                        // The first two blocks go over at the start; the rest as each block begins.
                        m_adma.fetched.assign(m_adma.size, 0u);
                        m_adma.fetch(m_adma.frames / 256u);
                        m_adma.fetch(m_adma.frames / 256u + 1u);
                        m_sdrReply = (mode & 0x10u) ? a[4] >> 1 : a[4];
                        NATIVE_AUDIO_TRACE("movie PCM ring 0x%x size 0x%x mode 0x%x\n", m_adma.base, m_adma.size, mode);
                    }
                    else
                    {
                        m_sdrReply = 0xFFFFFFFFu;
                    }
                    break;
                }
                case 0x8100:
                    m_sdrReply = (a[1] & 1u) == 0u ? admaStatus() : 0u;
                    break;
                case 0x8040: case 0x8050: case 0x8060: case 0x8070: case 0x8080:
                    m_sdrReply = 0;
                    break;
                default:
                    if (m_warnedSdr.insert(call).second)
                        std::fprintf(stderr, "[native-audio] SDRDRV call 0x%x not handled\n", call);
                    break;
                }
            }

            // ---------------------------------------------------------------------------------
            // MSTREAM (dispatch 0x4EE6C)
            // ---------------------------------------------------------------------------------

            // Voice volume from MSTREAM's volume and pan (0x4E6D8).
            void applyVolume(int32_t core, int32_t index, uint32_t volume, uint32_t pan)
            {
                volume &= 0xFFFFu;
                pan &= 0xFFFFu;
                int32_t left = 0, right = 0;
                const int32_t vol = static_cast<int32_t>(volume);
                if (vol == 0)
                {
                }
                else if (!m_stereo)
                {
                    int32_t d = static_cast<int32_t>(pan) - 0x3FFF;
                    if (d < 0)
                        d = -d;
                    left = ((vol - (((d * 100) / 0x3FFF) * 0x3333) / 100) >> 1) & 0xFFFF;
                    right = left;
                }
                else
                {
                    left = vol >> 1;
                    if (pan < 0x4000u)
                    {
                        const int32_t twice = static_cast<int32_t>(pan & 0x7FFFu) * 2;
                        right = twice < 0x7FFF - vol ? 0 : ((twice - (0x7FFF - vol)) >> 1) & 0xFFFF;
                    }
                    else
                    {
                        const int32_t far = static_cast<int32_t>((0x7FFFu - pan) & 0xFFFFu);
                        right = left;
                        left = far < 0x7FFF - vol ? 0 : ((far - (0x7FFF - vol)) >> 1) & 0xFFFF;
                    }
                }
                Voice &v = voice(core, index);
                v.volL = static_cast<uint16_t>(left);
                v.volR = static_cast<uint16_t>(right);
            }

            VoiceRecord &record(int32_t core, int32_t index)
            {
                return m_records[core & 1][static_cast<uint32_t>(index) % kVoices];
            }

            void setVolume(int32_t core, int32_t index, uint16_t volume, bool store) // 0x4E8A0
            {
                if (store)
                    record(core, index).volume = volume;
                applyVolume(core, index, volume, record(core, index).pan);
            }

            void setPan(int32_t core, int32_t index, uint16_t pan, bool store) // 0x4E90C
            {
                if (store)
                    record(core, index).pan = pan;
                applyVolume(core, index, record(core, index).volume, pan);
            }

            void setMute(int32_t core, int32_t index, int32_t mute) // 0x4E574
            {
                record(core, index).mute = mute;
                setVolume(core, index, mute == 0 ? record(core, index).volume : 0, false);
            }

            void setMasterVolume(uint16_t volume) // 0x4E97C
            {
                m_masterVolume = volume;
                const uint16_t reg = static_cast<uint16_t>(volume >> 1);
                m_cores[0].mvolL = m_cores[0].mvolR = m_cores[1].mvolL = m_cores[1].mvolR = reg;
            }

            // A voice for a stream channel (0x4E3D0): the first free one on the core, silent until
            // the game sets a volume.
            int32_t allocVoice(int32_t core, uint16_t pitch)
            {
                for (int32_t i = 0; i < kVoices; ++i)
                {
                    VoiceRecord &r = record(core, i);
                    if (r.used)
                        continue;
                    r.used = true;
                    Voice &v = voice(core, i);
                    v.pitch = pitch;
                    applyVolume(core, i, 0x7FFF, 0x3FFF);
                    setMute(core, i, 1);
                    return i;
                }
                return -1;
            }

            uint32_t voiceMask(const Slot &s) const // 0x4F808
            {
                uint32_t mask = 0;
                for (int32_t c = 0; c < s.channels && c < 8; ++c)
                    mask |= 1u << (static_cast<uint32_t>(s.voices[c]) & 0x1Fu);
                return mask;
            }

            int32_t freeSlot() // 0x4AD00: free means a zero length word
            {
                for (int32_t i = 0; i < kSlots; ++i)
                    if (m_slots[i].length == 0)
                        return i;
                return -1;
            }

            // FUN_0004ad64: header checks against the stream it shares voices with, derived sizes.
            bool deriveSizes(int32_t id, bool share)
            {
                Slot &s = m_slots[id];
                if (share)
                {
                    const Slot &b = m_slots[s.shareBase & 0xFF];
                    if (s.channels != b.channels || s.rate != b.rate || s.interleave != b.interleave)
                    {
                        NATIVE_AUDIO_TRACE("stream %d does not match the stream %d it shares voices with\n", id, s.shareBase);
                        closeSlot(id);
                        return false;
                    }
                }
                s.rows = s.channels ? s.length / s.channels : 0;
                s.pitch = pitchForRate(s.rate);
                return true;
            }

            bool lookup(Slot &s, const char *name)
            {
                for (int32_t i = 0; i < m_libraryCount && i < kLibraryEntries; ++i)
                {
                    if (std::strcmp(m_library[i].name, name) == 0)
                    {
                        s.fromLibrary = 1;
                        s.entry = i;
                        break;
                    }
                }
                return s.fromLibrary != 0;
            }

            bool readHeader(Slot &s)
            {
                std::vector<uint8_t> header;
                const LibraryEntry &e = m_library[s.entry % kLibraryEntries];
                if (!Loader::readRange(m_libraryPath, e.offset, 16u, header) || header.size() < 16u)
                    return false;
                s.channels = static_cast<int32_t>(readU32(header.data()));
                s.rate = static_cast<int32_t>(readU32(header.data() + 4));
                s.interleave = static_cast<int32_t>(readU32(header.data() + 8));
                s.length = static_cast<int32_t>(readU32(header.data() + 12));
                return s.channels >= 1 && s.channels <= 8;
            }

            void setOpenReply(int32_t id)
            {
                m_lastChannels = static_cast<uint32_t>(m_slots[id].channels);
                m_lastRate = static_cast<uint32_t>(m_slots[id].rate);
                m_lastId = static_cast<uint32_t>(id);
            }

            // 0x1400 (0x4B4A8): a stream from the library, read as it plays.
            uint32_t openStream(const char *name, int32_t share, int32_t base)
            {
                const int32_t id = freeSlot();
                if (id < 0)
                    return 1;
                Slot &s = m_slots[id];
                s.core = 0;
                if (share == 0)
                    s.shareBase = -1;
                else
                {
                    s.shareBase = base;
                    m_slots[base & 0xFF].shareBase = base;
                }
                if (!lookup(s, name))
                    return 1;
                if (!readHeader(s))
                {
                    std::fprintf(stderr, "[native-audio] stream %s: bad header\n", name);
                    return 1;
                }
                s.kind = 1;
                if (!deriveSizes(id, share != 0))
                    return 1;
                if (share == 0)
                {
                    for (int32_t c = 0; c < s.channels; ++c)
                    {
                        const int32_t v = allocVoice(s.core, s.pitch);
                        if (v < 0)
                        {
                            std::fprintf(stderr, "[native-audio] FATAL: no free voice for stream %s\n", name);
                            return 1;
                        }
                        s.voices[c] = v;
                    }
                }
                else
                {
                    std::copy(std::begin(m_slots[base & 0xFF].voices), std::end(m_slots[base & 0xFF].voices), s.voices);
                }
                auto data = std::make_shared<SoundData>();
                data->channels = static_cast<uint32_t>(s.channels);
                data->span = static_cast<uint32_t>(std::max(s.interleave, 1));
                data->blocksPerChannel = static_cast<uint32_t>(s.rows) * data->span;
                s.data = data;
                const uint64_t bytes = static_cast<uint64_t>(s.rows) * s.channels * data->span * 16u;
                m_loader.enqueue({data, m_libraryPath, m_library[s.entry].offset + 16ull, bytes});
                setOpenReply(id);
                NATIVE_AUDIO_TRACE("open stream %s as %d: %d ch, %d Hz, interleave 0x%x, %d rows, voices %d.. on core 0%s\n",
                                   name, id, s.channels, s.rate, s.interleave, s.rows, s.voices[0],
                                   share ? " (shared)" : "");
                return 0;
            }

            // 0x1500 (0x4C064): a sound loaded whole into its voices' memory.
            uint32_t openLocal(const char *name, int32_t loop)
            {
                const int32_t id = freeSlot();
                if (id < 0)
                    return 1;
                Slot &s = m_slots[id];
                s.core = 1;
                s.shareBase = -1;
                if (!lookup(s, name))
                {
                    std::fprintf(stderr, "[native-audio] could not find %s in the library\n", name);
                    return 1;
                }
                if (!readHeader(s))
                {
                    std::fprintf(stderr, "[native-audio] sound %s: bad header\n", name);
                    return 1;
                }
                s.kind = 3;
                if (!deriveSizes(id, false))
                    return 1;
                for (int32_t c = 0; c < s.channels; ++c)
                {
                    const int32_t v = allocVoice(s.core, s.pitch);
                    if (v < 0)
                    {
                        std::fprintf(stderr, "[native-audio] FATAL: no free voice for sound %s\n", name);
                        return 1;
                    }
                    s.voices[c] = v;
                }

                // Each channel's rows go to its voice's memory one after the other, with the loop
                // flags MSTREAM writes over the first and last blocks of rows.
                const uint32_t span = static_cast<uint32_t>(std::max(s.interleave, 1));
                const uint32_t rowBytes = span * 16u;
                const uint32_t rows = static_cast<uint32_t>(std::max(s.rows, 0));
                std::vector<uint8_t> file;
                const uint64_t total = static_cast<uint64_t>(rows) * s.channels * rowBytes;
                (void)Loader::readRange(m_libraryPath, m_library[s.entry].offset + 16ull, total, file);
                file.resize(static_cast<size_t>(total), 0u);
                auto data = std::make_shared<SoundData>();
                data->channels = static_cast<uint32_t>(s.channels);
                data->span = rows * span;
                data->blocksPerChannel = rows * span;
                data->bytes.resize(static_cast<size_t>(total));
                for (uint32_t r = 0; r < rows; ++r)
                {
                    for (uint32_t c = 0; c < static_cast<uint32_t>(s.channels); ++c)
                    {
                        uint8_t *row = data->bytes.data() + (static_cast<size_t>(c) * rows + r) * rowBytes;
                        std::memcpy(row, file.data() + (static_cast<size_t>(r) * s.channels + c) * rowBytes, rowBytes);
                        if (rowBytes < 0x20u)
                            continue;
                        uint8_t &first = row[1], &second = row[0x11], &last = row[rowBytes - 0xF];
                        if (rows == 1u)
                        {
                            second = 2;
                            first = 6;
                            last = loop ? 3 : 7;
                        }
                        else
                        {
                            if (r == 0u)
                            {
                                second = 2;
                                first = 6;
                                last = 2;
                            }
                            if (loop && r == rows - 1u)
                            {
                                second = 2;
                                first = 2;
                                last = 3;
                            }
                        }
                    }
                }
                data->ready.store(true);
                s.data = data;
                setOpenReply(id);
                NATIVE_AUDIO_TRACE("open sound %s as %d: %d ch, %d Hz, %u blocks, voices %d.. on core 1%s\n", name, id,
                                   s.channels, s.rate, rows * span, s.voices[0], loop ? ", looped" : "");
                return 0;
            }

            // 0x0A00 (0x4D694).
            void stopSlot(int32_t id)
            {
                Slot &s = m_slots[id & 0xFF];
                keyOff(s.core, voiceMask(s));
                if ((s.kind == 1 || s.kind == 4) && s.status != 0)
                {
                    m_clockRunning = false;
                    removeRun(id & 0xFF);
                    s.status = 0;
                }
            }

            // 0x1F00 (0x4C90C).
            void closeSlot(int32_t id)
            {
                Slot &s = m_slots[id & 0xFF];
                if (s.kind == 0)
                    return;
                if (s.shareBase == id || s.shareBase == -1)
                {
                    stopSlot(id);
                    for (int32_t c = 0; c < s.channels && c < 8; ++c)
                        record(s.core, s.voices[c]).used = false;
                }
                s.entry = 0;
                s.length = 0;
                s.kind = 0;
                s.data.reset();
            }

            // 0x0800 (0x4D4AC).
            void play(int32_t id, uint32_t streaming, uint32_t resetClock, uint32_t signal)
            {
                Slot &s = m_slots[id & 0xFF];
                if (signal)
                    ++m_eventSignals;
                if (streaming)
                {
                    m_clockRunning = true;
                    removeRun(id & 0xFF);
                    Run run;
                    run.slot = id & 0xFF;
                    run.span = static_cast<uint32_t>(std::max(s.interleave, 0));
                    run.limit = s.rows - (s.channels && s.interleave ? (s.consumed / s.channels) / s.interleave : 0);
                    m_runs.push_back(run);
                }
                if (resetClock)
                {
                    m_queueRead = m_queueWrite = 0;
                    m_clock = 0;
                }
                for (int32_t c = 0; c < s.channels && c < 8; ++c)
                {
                    Voice &v = voice(s.core, s.voices[c]);
                    v.data = s.data;
                    v.channel = static_cast<uint32_t>(c);
                }
                keyOn(s.core, voiceMask(s));
                s.status = 1;
                NATIVE_AUDIO_TRACE("play %d (kind %d)%s%s\n", id, s.kind, streaming ? " streaming" : "",
                                   resetClock ? ", clock reset" : "");
            }

            // 0x2000 and the event thread (0x4C7C4): read a stream's start and, with `upload`, put it
            // in its voices' buffers. Only MSTREAM's IOP buffer count survives of this (+0x7C, which
            // decides how many refill rounds the stream gets).
            void preload(int32_t id, uint32_t upload)
            {
                Slot &s = m_slots[id & 0xFF];
                s.consumed = 0;
                if (s.kind == 1 && s.data && upload)
                {
                    // MSTREAM reads the start before replying; here the whole stream is loading.
                    const std::shared_ptr<SoundData> data = s.data;
                    m_mutex.unlock();
                    m_loader.wait(data);
                    m_mutex.lock();
                    // Two buffers uploaded; with two IOP buffers (interleave >= 0x300) the second
                    // upload empties the first IOP buffer and the count starts again.
                    s.consumed = s.interleave >= 0x300 ? 0 : 2 * s.channels * s.interleave;
                }
            }

            // The event thread (0x4DCF4), one wake-up per signal.
            void eventThread()
            {
                if (m_queueRead < m_queueWrite)
                {
                    const QueueEntry &e = m_queue[m_queueRead % kQueueEntries];
                    m_eventSlot = static_cast<int32_t>(e.stream & 0xFF);
                    m_eventPos = e.position;
                    NATIVE_AUDIO_TRACE("next section: stream %u at block %u\n", e.stream, e.position);
                    preload(static_cast<int32_t>(e.stream), 0);
                    ++m_queueRead;
                }
                else if (m_queueRead == m_queueWrite)
                {
                    m_eventPos = 0xFFFFFFFFu;
                }
            }

        public:
            // MSTREAM's vblank handler (0x4DA4C -> 0x4D830): the clock voice's position, in blocks of
            // 28 samples, never going back. The game reads the value of the last vblank.
            void onVBlank()
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_clockRunning || !m_threads)
                    return;
                const Voice &v = voice(m_clockCore, m_clockVoice);
                const int64_t behind = m_clockOffsetFrames * v.pitch / 0x1000;
                const int64_t samples = static_cast<int64_t>(v.samples) - behind;
                if (samples > 0)
                    m_clock = std::max<uint32_t>(m_clock, static_cast<uint32_t>(samples / 28) * 28u);
            }

        private:

            void resetClockState() // 0x4DA04
            {
                m_clock = 0;
                m_eventPos = 0xFFFFFFFFu;
                m_clockRunning = false;
            }

            void init() // 0x0100
            {
                sdInit();
                setMasterVolume(m_masterVolume);
                for (int32_t core = 0; core < 2; ++core)
                    for (int32_t i = 0; i < kVoices; ++i)
                        m_records[core][i].used = false;
                // The init clears 0xC0 bytes of the volume records: core 0's first 16 voices.
                for (int32_t i = 0; i < 16; ++i)
                {
                    m_records[0][i].volume = m_records[0][i].pan = 0;
                    m_records[0][i].mute = 0;
                }
                for (Slot &s : m_slots)
                    s = Slot{};
                m_library = {};
                m_preloadBusy = 0;
                m_runs.clear();
                m_threads = true;
                resetClockState();
                m_queueRead = m_queueWrite = 0;
            }

            void sleep() // 0x0E00
            {
                m_threads = false;
                m_runs.clear();
                for (Slot &s : m_slots)
                    s = Slot{};
                m_library = {};
            }

            bool openLibrary(PS2Runtime &runtime, const std::array<uint8_t, 64> &raw) // 0x1A00 (0x4B138)
            {
                std::string path(reinterpret_cast<const char *>(raw.data()), strnlen(reinterpret_cast<const char *>(raw.data()), raw.size()));
                const PS2Runtime::IoPaths &paths = PS2Runtime::getIoPaths();
                std::filesystem::path host;
                if (!runtime.vfs().resolveHostPath(path, {paths.hostRoot, paths.cdRoot, paths.mcRoot}, host))
                {
                    std::fprintf(stderr, "[native-audio] cannot resolve library %s\n", path.c_str());
                    return false;
                }
                std::vector<uint8_t> head;
                if (!Loader::readRange(host.string(), 0, 4, head) || head.size() < 4)
                {
                    std::fprintf(stderr, "[native-audio] cannot read library %s\n", host.string().c_str());
                    return false;
                }
                m_libraryPath = host.string();
                m_libraryCount = static_cast<int32_t>(readU32(head.data()));
                std::vector<uint8_t> table;
                const uint32_t count = static_cast<uint32_t>(std::clamp(m_libraryCount, 0, kLibraryEntries));
                (void)Loader::readRange(m_libraryPath, 4, count * 32ull, table);
                table.resize(count * 32u, 0u);
                for (uint32_t i = 0; i < count; ++i)
                {
                    LibraryEntry &e = m_library[i];
                    const uint8_t *p = table.data() + i * 32u;
                    // The name is upper-cased over its first 8 bytes; byte 8 is the 8-character
                    // names' terminator.
                    for (uint32_t k = 0; k < 8; ++k)
                        e.name[k] = static_cast<char>(std::toupper(p[k]));
                    e.name[8] = '\0';
                    e.offset = readU32(p + 0x10);
                    e.kind = readU32(p + 0x1C);
                }
                NATIVE_AUDIO_TRACE("library %s: %d entries\n", path.c_str(), m_libraryCount);
                return true;
            }

            void mstream(PS2Runtime &runtime, uint32_t fno, const std::array<uint32_t, 16> &a, const std::array<uint8_t, 64> &raw)
            {
                const uint32_t cmd = fno & 0xFFFFFF00u;
                const int32_t id = static_cast<int32_t>(a[0]);
                const auto name = [&]() -> std::string
                { return std::string(reinterpret_cast<const char *>(raw.data()), strnlen(reinterpret_cast<const char *>(raw.data()), 0x3E)); };
                const auto shareFlag = static_cast<int32_t>(static_cast<int8_t>(raw[0x3E]));
                const auto shareBase = static_cast<int32_t>(static_cast<int8_t>(raw[0x3F]));
                switch (cmd)
                {
                case 0xF100:
                    break;
                case 0x5C00:
                    m_queueRead = m_queueWrite = 0;
                    m_reply[0] = cmd; // the void call leaves v0 = the command
                    break;
                case 0x5B00:
                    m_stereo = a[0] != 0;
                    break;
                case 0x5800:
                    m_queue[m_queueWrite % kQueueEntries] = {a[0], a[1], a[2]};
                    ++m_queueWrite;
                    if (a[3])
                        ++m_eventSignals;
                    break;
                case 0x5300:
                    if (id < kSlots)
                        setMute(m_slots[id].core, m_slots[id].voices[a[1] & 7], static_cast<int32_t>(a[2]));
                    break;
                case 0x5200:
                    if (id < kSlots)
                        setPan(m_slots[id].core, m_slots[id].voices[a[1] & 7], static_cast<uint16_t>(a[2]), true);
                    break;
                case 0x5100:
                    if (id < kSlots)
                        setVolume(m_slots[id].core, m_slots[id].voices[a[1] & 7], static_cast<uint16_t>(a[2]), true);
                    break;
                case 0x5A00:
                    setMasterVolume(static_cast<uint16_t>(a[0]));
                    break;
                case 0x0800:
                    if (id < kSlots)
                        play(id, a[1], a[2], a[3]);
                    break;
                case 0x5600:
                    m_reply[0] = m_clock;
                    break;
                case 0x2200:
                    m_reply[0] = m_preloadBusy;
                    break;
                case 0x2000:
                    m_slots[id & 0xFF].preloading = 1;
                    preload(id, a[2]);
                    m_slots[id & 0xFF].preloading = 0;
                    m_reply[0] = 0;
                    break;
                case 0x2100:
                    m_slots[id & 0xFF].preloading = 1;
                    m_reply[0] = 1; // queued for MSTREAM's command thread, which preloads at once here
                    preload(id, a[2]);
                    m_slots[id & 0xFF].preloading = 0;
                    m_preloadBusy = 0;
                    break;
                case 0x0900:
                    m_reply[0] = static_cast<uint32_t>(m_slots[id & 0xFF].status);
                    break;
                case 0x0A00:
                    stopSlot(id);
                    break;
                case 0x1F00:
                    closeSlot(id);
                    break;
                case 0x0D00:
                    for (int32_t i = kSlots - 1; i >= 0; --i)
                        closeSlot(i);
                    break;
                case 0x1B00:
                    for (int32_t i = kSlots - 1; i >= 0; --i)
                        closeSlot(i);
                    m_reply[0] = 0;
                    break;
                case 0x1A00:
                    m_reply[0] = openLibrary(runtime, raw) ? 0u : 1u;
                    break;
                case 0x1500:
                case 0x1400:
                {
                    const std::string n = name();
                    m_reply[0] = cmd == 0x1400 ? openStream(n.c_str(), shareFlag, shareBase) : openLocal(n.c_str(), shareBase);
                    m_reply[1] = m_lastChannels;
                    m_reply[2] = m_lastRate;
                    m_reply[3] = m_lastId;
                    break;
                }
                case 0x1000:
                case 0x1100:
                case 0x1300:
                    std::fprintf(stderr, "[native-audio] MSTREAM open 0x%x (by path or from EE memory) is not supported\n", cmd);
                    m_reply[0] = 1;
                    break;
                case 0x5700:
                {
                    const Slot &s = m_slots[id & 0xFF];
                    m_clockCore = s.core;
                    m_clockVoice = s.voices[0];
                    resetClockState();
                    m_reply[0] = 0;
                    break;
                }
                case 0x3200:
                    m_threads = false;
                    break;
                case 0x3100:
                    m_threads = true;
                    resetClockState();
                    break;
                case 0x0E00:
                    sleep();
                    break;
                case 0x0100:
                    init();
                    break;
                default:
                    std::fprintf(stderr, "[native-audio] MSTREAM command 0x%x not handled\n", cmd);
                    m_reply[0] = 1;
                    break;
                }
                // The event thread runs once the RPC thread has replied.
                while (m_eventSignals > 0)
                {
                    --m_eventSignals;
                    eventThread();
                }
            }

            struct QueueEntry
            {
                uint32_t stream = 0, position = 0, flag = 0;
            };

            std::mutex m_mutex;
            Loader m_loader;
            PS2Runtime *m_runtime = nullptr;
            bool m_eeClock = false;
            uint64_t m_eeCycles = 0;
            std::vector<int16_t> m_scratch;
            uint64_t m_framesRendered = 0;
            int64_t m_clockOffsetFrames = 0;

            std::array<std::array<Voice, kVoices>, 2> m_voices{};
            std::array<CoreRegs, 2> m_cores{};
            AutoDma m_adma;
            uint32_t m_sdrReply = 0;

            std::array<uint32_t, 4> m_reply{};
            std::array<std::array<VoiceRecord, kVoices>, 2> m_records{};
            std::array<Slot, kSlots> m_slots{};
            std::array<LibraryEntry, kLibraryEntries> m_library{};
            int32_t m_libraryCount = 0;
            std::string m_libraryPath;
            uint32_t m_lastChannels = 0, m_lastRate = 0, m_lastId = 0;
            uint32_t m_preloadBusy = 0;
            bool m_stereo = true;
            uint16_t m_masterVolume = 0x7FFF;
            bool m_threads = false;
            std::vector<Run> m_runs;
            std::array<QueueEntry, kQueueEntries> m_queue{};
            uint32_t m_queueRead = 0, m_queueWrite = 0;
            uint32_t m_eventSignals = 0;
            int32_t m_eventSlot = -1;
            uint32_t m_eventPos = 0xFFFFFFFFu;
            int32_t m_clockCore = 0, m_clockVoice = 0;
            bool m_clockRunning = false;
            uint32_t m_clock = 0;

            bool m_warnedUnloaded = false;
            bool m_warnedReverb = false;
            std::set<uint32_t> m_warnedSdr;
        };

        Engine &engine()
        {
            // Never destroyed: the device thread may still pull from it while the process exits.
            static Engine *instance = new Engine();
            return *instance;
        }
    }

    bool enabled()
    {
        // PS2X_NATIVE_AUDIO=0 runs the drivers on the IOP emulator instead (the reference).
        static const bool on = []
        {
            const char *value = std::getenv("PS2X_NATIVE_AUDIO");
            return !value || std::strcmp(value, "0") != 0;
        }();
        return on;
    }

    bool replacesModule(std::string_view path)
    {
        if (!enabled())
            return false;
        std::string upper(path);
        for (char &c : upper)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        for (const char *module : {"\\LIBSD.IRX", "\\SDRDRV.IRX", "\\MSTREAM.IRX", "/LIBSD.IRX", "/SDRDRV.IRX", "/MSTREAM.IRX"})
            if (upper.find(module) != std::string::npos)
                return true;
        return false;
    }

    bool servesSid(uint32_t sid)
    {
        return enabled() && (sid == kMstreamSid || sid == kSdrSid);
    }

    ps2x::iop::RpcResult handleRpc(PS2Runtime &runtime, uint8_t *rdram, const ps2x::iop::RpcRequest &request)
    {
        return engine().handle(runtime, rdram, request);
    }

    void onSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
    {
        if (enabled())
            engine().onSifTransfer(rdram, transfer);
    }

    void advanceEeCycles(uint64_t eeCycles)
    {
        if (enabled())
            engine().advanceEeCycles(eeCycles);
    }

    void onVBlank()
    {
        if (enabled())
            engine().onVBlank();
    }
}
