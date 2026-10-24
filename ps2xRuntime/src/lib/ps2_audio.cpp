#include "runtime/ps2_audio.h"
#include "runtime/ps2_memory.h"
#include "ps2_host_backend.h"
#include "ps2x/iop/iop_audio.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(PLATFORM_VITA)
// raylib 5.5 compiles miniaudio (and exports it) with these options; they change the layout of
// ma_device, so they must match its build.
#define MA_NO_JACK
#define MA_NO_WAV
#define MA_NO_FLAC
#define MA_NO_MP3
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#include "external/miniaudio.h"
#endif

namespace
{
    std::vector<uint8_t> buildWavFromPcm(const int16_t *pcm, size_t sampleCount, uint32_t sampleRate)
    {
        const uint32_t dataSize = static_cast<uint32_t>(sampleCount * 2);
        const uint32_t fileSize = 36 + dataSize;
        std::vector<uint8_t> wav(8 + fileSize);

        uint8_t *p = wav.data();
        p[0] = 'R';
        p[1] = 'I';
        p[2] = 'F';
        p[3] = 'F';
        p[4] = static_cast<uint8_t>(fileSize);
        p[5] = static_cast<uint8_t>(fileSize >> 8);
        p[6] = static_cast<uint8_t>(fileSize >> 16);
        p[7] = static_cast<uint8_t>(fileSize >> 24);
        p[8] = 'W';
        p[9] = 'A';
        p[10] = 'V';
        p[11] = 'E';
        p[12] = 'f';
        p[13] = 'm';
        p[14] = 't';
        p[15] = ' ';
        p[16] = 16;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 1;
        p[21] = 0;
        p[22] = 1;
        p[23] = 0;
        p[24] = static_cast<uint8_t>(sampleRate);
        p[25] = static_cast<uint8_t>(sampleRate >> 8);
        p[26] = static_cast<uint8_t>(sampleRate >> 16);
        p[27] = static_cast<uint8_t>(sampleRate >> 24);
        const uint32_t byteRate = sampleRate * 2;
        p[28] = static_cast<uint8_t>(byteRate);
        p[29] = static_cast<uint8_t>(byteRate >> 8);
        p[30] = static_cast<uint8_t>(byteRate >> 16);
        p[31] = static_cast<uint8_t>(byteRate >> 24);
        p[32] = 2;
        p[33] = 0;
        p[34] = 16;
        p[35] = 0;
        p[36] = 'd';
        p[37] = 'a';
        p[38] = 't';
        p[39] = 'a';
        p[40] = static_cast<uint8_t>(dataSize);
        p[41] = static_cast<uint8_t>(dataSize >> 8);
        p[42] = static_cast<uint8_t>(dataSize >> 16);
        p[43] = static_cast<uint8_t>(dataSize >> 24);
        std::memcpy(p + 44, pcm, dataSize);
        return wav;
    }
}

namespace
{
    // Host side of the emulated SPU2 (ps2xIOP): frames arrive from the IOP
    // thread in emulated time and are buffered for a raylib callback stream.
    // PS2X_AUDIO_DUMP=<file.wav> additionally records the exact SPU2 output.
    class Spu2HostOutput
    {
    public:
        static Spu2HostOutput &instance()
        {
            static Spu2HostOutput output;
            return output;
        }

        static void sink(void *user, const int16_t *frames, size_t frameCount)
        {
            (void)user;
            instance().push(frames, frameCount);
        }

        void push(const int16_t *frames, size_t frameCount)
        {
            writeDump(frames, frameCount);
            ensureStream();
            if (!m_streamReady)
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            {
                const auto now = std::chrono::steady_clock::now();
                const double gap = std::chrono::duration<double, std::milli>(now - m_lastPush).count();
                m_lastPush = now;
                m_statMaxGap = std::max(m_statMaxGap, gap);
                if (gap > 30.0)
                    ++m_statLongGaps;
            }
            for (size_t i = 0; i < frameCount * 2u; ++i)
            {
                m_ring[m_write] = frames[i];
                m_write = (m_write + 1u) % m_ring.size();
                if (m_write == m_read)
                    m_read = (m_read + 2u) % m_ring.size(); // overrun: drop oldest frame
            }
        }

        void shutdown()
        {
#if !defined(PLATFORM_VITA)
            if (m_deviceOpen)
            {
                ma_device_uninit(&m_device);
                m_deviceOpen = false;
            }
#endif
            std::lock_guard<std::mutex> lock(m_dumpMutex);
            finalizeDump();
        }

    private:
        static constexpr size_t kSampleRate = ps2x::iop::kSpu2SampleRate;
        static constexpr double kMaxExcessMs = 30.0;
        static constexpr double kMaxRateAdjust = 0.005;

        // PS2X_AUDIO_LATENCY_MS: audio buffered ahead of the device (default 40; raise it if it crackles).
        static double targetLatencyMs()
        {
            const char *env = std::getenv("PS2X_AUDIO_LATENCY_MS");
            const double ms = env ? std::strtod(env, nullptr) : 40.0;
            return std::clamp(ms, 10.0, 250.0);
        }

        Spu2HostOutput()
            : m_ring(kSampleRate * 2u)
        {
            const char *dumpPath = std::getenv("PS2X_AUDIO_DUMP");
            if (dumpPath && *dumpPath)
            {
                m_dump = std::fopen(dumpPath, "wb");
                if (m_dump)
                    writeWavHeader();
            }
        }

        static void streamCallback(void *buffer, unsigned int frames)
        {
            instance().pull(static_cast<int16_t *>(buffer), frames);
        }

        // Everything buffered here is heard that much later than the game produced it, and the game
        // judges button presses against its own clock, so the buffer is kept short: the device pulls at
        // a rate nudged by up to +-0.5% (inaudible) to hold the smoothed fill at the target, instead of
        // letting it drift up and snapping it back.
        void pull(int16_t *out, unsigned int frames)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const size_t want = static_cast<size_t>(frames) * 2u;
            size_t fill = (m_write + m_ring.size() - m_read) % m_ring.size();
            const double fillMs = static_cast<double>(fill / 2u) * 1000.0 / kSampleRate;
            if (m_priming)
            {
                // At start and after an underrun, wait for the target amount before playing.
                if (fillMs < m_targetMs)
                {
                    std::memset(out, 0, want * sizeof(int16_t));
                    return;
                }
                m_priming = false;
                m_fillEma = fillMs;
            }
            m_fillEma += (fillMs - m_fillEma) * 0.05;
            if (m_fillEma > m_targetMs + kMaxExcessMs)
            {
                // Far behind (the emulation ran ahead, e.g. after a stall): drop to the target at once.
                const size_t keep = static_cast<size_t>(m_targetMs * kSampleRate / 1000.0) * 2u;
                m_read = (m_write + m_ring.size() - keep) % m_ring.size();
                fill = keep;
                m_fillEma = m_targetMs;
                ++m_statSnaps;
            }
            const double error = (m_fillEma - m_targetMs) / m_targetMs;
            const double rate = 1.0 + std::clamp(error * 0.01, -kMaxRateAdjust, kMaxRateAdjust);

            size_t i = 0;
            for (; i < want; i += 2u)
            {
                if (fill < 4u)
                    break;
                const size_t next = (m_read + 2u) % m_ring.size();
                const double frac = m_phase;
                out[i] = static_cast<int16_t>(m_ring[m_read] + (m_ring[next] - m_ring[m_read]) * frac);
                out[i + 1] = static_cast<int16_t>(m_ring[m_read + 1] + (m_ring[next + 1] - m_ring[m_read + 1]) * frac);
                m_phase += rate;
                while (m_phase >= 1.0 && fill >= 4u)
                {
                    m_phase -= 1.0;
                    m_read = (m_read + 2u) % m_ring.size();
                    fill -= 2u;
                }
            }
            if (i < want)
            {
                std::memset(out + i, 0, (want - i) * sizeof(int16_t));
                m_priming = true;
                m_phase = 0.0;
                ++m_statUnderruns;
            }
            traceFill(fill, frames);
        }

        // PS2X_TRACE_AUDIO_LATENCY=1: once a second, the buffered audio (ms) seen by the device callback.
        void traceFill(size_t fill, unsigned int frames)
        {
            static const bool s_trace = std::getenv("PS2X_TRACE_AUDIO_LATENCY") != nullptr;
            if (!s_trace)
                return;
            const double ms = static_cast<double>(fill / 2u) * 1000.0 / kSampleRate;
            m_statMin = std::min(m_statMin, ms);
            m_statMax = std::max(m_statMax, ms);
            m_statSum += ms;
            ++m_statCalls;
            m_statFrames += frames;
            if (m_statFrames < kSampleRate)
                return;
            std::printf("[audio] buffered ms min=%.1f avg=%.1f max=%.1f callback=%u frames underruns=%u snaps=%u"
                        " producer max gap %.1f ms, gaps>30ms %u\n",
                        m_statMin, m_statSum / m_statCalls, m_statMax, frames, m_statUnderruns, m_statSnaps,
                        m_statMaxGap, m_statLongGaps);
            std::fflush(stdout);
            m_statMaxGap = 0.0;
            m_statLongGaps = 0;
            m_statMin = 1e9;
            m_statMax = 0.0;
            m_statSum = 0.0;
            m_statCalls = 0;
            m_statFrames = 0;
        }

        // The SPU2 output gets its own miniaudio device with short periods: raylib's shared device
        // buffers ~75 ms (3600 frames on PulseAudio/PipeWire), which is heard as that much lag.
        // PS2X_AUDIO_PERIOD_MS sets the period (default 5, three periods); PS2X_AUDIO_RAYLIB=1 plays
        // through a raylib stream instead.
        bool openDevice()
        {
#if defined(PLATFORM_VITA)
            return false;
#else
            if (std::getenv("PS2X_AUDIO_RAYLIB"))
                return false;
            const char *periodEnv = std::getenv("PS2X_AUDIO_PERIOD_MS");
            const long periodMs = periodEnv ? std::clamp(std::strtol(periodEnv, nullptr, 10), 1L, 100L) : 5L;
            ma_device_config config = ma_device_config_init(ma_device_type_playback);
            config.playback.format = ma_format_s16;
            config.playback.channels = 2;
            config.sampleRate = static_cast<ma_uint32>(kSampleRate);
            config.periodSizeInMilliseconds = static_cast<ma_uint32>(periodMs);
            config.periods = 3;
            config.performanceProfile = ma_performance_profile_low_latency;
            config.dataCallback = [](ma_device *, void *output, const void *, ma_uint32 frames)
            { instance().pull(static_cast<int16_t *>(output), frames); };
            if (ma_device_init(nullptr, &config, &m_device) != MA_SUCCESS)
                return false;
            if (ma_device_start(&m_device) != MA_SUCCESS)
            {
                ma_device_uninit(&m_device);
                return false;
            }
            m_deviceOpen = true;
            const ma_uint32 buffered = m_device.playback.internalPeriodSizeInFrames * m_device.playback.internalPeriods;
            std::printf("[audio] %s device, %u Hz, %u frames buffered (%.1f ms)\n",
                        ma_get_backend_name(m_device.pContext->backend), m_device.playback.internalSampleRate, buffered,
                        buffered * 1000.0 / m_device.playback.internalSampleRate);
            return true;
#endif
        }

        void ensureStream()
        {
#if defined(PLATFORM_VITA)
            return;
#else
            if (m_streamTried)
                return;
            if (!m_deviceTried)
            {
                m_deviceTried = true;
                if (openDevice())
                {
                    m_streamTried = true;
                    m_streamReady = true;
                    return;
                }
            }
            if (!IsAudioDeviceReady())
                return;
            m_streamTried = true;
            SetAudioStreamBufferSizeDefault(1024);
            m_stream = LoadAudioStream(static_cast<unsigned int>(kSampleRate), 16u, 2u);
            if (!IsAudioStreamValid(m_stream))
                return;
            SetAudioStreamCallback(m_stream, &Spu2HostOutput::streamCallback);
            PlayAudioStream(m_stream);
            m_streamReady = true;
#endif
        }

        void writeDump(const int16_t *frames, size_t frameCount)
        {
            if (!m_dump)
                return;
            std::lock_guard<std::mutex> lock(m_dumpMutex);
            if (!m_dump)
                return;
            std::fwrite(frames, sizeof(int16_t) * 2u, frameCount, m_dump);
            m_dumpFrames += static_cast<uint32_t>(frameCount);
            if (m_dumpFrames - m_dumpFramesAtHeader >= kSampleRate / 2u)
            {
                writeWavHeader();
                m_dumpFramesAtHeader = m_dumpFrames;
            }
        }

        void writeWavHeader()
        {
            const uint32_t dataBytes = m_dumpFrames * 4u;
            uint8_t header[44];
            const auto put32 = [&](size_t offset, uint32_t value)
            {
                for (int i = 0; i < 4; ++i)
                    header[offset + i] = static_cast<uint8_t>(value >> (8 * i));
            };
            const auto put16 = [&](size_t offset, uint16_t value)
            {
                header[offset] = static_cast<uint8_t>(value);
                header[offset + 1] = static_cast<uint8_t>(value >> 8);
            };
            std::memcpy(header, "RIFF", 4);
            put32(4, 36u + dataBytes);
            std::memcpy(header + 8, "WAVEfmt ", 8);
            put32(16, 16u);
            put16(20, 1u);
            put16(22, 2u);
            put32(24, static_cast<uint32_t>(kSampleRate));
            put32(28, static_cast<uint32_t>(kSampleRate) * 4u);
            put16(32, 4u);
            put16(34, 16u);
            std::memcpy(header + 36, "data", 4);
            put32(40, dataBytes);
            const long position = std::ftell(m_dump);
            std::fseek(m_dump, 0, SEEK_SET);
            std::fwrite(header, 1, sizeof(header), m_dump);
            if (position > static_cast<long>(sizeof(header)))
                std::fseek(m_dump, position, SEEK_SET);
            std::fflush(m_dump);
        }

        void finalizeDump()
        {
            if (!m_dump)
                return;
            writeWavHeader();
            std::fclose(m_dump);
            m_dump = nullptr;
        }

        std::mutex m_mutex;
        std::vector<int16_t> m_ring;
        size_t m_read = 0;
        size_t m_write = 0;
        bool m_priming = true;
        double m_targetMs = targetLatencyMs();
        double m_fillEma = 0.0;
        double m_phase = 0.0;
        double m_statMin = 1e9;
        double m_statMax = 0.0;
        double m_statSum = 0.0;
        uint32_t m_statCalls = 0;
        size_t m_statFrames = 0;
        uint32_t m_statUnderruns = 0;
        uint32_t m_statSnaps = 0;
        std::chrono::steady_clock::time_point m_lastPush{};
        double m_statMaxGap = 0.0;
        uint32_t m_statLongGaps = 0;
        bool m_streamTried = false;
        bool m_streamReady = false;
        bool m_deviceTried = false;
        bool m_deviceOpen = false;
#if !defined(PLATFORM_VITA)
        ma_device m_device{};
        AudioStream m_stream{};
#endif
        std::mutex m_dumpMutex;
        std::FILE *m_dump = nullptr;
        uint32_t m_dumpFrames = 0;
        uint32_t m_dumpFramesAtHeader = 0;
    };
}

namespace ps2_vag
{
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate);
}

struct PS2AudioBackend::Impl
{
    struct TrackedSound
    {
        Sound snd;
        uint32_t sampleKey;
    };
    std::vector<TrackedSound> activeSounds;
};

PS2AudioBackend::PS2AudioBackend() : m_impl(std::make_unique<Impl>())
{
    (void)Spu2HostOutput::instance();
    ps2x::iop::setAudioSink(&Spu2HostOutput::sink, nullptr);
}

PS2AudioBackend::~PS2AudioBackend()
{
    ps2x::iop::setAudioSink(nullptr, nullptr);
    Spu2HostOutput::instance().shutdown();
    if (m_impl)
        stopAll();
}

void PS2AudioBackend::onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
{
    if (!rdram || sizeBytes < 48)
        return;

    const uint32_t physAddr = srcAddr & PS2_RAM_MASK;
    if (physAddr + sizeBytes > PS2_RAM_SIZE)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(rdram + physAddr, sizeBytes, pcm, sampleRate))
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = std::move(sample);
    m_mostRecentSampleKey = physAddr;
}

void PS2AudioBackend::onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr)
{
    if (!data || sizeBytes < 48)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(data, sizeBytes, pcm, sampleRate))
        return;

    const uint32_t physAddr = keyAddr & PS2_RAM_MASK;
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = sample;
    m_mostRecentSampleKey = physAddr;
    m_loadOrderSamples.push_back(std::move(sample));
    m_loadOrderSampleKeys.push_back(physAddr);
    constexpr size_t kMaxLoadOrderSamples = 32;
    if (m_loadOrderSamples.size() > kMaxLoadOrderSamples)
    {
        m_loadOrderSamples.erase(m_loadOrderSamples.begin());
        m_loadOrderSampleKeys.erase(m_loadOrderSampleKeys.begin());
    }
}

namespace
{
    constexpr uint32_t LIBSD_CMD_SET_VOICE = 0x8010u;
}

void PS2AudioBackend::onSoundCommand(uint32_t sid, uint32_t rpcNum,
                                     const uint8_t *sendBuf, uint32_t sendSize,
                                     uint8_t *recvBuf, uint32_t recvSize)
{
    if (sid != 0x80000701u)
        return;

    if ((rpcNum == LIBSD_CMD_SET_VOICE || (rpcNum & 0xFF00u) == 0x8100u) &&
        sendBuf && sendSize >= 20)
    {
        uint32_t sampleAddr = 0;
        uint32_t voiceIndex = 0xFFFFFFFFu;
        for (int vo = 4; vo >= 0 && voiceIndex == 0xFFFFFFFFu; vo -= 4)
        {
            if (vo < static_cast<int>(sendSize))
            {
                uint32_t v = 0;
                std::memcpy(&v, sendBuf + vo, sizeof(v));
                if (v < 24u)
                    voiceIndex = v;
            }
        }

        constexpr uint32_t kMinPlausibleAddr = 0x1000u;
        for (int off = 12; off <= 24 && sampleAddr == 0; off += 4)
        {
            if (sendSize >= static_cast<uint32_t>(off + 4))
            {
                uint32_t cand = 0;
                std::memcpy(&cand, sendBuf + off, sizeof(cand));
                if (cand >= kMinPlausibleAddr && (cand <= PS2_RAM_MASK || (cand & ~PS2_RAM_MASK) == 0))
                    sampleAddr = cand;
            }
        }
        if (sampleAddr == 0)
            sampleAddr = m_mostRecentSampleKey;

        float pitch = 1.0f;
        if (sendSize >= 12)
        {
            uint16_t pitchHalf = 0;
            std::memcpy(&pitchHalf, sendBuf + 8, sizeof(pitchHalf));
            if (pitchHalf != 0)
                pitch = 4096.0f / static_cast<float>(pitchHalf);
        }
        play(sampleAddr, pitch, 1.0f, voiceIndex);
    }
}

void PS2AudioBackend::play(uint32_t sampleAddr, float pitch, float volume, uint32_t voiceIndex)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample *sampleToPlay = nullptr;
    uint32_t sampleKey = 0;

    auto it = m_sampleBank.find(sampleAddr & PS2_RAM_MASK);
    if (it != m_sampleBank.end())
    {
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    else if (voiceIndex != 0xFFFFFFFFu &&
             voiceIndex < m_loadOrderSamples.size() &&
             voiceIndex < m_loadOrderSampleKeys.size())
    {
        sampleToPlay = &m_loadOrderSamples[voiceIndex];
        sampleKey = m_loadOrderSampleKeys[voiceIndex];
    }
    else
    {
        it = m_sampleBank.find(m_mostRecentSampleKey);
        if (it == m_sampleBank.end())
            return;
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    if (!sampleToPlay || sampleToPlay->pcm.empty())
        return;

    const bool isBgm = (sampleToPlay->pcm.size() > static_cast<size_t>(sampleToPlay->sampleRate * 5));
    playDecodedSample(sampleKey, *sampleToPlay, pitch, volume, isBgm);
}

void PS2AudioBackend::pruneFinishedSounds()
{
#if defined(PLATFORM_VITA)
    return;
#else
    auto &sounds = m_impl->activeSounds;
    auto it = sounds.begin();
    while (it != sounds.end())
    {
        if (!IsSoundPlaying(it->snd))
        {
            UnloadSound(it->snd);
            it = sounds.erase(it);
        }
        else
        {
            ++it;
        }
    }
#endif
}

void PS2AudioBackend::playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                                        bool isBgm)
{
#if defined(PLATFORM_VITA)
    (void)sampleKey;
    (void)sample;
    (void)pitch;
    (void)volume;
    (void)isBgm;
    return;
#else
    if (!m_audioReady || sample.pcm.empty())
        return;

    pruneFinishedSounds();

    for (const auto &t : m_impl->activeSounds)
    {
        if (t.sampleKey == sampleKey && IsSoundPlaying(t.snd))
            return;
    }

    auto &sounds = m_impl->activeSounds;
    if (isBgm)
    {
        for (auto it = sounds.begin(); it != sounds.end();)
        {
            if (IsSoundPlaying(it->snd))
            {
                StopSound(it->snd);
                UnloadSound(it->snd);
                it = sounds.erase(it);
            }
            else
                ++it;
        }
    }

    constexpr int kMaxConcurrentSounds = 4;
    while (static_cast<int>(sounds.size()) >= kMaxConcurrentSounds)
    {
        StopSound(sounds.front().snd);
        UnloadSound(sounds.front().snd);
        sounds.erase(sounds.begin());
    }

    std::vector<uint8_t> wav = buildWavFromPcm(sample.pcm.data(), sample.pcm.size(), sample.sampleRate);
    Wave wave = LoadWaveFromMemory(".wav", wav.data(), static_cast<int>(wav.size()));
    if (wave.frameCount <= 0)
        return;
    Sound snd = LoadSoundFromWave(wave);
    UnloadWave(wave);
    SetSoundPitch(snd, pitch);
    SetSoundVolume(snd, volume);
    m_impl->activeSounds.push_back({snd, sampleKey});
    PlaySound(snd);
#endif
}

void PS2AudioBackend::stop(uint32_t voiceId)
{
    (void)voiceId;
}

void PS2AudioBackend::stopAll()
{
    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(PLATFORM_VITA)
    return;
#else
    for (auto &t : m_impl->activeSounds)
    {
        StopSound(t.snd);
        UnloadSound(t.snd);
    }
    m_impl->activeSounds.clear();
#endif
}
