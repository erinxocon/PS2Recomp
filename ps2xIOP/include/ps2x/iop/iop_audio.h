#pragma once

#include <cstddef>
#include <cstdint>

namespace ps2x::iop
{
    // SPU2 output: 48 kHz interleaved stereo signed 16-bit frames, produced on
    // the thread that advances the IOP (the EE executor). The sink must not
    // block for long; the host is expected to buffer the frames for its own
    // audio device.
    using AudioSink = void (*)(void *user, const int16_t *frames, size_t frameCount);

    constexpr uint32_t kSpu2SampleRate = 48000u;

    void setAudioSink(AudioSink sink, void *user) noexcept;
}
