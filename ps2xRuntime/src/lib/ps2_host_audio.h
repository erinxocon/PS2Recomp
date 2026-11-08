#pragma once

#include <cstddef>
#include <cstdint>

// The host audio output (ps2_audio.cpp): one 48 kHz stereo device, fed either by pushed frames
// (the emulated SPU2, or a producer clocked by emulated time) through the latency-controlled ring,
// or by a pull source the device callback renders from directly.
namespace ps2_host_audio
{
    constexpr uint32_t kSampleRate = 48000u;

    // Renders `frames` interleaved stereo frames; called on the device thread.
    using PullSource = void (*)(void *user, int16_t *out, uint32_t frames);

    void push(const int16_t *frames, size_t frameCount);
    // Replaces the ring with `source` (nullptr goes back to the ring). Opens the device if needed;
    // returns false when no device could be opened.
    bool setPullSource(PullSource source, void *user);
    // Frames the device holds between a pull and the speaker (its buffer).
    uint32_t deviceBufferedFrames();
}
