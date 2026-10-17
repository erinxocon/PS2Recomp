#pragma once
#include <cstdint>

constexpr uint32_t kThreadReturnSentinel = 0x1FFFFF00u;
constexpr uint32_t kCallReturnSentinel = 0x1FFFFF04u;
constexpr uint64_t kIopClockHz = 36'864'000ull;
// NTSC field cadence (approximately 59.94 Hz).  VBlank imports are
// scheduler waits, not no-op timing hints: returning immediately lets
// high-priority IRX threads busy-loop and starve RPC server threads.
constexpr uint64_t kVblankPeriodCycles = (kIopClockHz * 1001ull + 30'000ull) / 60'000ull;
constexpr uint64_t kVblankEndPhaseCycles = kVblankPeriodCycles / 16ull;
constexpr uint32_t kDefaultSlice = 256u;
constexpr uint32_t kMaxCallInstructions = 2'000'000u;
// RPC server functions run synchronously to completion (on hardware they run on the RPC thread with no
// time limit). Some handlers do heavy work inline, e.g. MSTREAM's OpenLocalFromLib (cmd 0x1500) reads a
// whole clip from CD and uploads it to SPU2 RAM through LIBSD's PIO transfer loop (~6 instructions per
// halfword, >2M instructions for a 700 KB clip). Abandoning such a call returns a garbage reply.
constexpr uint32_t kMaxRpcCallInstructions = 400'000'000u;
constexpr uint32_t kModuleLoadBase = 0x00010000u;
constexpr uint32_t kStackGuardBytes = 64u;