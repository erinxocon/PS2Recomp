#pragma once

#include "ps2x/iop/iop_types.h"

#include <cstdint>
#include <string_view>

class PS2Runtime;

// Native replacement for the game's IOP sound drivers (Gitaroo Man: MSTREAM, SDRDRV, LIBSD), see
// docs/native-audio.md: their RPCs are answered here, the music and effects are read from the IMC
// libraries and mixed on the host. PS2X_NATIVE_AUDIO=0 runs the drivers on the IOP emulator
// instead, the reference this is checked against.
namespace ps2_native_sound
{
    [[nodiscard]] bool enabled();
    // True for the driver modules this replaces: their load succeeds without the IOP.
    [[nodiscard]] bool replacesModule(std::string_view path);
    [[nodiscard]] bool servesSid(uint32_t sid);
    [[nodiscard]] ps2x::iop::RpcResult handleRpc(PS2Runtime &runtime, uint8_t *rdram, const ps2x::iop::RpcRequest &request);
    void onSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer);
    // Emulated time, for PS2X_NATIVE_AUDIO_CLOCK=ee (mixing clocked like the SPU2 it replaces).
    void advanceEeCycles(uint64_t eeCycles);
    // The guest's vblank: MSTREAM samples the song clock here.
    void onVBlank();
}
