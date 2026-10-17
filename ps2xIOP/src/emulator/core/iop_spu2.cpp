#include <cstdio>
#include "iop_spu2.h"

#include "ps2x/iop/iop_audio.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace ps2x::iop
{
    namespace
    {
        std::atomic<AudioSink> g_audioSink{nullptr};
        std::atomic<void *> g_audioSinkUser{nullptr};
    }

    void setAudioSink(AudioSink sink, void *user) noexcept
    {
        g_audioSinkUser.store(user, std::memory_order_release);
        g_audioSink.store(sink, std::memory_order_release);
    }
}

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kDmaChcrStart = 1u << 24u;
        constexpr uint32_t kDmaChcrToDevice = 1u << 0u;
        constexpr uint32_t kInputHalfSamples = 256u;
        constexpr uint32_t kInputBlockBytes = 0x400u; // 256 L + 256 R halfwords
        constexpr size_t kOutputFlushFrames = 256u;

        constexpr int32_t kFilterPos[5] = {0, 60, 115, 98, 122};
        constexpr int32_t kFilterNeg[5] = {0, 0, -52, -55, -60};

        // Core-relative register offsets (0x000-0x3FF window).
        constexpr uint32_t kPmon = 0x180u;
        constexpr uint32_t kNon = 0x184u;
        constexpr uint32_t kVmixl = 0x188u;
        constexpr uint32_t kVmixel = 0x18Cu;
        constexpr uint32_t kVmixr = 0x190u;
        constexpr uint32_t kVmixer = 0x194u;
        constexpr uint32_t kMmix = 0x198u;
        constexpr uint32_t kAttr = 0x19Au;
        constexpr uint32_t kIrqa = 0x19Cu;
        constexpr uint32_t kKon = 0x1A0u;
        constexpr uint32_t kKoff = 0x1A4u;
        constexpr uint32_t kTsa = 0x1A8u;
        constexpr uint32_t kData = 0x1ACu;
        constexpr uint32_t kAdmas = 0x1B0u;
        constexpr uint32_t kVoiceAddrBase = 0x1C0u;
        constexpr uint32_t kEsa = 0x2E0u;
        constexpr uint32_t kReverbAddrBase = 0x2E4u;
        constexpr uint32_t kEea = 0x33Cu;
        constexpr uint32_t kEndx = 0x340u;
        constexpr uint32_t kStatx = 0x344u;

        // Reverb address register indices (PS1 naming in comments).
        enum ReverbAddr : uint32_t
        {
            ApfOffset1 = 0, // dAPF1
            ApfOffset2,     // dAPF2
            SameLDst,       // mLSAME
            SameRDst,       // mRSAME
            Comb1L,         // mLCOMB1
            Comb1R,         // mRCOMB1
            Comb2L,         // mLCOMB2
            Comb2R,         // mRCOMB2
            SameLSrc,       // dLSAME
            SameRSrc,       // dRSAME
            DiffLDst,       // mLDIFF
            DiffRDst,       // mRDIFF
            Comb3L,         // mLCOMB3
            Comb3R,         // mRCOMB3
            Comb4L,         // mLCOMB4
            Comb4R,         // mRCOMB4
            DiffLSrc,       // dLDIFF
            DiffRSrc,       // dRDIFF
            Apf1LDst,       // mLAPF1
            Apf1RDst,       // mRAPF1
            Apf2LDst,       // mLAPF2
            Apf2RDst,       // mRAPF2
        };

        enum ReverbCoef : uint32_t
        {
            IirVol = 0, // vIIR
            Comb1Vol,
            Comb2Vol,
            Comb3Vol,
            Comb4Vol,
            WallVol, // vWALL
            Apf1Vol,
            Apf2Vol,
            InCoefL,
            InCoefR,
        };

        int32_t clamp16(int32_t value) noexcept
        {
            return std::clamp(value, -0x8000, 0x7FFF);
        }

        int32_t mulVol(int32_t sample, int32_t volume) noexcept
        {
            return (sample * volume) >> 15;
        }

        // PS1/PS2 envelope stepping shared by ADSR and volume sweeps.
        void envelopeTick(int32_t &level, uint32_t &counter, bool exponential, bool decrease, uint32_t shift,
                          int32_t step) noexcept
        {
            uint32_t cycles = 1u << (shift > 11u ? std::min<uint32_t>(shift - 11u, 30u) : 0u);
            int32_t delta = step << (shift < 11u ? 11u - shift : 0u);
            if (exponential && !decrease && level > 0x6000)
                cycles *= 4u;
            if (exponential && decrease)
                delta = (delta * level) >> 15;
            if (++counter < cycles)
                return;
            counter = 0u;
            level = std::clamp(level + delta, 0, 0x7FFF);
        }

        int32_t hermite(const std::array<int32_t, 4> &w, uint32_t frac) noexcept
        {
            // Catmull-Rom between w[1] and w[2]; frac is 12-bit.
            const int64_t t = frac;
            const int64_t p0 = w[0], p1 = w[1], p2 = w[2], p3 = w[3];
            const int64_t a = -p0 + 3 * p1 - 3 * p2 + p3;
            const int64_t b = 2 * p0 - 5 * p1 + 4 * p2 - p3;
            const int64_t c = -p0 + p2;
            // ((a*t^3 + b*t^2) / 4096^2 + c*t) / 4096 / 2 + p1
            int64_t v = (a * t) >> 12;
            v = ((v + b) * t) >> 12;
            v = ((v + c) * t) >> 12;
            return clamp16(static_cast<int32_t>(p1 + v / 2));
        }
    }

    IopSpu2::IopSpu2(std::vector<uint8_t> &iopRam)
        : m_iopRam(iopRam), m_ram(RamHalfwords)
    {
        reset();
    }

    void IopSpu2::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        for (Core &core : m_cores)
            core = Core{};
        m_regs.fill(0u);
        m_sampleCycle = 0u;
        m_samplesGenerated = 0u;
        m_pendingIrqs = 0u;
        m_spdifIrqInfo = 0u;
        m_output.clear();
        m_output.reserve(kOutputFlushFrames * 2u * 2u);
        for (Core &core : m_cores)
            core.statx = 0x80u;
    }

    // ----------------------------------------------------------------------------------------
    // Register interface
    // ----------------------------------------------------------------------------------------

    uint16_t IopSpu2::read16(uint32_t phys)
    {
        advanceTo(now());
        const uint32_t offset = (phys - RegBase) & (RegSize - 2u);
        if (offset < 0x760u)
            return readCoreReg(offset >= 0x400u ? 1u : 0u, offset & 0x3FFu);
        if (offset < 0x7B0u)
        {
            const uint32_t core = (offset - 0x760u) / 0x28u;
            const uint32_t reg = (offset - 0x760u) % 0x28u;
            const Core &c = m_cores[core];
            switch (reg)
            {
            case 0x00u:
                return c.mvolL.reg;
            case 0x02u:
                return c.mvolR.reg;
            case 0x10u:
                return static_cast<uint16_t>(c.mvolL.level);
            case 0x12u:
                return static_cast<uint16_t>(c.mvolR.level);
            default:
                return m_regs[offset / 2u];
            }
        }
        if (offset == 0x7C2u)
            return m_spdifIrqInfo;
        return m_regs[offset / 2u];
    }

    void IopSpu2::write16(uint32_t phys, uint16_t value)
    {
        advanceTo(now());
        const uint32_t offset = (phys - RegBase) & (RegSize - 2u);
        m_regs[offset / 2u] = value;
        if (offset < 0x760u)
        {
            writeCoreReg(offset >= 0x400u ? 1u : 0u, offset & 0x3FFu, value);
            return;
        }
        if (offset < 0x7B0u)
        {
            const uint32_t core = (offset - 0x760u) / 0x28u;
            const uint32_t reg = (offset - 0x760u) % 0x28u;
            Core &c = m_cores[core];
            switch (reg)
            {
            case 0x00u:
                c.mvolL.reg = value;
                if ((value & 0x8000u) == 0u)
                    c.mvolL.level = static_cast<int16_t>(value << 1u);
                break;
            case 0x02u:
                c.mvolR.reg = value;
                if ((value & 0x8000u) == 0u)
                    c.mvolR.level = static_cast<int16_t>(value << 1u);
                break;
            case 0x04u:
                c.evolL = static_cast<int16_t>(value);
                break;
            case 0x06u:
                c.evolR = static_cast<int16_t>(value);
                break;
            case 0x08u:
                c.avolL = static_cast<int16_t>(value);
                break;
            case 0x0Au:
                c.avolR = static_cast<int16_t>(value);
                break;
            case 0x0Cu:
                c.bvolL = static_cast<int16_t>(value);
                break;
            case 0x0Eu:
                c.bvolR = static_cast<int16_t>(value);
                break;
            default:
                if (reg >= 0x14u && reg < 0x28u)
                    c.reverbCoef[(reg - 0x14u) / 2u] = static_cast<int16_t>(value);
                break;
            }
        }
    }

    uint16_t IopSpu2::readCoreReg(uint32_t coreIndex, uint32_t local)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t raw = m_regs[(coreIndex * 0x400u + local) / 2u];
        if (local < 0x180u)
        {
            const Voice &v = core.voices[local >> 4u];
            switch (local & 0xFu)
            {
            case 0x0u:
                return v.volL.reg;
            case 0x2u:
                return v.volR.reg;
            case 0x4u:
                return v.pitch;
            case 0x6u:
                return v.adsr1;
            case 0x8u:
                return v.adsr2;
            case 0xAu:
                return static_cast<uint16_t>(v.envLevel);
            case 0xCu:
                return static_cast<uint16_t>(v.volL.level);
            case 0xEu:
                return static_cast<uint16_t>(v.volR.level);
            default:
                return static_cast<uint16_t>(raw);
            }
        }
        if (local >= kVoiceAddrBase && local < kVoiceAddrBase + 24u * 0xCu)
        {
            const Voice &v = core.voices[(local - kVoiceAddrBase) / 0xCu];
            const uint32_t reg = (local - kVoiceAddrBase) % 0xCu;
            uint32_t address = 0u;
            switch (reg / 4u)
            {
            case 0u:
                address = v.ssa;
                break;
            case 1u:
                address = v.lsa;
                break;
            default:
                address = v.nax;
                break;
            }
            return (reg & 2u) ? static_cast<uint16_t>(address & 0xFFFFu)
                              : static_cast<uint16_t>((address >> 16u) & 0xFu);
        }
        switch (local)
        {
        case kAttr:
            return core.attr;
        case kStatx:
            return core.statx;
        case kEndx:
            return static_cast<uint16_t>(core.endx & 0xFFFFu);
        case kEndx + 2u:
            return static_cast<uint16_t>(core.endx >> 16u);
        case kAdmas:
            return core.admas;
        case kIrqa:
            return static_cast<uint16_t>((core.irqa >> 16u) & 0xFu);
        case kIrqa + 2u:
            return static_cast<uint16_t>(core.irqa & 0xFFFFu);
        case kTsa:
            return static_cast<uint16_t>((core.tsa >> 16u) & 0xFu);
        case kTsa + 2u:
            return static_cast<uint16_t>(core.tsa & 0xFFFFu);
        case kData:
        {
            const uint16_t value = ramRead(core.tsa);
            core.tsa = (core.tsa + 1u) & (RamHalfwords - 1u);
            return value;
        }
        case kKon:
        case kKon + 2u:
        case kKoff:
        case kKoff + 2u:
            return 0u;
        default:
            return static_cast<uint16_t>(raw);
        }
    }

    namespace
    {
        bool traceSpu2()
        {
            static const bool enabled = std::getenv("PS2X_TRACE_SPU2") != nullptr;
            return enabled;
        }
        uint32_t g_spu2TraceCount = 0u;
    }

    void IopSpu2::writeCoreReg(uint32_t coreIndex, uint32_t local, uint16_t value)
    {
        if (traceSpu2() && g_spu2TraceCount < 4000u)
        {
            ++g_spu2TraceCount;
            std::fprintf(stderr, "[spu2] w core=%u reg=0x%03x val=0x%04x\n", coreIndex, local, value);
        }
        Core &core = m_cores[coreIndex];
        const auto setHi = [&](uint32_t &target)
        {
            target = (target & 0xFFFFu) | (static_cast<uint32_t>(value & 0xFu) << 16u);
        };
        const auto setLo = [&](uint32_t &target)
        {
            target = (target & 0xF0000u) | value;
        };
        const auto setMaskLo = [&](uint32_t &target)
        {
            target = (target & 0xFF0000u) | value;
        };
        const auto setMaskHi = [&](uint32_t &target)
        {
            target = (target & 0xFFFFu) | (static_cast<uint32_t>(value & 0xFFu) << 16u);
        };

        if (local < 0x180u)
        {
            Voice &v = core.voices[local >> 4u];
            switch (local & 0xFu)
            {
            case 0x0u:
                v.volL.reg = value;
                if ((value & 0x8000u) == 0u)
                    v.volL.level = static_cast<int16_t>(value << 1u);
                break;
            case 0x2u:
                v.volR.reg = value;
                if ((value & 0x8000u) == 0u)
                    v.volR.level = static_cast<int16_t>(value << 1u);
                break;
            case 0x4u:
                v.pitch = value;
                break;
            case 0x6u:
                v.adsr1 = value;
                break;
            case 0x8u:
                v.adsr2 = value;
                break;
            case 0xAu:
                v.envLevel = static_cast<int16_t>(value) < 0 ? 0 : static_cast<int16_t>(value);
                break;
            default:
                break;
            }
            return;
        }
        if (local >= kVoiceAddrBase && local < kVoiceAddrBase + 24u * 0xCu)
        {
            Voice &v = core.voices[(local - kVoiceAddrBase) / 0xCu];
            const uint32_t reg = (local - kVoiceAddrBase) % 0xCu;
            uint32_t *target = nullptr;
            switch (reg / 4u)
            {
            case 0u:
                target = &v.ssa;
                break;
            case 1u:
                target = &v.lsa;
                break;
            default:
                target = &v.nax;
                break;
            }
            if (reg & 2u)
                setLo(*target);
            else
                setHi(*target);
            return;
        }
        if (local >= kReverbAddrBase && local < kEea)
        {
            uint32_t &target = core.reverbAddr[(local - kReverbAddrBase) / 4u];
            if (local & 2u)
                setLo(target);
            else
                setHi(target);
            return;
        }

        switch (local)
        {
        case kPmon:
            setMaskLo(core.pmon);
            break;
        case kPmon + 2u:
            setMaskHi(core.pmon);
            break;
        case kNon:
            setMaskLo(core.non);
            break;
        case kNon + 2u:
            setMaskHi(core.non);
            break;
        case kVmixl:
            setMaskLo(core.vmixl);
            break;
        case kVmixl + 2u:
            setMaskHi(core.vmixl);
            break;
        case kVmixel:
            setMaskLo(core.vmixel);
            break;
        case kVmixel + 2u:
            setMaskHi(core.vmixel);
            break;
        case kVmixr:
            setMaskLo(core.vmixr);
            break;
        case kVmixr + 2u:
            setMaskHi(core.vmixr);
            break;
        case kVmixer:
            setMaskLo(core.vmixer);
            break;
        case kVmixer + 2u:
            setMaskHi(core.vmixer);
            break;
        case kMmix:
            core.mmix = value;
            break;
        case kAttr:
            if (value & 0x8000u)
                core.statx = 0u;
            core.attr = value & 0x7FFFu;
            if ((value & 0x40u) == 0u)
                m_spdifIrqInfo &= static_cast<uint16_t>(~(4u << coreIndex));
            break;
        case kIrqa:
            setHi(core.irqa);
            break;
        case kIrqa + 2u:
            setLo(core.irqa);
            break;
        case kKon:
            keyOn(core, coreIndex, value);
            break;
        case kKon + 2u:
            keyOn(core, coreIndex, static_cast<uint32_t>(value & 0xFFu) << 16u);
            break;
        case kKoff:
            keyOff(core, value);
            break;
        case kKoff + 2u:
            keyOff(core, static_cast<uint32_t>(value & 0xFFu) << 16u);
            break;
        case kTsa:
            setHi(core.tsa);
            break;
        case kTsa + 2u:
            setLo(core.tsa);
            break;
        case kData:
            testIrq(core.tsa);
            ramWrite(core.tsa, value);
            core.tsa = (core.tsa + 1u) & (RamHalfwords - 1u);
            break;
        case kAdmas:
            core.admas = value;
            if ((value & (1u << coreIndex)) == 0u)
            {
                core.inputReady = {false, false};
                core.inputHalf = 0u;
                core.inputPos = 0u;
            }
            break;
        case kEsa:
            setHi(core.esa);
            break;
        case kEsa + 2u:
            setLo(core.esa);
            break;
        case kEea:
            core.eea = (static_cast<uint32_t>(value & 0xFu) << 16u) | 0xFFFFu;
            break;
        case kEndx:
            core.endx &= 0xFF0000u;
            break;
        case kEndx + 2u:
            core.endx &= 0xFFFFu;
            break;
        case kStatx:
            break;
        default:
            break;
        }
    }

    void IopSpu2::keyOn(Core &core, uint32_t coreIndex, uint32_t mask)
    {
        (void)coreIndex;
        for (uint32_t i = 0; i < 24u; ++i)
        {
            if ((mask & (1u << i)) == 0u)
                continue;
            Voice &v = core.voices[i];
            v.phase = EnvPhase::Attack;
            v.envLevel = 0;
            v.envCounter = 0u;
            v.nax = v.ssa;
            v.pitchCounter = 0u;
            v.decodedPos = 28u;
            v.blockFlags = 0u;
            v.freshStart = true;
            v.hist1 = 0;
            v.hist2 = 0;
            v.window = {};
            v.lastOut = 0;
            v.volL.counter = 0u;
            v.volR.counter = 0u;
            core.endx &= ~(1u << i);
        }
    }

    void IopSpu2::keyOff(Core &core, uint32_t mask)
    {
        for (uint32_t i = 0; i < 24u; ++i)
        {
            if ((mask & (1u << i)) == 0u)
                continue;
            Voice &v = core.voices[i];
            if (v.phase != EnvPhase::Off)
            {
                v.phase = EnvPhase::Release;
                v.envCounter = 0u;
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // DMA (IOP channel 4 -> core 0, channel 7 -> core 1)
    // ----------------------------------------------------------------------------------------

    uint32_t IopSpu2::readDma32(uint32_t phys)
    {
        advanceTo(now());
        const uint32_t coreIndex = phys >= 0x1F801500u ? 1u : 0u;
        const Core &core = m_cores[coreIndex];
        switch (phys & 0xFu)
        {
        case 0x0u:
            return core.madr;
        case 0x4u:
            return core.bcr;
        case 0x8u:
            return core.chcr;
        default:
            return 0u;
        }
    }

    void IopSpu2::writeDma32(uint32_t phys, uint32_t value)
    {
        advanceTo(now());
        const uint32_t coreIndex = phys >= 0x1F801500u ? 1u : 0u;
        Core &core = m_cores[coreIndex];
        switch (phys & 0xFu)
        {
        case 0x0u:
            core.madr = value & 0x00FFFFFFu;
            break;
        case 0x4u:
            core.bcr = value;
            break;
        case 0x8u:
        {
            const bool wasRunning = (core.chcr & kDmaChcrStart) != 0u;
            core.chcr = value;
            if ((value & kDmaChcrStart) == 0u)
            {
                core.dmaActive = false;
            }
            else if (!wasRunning || !core.dmaActive)
            {
                startDma(coreIndex);
            }
            break;
        }
        default:
            break;
        }
    }

    void IopSpu2::startDma(uint32_t coreIndex)
    {
        Core &core = m_cores[coreIndex];
        uint32_t blockWords = core.bcr & 0xFFFFu;
        uint32_t blockCount = core.bcr >> 16u;
        if (blockWords == 0u)
            blockWords = 0x10000u;
        if (blockCount == 0u)
            blockCount = 1u;
        const uint32_t bytes = std::min<uint32_t>(blockWords * 4u * blockCount, static_cast<uint32_t>(m_iopRam.size()));

        core.statx &= static_cast<uint16_t>(~0x80u);
        const bool toDevice = (core.chcr & kDmaChcrToDevice) != 0u;
        if (traceSpu2() && g_spu2TraceCount < 4000u)
        {
            ++g_spu2TraceCount;
            std::fprintf(stderr, "[spu2] dma core=%u madr=0x%x bytes=0x%x toDev=%d admas=0x%x tsa=0x%x\n", coreIndex,
                         core.madr, bytes, toDevice ? 1 : 0, core.admas, core.tsa);
        }
        if (toDevice && (core.admas & (1u << coreIndex)) != 0u)
        {
            core.dmaActive = true;
            core.dmaAdma = true;
            core.dmaRemaining = bytes;
            fillInput(coreIndex);
            return;
        }

        const uint32_t halfwords = bytes / 2u;
        const uint32_t ramSize = static_cast<uint32_t>(m_iopRam.size());
        uint32_t source = core.madr & (ramSize - 1u);
        if (toDevice)
        {
            testIrqRange(core.tsa, halfwords);
            for (uint32_t i = 0; i < halfwords; ++i)
            {
                uint16_t sample = 0u;
                std::memcpy(&sample, m_iopRam.data() + ((source + i * 2u) & (ramSize - 2u)), sizeof(sample));
                ramWrite(core.tsa + i, sample);
            }
        }
        else
        {
            testIrqRange(core.tsa, halfwords);
            for (uint32_t i = 0; i < halfwords; ++i)
            {
                const uint16_t sample = ramRead(core.tsa + i);
                std::memcpy(m_iopRam.data() + ((source + i * 2u) & (ramSize - 2u)), &sample, sizeof(sample));
            }
        }
        core.tsa = (core.tsa + halfwords) & (RamHalfwords - 1u);
        core.dmaActive = true;
        core.dmaAdma = false;
        core.dmaRemaining = 0u;
        core.madr = (core.madr + bytes) & 0x00FFFFFFu;
        // Rough bus time: SPU2 DMA moves about one halfword per IOP cycle.
        core.dmaDoneCycle = now() + std::max<uint64_t>(64u, halfwords);
    }

    void IopSpu2::finishDma(uint32_t coreIndex)
    {
        Core &core = m_cores[coreIndex];
        core.dmaActive = false;
        core.dmaAdma = false;
        core.chcr &= ~kDmaChcrStart;
        core.bcr &= 0xFFFFu;
        core.statx |= 0x80u;
        m_pendingIrqs |= coreIndex == 0u ? 2u : 4u;
    }

    void IopSpu2::fillInput(uint32_t coreIndex)
    {
        Core &core = m_cores[coreIndex];
        if (!core.dmaActive || !core.dmaAdma)
            return;
        const uint32_t ramSize = static_cast<uint32_t>(m_iopRam.size());
        const uint32_t base = 0x2000u + coreIndex * 0x400u;

        for (int attempt = 0; attempt < 2 && core.dmaRemaining != 0u; ++attempt)
        {
            uint32_t half = 0u;
            if (!core.inputReady[core.inputHalf] && core.inputPos == 0u)
                half = core.inputHalf;
            else if (!core.inputReady[core.inputHalf ^ 1u])
                half = core.inputHalf ^ 1u;
            else
                break;

            const uint32_t bytes = std::min(core.dmaRemaining, kInputBlockBytes);
            const uint32_t left = base + half * kInputHalfSamples;
            const uint32_t right = left + 0x200u;
            for (uint32_t i = 0; i < kInputHalfSamples; ++i)
            {
                uint16_t l = 0u;
                uint16_t r = 0u;
                const uint32_t lo = i * 2u;
                const uint32_t ro = 0x200u + i * 2u;
                if (lo + 2u <= bytes)
                    std::memcpy(&l, m_iopRam.data() + ((core.madr + lo) & (ramSize - 2u)), sizeof(l));
                if (ro + 2u <= bytes)
                    std::memcpy(&r, m_iopRam.data() + ((core.madr + ro) & (ramSize - 2u)), sizeof(r));
                ramWrite(left + i, l);
                ramWrite(right + i, r);
            }
            {
                // PS2X_SPU2_INPUT_DUMP=<file>: raw interleaved L/R int16 of every AutoDMA block (diagnostics).
                static FILE *s_inputDump = []() -> FILE *
                {
                    const char *path = std::getenv("PS2X_SPU2_INPUT_DUMP");
                    return path ? std::fopen(path, "wb") : nullptr;
                }();
                if (s_inputDump && coreIndex == 0u)
                {
                    for (uint32_t i = 0; i < kInputHalfSamples; ++i)
                    {
                        const uint16_t pair[2] = {ramRead(left + i), ramRead(right + i)};
                        std::fwrite(pair, sizeof(pair), 1, s_inputDump);
                    }
                    std::fflush(s_inputDump);
                }
            }
            testIrqRange(left, kInputHalfSamples);
            testIrqRange(right, kInputHalfSamples);
            core.inputReady[half] = true;
            core.madr = (core.madr + bytes) & 0x00FFFFFFu;
            core.dmaRemaining -= bytes;
        }
        if (core.dmaRemaining == 0u)
            finishDma(coreIndex);
    }

    // ----------------------------------------------------------------------------------------
    // Timing
    // ----------------------------------------------------------------------------------------

    uint64_t IopSpu2::nextEventCycle() const noexcept
    {
        uint64_t next = m_sampleCycle + CyclesPerSample * 64u;
        for (uint32_t i = 0; i < 2u; ++i)
        {
            const Core &core = m_cores[i];
            if (core.dmaActive && !core.dmaAdma)
                next = std::min(next, core.dmaDoneCycle);
            if ((core.admas & (1u << i)) != 0u)
                next = std::min(next, m_sampleCycle + (kInputHalfSamples - core.inputPos) * CyclesPerSample);
            if ((core.attr & 0x40u) != 0u)
                next = std::min(next, m_sampleCycle + CyclesPerSample);
        }
        return next;
    }

    void IopSpu2::advanceTo(uint64_t cycle)
    {
        if (m_sampleCycle == 0u && m_samplesGenerated == 0u && cycle > CyclesPerSample * 4800u)
        {
            // First use well after boot: start the sample clock now rather than
            // generating seconds of silence in one burst.
            m_sampleCycle = cycle;
        }
        for (;;)
        {
            const uint64_t nextSample = m_sampleCycle + CyclesPerSample;
            for (uint32_t i = 0; i < 2u; ++i)
            {
                Core &core = m_cores[i];
                if (core.dmaActive && !core.dmaAdma && core.dmaDoneCycle <= std::min(nextSample, cycle))
                    finishDma(i);
            }
            if (nextSample > cycle)
                break;
            m_sampleCycle = nextSample;
            generateSample();
        }
        if (m_output.size() >= kOutputFlushFrames * 2u)
            flushOutput();
    }

    void IopSpu2::flushOutput()
    {
        if (m_output.empty())
            return;
        const AudioSink sink = g_audioSink.load(std::memory_order_acquire);
        if (sink)
            sink(g_audioSinkUser.load(std::memory_order_acquire), m_output.data(), m_output.size() / 2u);
        m_output.clear();
    }

    // ----------------------------------------------------------------------------------------
    // IRQ
    // ----------------------------------------------------------------------------------------

    void IopSpu2::testIrq(uint32_t address)
    {
        address &= RamHalfwords - 1u;
        for (uint32_t i = 0; i < 2u; ++i)
        {
            const Core &core = m_cores[i];
            if ((core.attr & 0x40u) != 0u && core.irqa == address)
            {
                m_spdifIrqInfo |= static_cast<uint16_t>(4u << i);
                m_pendingIrqs |= 1u;
            }
        }
    }

    void IopSpu2::testIrqRange(uint32_t address, uint32_t count)
    {
        for (uint32_t i = 0; i < 2u; ++i)
        {
            const Core &core = m_cores[i];
            if ((core.attr & 0x40u) == 0u)
                continue;
            const uint32_t offset = (core.irqa - address) & (RamHalfwords - 1u);
            if (offset < count)
            {
                m_spdifIrqInfo |= static_cast<uint16_t>(4u << i);
                m_pendingIrqs |= 1u;
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // Voices
    // ----------------------------------------------------------------------------------------

    void IopSpu2::decodeBlock(uint32_t coreIndex, uint32_t voiceIndex, Voice &voice)
    {
        Core &core = m_cores[coreIndex];
        if (!voice.freshStart)
        {
            // Leaving the previous block: apply its loop flags.
            const uint8_t flags = voice.blockFlags;
            if (flags & 1u)
            {
                core.endx |= 1u << voiceIndex;
                if (flags & 2u)
                {
                    voice.nax = voice.lsa;
                }
                else
                {
                    voice.phase = EnvPhase::Off;
                    voice.envLevel = 0;
                    voice.nax = voice.lsa;
                }
            }
            else
            {
                voice.nax = (voice.blockAddress + 8u) & (RamHalfwords - 1u);
            }
        }

        const uint32_t address = voice.nax & (RamHalfwords - 8u);
        testIrqRange(address, 8u);
        const uint16_t header = ramRead(address);
        uint32_t shift = header & 0xFu;
        const uint32_t filter = std::min<uint32_t>((header >> 4u) & 7u, 4u);
        const uint8_t flags = static_cast<uint8_t>(header >> 8u);
        if (shift > 12u)
            shift = 9u;
        if (flags & 4u)
            voice.lsa = address;

        const int32_t f0 = kFilterPos[filter];
        const int32_t f1 = kFilterNeg[filter];
        for (uint32_t i = 0; i < 28u; ++i)
        {
            const uint16_t word = ramRead(address + 1u + i / 4u);
            const int32_t nibble = (word >> ((i & 3u) * 4u)) & 0xF;
            int32_t sample = static_cast<int16_t>(static_cast<uint16_t>(nibble << 12u)) >> shift;
            sample += (voice.hist1 * f0 + voice.hist2 * f1 + 32) >> 6;
            sample = clamp16(sample);
            voice.hist2 = voice.hist1;
            voice.hist1 = sample;
            voice.decoded[i] = static_cast<int16_t>(sample);
        }
        voice.blockAddress = address;
        voice.blockFlags = flags;
        voice.freshStart = false;
        voice.decodedPos = 0u;
        voice.nax = address;
    }

    void IopSpu2::advanceEnvelope(Voice &v)
    {
        switch (v.phase)
        {
        case EnvPhase::Attack:
        {
            const uint32_t rate = (v.adsr1 >> 8u) & 0x7Fu;
            envelopeTick(v.envLevel, v.envCounter, (v.adsr1 & 0x8000u) != 0u, false, rate >> 2u,
                         7 - static_cast<int32_t>(rate & 3u));
            if (v.envLevel >= 0x7FFF)
            {
                v.envLevel = 0x7FFF;
                v.phase = EnvPhase::Decay;
                v.envCounter = 0u;
            }
            break;
        }
        case EnvPhase::Decay:
        {
            const int32_t sustainLevel = std::min<int32_t>(((v.adsr1 & 0xFu) + 1) * 0x800, 0x7FFF);
            envelopeTick(v.envLevel, v.envCounter, true, true, (v.adsr1 >> 4u) & 0xFu, -8);
            if (v.envLevel <= sustainLevel)
            {
                v.phase = EnvPhase::Sustain;
                v.envCounter = 0u;
            }
            break;
        }
        case EnvPhase::Sustain:
        {
            const uint32_t rate = (v.adsr2 >> 6u) & 0x7Fu;
            const bool decrease = (v.adsr2 & 0x4000u) != 0u;
            envelopeTick(v.envLevel, v.envCounter, (v.adsr2 & 0x8000u) != 0u, decrease, rate >> 2u,
                         decrease ? -8 + static_cast<int32_t>(rate & 3u) : 7 - static_cast<int32_t>(rate & 3u));
            break;
        }
        case EnvPhase::Release:
            envelopeTick(v.envLevel, v.envCounter, (v.adsr2 & 0x20u) != 0u, true, v.adsr2 & 0x1Fu, -8);
            if (v.envLevel <= 0)
            {
                v.envLevel = 0;
                v.phase = EnvPhase::Off;
            }
            break;
        case EnvPhase::Off:
            break;
        }
    }

    namespace
    {
        int32_t sweepVolume(int32_t &level, uint32_t &counter, uint16_t reg) noexcept
        {
            if ((reg & 0x8000u) == 0u)
                return level;
            const bool exponential = (reg & 0x4000u) != 0u;
            const bool decrease = (reg & 0x2000u) != 0u;
            const bool negative = (reg & 0x1000u) != 0u;
            const uint32_t rate = reg & 0x7Fu;
            int32_t magnitude = std::abs(level);
            envelopeTick(magnitude, counter, exponential, decrease, rate >> 2u,
                         decrease ? -8 + static_cast<int32_t>(rate & 3u) : 7 - static_cast<int32_t>(rate & 3u));
            level = negative ? -magnitude : magnitude;
            return level;
        }
    }

    int32_t IopSpu2::voiceSample(uint32_t coreIndex, uint32_t voiceIndex, Voice &v, bool noise, int32_t noiseValue,
                                 int32_t modulator)
    {
        Core &core = m_cores[coreIndex];
        int32_t step = std::min<int32_t>(v.pitch, 0x3FFF);
        if (voiceIndex > 0u && (core.pmon & (1u << voiceIndex)) != 0u)
            step = std::clamp<int32_t>((step * (0x8000 + modulator)) >> 15, 0, 0x3FFF);

        const int32_t sample = noise ? noiseValue : hermite(v.window, v.pitchCounter & 0xFFFu);
        v.pitchCounter += static_cast<uint32_t>(step);
        while (v.pitchCounter >= 0x1000u && v.phase != EnvPhase::Off)
        {
            v.pitchCounter -= 0x1000u;
            if (v.decodedPos >= 28u)
                decodeBlock(coreIndex, voiceIndex, v);
            v.window[0] = v.window[1];
            v.window[1] = v.window[2];
            v.window[2] = v.window[3];
            v.window[3] = v.decoded[v.decodedPos++];
        }

        const int32_t out = (sample * v.envLevel) >> 15;
        advanceEnvelope(v);
        v.lastOut = out;
        return out;
    }

    void IopSpu2::stepNoise(Core &core)
    {
        const uint32_t clock = (core.attr >> 8u) & 0x3Fu;
        const uint32_t shift = clock >> 2u;
        const int32_t step = static_cast<int32_t>(clock & 3u) + 4;
        core.noiseTimer -= step;
        if (core.noiseTimer < 0)
        {
            const uint32_t l = core.noiseLevel;
            const uint32_t parity = ((l >> 15u) ^ (l >> 12u) ^ (l >> 11u) ^ (l >> 10u) ^ 1u) & 1u;
            core.noiseLevel = ((l << 1u) | parity) & 0xFFFFu;
            core.noiseTimer += static_cast<int32_t>(0x20000u >> shift);
            if (core.noiseTimer < 0)
                core.noiseTimer += static_cast<int32_t>(0x20000u >> shift);
        }
    }

    void IopSpu2::mixCoreVoices(uint32_t coreIndex, int32_t &dryL, int32_t &dryR, int32_t &wetL, int32_t &wetR)
    {
        Core &core = m_cores[coreIndex];
        stepNoise(core);
        const int32_t noiseValue = static_cast<int16_t>(core.noiseLevel);
        int32_t previous = 0;
        for (uint32_t i = 0; i < 24u; ++i)
        {
            Voice &v = core.voices[i];
            if (v.phase == EnvPhase::Off)
            {
                v.lastOut = 0;
                previous = 0;
                continue;
            }
            const int32_t out = voiceSample(coreIndex, i, v, (core.non & (1u << i)) != 0u, noiseValue, previous);
            previous = out;
            const int32_t volL = sweepVolume(v.volL.level, v.volL.counter, v.volL.reg);
            const int32_t volR = sweepVolume(v.volR.level, v.volR.counter, v.volR.reg);
            const int32_t l = mulVol(out, volL);
            const int32_t r = mulVol(out, volR);
            const uint32_t bit = 1u << i;
            if (core.vmixl & bit)
                dryL += l;
            if (core.vmixr & bit)
                dryR += r;
            if (core.vmixel & bit)
                wetL += l;
            if (core.vmixer & bit)
                wetR += r;
        }
    }

    // ----------------------------------------------------------------------------------------
    // Reverb (PS1-style algorithm at 24 kHz, both sides on alternate samples)
    // ----------------------------------------------------------------------------------------

    void IopSpu2::runReverb(uint32_t coreIndex, int32_t inL, int32_t inR, int32_t &outL, int32_t &outR)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t start = core.esa;
        const uint32_t end = core.eea;
        if (end <= start || end - start < 16u)
        {
            outL = 0;
            outR = 0;
            return;
        }
        const uint32_t size = end - start + 1u;

        core.reverbInL += inL;
        core.reverbInR += inR;
        core.reverbPhase = !core.reverbPhase;
        if (core.reverbPhase)
        {
            outL = core.reverbOutL;
            outR = core.reverbOutR;
            return;
        }
        const int32_t sideInL = core.reverbInL / 2;
        const int32_t sideInR = core.reverbInR / 2;
        core.reverbInL = 0;
        core.reverbInR = 0;

        const auto &a = core.reverbAddr;
        const auto &k = core.reverbCoef;
        const auto at = [&](uint32_t offset) -> uint32_t
        {
            return start + ((core.reverbPos + offset) % size);
        };
        const auto rd = [&](uint32_t offset) -> int32_t
        {
            return static_cast<int16_t>(ramRead(at(offset)));
        };
        const bool write = (core.attr & 0x80u) != 0u;
        const auto wr = [&](uint32_t offset, int32_t value)
        {
            if (write)
                ramWrite(at(offset), static_cast<uint16_t>(clamp16(value)));
        };
        const auto mul = [](int32_t x, int32_t y)
        { return (x * y) >> 15; };

        int32_t results[2] = {};
        for (uint32_t side = 0; side < 2u; ++side)
        {
            const bool right = side != 0u;
            const int32_t in = mul(right ? k[InCoefR] : k[InCoefL], right ? sideInR : sideInL);
            const uint32_t sameDst = a[right ? SameRDst : SameLDst];
            const uint32_t sameSrc = a[right ? SameRSrc : SameLSrc];
            const uint32_t diffDst = a[right ? DiffRDst : DiffLDst];
            const uint32_t diffSrc = a[right ? DiffLSrc : DiffRSrc];
            const int32_t samePrev = rd(sameDst + size - 1u);
            const int32_t diffPrev = rd(diffDst + size - 1u);
            const int32_t same = mul(k[IirVol], in + mul(k[WallVol], rd(sameSrc)) - samePrev) + samePrev;
            const int32_t diff = mul(k[IirVol], in + mul(k[WallVol], rd(diffSrc)) - diffPrev) + diffPrev;
            int32_t out = mul(k[Comb1Vol], rd(a[right ? Comb1R : Comb1L])) +
                          mul(k[Comb2Vol], rd(a[right ? Comb2R : Comb2L])) +
                          mul(k[Comb3Vol], rd(a[right ? Comb3R : Comb3L])) +
                          mul(k[Comb4Vol], rd(a[right ? Comb4R : Comb4L]));
            const uint32_t apf1Dst = a[right ? Apf1RDst : Apf1LDst];
            const uint32_t apf2Dst = a[right ? Apf2RDst : Apf2LDst];
            const uint32_t apf1Src = apf1Dst + size - (a[ApfOffset1] % size);
            const uint32_t apf2Src = apf2Dst + size - (a[ApfOffset2] % size);
            const int32_t apf1 = out - mul(k[Apf1Vol], rd(apf1Src));
            out = rd(apf1Src) + mul(k[Apf1Vol], apf1);
            const int32_t apf2 = out - mul(k[Apf2Vol], rd(apf2Src));
            out = rd(apf2Src) + mul(k[Apf2Vol], apf2);
            wr(sameDst, same);
            wr(diffDst, diff);
            wr(apf1Dst, apf1);
            wr(apf2Dst, apf2);
            results[side] = clamp16(out);
        }
        core.reverbPos = (core.reverbPos + 1u) % size;
        core.reverbOutL = results[0];
        core.reverbOutR = results[1];
        outL = results[0];
        outR = results[1];
    }

    // ----------------------------------------------------------------------------------------
    // Mixer
    // ----------------------------------------------------------------------------------------

    void IopSpu2::generateSample()
    {
        ++m_samplesGenerated;
        int32_t ext[2] = {0, 0};
        int32_t final[2] = {0, 0};

        for (uint32_t c = 0; c < 2u; ++c)
        {
            Core &core = m_cores[c];
            int32_t dryL = 0, dryR = 0, wetL = 0, wetR = 0;
            mixCoreVoices(c, dryL, dryR, wetL, wetR);
            dryL = clamp16(dryL);
            dryR = clamp16(dryR);
            wetL = clamp16(wetL);
            wetR = clamp16(wetR);

            int32_t inL = 0, inR = 0;
            if ((core.admas & (1u << c)) != 0u)
            {
                const uint32_t base = 0x2000u + c * 0x400u + core.inputHalf * kInputHalfSamples + core.inputPos;
                if (core.inputReady[core.inputHalf])
                {
                    inL = static_cast<int16_t>(ramRead(base));
                    inR = static_cast<int16_t>(ramRead(base + 0x200u));
                }
                testIrq(base);
                testIrq(base + 0x200u);
                if (++core.inputPos >= kInputHalfSamples)
                {
                    core.inputReady[core.inputHalf] = false;
                    core.inputHalf ^= 1u;
                    core.inputPos = 0u;
                    fillInput(c);
                }
                inL = mulVol(inL, core.bvolL);
                inR = mulVol(inR, core.bvolR);
            }

            const uint32_t gate = core.mmix & (c == 0u ? 0xFF0u : 0xFFFu);
            const int32_t xL = c == 0u ? 0 : mulVol(ext[0], core.avolL);
            const int32_t xR = c == 0u ? 0 : mulVol(ext[1], core.avolR);

            int32_t tdL = 0, tdR = 0, twL = 0, twR = 0;
            if (gate & 0x080u)
                tdL += inL;
            if (gate & 0x040u)
                tdR += inR;
            if (gate & 0x800u)
                tdL += dryL;
            if (gate & 0x400u)
                tdR += dryR;
            if (gate & 0x008u)
                tdL += xL;
            if (gate & 0x004u)
                tdR += xR;
            if (gate & 0x020u)
                twL += inL;
            if (gate & 0x010u)
                twR += inR;
            if (gate & 0x200u)
                twL += wetL;
            if (gate & 0x100u)
                twR += wetR;
            if (gate & 0x002u)
                twL += xL;
            if (gate & 0x001u)
                twR += xR;

            int32_t rvL = 0, rvR = 0;
            runReverb(c, twL, twR, rvL, rvR);
            int32_t outL = tdL + mulVol(rvL, core.evolL);
            int32_t outR = tdR + mulVol(rvR, core.evolR);

            const int32_t mvL = sweepVolume(core.mvolL.level, core.mvolL.counter, core.mvolL.reg);
            const int32_t mvR = sweepVolume(core.mvolR.level, core.mvolR.counter, core.mvolR.reg);
            outL = clamp16(mulVol(clamp16(outL), mvL));
            outR = clamp16(mulVol(clamp16(outR), mvR));
            if (c == 0u)
            {
                ext[0] = outL;
                ext[1] = outR;
            }
            else
            {
                final[0] = outL;
                final[1] = outR;
            }
        }

        m_output.push_back(static_cast<int16_t>(final[0]));
        m_output.push_back(static_cast<int16_t>(final[1]));
    }
}
