#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps2x::iop::detail
{
    // SPU2 sound processor plus the two IOP DMA channels wired to it
    // (ch4 -> core 0, ch7 -> core 1). Runs in IOP cycle time: one 48 kHz output
    // sample every 768 IOP cycles. Voices decode PS-ADPCM from SPU RAM with ADSR
    // envelopes; AutoDMA (ADMA) streams PCM from IOP RAM into the core input
    // areas; the cores are mixed as on hardware (core 0 output feeds core 1's
    // external input) and the final core 1 output goes to the host sink.
    class IopSpu2
    {
    public:
        static constexpr uint32_t RegBase = 0x1F900000u;
        static constexpr uint32_t RegSize = 0x800u;
        static constexpr uint32_t RamHalfwords = 1u << 20u; // 2 MiB
        static constexpr uint64_t CyclesPerSample = 768u;    // 36.864 MHz / 48 kHz

        static constexpr int IrqSpu = 9;
        static constexpr int IrqDmaCore0 = 0x24;
        static constexpr int IrqDmaCore1 = 0x28;

        explicit IopSpu2(std::vector<uint8_t> &iopRam);

        void reset();
        void setClock(const uint64_t *cycles) noexcept { m_clock = cycles; }

        [[nodiscard]] static bool isRegister(uint32_t phys) noexcept
        {
            return phys >= RegBase && phys < RegBase + RegSize;
        }
        [[nodiscard]] static bool isDmaRegister(uint32_t phys) noexcept
        {
            return (phys >= 0x1F8010C0u && phys < 0x1F8010CCu) ||
                   (phys >= 0x1F801500u && phys < 0x1F80150Cu);
        }

        [[nodiscard]] uint16_t read16(uint32_t phys);
        void write16(uint32_t phys, uint16_t value);
        [[nodiscard]] uint32_t readDma32(uint32_t phys);
        void writeDma32(uint32_t phys, uint32_t value);

        // Generate output up to the given IOP cycle.
        void advanceTo(uint64_t cycle);
        // Earliest cycle at which the SPU2 wants to be advanced again.
        [[nodiscard]] uint64_t nextEventCycle() const noexcept;

        // Bitmask of raised interrupts: bit0 SPU (IRQ 9), bit1 DMA ch4, bit2 DMA ch7.
        [[nodiscard]] uint32_t pendingIrqs() const noexcept { return m_pendingIrqs; }
        [[nodiscard]] uint32_t takePendingIrqs() noexcept
        {
            const uint32_t value = m_pendingIrqs;
            m_pendingIrqs = 0u;
            return value;
        }

        [[nodiscard]] uint64_t samplesGenerated() const noexcept { return m_samplesGenerated; }

    private:
        enum class EnvPhase : uint8_t
        {
            Off,
            Attack,
            Decay,
            Sustain,
            Release,
        };

        struct VolumeSlide
        {
            uint16_t reg = 0;
            int32_t level = 0; // current value, -0x8000..0x7FFF
            uint32_t counter = 0;
        };

        struct Voice
        {
            VolumeSlide volL;
            VolumeSlide volR;
            uint16_t pitch = 0;
            uint16_t adsr1 = 0;
            uint16_t adsr2 = 0;
            uint32_t ssa = 0;
            uint32_t lsa = 0;
            uint32_t nax = 0;

            EnvPhase phase = EnvPhase::Off;
            int32_t envLevel = 0;
            uint32_t envCounter = 0;

            uint32_t pitchCounter = 0;
            std::array<int16_t, 28> decoded{};
            uint32_t decodedPos = 28;
            uint8_t blockFlags = 0;
            bool freshStart = true;
            uint32_t blockAddress = 0;
            int32_t hist1 = 0;
            int32_t hist2 = 0;
            std::array<int32_t, 4> window{};
            int32_t lastOut = 0;
        };

        struct Core
        {
            std::array<Voice, 24> voices{};
            uint32_t pmon = 0;
            uint32_t non = 0;
            uint32_t vmixl = 0;
            uint32_t vmixel = 0;
            uint32_t vmixr = 0;
            uint32_t vmixer = 0;
            uint32_t endx = 0;
            uint16_t mmix = 0;
            uint16_t attr = 0;
            uint16_t statx = 0;
            uint16_t admas = 0;
            uint32_t irqa = 0;
            uint32_t tsa = 0;
            VolumeSlide mvolL;
            VolumeSlide mvolR;
            int16_t evolL = 0;
            int16_t evolR = 0;
            int16_t avolL = 0;
            int16_t avolR = 0;
            int16_t bvolL = 0;
            int16_t bvolR = 0;

            // Reverb work area (halfword addresses) and registers.
            uint32_t esa = 0;
            uint32_t eea = 0;
            uint32_t reverbPos = 0;
            std::array<uint32_t, 22> reverbAddr{};
            std::array<int16_t, 10> reverbCoef{};
            int32_t reverbInL = 0;
            int32_t reverbInR = 0;
            int32_t reverbOutL = 0;
            int32_t reverbOutR = 0;
            bool reverbPhase = false;

            // IOP DMA channel feeding this core.
            uint32_t madr = 0;
            uint32_t bcr = 0;
            uint32_t chcr = 0;
            bool dmaActive = false;
            bool dmaAdma = false;
            uint32_t dmaRemaining = 0;
            uint64_t dmaDoneCycle = 0;

            // AutoDMA input double buffer: 2 halves of 256 stereo samples.
            std::array<bool, 2> inputReady{};
            uint32_t inputHalf = 0;
            uint32_t inputPos = 0;

            uint32_t noiseLevel = 1;
            int32_t noiseTimer = 0;
        };

        [[nodiscard]] uint64_t now() const noexcept { return m_clock ? *m_clock : m_sampleCycle; }

        uint16_t readCoreReg(uint32_t core, uint32_t local);
        void writeCoreReg(uint32_t core, uint32_t local, uint16_t value);
        void keyOn(Core &core, uint32_t coreIndex, uint32_t mask);
        void keyOff(Core &core, uint32_t mask);

        void startDma(uint32_t coreIndex);
        void finishDma(uint32_t coreIndex);
        void fillInput(uint32_t coreIndex);

        void generateSample();
        void mixCoreVoices(uint32_t coreIndex, int32_t &dryL, int32_t &dryR, int32_t &wetL, int32_t &wetR);
        int32_t voiceSample(uint32_t coreIndex, uint32_t voiceIndex, Voice &voice, bool noise, int32_t noiseValue,
                            int32_t modulator);
        void decodeBlock(uint32_t coreIndex, uint32_t voiceIndex, Voice &voice);
        void advanceEnvelope(Voice &voice);
        void runReverb(uint32_t coreIndex, int32_t inL, int32_t inR, int32_t &outL, int32_t &outR);
        void stepNoise(Core &core);

        void testIrq(uint32_t address);
        void testIrqRange(uint32_t address, uint32_t count);
        void flushOutput();

        uint16_t ramRead(uint32_t address) const noexcept { return m_ram[address & (RamHalfwords - 1u)]; }
        void ramWrite(uint32_t address, uint16_t value) noexcept { m_ram[address & (RamHalfwords - 1u)] = value; }

        std::vector<uint8_t> &m_iopRam;
        std::vector<uint16_t> m_ram;
        std::array<Core, 2> m_cores{};
        std::array<uint16_t, RegSize / 2u> m_regs{};
        const uint64_t *m_clock = nullptr;
        uint64_t m_sampleCycle = 0;
        uint64_t m_samplesGenerated = 0;
        uint32_t m_pendingIrqs = 0;
        uint16_t m_spdifIrqInfo = 0;
        std::vector<int16_t> m_output;
    };
}
