#pragma once

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ps2x::iop::detail
{
    // IOP modules compiled to C++ by tools/ioplift/ioplift.py (Gitaroo's MSTREAM, SDRDRV, LIBSD).
    // A block runs from its start address to the end of a basic block with the interpreter's exact
    // semantics and returns how many instructions it executed (0: the interpreter must run the
    // instruction at cpu.pc). A module's blocks are used only while its text in IOP RAM matches the
    // image they were generated from (on by default; PS2X_IOP_LIFT=0 turns them off). PS2X_IOP_LIFT_VERIFY=1 runs the
    // interpreter for real and checks every block against it (registers and stores).
    class IopLifted
    {
    public:
        explicit IopLifted(IopMemory &memory) noexcept;

        // Re-check the module images (call after a module is loaded or unloaded).
        void revalidate();

        // Runs the block at cpu.pc if there is one; returns instructions executed (0 = none).
        // In verify mode `interpret` executes one instruction on the real state.
        template <class Interpret>
        uint32_t run(IopCpuState &cpu, Interpret &&interpret)
        {
            if (!m_enabled || cpu.branchPending)
                return 0u;
            const Lookup found = lookup(cpu.pc);
            if (!found.block)
                return 0u;
            if (!m_verify)
            {
                const uint32_t executed = found.block(cpu, m_memory);
                if (m_stats)
                    noteLifted(executed);
                return executed;
            }
            return runVerified(cpu, found, interpret);
        }

        // PS2X_IOP_LIFT_STATS=1: instructions run compiled vs. interpreted inside the lifted modules,
        // with the addresses the interpreter runs most (missed block starts).
        void noteInterpreted(uint32_t pc)
        {
            if (m_stats)
                countInterpreted(pc);
        }

        struct Shadow;
        struct Replay;
        using RealFn = uint32_t (*)(IopCpuState &, IopMemory &);
        using ShadowFn = uint32_t (*)(IopCpuState &, Shadow &);
        using ReplayFn = uint32_t (*)(IopCpuState &, Replay &);
        struct Lookup
        {
            RealFn block = nullptr;
            ShadowFn shadow = nullptr;
            ReplayFn replay = nullptr;
        };

    private:
        Lookup lookup(uint32_t pc) const;
        bool lookupModule(uint32_t pc) const;

        // Verification: a dry run of the block (stores held back) gives its instruction count; the
        // interpreter then runs that many instructions for real while IOP memory records every read
        // and store; finally the block runs again from the saved state, fed the recorded reads, and
        // its registers and stores must equal the interpreter's. Hardware is touched once.
        template <class Interpret>
        uint32_t runVerified(IopCpuState &cpu, const Lookup &found, Interpret &interpret)
        {
            IopCpuState probe = cpu;
            const uint32_t count = runShadow(found.shadow, probe);
            if (count == 0u)
                return 0u;
            const IopCpuState before = cpu;
            std::vector<IopMemory::Write> reads, realWrites;
            m_memory.setJournals(&reads, &realWrites);
            uint32_t executed = 0u;
            while (executed < count)
            {
                // The interpreter's instruction fetch is not a data read.
                const size_t mark = reads.size();
                const uint32_t fetchPc = cpu.pc;
                interpret(cpu);
                if (reads.size() > mark && reads[mark].address == fetchPc && reads[mark].size == 4u)
                    reads.erase(reads.begin() + static_cast<std::ptrdiff_t>(mark));
                ++executed;
                if (cpu.exception || cpu.stopped)
                    break;
            }
            m_memory.setJournals(nullptr, nullptr);
            IopCpuState lifted = before;
            std::vector<IopMemory::Write> liftedWrites;
            const bool readsMatched = runReplay(found.replay, lifted, reads, liftedWrites);
            compare(before, lifted, cpu, liftedWrites, realWrites, executed == count && readsMatched);
            return executed;
        }

        uint32_t runShadow(ShadowFn fn, IopCpuState &cpu);
        bool runReplay(ReplayFn fn, IopCpuState &cpu, const std::vector<IopMemory::Write> &reads,
                       std::vector<IopMemory::Write> &writes);
        void compare(const IopCpuState &before, const IopCpuState &lifted, const IopCpuState &real,
                     const std::vector<IopMemory::Write> &liftedWrites, const std::vector<IopMemory::Write> &realWrites,
                     bool sameCount);

        void noteLifted(uint32_t executed);
        void countInterpreted(uint32_t pc);
        void reportStats();

        IopMemory &m_memory;
        bool m_stats = false;
        uint64_t m_liftedInstructions = 0;
        uint64_t m_interpretedInstructions = 0;
        std::unordered_map<uint32_t, uint64_t> m_interpretedPcs;
        bool m_enabled = false;
        bool m_verify = false;
        std::vector<bool> m_valid;
        uint64_t m_verified = 0;
        uint64_t m_mismatches = 0;
    };
}
