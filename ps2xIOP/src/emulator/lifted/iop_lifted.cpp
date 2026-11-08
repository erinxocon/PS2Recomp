#include "iop_lifted.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <unordered_map>

namespace ps2x::iop::detail
{
    // Dry run: reads come from IOP memory (side-effect free); stores are held back and seen by the
    // block's own later reads. Only the instruction count is used.
    struct IopLifted::Shadow
    {
        const IopMemory &memory;
        std::unordered_map<uint32_t, uint8_t> bytes;

        static uint32_t key(uint32_t address) { return address & 0x1FFFFFFFu; }
        uint32_t readBytes(uint32_t address, uint32_t size, uint32_t fallback) const
        {
            uint32_t value = fallback;
            for (uint32_t i = 0; i < size; ++i)
            {
                const auto it = bytes.find(key(address + i));
                if (it != bytes.end())
                    value = (value & ~(0xFFu << (8u * i))) | (static_cast<uint32_t>(it->second) << (8u * i));
            }
            return value;
        }
        uint8_t read8(uint32_t a) const { return static_cast<uint8_t>(readBytes(a, 1u, memory.read8(a))); }
        uint16_t read16(uint32_t a) const { return static_cast<uint16_t>(readBytes(a, 2u, memory.read16(a))); }
        uint32_t read32(uint32_t a) const { return readBytes(a, 4u, memory.read32(a)); }
        void store(uint32_t a, uint32_t size, uint32_t value)
        {
            for (uint32_t i = 0; i < size; ++i)
                bytes[key(a + i)] = static_cast<uint8_t>(value >> (8u * i));
        }
        void write8(uint32_t a, uint8_t v) { store(a, 1u, v); }
        void write16(uint32_t a, uint16_t v) { store(a, 2u, v); }
        void write32(uint32_t a, uint32_t v) { store(a, 4u, v); }
    };

    // Replay: the reads the interpreter made, in order (address and size must match), and the
    // block's stores recorded for comparison.
    struct IopLifted::Replay
    {
        const std::vector<IopMemory::Write> &reads;
        std::vector<IopMemory::Write> &writes;
        size_t next = 0;
        bool matched = true;

        uint32_t read(uint32_t a, uint32_t size)
        {
            if (next >= reads.size() || reads[next].address != a || reads[next].size != size)
            {
                matched = false;
                return 0u;
            }
            return reads[next++].value;
        }
        uint8_t read8(uint32_t a) { return static_cast<uint8_t>(read(a, 1u)); }
        uint16_t read16(uint32_t a) { return static_cast<uint16_t>(read(a, 2u)); }
        uint32_t read32(uint32_t a) { return read(a, 4u); }
        void write8(uint32_t a, uint8_t v) { writes.push_back({a, 1u, v}); }
        void write16(uint32_t a, uint16_t v) { writes.push_back({a, 2u, v}); }
        void write32(uint32_t a, uint32_t v) { writes.push_back({a, 4u, v}); }
    };

    namespace
    {
        // ---- helpers the generated blocks use (semantics of IopCpuCore::executeInstruction) ----
        inline bool add_overflows(uint32_t a, uint32_t b)
        {
            const int64_t r = static_cast<int64_t>(static_cast<int32_t>(a)) + static_cast<int32_t>(b);
            return r > std::numeric_limits<int32_t>::max() || r < std::numeric_limits<int32_t>::min();
        }
        inline bool sub_overflows(uint32_t a, uint32_t b)
        {
            const int64_t r = static_cast<int64_t>(static_cast<int32_t>(a)) - static_cast<int32_t>(b);
            return r > std::numeric_limits<int32_t>::max() || r < std::numeric_limits<int32_t>::min();
        }
        inline void div_s(IopCpuState &cpu, uint32_t a, uint32_t b)
        {
            const int32_t lhs = static_cast<int32_t>(a), rhs = static_cast<int32_t>(b);
            if (rhs == 0)
            {
                cpu.lo = lhs >= 0 ? 0xFFFFFFFFu : 1u;
                cpu.hi = static_cast<uint32_t>(lhs);
            }
            else if (lhs == std::numeric_limits<int32_t>::min() && rhs == -1)
            {
                cpu.lo = static_cast<uint32_t>(lhs);
                cpu.hi = 0u;
            }
            else
            {
                cpu.lo = static_cast<uint32_t>(lhs / rhs);
                cpu.hi = static_cast<uint32_t>(lhs % rhs);
            }
        }
        inline void div_u(IopCpuState &cpu, uint32_t a, uint32_t b)
        {
            if (b == 0u)
            {
                cpu.lo = 0xFFFFFFFFu;
                cpu.hi = a;
            }
            else
            {
                cpu.lo = a / b;
                cpu.hi = a % b;
            }
        }
        template <class M>
        inline uint32_t lwl_value(M &mem, uint32_t address, uint32_t old)
        {
            static constexpr uint32_t masks[4] = {0x00FFFFFFu, 0x0000FFFFu, 0x000000FFu, 0x00000000u};
            static constexpr uint32_t shifts[4] = {24u, 16u, 8u, 0u};
            return (old & masks[address & 3u]) | (mem.read32(address & ~3u) << shifts[address & 3u]);
        }
        template <class M>
        inline uint32_t lwr_value(M &mem, uint32_t address, uint32_t old)
        {
            static constexpr uint32_t masks[4] = {0x00000000u, 0xFF000000u, 0xFFFF0000u, 0xFFFFFF00u};
            static constexpr uint32_t shifts[4] = {0u, 8u, 16u, 24u};
            return (old & masks[address & 3u]) | (mem.read32(address & ~3u) >> shifts[address & 3u]);
        }
        template <class M>
        inline void swl_store(M &mem, uint32_t address, uint32_t value)
        {
            const uint32_t aligned = address & ~3u;
            const uint32_t old = mem.read32(aligned);
            static constexpr uint32_t keep[4] = {0xFFFFFF00u, 0xFFFF0000u, 0xFF000000u, 0x00000000u};
            static constexpr uint32_t shifts[4] = {24u, 16u, 8u, 0u};
            mem.write32(aligned, (old & keep[address & 3u]) | (value >> shifts[address & 3u]));
        }
        template <class M>
        inline void swr_store(M &mem, uint32_t address, uint32_t value)
        {
            const uint32_t aligned = address & ~3u;
            const uint32_t old = mem.read32(aligned);
            static constexpr uint32_t keep[4] = {0x00000000u, 0x000000FFu, 0x0000FFFFu, 0x00FFFFFFu};
            static constexpr uint32_t shifts[4] = {0u, 8u, 16u, 24u};
            mem.write32(aligned, (old & keep[address & 3u]) | (value << shifts[address & 3u]));
        }

        template <class M>
        using BlockFn = uint32_t (*)(IopCpuState &, M &);

        template <class M>
        struct LiftedModule
        {
            const char *name;
            uint32_t start;
            uint32_t end;
            uint64_t hash;
            const BlockFn<M> *blocks;
        };

// Load delay as the interpreter does it: a loaded value lands after the next instruction, unless
// that instruction wrote the same register.
#define BLOCK_BEGIN()                                                                          \
    uint32_t *g = cpu.gpr.data();                                                              \
    bool pl = cpu.pendingLoad;                                                                 \
    uint32_t plr = cpu.pendingLoadReg, plv = cpu.pendingLoadValue;                             \
    cpu.yielded = false;                                                                       \
    cpu.exception = false;                                                                     \
    (void)g
#define STEP(written)                                    \
    do                                                   \
    {                                                    \
        if (pl)                                          \
        {                                                \
            if (plr != 0u && plr != (written))           \
                g[plr] = plv;                            \
            pl = false;                                  \
        }                                                \
    } while (0)
#define LOAD(reg, value) \
    do                   \
    {                    \
        pl = true;       \
        plr = (reg);     \
        plv = (value);   \
    } while (0)
#define SYNC_PENDING()               \
    do                               \
    {                                \
        cpu.pendingLoad = pl;        \
        cpu.pendingLoadReg = plr;    \
        cpu.pendingLoadValue = plv;  \
    } while (0)

#if __has_include("generated/iop_lifted_gitaroo.inc")
#define PS2X_HAVE_LIFTED_IOP 1
#include "generated/iop_lifted_gitaroo.inc"
#endif

#undef BLOCK_BEGIN
#undef STEP
#undef LOAD
#undef SYNC_PENDING

        uint64_t textHash(const IopMemory &memory, uint32_t start, uint32_t end)
        {
            uint64_t h = 0xCBF29CE484222325ull;
            for (uint32_t a = start; a < end; a += 4u)
                h = (h ^ memory.read32(a)) * 0x100000001B3ull;
            return h;
        }

        bool envFlag(const char *name, bool fallback)
        {
            const char *v = std::getenv(name);
            if (!v || !*v)
                return fallback;
            return std::strcmp(v, "0") != 0;
        }
    }

    IopLifted::IopLifted(IopMemory &memory) noexcept
        : m_memory(memory)
    {
#if defined(PS2X_HAVE_LIFTED_IOP)
        m_verify = envFlag("PS2X_IOP_LIFT_VERIFY", false);
        m_enabled = envFlag("PS2X_IOP_LIFT", true) || m_verify;
        m_stats = m_enabled && envFlag("PS2X_IOP_LIFT_STATS", false);
        m_valid.assign(std::size(kLiftedModules<IopMemory>), false);
#endif
    }

    void IopLifted::revalidate()
    {
#if defined(PS2X_HAVE_LIFTED_IOP)
        if (!m_enabled)
            return;
        for (size_t i = 0; i < m_valid.size(); ++i)
        {
            const LiftedModule<IopMemory> &module = kLiftedModules<IopMemory>[i];
            const bool valid = textHash(m_memory, module.start, module.end) == module.hash;
            if (valid != m_valid[i])
                std::fprintf(stderr, "[iop-lift] %s %s\n", module.name, valid ? "runs compiled" : "no longer matches, interpreted");
            m_valid[i] = valid;
        }
#endif
    }

    IopLifted::Lookup IopLifted::lookup(uint32_t pc) const
    {
#if defined(PS2X_HAVE_LIFTED_IOP)
        for (size_t i = 0; i < m_valid.size(); ++i)
        {
            const LiftedModule<IopMemory> &module = kLiftedModules<IopMemory>[i];
            if (pc < module.start || pc >= module.end || !m_valid[i])
                continue;
            const size_t index = (pc - module.start) >> 2u;
            return {module.blocks[index], kLiftedModules<Shadow>[i].blocks[index], kLiftedModules<Replay>[i].blocks[index]};
        }
#else
        (void)pc;
#endif
        return {};
    }

    void IopLifted::noteLifted(uint32_t executed)
    {
        m_liftedInstructions += executed;
        if (((m_liftedInstructions + m_interpretedInstructions) >> 20) !=
            ((m_liftedInstructions + m_interpretedInstructions - executed) >> 20))
            reportStats();
    }

    void IopLifted::countInterpreted(uint32_t pc)
    {
        if (!lookupModule(pc))
            return;
        ++m_interpretedInstructions;
        ++m_interpretedPcs[pc];
    }

    void IopLifted::reportStats()
    {
        const uint64_t total = m_liftedInstructions + m_interpretedInstructions;
        std::vector<std::pair<uint64_t, uint32_t>> top;
        for (const auto &[pc, count] : m_interpretedPcs)
            top.push_back({count, pc});
        std::sort(top.rbegin(), top.rend());
        std::fprintf(stderr, "[iop-lift] module instructions: %.2f%% compiled (%llu interpreted); top interpreted:",
                     total ? 100.0 * m_liftedInstructions / total : 0.0,
                     static_cast<unsigned long long>(m_interpretedInstructions));
        for (size_t i = 0; i < top.size() && i < 8u; ++i)
            std::fprintf(stderr, " %x(%llu)", top[i].second, static_cast<unsigned long long>(top[i].first));
        std::fprintf(stderr, "\n");
    }

    bool IopLifted::lookupModule(uint32_t pc) const
    {
#if defined(PS2X_HAVE_LIFTED_IOP)
        for (size_t i = 0; i < m_valid.size(); ++i)
            if (m_valid[i] && pc >= kLiftedModules<IopMemory>[i].start && pc < kLiftedModules<IopMemory>[i].end)
                return true;
#else
        (void)pc;
#endif
        return false;
    }

    uint32_t IopLifted::runShadow(ShadowFn fn, IopCpuState &cpu)
    {
        Shadow shadow{m_memory, {}};
        return fn(cpu, shadow);
    }

    bool IopLifted::runReplay(ReplayFn fn, IopCpuState &cpu, const std::vector<IopMemory::Write> &reads,
                              std::vector<IopMemory::Write> &writes)
    {
        Replay replay{reads, writes};
        fn(cpu, replay);
        return replay.matched && replay.next == reads.size();
    }

    void IopLifted::compare(const IopCpuState &before, const IopCpuState &lifted, const IopCpuState &real,
                            const std::vector<IopMemory::Write> &liftedWrites,
                            const std::vector<IopMemory::Write> &realWrites, bool sameCount)
    {
        ++m_verified;
        bool same = sameCount && lifted.pc == real.pc && lifted.gpr == real.gpr && lifted.hi == real.hi &&
                    lifted.lo == real.lo && lifted.cop0 == real.cop0 && lifted.pendingLoad == real.pendingLoad &&
                    lifted.branchPending == real.branchPending && liftedWrites == realWrites;
        if (same && real.pendingLoad)
            same = lifted.pendingLoadReg == real.pendingLoadReg && lifted.pendingLoadValue == real.pendingLoadValue;
        if (same && real.branchPending)
            same = lifted.branchTarget == real.branchTarget;
        if (!same)
        {
            ++m_mismatches;
            if (m_mismatches <= 20u)
            {
                std::fprintf(stderr, "[iop-lift] MISMATCH block 0x%x%s: pc lifted 0x%x real 0x%x\n", before.pc,
                             sameCount ? "" : " (different count or read sequence)", lifted.pc, real.pc);
                for (int r = 0; r < 32; ++r)
                    if (lifted.gpr[r] != real.gpr[r])
                        std::fprintf(stderr, "    r%d lifted 0x%x real 0x%x\n", r, lifted.gpr[r], real.gpr[r]);
                if (lifted.hi != real.hi || lifted.lo != real.lo)
                    std::fprintf(stderr, "    hi/lo lifted 0x%x/0x%x real 0x%x/0x%x\n", lifted.hi, lifted.lo, real.hi, real.lo);
                if (lifted.pendingLoad != real.pendingLoad || lifted.pendingLoadReg != real.pendingLoadReg ||
                    lifted.pendingLoadValue != real.pendingLoadValue)
                    std::fprintf(stderr, "    pending load lifted %d r%u=0x%x real %d r%u=0x%x\n", lifted.pendingLoad,
                                 lifted.pendingLoadReg, lifted.pendingLoadValue, real.pendingLoad, real.pendingLoadReg,
                                 real.pendingLoadValue);
                if (lifted.branchPending != real.branchPending)
                    std::fprintf(stderr, "    branch pending lifted %d real %d\n", lifted.branchPending, real.branchPending);
                if (liftedWrites != realWrites)
                {
                    std::fprintf(stderr, "    stores: lifted %zu real %zu\n", liftedWrites.size(), realWrites.size());
                    for (size_t i = 0; i < std::max(liftedWrites.size(), realWrites.size()) && i < 8u; ++i)
                    {
                        const IopMemory::Write l = i < liftedWrites.size() ? liftedWrites[i] : IopMemory::Write{};
                        const IopMemory::Write r = i < realWrites.size() ? realWrites[i] : IopMemory::Write{};
                        std::fprintf(stderr, "      [%zu] lifted %08x/%u=%08x real %08x/%u=%08x\n", i, l.address, l.size,
                                     l.value, r.address, r.size, r.value);
                    }
                }
            }
        }
        if ((m_verified & ((1u << 20) - 1u)) == 0u)
            std::fprintf(stderr, "[iop-lift] %llu blocks verified, %llu mismatches\n",
                         static_cast<unsigned long long>(m_verified), static_cast<unsigned long long>(m_mismatches));
    }
}
