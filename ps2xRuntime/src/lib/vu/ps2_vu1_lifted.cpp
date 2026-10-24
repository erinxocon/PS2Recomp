#include "ps2_vu1_lifted.h"

#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define VF(r) lvf[r]
#define VI(r) lvi[r]

// ---- helpers for the generated code ---------------------------------------------------------------

inline void VU1Lifted::step(VU1Interpreter &vu)
{
    if (vu.m_cycle >= vu.m_nextCommitCycle)
        vu.commitReadyPipelines();
}

inline void VU1Lifted::tick(VU1Interpreter &vu)
{
    ++vu.m_cycle;
}

// An explicit stall (WAITQ/WAITP, busy FDIV/EFU): one cycle at a time, so pipeline entries commit in
// the order they would in the interpreter.
inline void VU1Lifted::advance(VU1Interpreter &vu, uint64_t target)
{
    while (vu.m_cycle < target)
    {
        ++vu.m_cycle;
        vu.commitReadyPipelines();
    }
}

inline void VU1Lifted::waitFdiv(VU1Interpreter &vu)
{
    if (vu.m_fdiv.valid)
        advance(vu, vu.m_fdiv.readyCycle);
}

inline void VU1Lifted::waitEfuResource(VU1Interpreter &vu)
{
    advance(vu, vu.m_efuResourceReady);
}

inline void VU1Lifted::waitP(VU1Interpreter &vu)
{
    uint64_t target = vu.m_efuResourceReady;
    for (const auto &entry : vu.m_efu)
        if (entry.valid)
            target = std::max(target, entry.readyCycle);
    advance(vu, target);
}

inline int16_t VU1Lifted::bv(const int32_t *vi, const VU1Interpreter &vu, const BranchBackup &bk, uint8_t reg)
{
    if (reg == 0u)
        return 0;
    if (bk.reg == reg && bk.cycle + 1u == vu.m_cycle)
        return static_cast<int16_t>(bk.value);
    return static_cast<int16_t>(vi[reg]);
}

// AC5's vu_do_xgkick: the whole packet leaves at the XGKICK. The census found no store into a kicked
// packet before the interpreter's PATH1 transfer had copied it, so the bytes are the same.
inline void VU1Lifted::xgkick(VU1Interpreter &vu, uint32_t qwordAddress)
{
    vu.startXgkick(qwordAddress);
    vu.m_xgkick.cycleCredit = 0xFFFFFFF0u;
    vu.progressXgkick();
}

inline int VU1Lifted::handoff(VU1Interpreter &vu, const BranchBackup &bk, uint32_t pc)
{
    vu.m_state.pc = pc;
    vu.m_state.branchPending = false;
    vu.m_viBranchBackupValid = bk.reg != 0u && bk.cycle + 1u == vu.m_cycle;
    vu.m_viBranchBackupReg = bk.reg;
    vu.m_viBranchBackupValue = bk.value;
    return kLiftHandoff;
}

inline int VU1Lifted::yield(VU1Interpreter &vu, const BranchBackup &bk, uint32_t pc)
{
    handoff(vu, bk, pc);
    return kLiftYield;
}

inline uint32_t VU1Lifted::dynamicBranchTarget(const LiftedRun &run, uint32_t pc)
{
    uint32_t lower = 0;
    std::memcpy(&lower, run.code + pc, 4);
    const int32_t imm = static_cast<int32_t>(lower << 21) >> 21;
    return (pc + 8u + static_cast<uint32_t>(imm * 8)) & 0x3FFFu;
}

inline uint32_t VU1Lifted::lowBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float VU1Lifted::nrm(float value)
{
    uint32_t bits = lowBits(value);
    const uint32_t exponent = bits & 0x7F800000u;
    if (exponent == 0u)
        bits &= 0x80000000u;
    else if (exponent == 0x7F800000u)
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// vuFloatToInt (ps2_vu1_upper.cpp), returning the integer's bits as a float.
inline float VU1Lifted::ftoi(float value, float scale)
{
    const double scaled = static_cast<double>(value) * static_cast<double>(scale);
    int32_t result = 0;
    if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
        result = std::numeric_limits<int32_t>::max();
    else if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
        result = std::numeric_limits<int32_t>::min();
    else
        result = static_cast<int32_t>(scaled);
    float out = 0.0f;
    std::memcpy(&out, &result, sizeof(out));
    return out;
}

// applyFmacDest / applyFmacDestAcc: normalize each written lane (exactly, through the interpreter,
// only when the result is outside the plainly normal range), queue the flags if observable, write.
// The common case (every lane plainly normal, flags not observed) stays inline.
template <uint8_t Dest, bool Flags>
inline void VU1Lifted::fmacStore(VU1Interpreter &vu, uint32_t word, float *result, float *dst, const float (*lvf)[4],
                                 const float *lacc)
{
    if constexpr (!Flags)
    {
        bool plain = true;
        for (uint32_t c = 0; c < 4u; ++c)
            if (Dest & (8u >> c))
                plain &= ((lowBits(result[c]) >> 23) & 0xFFu) - 2u <= 251u;
        if (plain)
        {
            for (uint32_t c = 0; c < 4u; ++c)
                if (Dest & (8u >> c))
                    dst[c] = result[c];
            return;
        }
    }
    fmacStoreSlow(vu, word, result, dst, lvf, lacc, Dest, Flags);
}

void VU1Lifted::fmacStoreSlow(VU1Interpreter &vu, uint32_t word, float *result, float *dst, const float (*lvf)[4],
                              const float *lacc, uint8_t dest, bool flags)
{
    uint8_t laneFlags[4] = {0u, 0u, 0u, 0u};
    bool synced = false;
    auto sync = [&]()
    {
        if (!synced)
        {
            std::memcpy(vu.m_state.vf, lvf, sizeof(vu.m_state.vf));
            std::memcpy(vu.m_state.acc, lacc, sizeof(vu.m_state.acc));
            synced = true;
        }
    };
    for (uint32_t c = 0; c < 4u; ++c)
    {
        if ((dest & (8u >> c)) == 0u)
            continue;
        const uint32_t bits = lowBits(result[c]);
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent >= 2u && exponent <= 253u)
        {
            laneFlags[c] = (bits >> 31) != 0u ? 0x2u : 0u;
            continue;
        }
        sync();
        vu.m_currentUpperInstruction = word;
        double exact = 0.0;
        if (vu.calculateFmacExactResult(c, exact))
            laneFlags[c] = vu.normalizeFmacExactResult(result[c], exact);
        else
        {
            uint32_t laneFlagBits = 0u;
            result[c] = vu.normalizeResult(result[c], laneFlagBits);
            laneFlags[c] = static_cast<uint8_t>(laneFlagBits);
        }
    }
    if (flags && vu.m_flagsObservable)
    {
        sync();
        vu.m_currentUpperInstruction = word;
        vu.updateFmacFlags(laneFlags, dest, vu.calculateFmacProductSticky(dest));
    }
    for (uint32_t c = 0; c < 4u; ++c)
        if ((dest & (8u >> c)) != 0u)
            dst[c] = result[c];
}

inline void VU1Lifted::opDiv(VU1Interpreter &vu, float numRaw, float denRaw)
{
    const float num = vu.normalizeOperand(numRaw);
    const float den = vu.normalizeOperand(denRaw);
    uint32_t statusDi = 0u;
    float result = 0.0f;
    if (den == 0.0f)
    {
        statusDi = num == 0.0f ? 0x10u : 0x20u;
        result = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max()
                                                        : std::numeric_limits<float>::max();
    }
    else
        result = num / den;
    uint32_t ignoredFlags = 0u;
    result = vu.normalizeResult(result, ignoredFlags);
    vu.queueQ(result, 7u, statusDi);
}

inline void VU1Lifted::opSqrt(VU1Interpreter &vu, float valueRaw)
{
    const float val = vu.normalizeOperand(valueRaw);
    vu.queueQ(std::sqrt(std::fabs(val)), 7u, val < 0.0f ? 0x10u : 0u);
}

inline void VU1Lifted::opRsqrt(VU1Interpreter &vu, float numRaw, float radicandRaw)
{
    const float num = vu.normalizeOperand(numRaw);
    const float radicand = vu.normalizeOperand(radicandRaw);
    const float den = std::sqrt(std::fabs(radicand));
    uint32_t statusDi = radicand < 0.0f ? 0x10u : 0u;
    float result = 0.0f;
    if (den != 0.0f)
        result = num / den;
    else
    {
        statusDi = num == 0.0f ? 0x10u : 0x20u;
        result = std::signbit(num) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
    }
    uint32_t ignoredFlags = 0u;
    result = vu.normalizeResult(result, ignoredFlags);
    vu.queueQ(result, 13u, statusDi);
}

inline void VU1Lifted::opErleng(VU1Interpreter &vu, const float *v)
{
    const float x = vu.normalizeOperand(v[0]);
    const float y = vu.normalizeOperand(v[1]);
    const float z = vu.normalizeOperand(v[2]);
    const float len = std::sqrt(x * x + y * y + z * z);
    vu.queueP(len != 0.0f ? 1.0f / len : len, 24u);
}

inline void VU1Lifted::opEsqrt(VU1Interpreter &vu, float valueRaw)
{
    const float value = vu.normalizeOperand(valueRaw);
    vu.queueP(value >= 0.0f ? std::sqrt(value) : value, 12u);
}

inline void VU1Lifted::opClip(VU1Interpreter &vu, const float *fs, float ftw)
{
    uint32_t wBits = lowBits(ftw);
    const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
    auto exceeds = [limit](float value, uint32_t signMask)
    {
        const int32_t ordered = static_cast<int32_t>(lowBits(value) ^ signMask);
        return ordered > limit;
    };
    uint32_t flags = 0u;
    if (exceeds(fs[0], 0x00000000u))
        flags |= 0x01u;
    if (exceeds(fs[0], 0x80000000u))
        flags |= 0x02u;
    if (exceeds(fs[1], 0x00000000u))
        flags |= 0x04u;
    if (exceeds(fs[1], 0x80000000u))
        flags |= 0x08u;
    if (exceeds(fs[2], 0x00000000u))
        flags |= 0x10u;
    if (exceeds(fs[2], 0x80000000u))
        flags |= 0x20u;
    vu.queueClip(flags);
}

// ---- the generated program ------------------------------------------------------------------------

#if __has_include("generated/ps2_vu1_lifted_gitaroo.inc")
#define PS2X_HAVE_LIFTED_VU1 1
#include "generated/ps2_vu1_lifted_gitaroo.inc"
#else
constexpr uint32_t kImageBytes = 0;
static const uint32_t kImage[1] = {0};
static const uint16_t kDynamicPairs[1] = {0};
static bool isBlockStart(uint32_t) { return false; }
constexpr bool kLiftedFlagsExact = true;
int VU1Lifted::runProgram(VU1Interpreter &, LiftedRun &, uint32_t) { return kLiftNotLifted; }
#endif

namespace vu1lift
{
    struct Settings
    {
        bool lift = false;
        bool verify = false;
        Settings()
        {
            const char *lift = std::getenv("PS2X_VU1_LIFT");
            const char *verifyEnv = std::getenv("PS2X_VU1_VERIFY");
            verify = verifyEnv && *verifyEnv && std::strcmp(verifyEnv, "0") != 0;
            this->lift = !(lift && std::strcmp(lift, "0") == 0);
#ifndef PS2X_HAVE_LIFTED_VU1
            this->lift = verify = false;
#endif
            if (verify)
                std::fprintf(stderr, "[vu1-lift] lockstep verification on (PS2X_VU1_VERIFY)\n");
            else if (this->lift)
                std::fprintf(stderr, "[vu1-lift] recompiled VU1 microprogram enabled (PS2X_VU1_LIFT=0 disables)\n");
        }
    };

    const Settings &settings()
    {
        static const Settings s;
        return s;
    }

    struct Stats
    {
        uint64_t runs = 0, handoffs = 0, yields = 0, verified = 0, mismatches = 0, budgetSkips = 0;
        ~Stats()
        {
            if (runs || verified)
                std::fprintf(stderr, "[vu1-lift] %llu lifted runs (%llu handed to the interpreter, %llu out of budget); "
                                     "verified %llu, mismatches %llu, budget-skipped %llu\n",
                             static_cast<unsigned long long>(runs), static_cast<unsigned long long>(handoffs),
                             static_cast<unsigned long long>(yields), static_cast<unsigned long long>(verified),
                             static_cast<unsigned long long>(mismatches), static_cast<unsigned long long>(budgetSkips));
        }
    } g_stats;

    // Whether the loaded micro image is the lifted program: every word equal except the dynamic
    // branches, which must still be a B with the same upper word. Cached per code generation.
    bool imageMatches(const uint8_t *code, uint32_t codeSize, uint64_t generation)
    {
        static uint64_t s_generation = ~0ull;
        static const uint8_t *s_code = nullptr;
        static bool s_match = false;
        if (generation == s_generation && code == s_code)
            return s_match;
        s_generation = generation;
        s_code = code;
        s_match = false;
        if (kImageBytes == 0 || codeSize < kImageBytes)
            return false;
        const uint8_t *image = reinterpret_cast<const uint8_t *>(kImage);
        uint32_t from = 0;
        for (uint16_t pc : kDynamicPairs)
        {
            if (std::memcmp(code + from, image + from, pc - from) != 0)
                return false;
            uint32_t lower = 0, upper = 0, wantUpper = 0;
            std::memcpy(&lower, code + pc, 4);
            std::memcpy(&upper, code + pc + 4, 4);
            std::memcpy(&wantUpper, image + pc + 4, 4);
            if ((lower >> 25) != 0x20u || upper != wantUpper)
                return false;
            from = pc + 8u;
        }
        s_match = std::memcmp(code + from, image + from, kImageBytes - from) == 0;
        return s_match;
    }
}

using namespace vu1lift;


// ---- running ---------------------------------------------------------------------------------------

int VU1Lifted::runLifted(VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData,
                         uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    vu.m_activeVuData = vuData;
    vu.m_activeVuDataSize = dataSize;
    vu.m_activeGs = &gs;
    vu.m_activeMemory = memory;
    vu.refreshFlagObservability(vuCode, codeSize, memory);

    LiftedRun run{vuData, dataSize, dataSize - 1u, vuCode, vu.m_cycle + maxCycles, &gs, memory};
    const int previousRounding = std::fegetround();
    std::fesetround(FE_TOWARDZERO);
    int result = vu.m_state.pc == 0x1B68u ? runNative(vu, run) : kLiftNotLifted;
    if (result == kLiftNotLifted || result == kLiftContinue)
        result = runProgram(vu, run, vu.m_state.pc);
    std::fesetround(previousRounding);

    ++g_stats.runs;
    if (result == kLiftEnd)
    {
        vu.flushPipelines();
        vu.m_state.ebit = false;
        vu.m_state.haltAfterDelaySlot = false;
        vu.m_pendingHaltD = false;
        vu.m_pendingHaltT = false;
    }
    else if (result == kLiftHandoff)
    {
        ++g_stats.handoffs;
        if (vu.m_cycle < run.budgetEnd)
            vu.run(vuCode, codeSize, vuData, dataSize, gs, memory, static_cast<uint32_t>(run.budgetEnd - vu.m_cycle));
    }
    else if (result == kLiftYield)
    {
        ++g_stats.yields;
    }
    vu.m_state.vf[0][0] = 0.0f;
    vu.m_state.vf[0][1] = 0.0f;
    vu.m_state.vf[0][2] = 0.0f;
    vu.m_state.vf[0][3] = 1.0f;
    vu.m_state.vi[0] = 0;
    vu.m_state.cycles = vu.m_cycle;
    return result;
}

bool VU1Lifted::tryExecute(VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData,
                           uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    const Settings &s = settings();
    if (!s.lift && !s.verify)
        return false;
    if (!memory || vuCode != memory->getVU1Code() || dataSize < 16u || (dataSize & (dataSize - 1u)) != 0u)
        return false;
    if (!imageMatches(vuCode, codeSize, memory->getVU1CodeGeneration()))
        return false;
    if (!isBlockStart(vu.m_state.pc) || vu.m_state.dBitEnabled || vu.m_state.tBitEnabled)
        return false;
    if (s.verify)
        verifyExecute(vu, vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
    else
        runLifted(vu, vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
    return true;
}

// ---- lockstep verification -------------------------------------------------------------------------

struct VU1Lifted::Snapshot
{
    VU1State state;
    decltype(VU1Interpreter::m_flagPipeline) flagPipeline;
    decltype(VU1Interpreter::m_fdiv) fdiv;
    decltype(VU1Interpreter::m_efu) efu;
    decltype(VU1Interpreter::m_storePipeline) storePipeline;
    decltype(VU1Interpreter::m_vfWritePipeline) vfWritePipeline;
    decltype(VU1Interpreter::m_viWritePipeline) viWritePipeline;
    decltype(VU1Interpreter::m_accWritePipeline) accWritePipeline;
    decltype(VU1Interpreter::m_vfReady) vfReady;
    decltype(VU1Interpreter::m_viReady) viReady;
    decltype(VU1Interpreter::m_accReady) accReady;
    decltype(VU1Interpreter::m_vfLatestWrite) vfLatestWrite;
    decltype(VU1Interpreter::m_viLatestWrite) viLatestWrite;
    decltype(VU1Interpreter::m_accLatestWrite) accLatestWrite;
    uint64_t cycle, nextCommitCycle, nextFlagCommit, nextStoreCommit, nextVfCommit, nextViCommit, nextAccCommit;
    uint64_t nextWriteSequence, efuResourceReady;
    uint32_t workingClip, currentUpperInstruction;
    int32_t viBranchBackupValue;
    uint8_t viBranchBackupReg;
    bool viBranchBackupValid, stopRequested, pendingHaltD, pendingHaltT;
    // XGKICK bookkeeping (the packet buffer is refilled by every kick, so it is not saved).
    uint32_t xgSource, xgTotal, xgCopied, xgTagEnd, xgCredit;
    uint64_t xgIssue;
    bool xgActive, xgTagEop;
    std::vector<uint8_t> data;
};

void VU1Lifted::takeSnapshot(const VU1Interpreter &vu, const uint8_t *vuData, uint32_t dataSize, Snapshot &s)
{
    s.state = vu.m_state;
    s.flagPipeline = vu.m_flagPipeline;
    s.fdiv = vu.m_fdiv;
    s.efu = vu.m_efu;
    s.storePipeline = vu.m_storePipeline;
    s.vfWritePipeline = vu.m_vfWritePipeline;
    s.viWritePipeline = vu.m_viWritePipeline;
    s.accWritePipeline = vu.m_accWritePipeline;
    s.vfReady = vu.m_vfReady;
    s.viReady = vu.m_viReady;
    s.accReady = vu.m_accReady;
    s.vfLatestWrite = vu.m_vfLatestWrite;
    s.viLatestWrite = vu.m_viLatestWrite;
    s.accLatestWrite = vu.m_accLatestWrite;
    s.cycle = vu.m_cycle;
    s.nextCommitCycle = vu.m_nextCommitCycle;
    s.nextFlagCommit = vu.m_nextFlagCommit;
    s.nextStoreCommit = vu.m_nextStoreCommit;
    s.nextVfCommit = vu.m_nextVfCommit;
    s.nextViCommit = vu.m_nextViCommit;
    s.nextAccCommit = vu.m_nextAccCommit;
    s.nextWriteSequence = vu.m_nextWriteSequence;
    s.efuResourceReady = vu.m_efuResourceReady;
    s.workingClip = vu.m_workingClip;
    s.currentUpperInstruction = vu.m_currentUpperInstruction;
    s.viBranchBackupValue = vu.m_viBranchBackupValue;
    s.viBranchBackupReg = vu.m_viBranchBackupReg;
    s.viBranchBackupValid = vu.m_viBranchBackupValid;
    s.stopRequested = vu.m_stopRequested;
    s.pendingHaltD = vu.m_pendingHaltD;
    s.pendingHaltT = vu.m_pendingHaltT;
    s.xgSource = vu.m_xgkick.sourceAddress;
    s.xgTotal = vu.m_xgkick.totalBytes;
    s.xgCopied = vu.m_xgkick.copiedBytes;
    s.xgTagEnd = vu.m_xgkick.currentTagEnd;
    s.xgCredit = vu.m_xgkick.cycleCredit;
    s.xgIssue = vu.m_xgkick.issueCycle;
    s.xgActive = vu.m_xgkick.active;
    s.xgTagEop = vu.m_xgkick.currentTagEop;
    s.data.assign(vuData, vuData + dataSize);
}

void VU1Lifted::restoreSnapshot(VU1Interpreter &vu, uint8_t *vuData, uint32_t dataSize, const Snapshot &s)
{
    vu.m_state = s.state;
    vu.m_flagPipeline = s.flagPipeline;
    vu.m_fdiv = s.fdiv;
    vu.m_efu = s.efu;
    vu.m_storePipeline = s.storePipeline;
    vu.m_vfWritePipeline = s.vfWritePipeline;
    vu.m_viWritePipeline = s.viWritePipeline;
    vu.m_accWritePipeline = s.accWritePipeline;
    vu.m_vfReady = s.vfReady;
    vu.m_viReady = s.viReady;
    vu.m_accReady = s.accReady;
    vu.m_vfLatestWrite = s.vfLatestWrite;
    vu.m_viLatestWrite = s.viLatestWrite;
    vu.m_accLatestWrite = s.accLatestWrite;
    vu.m_cycle = s.cycle;
    vu.m_nextCommitCycle = s.nextCommitCycle;
    vu.m_nextFlagCommit = s.nextFlagCommit;
    vu.m_nextStoreCommit = s.nextStoreCommit;
    vu.m_nextVfCommit = s.nextVfCommit;
    vu.m_nextViCommit = s.nextViCommit;
    vu.m_nextAccCommit = s.nextAccCommit;
    vu.m_nextWriteSequence = s.nextWriteSequence;
    vu.m_efuResourceReady = s.efuResourceReady;
    vu.m_workingClip = s.workingClip;
    vu.m_currentUpperInstruction = s.currentUpperInstruction;
    vu.m_viBranchBackupValue = s.viBranchBackupValue;
    vu.m_viBranchBackupReg = s.viBranchBackupReg;
    vu.m_viBranchBackupValid = s.viBranchBackupValid;
    vu.m_stopRequested = s.stopRequested;
    vu.m_pendingHaltD = s.pendingHaltD;
    vu.m_pendingHaltT = s.pendingHaltT;
    vu.m_xgkick.sourceAddress = s.xgSource;
    vu.m_xgkick.totalBytes = s.xgTotal;
    vu.m_xgkick.copiedBytes = s.xgCopied;
    vu.m_xgkick.currentTagEnd = s.xgTagEnd;
    vu.m_xgkick.cycleCredit = s.xgCredit;
    vu.m_xgkick.issueCycle = s.xgIssue;
    vu.m_xgkick.active = s.xgActive;
    vu.m_xgkick.currentTagEop = s.xgTagEop;
    std::memcpy(vuData, s.data.data(), std::min<size_t>(dataSize, s.data.size()));
}

namespace
{
    using PacketList = std::vector<std::vector<uint8_t>>;

    void capturePacket(void *ctx, const uint8_t *data, uint32_t size)
    {
        static_cast<PacketList *>(ctx)->emplace_back(data, data + size);
    }

    uint32_t floatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
}

void VU1Lifted::verifyExecute(VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData,
                              uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    static Snapshot before, interp;
    static PacketList interpPackets, liftedPackets;
    const uint32_t entry = vu.m_state.pc;
    uint32_t patch = 0;
    std::memcpy(&patch, vuCode + 0x1B70u, 4);

    takeSnapshot(vu, vuData, dataSize, before);

    interpPackets.clear();
    vu.m_gifCaptureFn = capturePacket;
    vu.m_gifCaptureCtx = &interpPackets;
    vu.run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
    const bool interpOutOfBudget = vu.m_cycle >= before.cycle + maxCycles;
    takeSnapshot(vu, vuData, dataSize, interp);

    restoreSnapshot(vu, vuData, dataSize, before);
    liftedPackets.clear();
    vu.m_gifCaptureCtx = &liftedPackets;
    const int result = runLifted(vu, vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
    vu.m_gifCaptureFn = nullptr;
    vu.m_gifCaptureCtx = nullptr;

    ++g_stats.verified;
    if (interpOutOfBudget || result == kLiftYield)
    {
        ++g_stats.budgetSkips;
    }
    else
    {
        char what[512];
        int n = 0;
        const auto note = [&](const char *fmt, auto... args)
        {
            if (n < static_cast<int>(sizeof(what)) - 64)
                n += std::snprintf(what + n, sizeof(what) - static_cast<size_t>(n), fmt, args...);
        };
        const VU1State &a = interp.state;
        const VU1State &b = vu.m_state;
        for (int r = 1; r < 32; ++r)
            for (int c = 0; c < 4; ++c)
                if (floatBits(a.vf[r][c]) != floatBits(b.vf[r][c]))
                {
                    note(" vf%d.%c=%08x/%08x", r, "xyzw"[c], floatBits(a.vf[r][c]), floatBits(b.vf[r][c]));
                    break;
                }
        for (int r = 1; r < 16; ++r)
            if (a.vi[r] != b.vi[r])
                note(" vi%d=%d/%d", r, a.vi[r], b.vi[r]);
        for (int c = 0; c < 4; ++c)
            if (floatBits(a.acc[c]) != floatBits(b.acc[c]))
            {
                note(" acc.%c=%08x/%08x", "xyzw"[c], floatBits(a.acc[c]), floatBits(b.acc[c]));
                break;
            }
        if (floatBits(a.q) != floatBits(b.q))
            note(" q=%08x/%08x", floatBits(a.q), floatBits(b.q));
        if (floatBits(a.p) != floatBits(b.p))
            note(" p=%08x/%08x", floatBits(a.p), floatBits(b.p));
        if (floatBits(a.i) != floatBits(b.i))
            note(" i");
        if (a.r != b.r)
            note(" r");
        // With unobservable flags elided (kLiftedFlagsExact false) MAC/status are only meaningful
        // through what FMAND & co. read, which lands in the VI registers compared above.
        if (kLiftedFlagsExact && a.mac != b.mac)
            note(" mac=%04x/%04x", a.mac, b.mac);
        if (kLiftedFlagsExact && a.status != b.status)
            note(" status=%03x/%03x", a.status, b.status);
        if (a.clip != b.clip || interp.workingClip != vu.m_workingClip)
            note(" clip=%06x/%06x", a.clip, b.clip);
        if (a.pc != b.pc)
            note(" pc=%04x/%04x", a.pc, b.pc);
        if (std::memcmp(interp.data.data(), vuData, dataSize) != 0)
        {
            uint32_t at = 0;
            while (at < dataSize && interp.data[at] == vuData[at])
                ++at;
            note(" mem@%04x", at & ~15u);
        }
        if (interpPackets.size() != liftedPackets.size())
            note(" kicks=%zu/%zu", interpPackets.size(), liftedPackets.size());
        else
            for (size_t k = 0; k < interpPackets.size(); ++k)
                if (interpPackets[k] != liftedPackets[k])
                {
                    note(" kick%zu(%zu/%zu bytes)", k, interpPackets[k].size(), liftedPackets[k].size());
                    break;
                }
        if (n > 0)
        {
            ++g_stats.mismatches;
            if (g_stats.mismatches <= 40u)
                std::fprintf(stderr, "[vu1-verify] MISMATCH run %llu entry %04x patch %08x lift=%d (interp/lifted):%s\n",
                             static_cast<unsigned long long>(g_stats.verified), entry, patch, result, what);
        }
        if ((g_stats.verified & 0xFFFFu) == 0u)
            std::fprintf(stderr, "[vu1-verify] %llu runs verified, %llu mismatches (%llu handed off, %llu out of budget)\n",
                         static_cast<unsigned long long>(g_stats.verified),
                         static_cast<unsigned long long>(g_stats.mismatches),
                         static_cast<unsigned long long>(g_stats.handoffs),
                         static_cast<unsigned long long>(g_stats.budgetSkips));
    }

    // Keep the interpreter's result and hand its packets to the GIF.
    restoreSnapshot(vu, vuData, dataSize, interp);
    for (const auto &packet : interpPackets)
    {
        if (memory)
            memory->submitGifPacket(GifPathId::Path1, packet.data(), static_cast<uint32_t>(packet.size()));
        else
            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));
    }
}
