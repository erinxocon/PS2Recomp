#ifndef PS2_VU1_LIFTED_H
#define PS2_VU1_LIFTED_H

#include "runtime/ps2_vu1.h"

#include <cstdint>

class GS;
class PS2Memory;

// Gitaroo Man's VU1 microprogram compiled to C++ by tools/vu1lift (generated/ps2_vu1_lifted_gitaroo.inc).
// It runs instead of the interpreter whenever the loaded micro image is that program (any value of
// the in-stream patched branch at 0x1B70) and the MSCAL entry is a lifted block.
//   PS2X_VU1_LIFT=0    interpreter only
//   PS2X_VU1_VERIFY=1  lockstep: run the interpreter and the lifted code from the same state, compare
//                      registers, flags, VU1 data memory and XGKICK packets, keep the interpreter's result
struct VU1Lifted
{
    static bool tryExecute(VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize,
                           uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory,
                           uint32_t maxCycles);

    // ---- used by the generated code -----------------------------------------------------------
    enum : int
    {
        kLiftNotLifted = 0,
        kLiftEnd = 1,     // E bit: the program finished
        kLiftHandoff = 2, // continue in the interpreter at m_state.pc (unknown JR / patch target)
        kLiftYield = 3,   // cycle budget used up, like the interpreter's run() returning early
        kLiftContinue = 4 // runNative: continue the lifted program at m_state.pc
    };

    struct LiftedRun
    {
        uint8_t *mem;
        uint32_t dataSize;
        uint32_t dataMask;
        const uint8_t *code;
        uint64_t budgetEnd;
        GS *gs;
        PS2Memory *memory;
    };

    // The interpreter's m_viBranchBackup*: a branch reads the old value of a VI written by an
    // integer op in the pair immediately before it (cycle stamp = that pair's clock).
    struct BranchBackup
    {
        uint64_t cycle = ~0ull;
        int32_t value = 0;
        uint8_t reg = 0;
        void record(uint8_t r, int32_t old, uint64_t c)
        {
            reg = r;
            value = old;
            cycle = c;
        }
    };

    struct Snapshot;

    static int runProgram(VU1Interpreter &vu, LiftedRun &run, uint32_t startPc);
    // ps2_vu1_native.cpp: entry 0x1B68 in plain C++; kLiftEnd, kLiftContinue (the lifted program
    // takes over at m_state.pc), or kLiftNotLifted with nothing changed.
    static int runNative(VU1Interpreter &vu, LiftedRun &run);
    static int runLifted(VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData,
                         uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t maxCycles);
    static void verifyExecute(VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData,
                              uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t maxCycles);
    static void takeSnapshot(const VU1Interpreter &vu, const uint8_t *vuData, uint32_t dataSize, Snapshot &s);
    static void restoreSnapshot(VU1Interpreter &vu, uint8_t *vuData, uint32_t dataSize, const Snapshot &s);

    static void step(VU1Interpreter &vu);
    static void tick(VU1Interpreter &vu);
    static void advance(VU1Interpreter &vu, uint64_t target);
    static void waitFdiv(VU1Interpreter &vu);
    static void waitEfuResource(VU1Interpreter &vu);
    static void waitP(VU1Interpreter &vu);
    static int16_t bv(const int32_t *vi, const VU1Interpreter &vu, const BranchBackup &bk, uint8_t reg);
    static void xgkick(VU1Interpreter &vu, uint32_t qwordAddress);
    static int handoff(VU1Interpreter &vu, const BranchBackup &bk, uint32_t pc);
    static int yield(VU1Interpreter &vu, const BranchBackup &bk, uint32_t pc);
    static uint32_t dynamicBranchTarget(const LiftedRun &run, uint32_t pc);
    static uint32_t lowBits(float value);

    // Upper (FMAC) ops, inlined. Same arithmetic as VU1Interpreter::execUpper: operands normalized
    // (denormal -> 0, Inf/NaN -> +-FLT_MAX), round toward zero, results checked against the exact
    // value only when the float result is near the edges of the range. Flags is false where the
    // generator proved no FMAND/FMEQ/FMOR can observe this op's MAC/status.
    static float nrm(float value);
    // execLower's DIV/SQRT/RSQRT/ERLENG/ESQRT and execUpper's CLIP on the lifted register copies.
    static void opDiv(VU1Interpreter &vu, float numRaw, float denRaw);
    static void opSqrt(VU1Interpreter &vu, float valueRaw);
    static void opRsqrt(VU1Interpreter &vu, float numRaw, float radicandRaw);
    static void opErleng(VU1Interpreter &vu, const float *v);
    static void opEsqrt(VU1Interpreter &vu, float valueRaw);
    static void opClip(VU1Interpreter &vu, const float *fs, float ftw);
    static float ftoi(float value, float scale);
    // lvf/lacc: the lifted program's register copies, synced into the interpreter state before its
    // exact-result and flag functions read it.
    template <uint8_t Dest, bool Flags>
    static void fmacStore(VU1Interpreter &vu, uint32_t word, float *result, float *dst, const float (*lvf)[4], const float *lacc);
    static void fmacStoreSlow(VU1Interpreter &vu, uint32_t word, float *result, float *dst, const float (*lvf)[4], const float *lacc,
                              uint8_t dest, bool flags);
};

#endif
