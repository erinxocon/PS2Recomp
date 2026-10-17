// Fallback EE interpreter for guest code that has no recompiled function: code the game builds in
// RAM at runtime (Gitaroo Man's AiCube renderer assembles its vertex-array draw loops from
// instruction templates). It keeps the calling conventions of generated code: calls go through
// dispatchGuestBranch, jumps into recompiled code hand control back with ctx->pc set, and backward
// branches are scheduler checkpoints. Instruction semantics mirror the recompiler's translators.
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_set>

namespace
{
    constexpr uint32_t kOpSpecial = 0, kOpRegimm = 1, kOpJ = 2, kOpJal = 3, kOpBeq = 4, kOpBne = 5,
                       kOpBlez = 6, kOpBgtz = 7, kOpAddi = 8, kOpAddiu = 9, kOpSlti = 10, kOpSltiu = 11,
                       kOpAndi = 12, kOpOri = 13, kOpXori = 14, kOpLui = 15, kOpCop1 = 17, kOpCop2 = 18,
                       kOpBeql = 20, kOpBnel = 21, kOpBlezl = 22, kOpBgtzl = 23, kOpDaddi = 24, kOpDaddiu = 25,
                       kOpMmi = 28, kOpLq = 30, kOpSq = 31, kOpLb = 32, kOpLh = 33, kOpLw = 35, kOpLbu = 36,
                       kOpLhu = 37, kOpLwu = 39, kOpSb = 40, kOpSh = 41, kOpSw = 43, kOpCache = 47,
                       kOpLwc1 = 49, kOpPref = 51, kOpLqc2 = 54, kOpLd = 55, kOpSwc1 = 57, kOpSqc2 = 62,
                       kOpSd = 63;

    struct Fields
    {
        uint32_t raw, op, rs, rt, rd, sa, funct;
        int32_t simm;
        uint32_t imm;
    };

    Fields decode(uint32_t raw)
    {
        return Fields{raw, raw >> 26, (raw >> 21) & 31u, (raw >> 16) & 31u, (raw >> 11) & 31u, (raw >> 6) & 31u,
                      raw & 63u, static_cast<int16_t>(raw & 0xFFFFu), raw & 0xFFFFu};
    }

    bool isControlTransfer(const Fields &f)
    {
        switch (f.op)
        {
        case kOpSpecial:
            return f.funct == 8u || f.funct == 9u; // JR, JALR
        case kOpRegimm:
            return f.rt <= 3u || (f.rt >= 16u && f.rt <= 19u);
        case kOpJ:
        case kOpJal:
        case kOpBeq:
        case kOpBne:
        case kOpBlez:
        case kOpBgtz:
        case kOpBeql:
        case kOpBnel:
        case kOpBlezl:
        case kOpBgtzl:
            return true;
        default:
            return false;
        }
    }

    void reportUnhandled(uint32_t pc, uint32_t raw)
    {
        static std::mutex mutex;
        static std::unordered_set<uint32_t> reported;
        std::lock_guard<std::mutex> lock(mutex);
        if (reported.insert(pc).second && reported.size() <= 64u)
        {
            std::cerr << "[ee-interp] unhandled instruction 0x" << std::hex << raw << " at pc=0x" << pc << std::dec
                      << " (treated as nop)" << std::endl;
        }
    }

    bool executeMmi(R5900Context *ctx, const Fields &f)
    {
        const uint32_t rs = f.rs, rt = f.rt, rd = f.rd;
        switch (f.funct)
        {
        case 9u: // MMI2
            switch (f.sa)
            {
            case 14u: // PCPYLD
                SET_GPR_VEC(ctx, rd, _mm_unpacklo_epi64(GPR_VEC(ctx, rt), GPR_VEC(ctx, rs)));
                return true;
            case 18u: // PAND
                SET_GPR_VEC(ctx, rd, PS2_PAND(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt)));
                return true;
            case 19u: // PXOR
                SET_GPR_VEC(ctx, rd, PS2_PXOR(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt)));
                return true;
            default:
                return false;
            }
        case 41u: // MMI3
            switch (f.sa)
            {
            case 14u: // PCPYUD
                SET_GPR_VEC(ctx, rd, _mm_unpackhi_epi64(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt)));
                return true;
            case 18u: // POR
                SET_GPR_VEC(ctx, rd, PS2_POR(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt)));
                return true;
            case 19u: // PNOR
                SET_GPR_VEC(ctx, rd, PS2_PNOR(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt)));
                return true;
            default:
                return false;
            }
        case 40u: // MMI1
            if (f.sa == 27u) // QFSRV
            {
                SET_GPR_VEC(ctx, rd, PS2_QFSRV(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt), ctx->sa & 0x7F));
                return true;
            }
            return false;
        case 8u: // MMI0
            switch (f.sa)
            {
            case 0u: // PADDW
                SET_GPR_VEC(ctx, rd, _mm_add_epi32(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt)));
                return true;
            case 1u: // PSUBW
                SET_GPR_VEC(ctx, rd, _mm_sub_epi32(GPR_VEC(ctx, rs), GPR_VEC(ctx, rt)));
                return true;
            case 18u: // PEXTLW
                SET_GPR_VEC(ctx, rd, _mm_unpacklo_epi32(GPR_VEC(ctx, rt), GPR_VEC(ctx, rs)));
                return true;
            default:
                return false;
            }
        default:
            return false;
        }
    }

    bool executeSpecial(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t pc, const Fields &f)
    {
        const uint32_t rs = f.rs, rt = f.rt, rd = f.rd, sa = f.sa;
        switch (f.funct)
        {
        case 0u: // SLL
            if (rd != 0u)
                SET_GPR_S32(ctx, rd, (int32_t)SLL32(GPR_U32(ctx, rt), sa));
            return true;
        case 2u: // SRL
            SET_GPR_S32(ctx, rd, (int32_t)SRL32(GPR_U32(ctx, rt), sa));
            return true;
        case 3u: // SRA
            SET_GPR_S32(ctx, rd, SRA32(GPR_S32(ctx, rt), sa));
            return true;
        case 4u: // SLLV
            SET_GPR_S32(ctx, rd, (int32_t)SLL32(GPR_U32(ctx, rt), GPR_U32(ctx, rs) & 0x1F));
            return true;
        case 6u: // SRLV
            SET_GPR_S32(ctx, rd, (int32_t)SRL32(GPR_U32(ctx, rt), GPR_U32(ctx, rs) & 0x1F));
            return true;
        case 7u: // SRAV
            SET_GPR_S32(ctx, rd, SRA32(GPR_S32(ctx, rt), GPR_U32(ctx, rs) & 0x1F));
            return true;
        case 10u: // MOVZ
            if (GPR_U64(ctx, rt) == 0)
                SET_GPR_VEC(ctx, rd, GPR_VEC(ctx, rs));
            return true;
        case 11u: // MOVN
            if (GPR_U64(ctx, rt) != 0)
                SET_GPR_VEC(ctx, rd, GPR_VEC(ctx, rs));
            return true;
        case 12u: // SYSCALL
            ctx->pc = pc + 4u;
            runtime->handleSyscall(rdram, ctx, (f.raw >> 6) & 0xFFFFFu);
            return true;
        case 15u: // SYNC
            return true;
        case 16u: // MFHI
            SET_GPR_U64(ctx, rd, ctx->hi);
            return true;
        case 17u: // MTHI
            ctx->hi = GPR_U64(ctx, rs);
            return true;
        case 18u: // MFLO
            SET_GPR_U64(ctx, rd, ctx->lo);
            return true;
        case 19u: // MTLO
            ctx->lo = GPR_U64(ctx, rs);
            return true;
        case 20u: // DSLLV
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) << (GPR_U32(ctx, rs) & 0x3F));
            return true;
        case 22u: // DSRLV
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) >> (GPR_U32(ctx, rs) & 0x3F));
            return true;
        case 23u: // DSRAV
            SET_GPR_S64(ctx, rd, GPR_S64(ctx, rt) >> (GPR_U32(ctx, rs) & 0x3F));
            return true;
        case 24u: // MULT (R5900 also writes rd)
        {
            const int64_t result = (int64_t)GPR_S32(ctx, rs) * (int64_t)GPR_S32(ctx, rt);
            ctx->lo = (uint64_t)(int64_t)(int32_t)result;
            ctx->hi = (uint64_t)(int64_t)(int32_t)(result >> 32);
            if (rd != 0u)
                SET_GPR_S32(ctx, rd, (int32_t)result);
            return true;
        }
        case 25u: // MULTU
        {
            const uint64_t result = (uint64_t)GPR_U32(ctx, rs) * (uint64_t)GPR_U32(ctx, rt);
            ctx->lo = (uint64_t)(int64_t)(int32_t)result;
            ctx->hi = (uint64_t)(int64_t)(int32_t)(result >> 32);
            if (rd != 0u)
                SET_GPR_S32(ctx, rd, (int32_t)result);
            return true;
        }
        case 26u: // DIV
        {
            const int32_t divisor = GPR_S32(ctx, rt);
            const int32_t dividend = GPR_S32(ctx, rs);
            if (divisor != 0)
            {
                if (divisor == -1 && dividend == INT32_MIN)
                {
                    ctx->lo = (uint64_t)(int64_t)INT32_MIN;
                    ctx->hi = 0;
                }
                else
                {
                    ctx->lo = (uint64_t)(int64_t)(dividend / divisor);
                    ctx->hi = (uint64_t)(int64_t)(dividend % divisor);
                }
            }
            else
            {
                ctx->lo = (dividend < 0) ? 1ull : 0xFFFFFFFFFFFFFFFFull;
                ctx->hi = (uint64_t)(int64_t)dividend;
            }
            return true;
        }
        case 27u: // DIVU
        {
            const uint32_t divisor = GPR_U32(ctx, rt);
            if (divisor != 0)
            {
                ctx->lo = (uint64_t)(int64_t)(int32_t)(GPR_U32(ctx, rs) / divisor);
                ctx->hi = (uint64_t)(int64_t)(int32_t)(GPR_U32(ctx, rs) % divisor);
            }
            else
            {
                ctx->lo = 0xFFFFFFFFFFFFFFFFull;
                ctx->hi = (uint64_t)(int64_t)(int32_t)GPR_U32(ctx, rs);
            }
            return true;
        }
        case 32u: // ADD
        {
            const int64_t result = (int64_t)GPR_S32(ctx, rs) + (int64_t)GPR_S32(ctx, rt);
            if (result > INT32_MAX || result < INT32_MIN)
                runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW);
            else
                SET_GPR_S32(ctx, rd, (int32_t)result);
            return true;
        }
        case 33u: // ADDU
            SET_GPR_S32(ctx, rd, (int32_t)ADD32(GPR_U32(ctx, rs), GPR_U32(ctx, rt)));
            return true;
        case 34u: // SUB
        {
            uint32_t tmp;
            bool ov;
            SUB32_OV(GPR_U32(ctx, rs), GPR_U32(ctx, rt), tmp, ov);
            if (ov)
                runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW);
            else
                SET_GPR_S32(ctx, rd, (int32_t)tmp);
            return true;
        }
        case 35u: // SUBU
            SET_GPR_S32(ctx, rd, (int32_t)SUB32(GPR_U32(ctx, rs), GPR_U32(ctx, rt)));
            return true;
        case 36u: // AND
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) & GPR_U64(ctx, rt));
            return true;
        case 37u: // OR
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) | GPR_U64(ctx, rt));
            return true;
        case 38u: // XOR
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) ^ GPR_U64(ctx, rt));
            return true;
        case 39u: // NOR
            SET_GPR_U64(ctx, rd, ~(GPR_U64(ctx, rs) | GPR_U64(ctx, rt)));
            return true;
        case 40u: // MFSA
            SET_GPR_U32(ctx, rd, ctx->sa);
            return true;
        case 41u: // MTSA
            ctx->sa = GPR_U32(ctx, rs) & 0x7F;
            return true;
        case 42u: // SLT
            SET_GPR_U64(ctx, rd, ((int64_t)GPR_S64(ctx, rs) < (int64_t)GPR_S64(ctx, rt)) ? 1 : 0);
            return true;
        case 43u: // SLTU
            SET_GPR_U64(ctx, rd, ((uint64_t)GPR_U64(ctx, rs) < (uint64_t)GPR_U64(ctx, rt)) ? 1 : 0);
            return true;
        case 44u: // DADD
        {
            const int64_t a = (int64_t)GPR_S64(ctx, rs);
            const int64_t b = (int64_t)GPR_S64(ctx, rt);
            const int64_t r = (int64_t)((uint64_t)a + (uint64_t)b);
            if (((a ^ b) >= 0) && ((a ^ r) < 0))
                runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW);
            else
                SET_GPR_S64(ctx, rd, r);
            return true;
        }
        case 45u: // DADDU
            SET_GPR_U64(ctx, rd, (uint64_t)GPR_U64(ctx, rs) + (uint64_t)GPR_U64(ctx, rt));
            return true;
        case 46u: // DSUB
        {
            const int64_t a = (int64_t)GPR_S64(ctx, rs);
            const int64_t b = (int64_t)GPR_S64(ctx, rt);
            const int64_t r = (int64_t)((uint64_t)a - (uint64_t)b);
            if (((a ^ b) < 0) && ((a ^ r) < 0))
                runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW);
            else
                SET_GPR_S64(ctx, rd, r);
            return true;
        }
        case 47u: // DSUBU
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rs) - GPR_U64(ctx, rt));
            return true;
        case 56u: // DSLL
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) << sa);
            return true;
        case 58u: // DSRL
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) >> sa);
            return true;
        case 59u: // DSRA
            SET_GPR_S64(ctx, rd, GPR_S64(ctx, rt) >> sa);
            return true;
        case 60u: // DSLL32
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) << (32 + sa));
            return true;
        case 62u: // DSRL32
            SET_GPR_U64(ctx, rd, GPR_U64(ctx, rt) >> (32 + sa));
            return true;
        case 63u: // DSRA32
            SET_GPR_S64(ctx, rd, GPR_S64(ctx, rt) >> (32 + sa));
            return true;
        default:
            return false;
        }
    }

    // Executes one instruction that is not a branch or jump.
    void executeSimple(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t pc, const Fields &f)
    {
        const uint32_t rs = f.rs, rt = f.rt;
        const int32_t simm = f.simm;
        bool handled = true;
        switch (f.op)
        {
        case kOpSpecial:
            handled = executeSpecial(rdram, ctx, runtime, pc, f);
            break;
        case kOpRegimm:
            if (f.rt == 24u) // MTSAB
                ctx->sa = ((GPR_U32(ctx, rs) ^ (uint32_t)simm) & 0xF) << 3;
            else if (f.rt == 25u) // MTSAH
                ctx->sa = ((GPR_U32(ctx, rs) ^ (uint32_t)simm) & 0x7) << 4;
            else
                handled = false;
            break;
        case kOpAddi:
            if (rt != 0u)
            {
                uint32_t tmp;
                bool ov;
                ADD32_OV(GPR_U32(ctx, rs), (int32_t)simm, tmp, ov);
                if (ov)
                    runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW);
                else
                    SET_GPR_S32(ctx, rt, (int32_t)tmp);
            }
            break;
        case kOpAddiu:
            if (rt != 0u)
                SET_GPR_S32(ctx, rt, (int32_t)ADD32(GPR_U32(ctx, rs), simm));
            break;
        case kOpSlti:
            SET_GPR_U64(ctx, rt, ((int64_t)GPR_S64(ctx, rs) < (int64_t)simm) ? 1 : 0);
            break;
        case kOpSltiu:
            SET_GPR_U64(ctx, rt, ((uint64_t)GPR_U64(ctx, rs) < (uint64_t)(int64_t)simm) ? 1 : 0);
            break;
        case kOpAndi:
            SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) & (uint64_t)f.imm);
            break;
        case kOpOri:
            SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) | (uint64_t)f.imm);
            break;
        case kOpXori:
            SET_GPR_U64(ctx, rt, GPR_U64(ctx, rs) ^ (uint64_t)f.imm);
            break;
        case kOpLui:
            SET_GPR_S32(ctx, rt, (int32_t)(f.imm << 16));
            break;
        case kOpDaddi:
        {
            const int64_t src = (int64_t)GPR_S64(ctx, rs);
            const int64_t res = (int64_t)((uint64_t)src + (uint64_t)(int64_t)simm);
            if (((src ^ simm) >= 0) && ((src ^ res) < 0))
                runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW);
            else
                SET_GPR_S64(ctx, rt, res);
            break;
        }
        case kOpDaddiu:
            SET_GPR_S64(ctx, rt, (int64_t)GPR_S64(ctx, rs) + (int64_t)simm);
            break;
        case kOpMmi:
            handled = executeMmi(ctx, f);
            break;
        case kOpLq:
            SET_GPR_VEC(ctx, rt, READ128(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpSq:
            WRITE128(ADD32(GPR_U32(ctx, rs), simm), GPR_VEC(ctx, rt));
            break;
        case kOpLb:
            SET_GPR_S32(ctx, rt, (int8_t)READ8(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpLh:
            SET_GPR_S32(ctx, rt, (int16_t)READ16(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpLw:
            SET_GPR_S32(ctx, rt, (int32_t)READ32(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpLbu:
            SET_GPR_ZE32(ctx, rt, (uint8_t)READ8(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpLhu:
            SET_GPR_ZE32(ctx, rt, (uint16_t)READ16(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpLwu:
            SET_GPR_ZE32(ctx, rt, READ32(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpLd:
            SET_GPR_U64(ctx, rt, READ64(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpSb:
            WRITE8(ADD32(GPR_U32(ctx, rs), simm), (uint8_t)GPR_U32(ctx, rt));
            break;
        case kOpSh:
            WRITE16(ADD32(GPR_U32(ctx, rs), simm), (uint16_t)GPR_U32(ctx, rt));
            break;
        case kOpSw:
            WRITE32(ADD32(GPR_U32(ctx, rs), simm), GPR_U32(ctx, rt));
            break;
        case kOpSd:
            WRITE64(ADD32(GPR_U32(ctx, rs), simm), GPR_U64(ctx, rt));
            break;
        case kOpLwc1:
        {
            const uint32_t bits = READ32(ADD32(GPR_U32(ctx, rs), simm));
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            ctx->f[rt] = value;
            break;
        }
        case kOpSwc1:
        {
            const float value = ctx->f[rt];
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            WRITE32(ADD32(GPR_U32(ctx, rs), simm), bits);
            break;
        }
        case kOpLqc2:
            ctx->vu0_vf[rt] = _mm_castsi128_ps(READ128(ADD32(GPR_U32(ctx, rs), simm)));
            break;
        case kOpSqc2:
            WRITE128(ADD32(GPR_U32(ctx, rs), simm), _mm_castps_si128(ctx->vu0_vf[rt]));
            break;
        case kOpCache:
        case kOpPref:
            break;
        default:
            handled = false;
            break;
        }
        if (!handled)
            reportUnhandled(pc, f.raw);
    }

    void noteEntry(uint32_t pc)
    {
        static std::mutex mutex;
        static std::unordered_set<uint32_t> seen;
        std::lock_guard<std::mutex> lock(mutex);
        if (seen.insert(pc).second && seen.size() <= 16u)
            std::cerr << "[ee-interp] interpreting runtime-generated code at 0x" << std::hex << pc << std::dec << std::endl;
    }
}

bool PS2Runtime::canInterpretGuestCode(uint32_t address) const
{
    static const bool s_disabled = std::getenv("PS2X_NO_EE_INTERP") != nullptr;
    if (s_disabled || address == 0u || (address & 3u) != 0u)
        return false;
    // Only plain RAM outside the recompiled code ranges: a gap inside the executable is a recompiler
    // problem to fix, not something to paper over.
    const uint32_t physical = address & 0x1FFFFFFFu;
    return physical >= 0x80000u && physical < PS2_RAM_SIZE && !m_memory.isCodeAddress(address);
}

void PS2Runtime::interpretGuestCode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
{
    noteEntry(ctx->pc);
    for (;;)
    {
        const uint32_t pc = ctx->pc;
        const Fields f = decode(FAST_READ32(pc));
        if (!isControlTransfer(f))
        {
            executeSimple(rdram, ctx, runtime, pc, f);
            if (ctx->pc == pc || ctx->pc == pc + 4u)
                ctx->pc = pc + 4u;
            else
                return; // an exception or syscall redirected the pc
            continue;
        }

        // Branch condition and jump register are read before the delay slot runs.
        const uint32_t delayPc = pc + 4u;
        const uint32_t fallthroughPc = pc + 8u;
        const uint32_t branchTarget = delayPc + (static_cast<uint32_t>(f.simm) << 2);
        bool taken = false;
        bool likely = false;
        bool link = false;
        bool registerJump = false;
        uint32_t target = branchTarget;
        uint32_t linkReg = 31u;
        switch (f.op)
        {
        case kOpSpecial:
            registerJump = true;
            target = GPR_U32(ctx, f.rs);
            taken = true;
            link = (f.funct == 9u);
            linkReg = f.rd;
            break;
        case kOpRegimm:
            likely = (f.rt & 2u) != 0u;
            link = f.rt >= 16u;
            taken = ((f.rt & 1u) != 0u) ? (GPR_S64(ctx, f.rs) >= 0) : (GPR_S64(ctx, f.rs) < 0);
            break;
        case kOpJ:
        case kOpJal:
            target = (delayPc & 0xF0000000u) | ((f.raw & 0x03FFFFFFu) << 2);
            taken = true;
            link = (f.op == kOpJal);
            break;
        case kOpBeq:
        case kOpBeql:
            taken = GPR_U64(ctx, f.rs) == GPR_U64(ctx, f.rt);
            likely = f.op == kOpBeql;
            break;
        case kOpBne:
        case kOpBnel:
            taken = GPR_U64(ctx, f.rs) != GPR_U64(ctx, f.rt);
            likely = f.op == kOpBnel;
            break;
        case kOpBlez:
        case kOpBlezl:
            taken = GPR_S64(ctx, f.rs) <= 0;
            likely = f.op == kOpBlezl;
            break;
        case kOpBgtz:
        case kOpBgtzl:
            taken = GPR_S64(ctx, f.rs) > 0;
            likely = f.op == kOpBgtzl;
            break;
        default:
            break;
        }

        if (link && linkReg != 0u)
            SET_GPR_U32(ctx, linkReg, fallthroughPc);

        if (!taken && likely)
        {
            ctx->pc = fallthroughPc;
            continue;
        }

        ctx->pc = delayPc;
        const Fields delay = decode(FAST_READ32(delayPc));
        if (isControlTransfer(delay))
            reportUnhandled(delayPc, delay.raw);
        else
            executeSimple(rdram, ctx, runtime, delayPc, delay);

        if (!taken)
        {
            ctx->pc = fallthroughPc;
            continue;
        }

        if (link)
        {
            const auto kind = registerJump ? GuestBranchKind::IndirectCall : GuestBranchKind::DirectCall;
            if (!runtime->dispatchGuestBranch(rdram, ctx, target, pc, fallthroughPc, kind, "ee-interp call"))
                return;
            ctx->pc = fallthroughPc;
            continue;
        }

        ctx->pc = target;
        if (!runtime->canInterpretGuestCode(target))
        {
            // Leaving for recompiled code (a return lands mid-function, so hand the pc back to the
            // caller); a direct jump to a function entry is a tail call, as in generated code.
            if (!registerJump && runtime->hasFunction(target))
                runtime->lookupFunction(target)(rdram, ctx, runtime);
            return;
        }
        if (target <= pc && runtime->eeCheckpointDue())
            return;
    }
}
