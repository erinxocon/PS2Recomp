#include "Common.h"
#include "Audio.h"
#include "../../ps2_iop_transport.h"

namespace ps2_stubs
{
    namespace
    {
        constexpr uint32_t kLibSdCmdSetParam = 0x8010u;
        constexpr uint32_t kLibSdCmdVoiceTrans = 0x80D0u;
        constexpr uint32_t kLibSdCmdBlockTrans = 0x80E0u;
        constexpr uint32_t kLibSdCmdVoiceTransStatus = 0x80F0u;
        constexpr uint32_t kLibSdCmdBlockTransStatus = 0x8100u;
        constexpr uint32_t kAudioPositionMask = 0x00FFFFFFu;
        constexpr uint32_t kAudioTransferUnit = 1024u;
        constexpr uint32_t kLibSdCoreCount = 2u;
        constexpr uint32_t kLibSdTransModeIo = 0x08u;
        constexpr uint32_t kLibSdTransDirectionMask = 0x03u;
        constexpr uint32_t kLibSdTransStop = 0x02u;
        constexpr uint32_t kLibSdTransLoop = 0x10u;

        struct VoiceTransferState
        {
            uint32_t sourceAddress = 0u;
            uint32_t destinationAddress = 0u;
            uint32_t size = 0u;
            uint16_t mode = 0u;
            bool completed = true;
        };

        struct BlockTransferState
        {
            uint32_t base = 0u;
            uint32_t size = 0u;
            uint32_t pauseBase = 0u;
            uint32_t offset = 0u;
            uint32_t statusTraceCount = 0u;
            uint16_t mode = 0u;
            bool active = false;
            bool loop = false;
        };

        struct AudioStubState
        {
            bool initialized = false;
            std::array<VoiceTransferState, kLibSdCoreCount> voiceTransfers{};
            std::array<BlockTransferState, kLibSdCoreCount> blockTransfers{};
        };

        std::mutex g_audio_stub_mutex;
        AudioStubState g_audio_stub_state;

        void resetAudioStubStateUnlocked()
        {
            g_audio_stub_state = {};
        }

        // libsdr (EE) <-> SDRDRV (IOP) remote sound RPC. When the SDRDRV IRX runs on the
        // emulated IOP, sceSdRemote is forwarded exactly like the SDK client does it:
        // sceSifCallRpc(&cd, command, 0, sbuf[16], 64, rbuf, 16, 0, 0) with
        // sbuf = {command, args...}, returning rbuf[0].
        constexpr uint32_t kSdrRemoteSid = 0x80000701u;
        constexpr uint32_t kSdrClientOffset = 0x00u;
        constexpr uint32_t kSdrSendOffset = 0x40u;
        constexpr uint32_t kSdrRecvOffset = 0x80u;
        constexpr uint32_t kSdrBlockSize = 0xC0u;
        uint32_t g_sdrRemoteBlock = 0u;
        bool g_sdrRemoteBound = false;

        bool sdrRemoteAvailable(PS2Runtime *runtime)
        {
            return runtime && PS2IopTransport::canBindRpc(runtime, kSdrRemoteSid);
        }

        bool sdrRemoteEnsureBound(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
        {
            if (g_sdrRemoteBound)
                return true;
            if (!sdrRemoteAvailable(runtime))
                return false;
            if (g_sdrRemoteBlock == 0u)
            {
                g_sdrRemoteBlock = runtime->guestMalloc(kSdrBlockSize, 64u);
                if (g_sdrRemoteBlock == 0u)
                    return false;
            }
            uint8_t *block = getMemPtr(rdram, g_sdrRemoteBlock);
            if (!block)
                return false;
            std::memset(block, 0, kSdrBlockSize);

            const R5900Context saved = *ctx;
            setReturnU32(ctx, 0u);
            SET_GPR_U32(ctx, 4, g_sdrRemoteBlock + kSdrClientOffset);
            SET_GPR_U32(ctx, 5, kSdrRemoteSid);
            SET_GPR_U32(ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(rdram, ctx, runtime);
            *ctx = saved;

            uint32_t server = 0u;
            std::memcpy(&server, block + kSdrClientOffset + 36u /* t_SifRpcClientData::server */, sizeof(server));
            g_sdrRemoteBound = server != 0u;
            return g_sdrRemoteBound;
        }

        bool sdrRemoteCall(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
                           uint32_t command, const uint32_t (&args)[6], uint32_t &result)
        {
            if (!sdrRemoteEnsureBound(rdram, ctx, runtime))
                return false;
            uint8_t *block = getMemPtr(rdram, g_sdrRemoteBlock);
            if (!block)
                return false;
            uint32_t send[16]{};
            send[0] = command;
            for (size_t i = 0; i < 6u; ++i)
                send[i + 1u] = args[i];
            std::memcpy(block + kSdrSendOffset, send, sizeof(send));
            std::memset(block + kSdrRecvOffset, 0, 16u);

            const R5900Context saved = *ctx;
            // Give SifCallRpc a private frame so its stack-argument probing
            // (end_param at 0(sp)) reads zeros rather than caller data.
            const uint32_t frame = (getRegU32(ctx, 29) - 0x80u) & ~0xFu;
            uint8_t *framePtr = getMemPtr(rdram, frame);
            uint8_t savedFrame[0x40]{};
            if (framePtr)
            {
                std::memcpy(savedFrame, framePtr, sizeof(savedFrame));
                std::memset(framePtr, 0, sizeof(savedFrame));
            }
            SET_GPR_U32(ctx, 29, frame);
            SET_GPR_U32(ctx, 4, g_sdrRemoteBlock + kSdrClientOffset);
            SET_GPR_U32(ctx, 5, command);
            SET_GPR_U32(ctx, 6, 0u);
            SET_GPR_U32(ctx, 7, g_sdrRemoteBlock + kSdrSendOffset);
            SET_GPR_U32(ctx, 8, 64u);
            SET_GPR_U32(ctx, 9, g_sdrRemoteBlock + kSdrRecvOffset);
            SET_GPR_U32(ctx, 10, 16u);
            SET_GPR_U32(ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(rdram, ctx, runtime);
            *ctx = saved;
            if (framePtr)
                std::memcpy(framePtr, savedFrame, sizeof(savedFrame));

            std::memcpy(&result, block + kSdrRecvOffset, sizeof(result));
            return true;
        }

        uint32_t currentBlockStatus(const BlockTransferState &transfer)
        {
            const uint32_t position = (transfer.base + transfer.offset) & kAudioPositionMask;
            const uint32_t halfSize = transfer.size / 2u;
            const uint32_t bank = (transfer.loop && halfSize != 0u && transfer.offset >= halfSize) ? 1u : 0u;
            return (bank << 24u) | position;
        }
    }

    void resetAudioStubState()
    {
        std::lock_guard<std::mutex> lock(g_audio_stub_mutex);
        resetAudioStubStateUnlocked();
    }

    void sceSdCallBack(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSdCallBack", rdram, ctx, runtime);
    }

    void sceSdRemote(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cmd = getRegU32(ctx, 5);
        const uint32_t cmdArg0 = getRegU32(ctx, 6);
        const uint32_t cmdArg1 = getRegU32(ctx, 7);

        {
            // EE n32 varargs: sceSdRemote(arg, command, a2, a3, t0, t1, t2, t3).
            const uint32_t args[6] = {
                cmdArg0, cmdArg1, getRegU32(ctx, 8), getRegU32(ctx, 9), getRegU32(ctx, 10), getRegU32(ctx, 11)};
            uint32_t remoteResult = 0u;
            if (sdrRemoteCall(rdram, ctx, runtime, cmd, args, remoteResult))
            {
                setReturnU32(ctx, remoteResult);
                return;
            }
        }

        const uint32_t sp = getRegU32(ctx, 29);
        const uint32_t arg4Reg = getRegU32(ctx, 8);
        const uint32_t arg5Reg = getRegU32(ctx, 9);
        const uint32_t arg6Reg = getRegU32(ctx, 10);
        const uint32_t arg4Stk = FAST_READ32(sp + 0x10u);
        const uint32_t arg5Stk = FAST_READ32(sp + 0x14u);
        const uint32_t arg6Stk = FAST_READ32(sp + 0x18u);
        const uint32_t arg4 = (arg4Reg != 0u) ? arg4Reg : arg4Stk;
        const uint32_t arg5 = (arg5Reg != 0u) ? arg5Reg : arg5Stk;
        const uint32_t arg6 = (arg6Reg != 0u) ? arg6Reg : arg6Stk;

        std::lock_guard<std::mutex> lock(g_audio_stub_mutex);
        g_audio_stub_state.initialized = true;
        const uint32_t core = cmdArg0 & (kLibSdCoreCount - 1u);
        VoiceTransferState &voiceTransfer = g_audio_stub_state.voiceTransfers[core];
        BlockTransferState &blockTransfer = g_audio_stub_state.blockTransfers[core];
        uint32_t returnValue = 0u;

        if (cmd == kLibSdCmdVoiceTrans)
        {
            voiceTransfer.sourceAddress = arg4;
            voiceTransfer.destinationAddress = arg5;
            voiceTransfer.size = arg6;
            voiceTransfer.mode = static_cast<uint16_t>(cmdArg1);
            voiceTransfer.completed = true;

            // sceSdVoiceTrans returns the transferred byte count for DMA
            // mode and zero for its synchronous programmed-I/O mode.
            returnValue = ((voiceTransfer.mode & kLibSdTransModeIo) != 0u)
                              ? 0u
                              : voiceTransfer.size;

            PS2_IF_AGRESSIVE_LOGS({
                std::cerr << "[Audio:VoiceTrans] core=" << core
                          << " src=0x" << std::hex << voiceTransfer.sourceAddress
                          << " dest=0x" << voiceTransfer.destinationAddress
                          << " size=0x" << voiceTransfer.size
                          << " mode=0x" << voiceTransfer.mode
                          << std::dec << std::endl;
            });
        }
        else if (cmd == kLibSdCmdBlockTrans)
        {
            const uint32_t direction = cmdArg1 & kLibSdTransDirectionMask;
            if (direction == kLibSdTransStop)
            {
                returnValue = currentBlockStatus(blockTransfer);
                blockTransfer = {};
            }
            else if (arg4 != 0u && arg5 != 0u)
            {
                blockTransfer.base = arg4 & kAudioPositionMask;
                blockTransfer.size = arg5;
                blockTransfer.pauseBase =
                    ((arg6 != 0u) ? arg6 : arg4) & kAudioPositionMask;
                blockTransfer.mode = static_cast<uint16_t>(cmdArg1);
                blockTransfer.loop = (cmdArg1 & kLibSdTransLoop) != 0u;

                const uint32_t pauseOffset =
                    (blockTransfer.pauseBase - blockTransfer.base) & kAudioPositionMask;
                blockTransfer.offset = (pauseOffset < blockTransfer.size) ? pauseOffset : 0u;
                blockTransfer.statusTraceCount = 0u;
                blockTransfer.active = true;
                returnValue = 0u;
            }
            else
            {
                returnValue = std::numeric_limits<uint32_t>::max();
            }

            PS2_IF_AGRESSIVE_LOGS({
                std::cerr << "[Audio:BlockTrans] core=" << core
                          << " active=" << blockTransfer.active
                          << " base=0x" << std::hex << blockTransfer.base
                          << " size=0x" << blockTransfer.size
                          << " pause=0x" << blockTransfer.pauseBase
                          << " offset=0x" << blockTransfer.offset
                          << std::dec << std::endl;
            });
        }
        else if (cmd == kLibSdCmdVoiceTransStatus)
        {
            returnValue = voiceTransfer.completed ? 1u : 0u;
        }
        else if (cmd == kLibSdCmdBlockTransStatus)
        {
            if (blockTransfer.active && blockTransfer.size != 0u)
            {
                blockTransfer.offset = (blockTransfer.offset + kAudioTransferUnit) % blockTransfer.size;
                if (blockTransfer.statusTraceCount < 32u)
                {
                    PS2_IF_AGRESSIVE_LOGS({
                        std::cerr << "[Audio:BlockStatus] core=" << core
                                  << " status=0x" << std::hex << currentBlockStatus(blockTransfer)
                                  << " offset=0x" << blockTransfer.offset
                                  << " size=0x" << blockTransfer.size
                                  << std::dec << std::endl;
                    });
                    ++blockTransfer.statusTraceCount;
                }
            }
            returnValue = currentBlockStatus(blockTransfer);
        }
        else if (cmd == kLibSdCmdSetParam)
        {
            (void)cmdArg0;
            (void)cmdArg1;
            returnValue = 0u;
        }

        setReturnU32(ctx, returnValue);
    }

    void sceSdRemoteInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        std::lock_guard<std::mutex> lock(g_audio_stub_mutex);
        resetAudioStubStateUnlocked();
        g_audio_stub_state.initialized = true;
        g_sdrRemoteBound = false;
        (void)sdrRemoteEnsureBound(rdram, ctx, runtime);
        setReturnS32(ctx, 0);
    }

    void sceSdTransToIOP(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSdTransToIOP", rdram, ctx, runtime);
    }

    void sceSSyn_BreakAtick(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_BreakAtick", rdram, ctx, runtime);
    }

    void sceSSyn_ClearBreakAtick(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_ClearBreakAtick", rdram, ctx, runtime);
    }

    void sceSSyn_SendExcMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendExcMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SendNrpnMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendNrpnMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SendRpnMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendRpnMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SendShortMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendShortMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SetChPriority(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetChPriority", rdram, ctx, runtime);
    }

    void sceSSyn_SetMasterVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetMasterVolume", rdram, ctx, runtime);
    }

    void sceSSyn_SetOutPortVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetOutPortVolume", rdram, ctx, runtime);
    }

    void sceSSyn_SetOutputAssign(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetOutputAssign", rdram, ctx, runtime);
    }

    void sceSSyn_SetOutputMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSSyn_SetPortMaxPoly(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetPortMaxPoly", rdram, ctx, runtime);
    }

    void sceSSyn_SetPortVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetPortVolume", rdram, ctx, runtime);
    }

    void sceSSyn_SetTvaEnvMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetTvaEnvMode", rdram, ctx, runtime);
    }

    void sceSynthesizerAmpProcI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAmpProcI", rdram, ctx, runtime);
    }

    void sceSynthesizerAmpProcNI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAmpProcNI", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignAllNoteOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignAllNoteOff", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignAllSoundOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignAllSoundOff", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignHoldChange(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignHoldChange", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignNoteOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignNoteOff", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignNoteOn(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignNoteOn", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcPortamentPitch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcPortamentPitch", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcTvfCoefAll(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcTvfCoefAll", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcTvfCoefF0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcTvfCoefF0", rdram, ctx, runtime);
    }

    void sceSynthesizerCent2PhaseInc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCent2PhaseInc", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeEffectSend(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeEffectSend", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeHsPanpot(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeHsPanpot", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeNrpnCutOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeNrpnCutOff", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeNrpnLfoDepth(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeNrpnLfoDepth", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeNrpnLfoRate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeNrpnLfoRate", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeOutAttrib(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeOutAttrib", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeOutVol(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeOutVol", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePanpot(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePanpot", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartBendSens(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartBendSens", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartExpression(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartExpression", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartHsExpression(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartHsExpression", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartHsPitchBend(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartHsPitchBend", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartModuration(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartModuration", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartPitchBend(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartPitchBend", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartVolume", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePortamento(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePortamento", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePortamentoTime(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePortamentoTime", rdram, ctx, runtime);
    }

    void sceSynthesizerClearKeyMap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerClearKeyMap", rdram, ctx, runtime);
    }

    void sceSynthesizerClearSpr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerClearSpr", rdram, ctx, runtime);
    }

    void sceSynthesizerCopyOutput(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCopyOutput", rdram, ctx, runtime);
    }

    void sceSynthesizerDmaFromSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerDmaFromSPR", rdram, ctx, runtime);
    }

    void sceSynthesizerDmaSpr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerDmaSpr", rdram, ctx, runtime);
    }

    void sceSynthesizerDmaToSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerDmaToSPR", rdram, ctx, runtime);
    }

    void sceSynthesizerGetPartial(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerGetPartial", rdram, ctx, runtime);
    }

    void sceSynthesizerGetPartOutLevel(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerGetPartOutLevel", rdram, ctx, runtime);
    }

    void sceSynthesizerGetSampleParam(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerGetSampleParam", rdram, ctx, runtime);
    }

    void sceSynthesizerHsMessage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerHsMessage", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoNone(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoNone", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoProc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoProc", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoSawDown(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoSawDown", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoSawUp(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoSawUp", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoSquare(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoSquare", rdram, ctx, runtime);
    }

    void sceSynthesizerReadNoise(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadNoise", rdram, ctx, runtime);
    }

    void sceSynthesizerReadNoiseAdd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadNoiseAdd", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample16(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample16", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample16Add(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample16Add", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample8(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample8", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample8Add(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample8Add", rdram, ctx, runtime);
    }

    void sceSynthesizerResetPart(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerResetPart", rdram, ctx, runtime);
    }

    void sceSynthesizerRestorDma(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerRestorDma", rdram, ctx, runtime);
    }

    void sceSynthesizerSelectPatch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSelectPatch", rdram, ctx, runtime);
    }

    void sceSynthesizerSendShortMessage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSendShortMessage", rdram, ctx, runtime);
    }

    void sceSynthesizerSetMasterVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetMasterVolume", rdram, ctx, runtime);
    }

    void sceSynthesizerSetRVoice(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetRVoice", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupDma(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupDma", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupLfo(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupLfo", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupMidiModuration(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupMidiModuration", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupMidiPanpot(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupMidiPanpot", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupNewNoise(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupNewNoise", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupReleaseEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupReleaseEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerSetuptEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetuptEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupTruncateTvaEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupTruncateTvaEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupTruncateTvfPitchEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupTruncateTvfPitchEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerTonegenerator(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTonegenerator", rdram, ctx, runtime);
    }

    void sceSynthesizerTransposeMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTransposeMatrix", rdram, ctx, runtime);
    }

    void sceSynthesizerTvfProcI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTvfProcI", rdram, ctx, runtime);
    }

    void sceSynthesizerTvfProcNI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTvfProcNI", rdram, ctx, runtime);
    }

    void sceSynthesizerWaitDmaFromSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerWaitDmaFromSPR", rdram, ctx, runtime);
    }

    void sceSynthesizerWaitDmaToSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerWaitDmaToSPR", rdram, ctx, runtime);
    }

    void sceSynthsizerGetDrumPatch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthsizerGetDrumPatch", rdram, ctx, runtime);
    }

    void sceSynthsizerGetMeloPatch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthsizerGetMeloPatch", rdram, ctx, runtime);
    }

    void sceSynthsizerLfoNoise(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthsizerLfoNoise", rdram, ctx, runtime);
    }

    void sceSynthSizerLfoTriangle(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthSizerLfoTriangle", rdram, ctx, runtime);
    }
}
