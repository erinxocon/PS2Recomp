#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_cpu_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class GSGpuRenderer;

// GS backend that rasterizes on the GPU with OpenGL (through raylib's GL context), on the host
// (window) thread.
//
// The guest thread only queues commands; PumpHostThread(), called from the host loop, executes
// them in order. GS memory stays the source of truth for everything that is not drawing: a CPU
// backend over the same memory runs transfers, CLUT loads and presentation, and the renderer keeps
// its render targets and textures in sync with it (download a target before GS memory under it is
// read or written by anything else, upload it again before the next draw after GS memory changed).
// Calls that return data wait for the host thread; FINISH does not, since the guest cannot see GS
// memory without one of those calls.
//
//   (default; PS2X_GS_BACKEND=cpu selects the CPU rasterizer instead)
//   PS2X_GS_SCALE=N          internal resolution (default 2)
//   PS2X_GS_GPU_VERIFY=1     also run the CPU rasterizer on a copy of GS memory and compare every
//                            presented frame (stats on stderr, diff images into PS2X_DUMP_FRAMES)
class GSGpuBackend final : public GSRasterBackend
{
public:
    GSGpuBackend();
    ~GSGpuBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;
    bool SupportsAsyncPresent() const override { return true; }
    bool PresentAsync(const GSPresentationRequest &request, PresentCallback callback) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    // Host thread: executes queued commands until `deadline`, waking as commands are published.
    void PumpHostThread(std::chrono::steady_clock::time_point deadline);
    // Host thread, before the GL context goes away: releases GL objects and stops executing;
    // pending and later synchronous calls return without results.
    void ShutdownHostThread();

    // Host thread: whether the last presented frame came from the GPU, and drawing it (inside
    // raylib's BeginDrawing/EndDrawing) into the given window rectangle.
    bool HasPresentation() const;
    void DrawPresentation(float x, float y, float w, float h);

    static bool Requested();

private:
    enum class CommandType : uint8_t
    {
        Submit,
        LoadClut,
        BeginTransfer,
        UploadImage,
        TextureFlush,
        WriteVram,
        Present,
        Call,
    };

    // Everything but draws is rare; keep its payload out of line so queued draws stay small.
    struct CommandExtra
    {
        GSTex0Reg tex0{};
        GSTexClutReg texclut{};
        GSTransferCommand transfer{};
        std::vector<uint8_t> data;
        uint32_t args[6]{};
        GSPresentationRequest presentRequest{};
        PresentCallback presentCallback;
        std::function<void()> call;
    };

    struct Command
    {
        CommandType type = CommandType::TextureFlush;
        GSPrimitiveBatch batch{};
        std::unique_ptr<CommandExtra> extra;
    };

    // Queued batches beyond which the guest waits for the host (bounds memory when the host falls
    // behind, e.g. with verification on).
    static constexpr size_t kMaxReadyBatches = 64;

    void Enqueue(Command &&command);
    void PublishLocked();
    // Runs fn on the host thread behind everything queued so far and waits for it (inline when
    // called on the host thread, or before the host loop runs).
    void RunOnHost(const std::function<void()> &fn) const;
    void ExecuteBatch(std::vector<Command> &commands);
    void Execute(Command &command);
    bool OnHostThread() const;

    GSCpuBackend m_cpu;                       // GS memory side (host thread only once pumping)
    std::unique_ptr<GSGpuRenderer> m_renderer; // GL side (host thread only)
    std::unique_ptr<GSCpuBackend> m_shadow;   // PS2X_GS_GPU_VERIFY reference rasterizer
    std::vector<uint8_t> m_shadowVram;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;

    mutable std::mutex m_mutex;
    mutable std::condition_variable m_workCv;
    mutable std::condition_variable m_doneCv;
    mutable std::vector<Command> m_pending;
    mutable std::deque<std::vector<Command>> m_ready;
    mutable uint64_t m_queued = 0;
    uint64_t m_completed = 0;
    bool m_stopped = false;
    std::atomic<bool> m_hostStarted{false};
    std::thread::id m_hostThread{};
    std::atomic<bool> m_presentPending{false};
    uint32_t m_transferFirstPage = 0; // host thread: pages the current host->local transfer writes
    uint32_t m_transferPageCount = 0;
};
