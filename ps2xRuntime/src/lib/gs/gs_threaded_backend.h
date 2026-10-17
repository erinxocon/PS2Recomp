#pragma once

#include "runtime/gs/gs_backend.h"

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Runs a raster backend on worker threads. Each worker owns a backend instance over the same GS
// memory and rasterizes an interleaved set of 16-row bands, so draws touching different rows run in
// parallel. Commands that write GS memory outside of drawing (transfers, VRAM writes, TEXFLUSH) run
// on worker 0 between two barriers, so no draw overlaps them. Every call that returns data waits
// for the commands queued before it, so callers observe the same results as with one backend.
class GSThreadedBackend final : public GSRasterBackend
{
public:
    using Factory = std::function<std::unique_ptr<GSRasterBackend>()>;

    GSThreadedBackend(const Factory &factory, uint32_t workerCount);
    ~GSThreadedBackend() override;

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

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    // Worker count from PS2X_GS_THREADS (default 6); 0 disables threading.
    static uint32_t ConfiguredWorkers();

    using PresentCallback = std::function<void(PresentationFrame &&)>;
    // Presents behind the commands queued so far without waiting for them: worker 0 builds the
    // frame once those draws are done and hands it to the callback (on that worker thread). Returns
    // false, queueing nothing, while an earlier one is still pending.
    bool PresentAsync(const GSPresentationRequest &request, PresentCallback callback);

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
    };

    struct Command
    {
        CommandType type = CommandType::TextureFlush;
        GSPrimitiveBatch batch{};
        GSTex0Reg tex0{};
        GSTexClutReg texclut{};
        GSTransferCommand transfer{};
        std::vector<uint8_t> data;
        uint32_t args[6]{};
        GSPresentationRequest presentRequest{};
        PresentCallback presentCallback;
    };

    struct Segment
    {
        std::vector<Command> commands;
        uint32_t remainingWorkers = 0;
    };

    static bool IsExclusive(CommandType type);

    void Enqueue(Command &&command);
    void PublishLocked();
    // Blocks until every command queued so far has been executed by all workers.
    void Drain() const;
    void WorkerLoop(uint32_t worker);
    void RunSegment(uint32_t worker, const Segment &segment);
    void SubmitBand(uint32_t worker, const GSPrimitiveBatch &batch);
    void ExecuteExclusive(Command &command);

    std::vector<std::unique_ptr<GSRasterBackend>> m_instances;
    uint32_t m_workerCount = 1;

    mutable std::mutex m_mutex;
    mutable std::condition_variable m_workCv;
    mutable std::condition_variable m_doneCv;
    std::vector<Command> m_pending;
    std::deque<std::shared_ptr<Segment>> m_segments; // published, oldest first
    uint64_t m_firstSegment = 0;                      // sequence number of m_segments.front()
    uint64_t m_queued = 0;
    uint64_t m_completed = 0;
    bool m_stop = false;
    std::atomic<bool> m_presentPending{false};

    std::barrier<> m_barrier;
    std::vector<std::thread> m_workers;
};
