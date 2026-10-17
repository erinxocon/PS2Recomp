#include "gs_threaded_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace
{
    // Rows per band; band b is rasterized by worker b % workerCount.
    constexpr int kBandRows = 16;
    // Pending commands are handed to the workers once this many accumulate (or when they go idle,
    // or a caller needs results).
    constexpr size_t kPublishThreshold = 256u;
}

GSThreadedBackend::GSThreadedBackend(const Factory &factory, uint32_t workerCount)
    : m_workerCount(std::max<uint32_t>(workerCount, 1u)), m_barrier(static_cast<std::ptrdiff_t>(std::max<uint32_t>(workerCount, 1u)))
{
    for (uint32_t i = 0; i < m_workerCount; ++i)
        m_instances.push_back(factory());
    for (uint32_t i = 0; i < m_workerCount; ++i)
        m_workers.emplace_back([this, i]() { WorkerLoop(i); });
}

GSThreadedBackend::~GSThreadedBackend()
{
    Drain();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_workCv.notify_all();
    for (std::thread &worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
}

uint32_t GSThreadedBackend::ConfiguredWorkers()
{
    const char *value = std::getenv("PS2X_GS_THREADS");
    if (!value || !*value)
        return 6u;
    return static_cast<uint32_t>(std::min(std::strtoul(value, nullptr, 10), 32ul));
}

bool GSThreadedBackend::IsExclusive(CommandType type)
{
    return type == CommandType::BeginTransfer || type == CommandType::UploadImage ||
           type == CommandType::WriteVram || type == CommandType::TextureFlush || type == CommandType::Present;
}

void GSThreadedBackend::Enqueue(Command &&command)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_pending.push_back(std::move(command));
    ++m_queued;
    if (m_pending.size() >= kPublishThreshold || m_segments.empty())
        PublishLocked();
}

void GSThreadedBackend::PublishLocked()
{
    if (m_pending.empty())
        return;
    auto segment = std::make_shared<Segment>();
    segment->commands.swap(m_pending);
    segment->remainingWorkers = m_workerCount;
    m_segments.push_back(std::move(segment));
    m_workCv.notify_all();
}

void GSThreadedBackend::Drain() const
{
    auto *self = const_cast<GSThreadedBackend *>(this);
    std::unique_lock<std::mutex> lock(m_mutex);
    self->PublishLocked();
    const uint64_t target = m_queued;
    m_doneCv.wait(lock, [&]() { return m_completed >= target || m_stop; });
}

void GSThreadedBackend::WorkerLoop(uint32_t worker)
{
    uint64_t nextSegment = 0;
    for (;;)
    {
        std::shared_ptr<Segment> segment;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_workCv.wait(lock, [&]() { return m_stop || nextSegment < m_firstSegment + m_segments.size(); });
            if (nextSegment >= m_firstSegment + m_segments.size())
                return;
            segment = m_segments[static_cast<size_t>(nextSegment - m_firstSegment)];
        }

        RunSegment(worker, *segment);
        ++nextSegment;

        bool completedAny = false;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            --segment->remainingWorkers;
            // Segments retire in order, so m_completed never runs ahead of an unfinished segment.
            while (!m_segments.empty() && m_segments.front()->remainingWorkers == 0u)
            {
                m_completed += m_segments.front()->commands.size();
                m_segments.pop_front();
                ++m_firstSegment;
                completedAny = true;
            }
            if (completedAny && m_segments.empty())
                PublishLocked();
        }
        if (completedAny)
            m_doneCv.notify_all();
    }
}

void GSThreadedBackend::RunSegment(uint32_t worker, const Segment &segment)
{
    GSRasterBackend &backend = *m_instances[worker];
    const std::vector<Command> &commands = segment.commands;
    size_t i = 0;
    while (i < commands.size())
    {
        const Command &command = commands[i];
        if (!IsExclusive(command.type))
        {
            try
            {
                if (command.type == CommandType::Submit)
                    SubmitBand(worker, command.batch);
                else
                    backend.LoadClut(command.tex0, command.texclut);
            }
            catch (...)
            {
            }
            ++i;
            continue;
        }

        // A run of memory-writing commands: wait for every band to finish the draws before it,
        // let worker 0 apply it, then drop the texture caches that may now be stale.
        size_t end = i;
        bool writesMemory = false;
        while (end < commands.size() && IsExclusive(commands[end].type))
            writesMemory |= commands[end++].type != CommandType::Present;
        m_barrier.arrive_and_wait();
        if (worker == 0u)
        {
            for (size_t j = i; j < end; ++j)
            {
                try
                {
                    ExecuteExclusive(const_cast<Command &>(commands[j]));
                }
                catch (...)
                {
                }
            }
        }
        m_barrier.arrive_and_wait();
        if (writesMemory)
            backend.TextureFlush();
        i = end;
    }
}

void GSThreadedBackend::SubmitBand(uint32_t worker, const GSPrimitiveBatch &batch)
{
    GSRasterBackend &backend = *m_instances[worker];
    if (m_workerCount == 1u)
    {
        backend.Submit(batch);
        return;
    }

    const auto &scissor = batch.state.context.scissor;
    int top = static_cast<int>(scissor.y0);
    int bottom = static_cast<int>(scissor.y1);
    if (batch.vertexCount > 0u)
    {
        // Only visit the bands the primitive's vertices span (with a row of slack for rounding).
        const float ofy = static_cast<float>(batch.state.context.xyoffset.ofy >> 4);
        float minY = batch.vertices[0].y;
        float maxY = batch.vertices[0].y;
        for (uint32_t v = 1; v < batch.vertexCount && v < batch.vertices.size(); ++v)
        {
            minY = std::min(minY, batch.vertices[v].y);
            maxY = std::max(maxY, batch.vertices[v].y);
        }
        top = std::max(top, static_cast<int>(std::floor(minY - ofy)) - 1);
        bottom = std::min(bottom, static_cast<int>(std::ceil(maxY - ofy)) + 1);
    }
    if (top > bottom)
        return;

    const int firstBand = top / kBandRows;
    const int lastBand = bottom / kBandRows;
    for (int band = firstBand; band <= lastBand; ++band)
    {
        if (static_cast<uint32_t>(band) % m_workerCount != worker)
            continue;
        GSPrimitiveBatch clipped = batch;
        auto &clippedScissor = clipped.state.context.scissor;
        clippedScissor.y0 = static_cast<decltype(clippedScissor.y0)>(std::max<int>(scissor.y0, band * kBandRows));
        clippedScissor.y1 = static_cast<decltype(clippedScissor.y1)>(std::min<int>(scissor.y1, band * kBandRows + kBandRows - 1));
        if (clippedScissor.y0 > clippedScissor.y1)
            continue;
        backend.Submit(clipped);
    }
}

void GSThreadedBackend::ExecuteExclusive(Command &command)
{
    GSRasterBackend &primary = *m_instances[0];
    switch (command.type)
    {
    case CommandType::BeginTransfer:
        primary.BeginTransfer(command.transfer);
        break;
    case CommandType::UploadImage:
        primary.UploadImage(command.data.data(), static_cast<uint32_t>(command.data.size()));
        break;
    case CommandType::WriteVram:
        primary.WriteVram(command.args[0], command.args[1], command.args[2], command.args[3], command.args[4],
                          command.args[5]);
        break;
    case CommandType::TextureFlush:
        // Every worker invalidates its own texture cache after the barrier.
        break;
    case CommandType::Present:
    {
        PresentationFrame frame = primary.Present(command.presentRequest);
        PresentCallback callback = std::move(command.presentCallback);
        m_presentPending.store(false, std::memory_order_release);
        if (callback)
            callback(std::move(frame));
        break;
    }
    default:
        break;
    }
}

void GSThreadedBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    Drain();
    for (auto &instance : m_instances)
        instance->Initialize(vram, vramSize);
}

void GSThreadedBackend::Reset()
{
    Drain();
    for (auto &instance : m_instances)
        instance->Reset();
}

void GSThreadedBackend::Submit(const GSPrimitiveBatch &batch)
{
    Command command;
    command.type = CommandType::Submit;
    command.batch = batch;
    Enqueue(std::move(command));
}

void GSThreadedBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    Command command;
    command.type = CommandType::LoadClut;
    command.tex0 = tex0;
    command.texclut = texclut;
    Enqueue(std::move(command));
}

void GSThreadedBackend::BeginTransfer(const GSTransferCommand &transfer)
{
    Command command;
    command.type = CommandType::BeginTransfer;
    command.transfer = transfer;
    Enqueue(std::move(command));
}

void GSThreadedBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u)
        return;
    Command command;
    command.type = CommandType::UploadImage;
    command.data.assign(data, data + sizeBytes);
    Enqueue(std::move(command));
}

void GSThreadedBackend::Flush()
{
    // The CPU backends rasterize immediately; there is nothing to submit.
}

void GSThreadedBackend::TextureFlush()
{
    Command command;
    command.type = CommandType::TextureFlush;
    Enqueue(std::move(command));
}

void GSThreadedBackend::Sync(GSSyncReason reason)
{
    // FINISH only orders drawing; the guest cannot observe GS memory without going through one of
    // the draining calls below, so it does not have to wait for the workers.
    if (reason == GSSyncReason::Finish)
        return;
    Drain();
    m_instances[0]->Sync(reason);
}

bool GSThreadedBackend::PresentAsync(const GSPresentationRequest &request, PresentCallback callback)
{
    if (m_presentPending.exchange(true, std::memory_order_acq_rel))
        return false;
    Command command;
    command.type = CommandType::Present;
    command.presentRequest = request;
    command.presentCallback = std::move(callback);
    Enqueue(std::move(command));
    return true;
}

PresentationFrame GSThreadedBackend::Present(const GSPresentationRequest &request)
{
    Drain();
    return m_instances[0]->Present(request);
}

bool GSThreadedBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    Drain();
    return m_instances[0]->ClearFramebuffer(context, rgba);
}

uint32_t GSThreadedBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    Drain();
    return m_instances[0]->ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSThreadedBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    Drain();
    return m_instances[0]->ReadVram(psm, base, bw, x, y);
}

void GSThreadedBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    Command command;
    command.type = CommandType::WriteVram;
    command.args[0] = psm;
    command.args[1] = base;
    command.args[2] = bw;
    command.args[3] = x;
    command.args[4] = y;
    command.args[5] = value;
    Enqueue(std::move(command));
}

void GSThreadedBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    Drain();
    m_instances[0]->SnapshotVram(out);
}

GSTransferSnapshot GSThreadedBackend::GetTransferSnapshot() const
{
    Drain();
    return m_instances[0]->GetTransferSnapshot();
}
