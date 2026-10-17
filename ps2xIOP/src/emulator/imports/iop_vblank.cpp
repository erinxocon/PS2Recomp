#include "iop_vblank.h"

#include "../core/iop_cpu.h"
#include "../iop_emulator_const.h"
#include "../core/iop_kernel.h"
#include "../services/iop_rpc.h"

#include <algorithm>

namespace ps2x::iop::detail
{
    namespace
    {
        uint64_t nextEdgeAfter(uint64_t currentCycle, uint64_t phase) noexcept
        {
            const uint64_t fieldStart = currentCycle - (currentCycle % kVblankPeriodCycles);
            uint64_t edge = fieldStart + phase;
            if (edge <= currentCycle)
                edge += kVblankPeriodCycles;
            return edge;
        }
    }

    IopVblank::IopVblank(IopKernel &kernel) noexcept
        : m_kernel(kernel)
    {
    }

    void IopVblank::reset() noexcept
    {
        m_handlers.clear();
        m_nextStartCycle = 0u;
        m_nextEndCycle = 0u;
        m_servicing = false;
    }

    bool IopVblank::dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle)
    {
        switch (ordinal)
        {
        case 4: // WaitVblankStart
        case 5: // WaitVblankEnd
        case 6: // WaitVblank
        case 7: // WaitNonVblank
        {
            const bool waitForEnd = ordinal == 5u || ordinal == 7u;
            const uint64_t wakeCycle = nextEdgeAfter(currentCycle, waitForEnd ? kVblankEndPhaseCycles : 0u);
            m_kernel.delayCurrentUntil(wakeCycle, cpu);
            cpu.gpr[2] = 0u;
            return true;
        }
        case 8: // RegisterVblankHandler(edge, priority, handler, arg)
        {
            if (m_handlers.empty())
            {
                m_nextStartCycle = nextEdgeAfter(currentCycle, 0u);
                m_nextEndCycle = nextEdgeAfter(currentCycle, kVblankEndPhaseCycles);
            }
            const Handler handler{cpu.gpr[4] != 0u ? 1u : 0u, cpu.gpr[5], cpu.gpr[6], cpu.gpr[7], cpu.gpr[28]};
            // Lower priority values run first.
            const auto pos = std::upper_bound(m_handlers.begin(), m_handlers.end(), handler,
                                              [](const Handler &a, const Handler &b) { return a.priority < b.priority; });
            m_handlers.insert(pos, handler);
            cpu.gpr[2] = 0u;
            return true;
        }
        case 9: // ReleaseVblankHandler(edge, handler)
        {
            const uint32_t edge = cpu.gpr[4] != 0u ? 1u : 0u;
            const uint32_t function = cpu.gpr[5];
            std::erase_if(m_handlers, [&](const Handler &h) { return h.edge == edge && h.function == function; });
            cpu.gpr[2] = 0u;
            return true;
        }
        default:
            return false;
        }
    }

    void IopVblank::runHandlers(uint32_t edge, IopGuestExecutor &executor)
    {
        const std::vector<Handler> handlers = m_handlers;
        for (const Handler &handler : handlers)
        {
            if (handler.edge != edge || handler.function == 0u)
                continue;
            const uint32_t keep = executor.executeGuestFunctionWithBudget(handler.function, handler.argument,
                                                                           0u, 0u, 0u, handler.gp, 100000u);
            // A handler that returns 0 is unregistered, as in the IOP vblank module.
            if (keep == 0u)
            {
                std::erase_if(m_handlers, [&](const Handler &h)
                              { return h.edge == edge && h.function == handler.function && h.argument == handler.argument; });
            }
        }
    }

    void IopVblank::serviceDue(uint64_t currentCycle, IopGuestExecutor &executor)
    {
        if (m_servicing || m_handlers.empty())
            return;
        m_servicing = true;
        struct ServiceGuard
        {
            bool &flag;
            ~ServiceGuard() { flag = false; }
        } guard{m_servicing};

        // Deliver at most one edge of each kind per call; after a long stall, skip missed fields
        // rather than replaying them back to back.
        if (m_nextStartCycle <= currentCycle)
        {
            m_nextStartCycle = nextEdgeAfter(currentCycle, 0u);
            runHandlers(0u, executor);
        }
        if (m_nextEndCycle <= currentCycle)
        {
            m_nextEndCycle = nextEdgeAfter(currentCycle, kVblankEndPhaseCycles);
            runHandlers(1u, executor);
        }
    }

    uint64_t IopVblank::nextEventCycle(uint64_t fallback) const noexcept
    {
        if (m_handlers.empty())
            return fallback;
        return std::min({fallback, m_nextStartCycle, m_nextEndCycle});
    }
}
