#pragma once

#include <cstdint>
#include <vector>

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopKernel;
    class IopGuestExecutor;

    class IopVblank
    {
    public:
        explicit IopVblank(IopKernel &kernel) noexcept;

        void reset() noexcept;
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle);
        // Calls the handlers registered with RegisterVblankHandler once a vblank start/end edge
        // has been reached.
        void serviceDue(uint64_t currentCycle, IopGuestExecutor &executor);
        [[nodiscard]] uint64_t nextEventCycle(uint64_t fallback) const noexcept;

    private:
        struct Handler
        {
            uint32_t edge = 0u;
            uint32_t priority = 0u;
            uint32_t function = 0u;
            uint32_t argument = 0u;
            uint32_t gp = 0u;
        };

        void runHandlers(uint32_t edge, IopGuestExecutor &executor);

        IopKernel &m_kernel;
        std::vector<Handler> m_handlers;
        uint64_t m_nextStartCycle = 0u;
        uint64_t m_nextEndCycle = 0u;
        bool m_servicing = false;
    };
}
