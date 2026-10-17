#include "iop_stdio.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

namespace ps2x::iop::detail
{
    namespace
    {
        // Minimal printf for IOP modules: o32 varargs come from a1..a3, then the stack at sp+16.
        std::string formatIopPrintf(const IopMemory &memory, const IopCpuState &cpu, uint32_t formatAddress, uint32_t firstArgIndex)
        {
            const std::string format = memory.readString(formatAddress, 2048u);
            uint32_t argIndex = firstArgIndex;
            auto nextArg = [&]() -> uint32_t
            {
                const uint32_t index = argIndex++;
                if (index < 4u)
                    return cpu.gpr[4u + index];
                return memory.read32(cpu.gpr[29] + 16u + (index - 4u) * 4u);
            };

            std::string out;
            for (size_t i = 0; i < format.size(); ++i)
            {
                const char c = format[i];
                if (c != '%' || i + 1 >= format.size())
                {
                    out.push_back(c);
                    continue;
                }
                size_t j = i + 1;
                std::string spec = "%";
                while (j < format.size() && std::strchr("-+ #0", format[j]))
                    spec.push_back(format[j++]);
                while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j])))
                    spec.push_back(format[j++]);
                while (j < format.size() && (format[j] == 'l' || format[j] == 'h'))
                    ++j;
                if (j >= format.size())
                    break;
                const char conv = format[j];
                char buffer[512];
                switch (conv)
                {
                case 'd':
                case 'i':
                    std::snprintf(buffer, sizeof(buffer), (spec + "d").c_str(), static_cast<int32_t>(nextArg()));
                    out += buffer;
                    break;
                case 'u':
                case 'x':
                case 'X':
                case 'o':
                    std::snprintf(buffer, sizeof(buffer), (spec + conv).c_str(), nextArg());
                    out += buffer;
                    break;
                case 'p':
                    std::snprintf(buffer, sizeof(buffer), "0x%08x", nextArg());
                    out += buffer;
                    break;
                case 'c':
                    out.push_back(static_cast<char>(nextArg() & 0xFFu));
                    break;
                case 's':
                    std::snprintf(buffer, sizeof(buffer), (spec + "s").c_str(), memory.readString(nextArg(), 256u).c_str());
                    out += buffer;
                    break;
                case '%':
                    out.push_back('%');
                    break;
                default:
                    out += spec;
                    out.push_back(conv);
                    break;
                }
                i = j;
            }
            return out;
        }
    }

    IopStdio::IopStdio(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    bool IopStdio::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        const auto logString = [&](std::string_view prefix, uint32_t address, uint32_t resultBias = 0u)
        {
            const std::string text = m_memory.readString(address, 2048u);
            m_host.log(LogLevel::Info, std::string(prefix) + text);
            setV0(static_cast<uint32_t>(text.size()) + resultBias);
        };

        switch (ordinal)
        {
        case 4: // printf
        {
            const std::string text = formatIopPrintf(m_memory, cpu, a0, 1u);
            m_host.log(LogLevel::Info, "[IOP printf] " + text);
            setV0(static_cast<uint32_t>(text.size()));
            return true;
        }
        case 5: // getchar
        case 10:
            setV0(0xFFFFFFFFu);
            return true;
        case 6: // putchar
            m_host.log(LogLevel::Info, std::string("[IOP putchar] ") + static_cast<char>(a0 & 0xFFu));
            setV0(a0 & 0xFFu);
            return true;
        case 7: // puts
            logString("[IOP puts] ", a0, 1u);
            return true;
        case 8: // gets
        case 13:
            setV0(0u);
            return true;
        case 9: // fdprintf
            logString("[IOP fdprintf] ", a1);
            return true;
        case 11:
            setV0(a0 & 0xFFu);
            return true;
        case 12: // fdputs
            logString("[IOP fdputs] ", a0);
            return true;
        case 14: // vfdprintf
            logString("[IOP vfdprintf] ", a1);
            return true;
        default:
            return false;
        }
    }
}
