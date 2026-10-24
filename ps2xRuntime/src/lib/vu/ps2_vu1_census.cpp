#include "ps2_vu1_census.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kPairs = 0x4000u / 8u;

    // Readers whose value depends on pipeline timing. For each, the latest producer's distance is
    // measured both in instructions (AC5's lifted-code model: flags through a 4-slot ring, Q/P ready
    // after N instructions, no stalls) and in cycles (our interpreter). They disagree only when a
    // stall sits between producer and reader.
    enum Hazard : uint8_t
    {
        HazQ,      // upper *q op reads Q after DIV/SQRT/RSQRT
        HazP,      // MFP reads P after an EFU op
        HazClip,   // FCxxx / CLIP read the clip flag after CLIP/FCSET
        HazMac,    // FMxxx / FSxxx read MAC/status after an FMAC op
        HazCount
    };
    const char *const kHazardNames[HazCount] = {"q", "p", "clip", "mac"};

    struct HazardStat
    {
        uint64_t reads = 0;
        uint64_t instrPending = 0; // instruction model: producer not yet visible
        uint64_t cyclePending = 0; // cycle model: producer not yet visible
        uint64_t disagree = 0;
    };

    struct Producer
    {
        uint64_t index = 0;
        uint64_t cycle = 0;
        uint32_t latency = 0;
        bool valid = false;
    };

    struct EntryStat
    {
        uint64_t count = 0;
        uint64_t pairs = 0;
        uint64_t cycles = 0;
    };

    struct Image
    {
        uint64_t hash = 0;
        std::vector<uint8_t> bytes;
        uint32_t lo = 0, hi = 0;
        uint64_t activations = 0, insns = 0, kicks = 0, cycles = 0;
        std::map<uint32_t, EntryStat> entries;
        std::vector<uint64_t> pcCount = std::vector<uint64_t>(kPairs);
    };

    struct Census
    {
        bool checked = false;
        bool on = false;
        std::string dir;
        std::vector<Image> images;
        int current = -1;
        uint32_t currentEntry = 0;
        uint64_t lastGeneration = ~0ull;
        const uint8_t *lastCode = nullptr;
        uint64_t runCycleStart = 0;

        uint64_t index = 0; // pairs since the program started
        Producer producers[HazCount];
        std::map<std::pair<uint32_t, uint8_t>, HazardStat> hazards; // (pc, kind)
        std::map<uint32_t, uint64_t> viBackups;                     // branch pc -> count
        std::map<std::pair<uint32_t, uint32_t>, uint64_t> jumps;    // (pc, target) -> count

        // Last XGKICK, to spot stores into the part of its packet our interpreter has not copied
        // yet (AC5's vu_do_xgkick snapshots the whole packet at the XGKICK instead).
        uint32_t kickStart = 0, kickBytes = 0;
        uint64_t kickCycle = 0;
        bool kickValid = false;
        std::map<uint32_t, uint64_t> kickRaces; // store pc -> count

        std::chrono::steady_clock::time_point lastDump = std::chrono::steady_clock::now();
    };

    Census g_census;

    uint64_t fnv1a(const uint8_t *p, size_t n)
    {
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < n; ++i)
            h = (h ^ p[i]) * 0x100000001b3ull;
        return h;
    }

    uint32_t efuLatency(uint8_t special)
    {
        switch (special)
        {
        case 0x70: return 11u;               // ESADD
        case 0x71: case 0x72: return 18u;    // ERSADD, ELENG
        case 0x73: return 24u;               // ERLENG
        case 0x76: case 0x78: case 0x7A: return 12u; // ESUM, ESQRT, ERCPR
        case 0x79: return 18u;               // ERSQRT
        case 0x7C: return 29u;               // ESIN
        case 0x74: case 0x75: case 0x7D: return 54u; // EATANxy/xz, EATAN
        case 0x7E: return 44u;               // EEXP
        default: return 0u;
        }
    }

    // Upper ops that update MAC/status: everything but NOP, MAX/MINI, ITOF/FTOI, ABS and CLIP.
    bool upperWritesMac(uint32_t up)
    {
        const uint32_t funct = up & 0x3Fu;
        if ((funct & 0x3Cu) == 0x3Cu)
        {
            const uint32_t sub = (up & 3u) | ((up >> 4) & 0x7Cu);
            if (sub == 0x2Fu || sub == 0x1Du || sub == 0x1Fu || (sub >= 0x10u && sub <= 0x17u))
                return false;
            return sub <= 0x2Eu && sub != 0x2Bu;
        }
        if ((funct >= 0x10u && funct <= 0x17u) || funct == 0x1Du || funct == 0x1Fu ||
            funct == 0x2Bu || funct == 0x2Fu)
            return false;
        return true;
    }

    bool upperReadsQ(uint32_t up)
    {
        const uint32_t funct = up & 0x3Fu;
        uint32_t op = funct;
        if ((funct & 0x3Cu) == 0x3Cu)
            op = (up & 3u) | ((up >> 4) & 0x7Cu);
        return op == 0x1Cu || op == 0x20u || op == 0x21u || op == 0x24u || op == 0x25u;
    }

    // Size in bytes of the GIF packet at qword address qwAddr in VU1 data memory (through EOP).
    uint32_t gifPacketBytes(const uint8_t *data, uint32_t dataSize, uint32_t qwAddr)
    {
        uint32_t offset = 0;
        for (int tags = 0; tags < 256 && offset < dataSize; ++tags)
        {
            uint64_t tag = 0;
            std::memcpy(&tag, data + ((qwAddr * 16u + offset) % dataSize), 8);
            const uint32_t nloop = static_cast<uint32_t>(tag & 0x7FFFu);
            const bool eop = ((tag >> 15) & 1u) != 0u;
            const uint32_t flg = static_cast<uint32_t>((tag >> 58) & 3u);
            uint32_t nreg = static_cast<uint32_t>((tag >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            uint32_t qwords = flg == 0u ? nloop * nreg : flg == 1u ? (nloop * nreg + 1u) / 2u : nloop;
            offset += 16u + qwords * 16u;
            if (eop)
                break;
        }
        return offset;
    }

    void noteRead(Hazard kind, uint32_t pc, uint64_t cycle)
    {
        Census &c = g_census;
        const Producer &p = c.producers[kind];
        if (!p.valid)
            return;
        HazardStat &s = c.hazards[{pc, static_cast<uint8_t>(kind)}];
        const bool instrPending = c.index - p.index < p.latency;
        const bool cyclePending = cycle - p.cycle < p.latency;
        ++s.reads;
        s.instrPending += instrPending;
        s.cyclePending += cyclePending;
        s.disagree += instrPending != cyclePending;
    }

    void produce(Hazard kind, uint64_t cycle, uint32_t latency)
    {
        Census &c = g_census;
        c.producers[kind] = {c.index, cycle, latency, true};
    }

    void dump()
    {
        Census &c = g_census;
        std::error_code ec;
        std::filesystem::create_directories(c.dir, ec);
        const std::string manifestPath = c.dir + "/manifest.json";
        FILE *mf = std::fopen(manifestPath.c_str(), "wb");
        if (!mf)
            return;
        std::fprintf(mf, "{\n  \"programs\": [\n");
        for (size_t i = 0; i < c.images.size(); ++i)
        {
            const Image &im = c.images[i];
            char name[64];
            std::snprintf(name, sizeof(name), "vu1_%016llx.bin", static_cast<unsigned long long>(im.hash));
            if (FILE *bf = std::fopen((c.dir + "/" + name).c_str(), "wb"))
            {
                std::fwrite(im.bytes.data(), 1, im.bytes.size(), bf);
                std::fclose(bf);
            }
            std::fprintf(mf, "    {\n      \"crc\": \"%016llx\",\n      \"image\": \"%s\",\n      \"bytes\": %zu,\n"
                             "      \"code_lo\": %u,\n      \"code_hi\": %u,\n      \"activations\": %llu,\n"
                             "      \"instructions\": %llu,\n      \"xgkicks\": %llu,\n      \"entries\": [",
                         static_cast<unsigned long long>(im.hash), name, im.bytes.size(), im.lo, im.hi,
                         static_cast<unsigned long long>(im.activations), static_cast<unsigned long long>(im.insns),
                         static_cast<unsigned long long>(im.kicks));
            bool first = true;
            for (const auto &[pc, e] : im.entries)
            {
                std::fprintf(mf, "%s\n        {\"pc\": %u, \"count\": %llu}", first ? "" : ",", pc,
                             static_cast<unsigned long long>(e.count));
                first = false;
            }
            std::fprintf(mf, "\n      ]\n    }%s\n", i + 1 < c.images.size() ? "," : "");
        }
        std::fprintf(mf, "  ],\n  \"dropped\": 0\n}\n");
        std::fclose(mf);

        FILE *gf = std::fopen((c.dir + "/gitaroo_census.json").c_str(), "wb");
        if (!gf)
            return;
        std::fprintf(gf, "{\n  \"images\": [\n");
        for (size_t i = 0; i < c.images.size(); ++i)
        {
            const Image &im = c.images[i];
            uint32_t w1b70 = 0;
            if (im.bytes.size() >= 0x1B78u)
                std::memcpy(&w1b70, im.bytes.data() + 0x1B70u, 4);
            std::fprintf(gf, "    {\"crc\": \"%016llx\", \"lower_1b70\": %u, \"cycles\": %llu, \"entries\": {",
                         static_cast<unsigned long long>(im.hash), w1b70, static_cast<unsigned long long>(im.cycles));
            bool first = true;
            for (const auto &[pc, e] : im.entries)
            {
                std::fprintf(gf, "%s\"%u\": [%llu, %llu, %llu]", first ? "" : ", ", pc,
                             static_cast<unsigned long long>(e.count), static_cast<unsigned long long>(e.pairs),
                             static_cast<unsigned long long>(e.cycles));
                first = false;
            }
            std::fprintf(gf, "},\n     \"pc_count\": {");
            first = true;
            for (uint32_t p = 0; p < kPairs; ++p)
            {
                if (!im.pcCount[p])
                    continue;
                std::fprintf(gf, "%s\"%u\": %llu", first ? "" : ", ", p * 8u, static_cast<unsigned long long>(im.pcCount[p]));
                first = false;
            }
            std::fprintf(gf, "}}%s\n", i + 1 < c.images.size() ? "," : "");
        }
        std::fprintf(gf, "  ],\n  \"jumps\": [");
        bool first = true;
        for (const auto &[key, n] : c.jumps)
        {
            std::fprintf(gf, "%s[%u, %u, %llu]", first ? "" : ", ", key.first, key.second, static_cast<unsigned long long>(n));
            first = false;
        }
        std::fprintf(gf, "],\n  \"hazards\": [");
        first = true;
        for (const auto &[key, s] : c.hazards)
        {
            std::fprintf(gf, "%s\n    {\"pc\": %u, \"kind\": \"%s\", \"reads\": %llu, \"instr_pending\": %llu, "
                             "\"cycle_pending\": %llu, \"disagree\": %llu}",
                         first ? "" : ",", key.first, kHazardNames[key.second],
                         static_cast<unsigned long long>(s.reads), static_cast<unsigned long long>(s.instrPending),
                         static_cast<unsigned long long>(s.cyclePending), static_cast<unsigned long long>(s.disagree));
            first = false;
        }
        std::fprintf(gf, "\n  ],\n  \"vi_branch_backup\": {");
        first = true;
        for (const auto &[pc, n] : c.viBackups)
        {
            std::fprintf(gf, "%s\"%u\": %llu", first ? "" : ", ", pc, static_cast<unsigned long long>(n));
            first = false;
        }
        std::fprintf(gf, "},\n  \"xgkick_store_races\": {");
        first = true;
        for (const auto &[pc, n] : c.kickRaces)
        {
            std::fprintf(gf, "%s\"%u\": %llu", first ? "" : ", ", pc, static_cast<unsigned long long>(n));
            first = false;
        }
        std::fprintf(gf, "}\n}\n");
        std::fclose(gf);
    }

    struct DumpAtExit
    {
        ~DumpAtExit()
        {
            if (g_census.on && !g_census.images.empty())
                dump();
        }
    } g_dumpAtExit;
}

namespace vu1census
{
    bool enabled()
    {
        Census &c = g_census;
        if (!c.checked)
        {
            c.checked = true;
            const char *e = std::getenv("PS2X_VU1_CENSUS");
            c.on = e && *e && std::strcmp(e, "0") != 0;
            if (c.on)
            {
                c.dir = std::strcmp(e, "1") == 0 ? std::string("build/vu_census") : std::string(e);
                std::fprintf(stderr, "[vu1-census] armed, writing %s/manifest.json\n", c.dir.c_str());
            }
        }
        return c.on;
    }

    void onExecute(const uint8_t *code, uint32_t codeSize, uint64_t codeGeneration, uint32_t startPc)
    {
        Census &c = g_census;
        if (c.current < 0 || codeGeneration != c.lastGeneration || code != c.lastCode)
        {
            c.lastGeneration = codeGeneration;
            c.lastCode = code;
            const uint64_t h = fnv1a(code, codeSize);
            c.current = -1;
            for (size_t i = 0; i < c.images.size(); ++i)
                if (c.images[i].hash == h)
                    c.current = static_cast<int>(i);
            if (c.current < 0)
            {
                Image im;
                im.hash = h;
                im.bytes.assign(code, code + codeSize);
                im.lo = codeSize;
                for (uint32_t i = 0; i + 8u <= codeSize; i += 8u)
                {
                    uint64_t pair = 0;
                    std::memcpy(&pair, code + i, 8);
                    if (!pair)
                        continue;
                    im.lo = std::min(im.lo, i);
                    im.hi = std::max(im.hi, i + 8u);
                }
                if (im.lo > im.hi)
                    im.lo = im.hi = 0;
                c.images.push_back(std::move(im));
                c.current = static_cast<int>(c.images.size() - 1);
            }
        }
        Image &im = c.images[c.current];
        ++im.activations;
        ++im.entries[startPc].count;
        c.currentEntry = startPc;
        c.index = 0;
        for (Producer &p : c.producers)
            p.valid = false;
        c.kickValid = false;
    }

    void onPair(uint32_t pc, uint32_t lower, uint32_t upper, uint64_t cycle,
                const int32_t *vi, const uint8_t *data, uint32_t dataSize)
    {
        Census &c = g_census;
        if (c.current < 0)
            return;
        Image &im = c.images[c.current];
        ++im.insns;
        ++im.entries[c.currentEntry].pairs;
        if (pc / 8u < kPairs)
            ++im.pcCount[pc / 8u];
        ++c.index;

        const uint32_t up = upper & 0x07FFFFFFu;
        const bool ibit = (upper >> 31) != 0u;
        const uint32_t funct = up & 0x3Fu;
        const bool upperClip = (funct & 0x3Cu) == 0x3Cu && ((up & 3u) | ((up >> 4) & 0x7Cu)) == 0x1Fu;

        // Reads first (a pair sees state from before its own writes).
        if (upperReadsQ(up))
            noteRead(HazQ, pc, cycle);
        if (upperClip)
            noteRead(HazClip, pc, cycle);
        uint32_t storeAddr = ~0u;
        if (!ibit)
        {
            const uint32_t op = lower >> 25;
            const uint32_t it = (lower >> 16) & 0xFu, is = (lower >> 11) & 0xFu;
            const int32_t imm11 = static_cast<int32_t>(lower << 21) >> 21;
            switch (op)
            {
            case 0x10: case 0x12: case 0x13: case 0x1C:
                noteRead(HazClip, pc, cycle);
                break;
            case 0x14: case 0x16: case 0x17: case 0x18: case 0x1A: case 0x1B:
                noteRead(HazMac, pc, cycle);
                break;
            case 0x01: // SQ fs, imm(it)
                storeAddr = static_cast<uint32_t>(static_cast<uint16_t>(vi[it] + imm11)) * 16u;
                break;
            case 0x05: // ISW it, imm(is)
                storeAddr = static_cast<uint32_t>(static_cast<uint16_t>(vi[is] + imm11)) * 16u;
                break;
            case 0x40:
                if ((lower & 0x3Fu) >= 0x3Cu)
                {
                    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
                    if (special == 0x64u)
                        noteRead(HazP, pc, cycle);
                    else if (special == 0x35u) // SQI fs, (it++)
                        storeAddr = static_cast<uint32_t>(static_cast<uint16_t>(vi[it])) * 16u;
                    else if (special == 0x37u) // SQD fs, (--it)
                        storeAddr = static_cast<uint32_t>(static_cast<uint16_t>(vi[it] - 1)) * 16u;
                    else if (special == 0x3Fu) // ISWR it, (is)
                        storeAddr = static_cast<uint32_t>(static_cast<uint16_t>(vi[is])) * 16u;
                }
                break;
            default:
                break;
            }
        }

        if (storeAddr != ~0u && c.kickValid && dataSize)
        {
            const uint32_t copied = static_cast<uint32_t>(std::min<uint64_t>((cycle - c.kickCycle) / 2u, c.kickBytes / 16u)) * 16u;
            const uint32_t rel = (storeAddr % dataSize + dataSize - c.kickStart) % dataSize;
            if (rel >= copied && rel < c.kickBytes)
                ++c.kickRaces[pc];
        }

        // Then this pair's own productions.
        if (upperWritesMac(up))
            produce(HazMac, cycle, 4u);
        if (upperClip)
            produce(HazClip, cycle, 4u);
        if (!ibit)
        {
            const uint32_t op = lower >> 25;
            if (op == 0x11u)
                produce(HazClip, cycle, 4u);
            else if (op == 0x15u)
                produce(HazMac, cycle, 4u);
            else if (op == 0x40u && (lower & 0x3Fu) >= 0x3Cu)
            {
                const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
                if (special == 0x38u || special == 0x39u)
                    produce(HazQ, cycle, 7u);
                else if (special == 0x3Au)
                    produce(HazQ, cycle, 13u);
                else if (special == 0x3Bu) // WAITQ: both models are synchronised afterwards
                    c.producers[HazQ].valid = false;
                else if (special == 0x7Bu) // WAITP
                    c.producers[HazP].valid = false;
                else if (uint32_t lat = efuLatency(special))
                    produce(HazP, cycle, lat);
                else if (special == 0x6Cu) // XGKICK
                {
                    ++im.kicks;
                    const uint32_t is = (lower >> 11) & 0xFu;
                    c.kickStart = (static_cast<uint32_t>(static_cast<uint16_t>(vi[is])) * 16u) % (dataSize ? dataSize : 1u);
                    c.kickBytes = dataSize ? gifPacketBytes(data, dataSize, c.kickStart / 16u) : 0u;
                    c.kickCycle = cycle;
                    c.kickValid = true;
                }
            }
        }
    }

    void onBranchViBackup(uint32_t pc)
    {
        ++g_census.viBackups[pc];
    }

    void onJump(uint32_t pc, uint32_t target)
    {
        ++g_census.jumps[{pc, target}];
    }

    void onRunEnd(uint64_t cycles)
    {
        Census &c = g_census;
        if (c.current < 0)
            return;
        Image &im = c.images[c.current];
        im.cycles += cycles;
        im.entries[c.currentEntry].cycles += cycles;
        const auto now = std::chrono::steady_clock::now();
        if (now - c.lastDump > std::chrono::seconds(5))
        {
            c.lastDump = now;
            dump();
        }
    }
}
