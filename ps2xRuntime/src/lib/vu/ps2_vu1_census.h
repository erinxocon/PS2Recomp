#ifndef PS2_VU1_CENSUS_H
#define PS2_VU1_CENSUS_H

#include <cstdint>

// VU1 microprogram census (PS2X_VU1_CENSUS=<dir>, or 1 for build/vu_census), modelled on the Ace
// Combat 5 port's PS2_VU_CENSUS. Writes <dir>/manifest.json in AC5's format (one record per distinct
// micro image, with entry points and attributed work) so its tools/vurecomp runs on it unchanged,
// plus <dir>/gitaroo_census.json with what a lifter needs beyond that: per-pair execution counts,
// observed JR/JALR targets, and the places where an instruction-counted timing model (what AC5's
// lifted code uses) would read a different Q/P/flag/VI value than our cycle-accurate pipeline.
namespace vu1census
{
    bool enabled();

    // A new program starts at startPc (MSCAL/MSCNT/...). Re-hashes the image only when the code
    // generation changed.
    void onExecute(const uint8_t *code, uint32_t codeSize, uint64_t codeGeneration, uint32_t startPc);

    // One executed pair, at its issue cycle (after stalls). vi = current VI registers, data = VU1
    // data memory (for XGKICK packet sizes and store addresses).
    void onPair(uint32_t pc, uint32_t lower, uint32_t upper, uint64_t cycle,
                const int32_t *vi, const uint8_t *data, uint32_t dataSize);

    // A branch read the pre-write value of a VI register written by the previous pair.
    void onBranchViBackup(uint32_t pc);

    void onJump(uint32_t pc, uint32_t target);

    void onRunEnd(uint64_t cycles);
}

#endif
