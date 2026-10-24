#include "ps2_vu1_lifted.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// Native C++ for Gitaroo Man's VU1 mesh renderer, entry 0x1B68 (99.9% of all VU1 work), in the
// spirit of the Ace Combat 5 port's rn_vp: instead of executing the microprogram, read the vertex
// batch and constants the game uploaded and do the transform/light/clip/pack work directly.
// Everything the microprogram does is reproduced bit for bit -- the GIF packet, every store into
// VU1 data memory (the clipper and later passes read them) and the registers it leaves behind -- so
// PS2X_VU1_VERIFY checks this code against the interpreter like the lifted program.
//
// The program (docs/vu1-renderer.md). XTOP = batch of n vertices, 4 qwords each:
//   +0 position (xyzw)   +1 normal (xyz)   +2 STQ   +3 RGBA (float)
// Constants: vf1-vf4 model-view-projection rows, vf8-vf10 normal rotation (w: colour clamp,
// colour scale, alpha scale), vf11/vf12 screen scale/offset (w: fog), vf13-vf15 light directions,
// vf16-vf19 light colours, data[25] emissive colour, data[26] x/y double-buffered output address,
// data[29].x continuation (0x27C0 = kick), data[42] GIF tag template, data[202].x extra pass.
// The EE picks one of four modes by patching the branch at 0x1B70:
//   0x1B80 lit           colour = clamp(RGBA * max(lights(R * normal), 0) + emissive)
//   0x1E10 lit           same with the rotated normal normalized (ERLENG)
//   0x2108 unlit         colour = RGBA * (vf9.w, vf9.w, vf9.w, vf10.w)
//   0x2280 unlit         same, rotated normal normalized and stored for a later pass
// Per vertex out: STQ * Q, RGBA as integers, screen XYZF (FTOI4) with Q = 1 / clip w.
// Batches that need clipping, or an extra pass (data[202]), continue in the lifted program: this
// code then changes nothing and reports kLiftNotLifted, and the lifted program starts over.
//
//   PS2X_VU1_NATIVE=0         off (lifted program only)
//   PS2X_VU1_NATIVE_BREAK=1   negative test for PS2X_VU1_VERIFY: corrupt one colour bit per batch

namespace
{
    constexpr uint32_t kMaxVertices = 23; // keeps the +96 colour copies clear of the batch

    struct Settings
    {
        bool on = true, verbose = false, corrupt = false;
        Settings()
        {
            const char *e = std::getenv("PS2X_VU1_NATIVE");
            on = !(e && std::strcmp(e, "0") == 0);
            const char *v = std::getenv("PS2X_VU1_VERIFY");
            verbose = v && *v && std::strcmp(v, "0") != 0;
            const char *b = std::getenv("PS2X_VU1_NATIVE_BREAK");
            corrupt = b && *b && std::strcmp(b, "0") != 0;
        }
    };

    struct Stats
    {
        uint64_t drawn = 0, culled = 0, clipped = 0, extraPass = 0, other = 0;
        void print() const
        {
                std::fprintf(stderr, "[vu1-native] 1B68 batches: %llu drawn, %llu culled natively; to the lifted program: "
                                     "%llu clipped, %llu extra pass, %llu other\n",
                             static_cast<unsigned long long>(drawn), static_cast<unsigned long long>(culled),
                             static_cast<unsigned long long>(clipped), static_cast<unsigned long long>(extraPass),
                             static_cast<unsigned long long>(other));
        }
        ~Stats()
        {
            if (drawn || culled || clipped || extraPass || other)
                print();
        }
    } g_stats;

    inline uint32_t bitsOf(float f)
    {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        return b;
    }

    inline float fromBits(uint32_t b)
    {
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }

    // VU operand: denormal -> signed zero, Inf/NaN -> signed FLT_MAX.
    inline float vnrm(float value)
    {
        uint32_t b = bitsOf(value);
        const uint32_t e = b & 0x7F800000u;
        if (e == 0u)
            b &= 0x80000000u;
        else if (e == 0x7F800000u)
            b = (b & 0x80000000u) | 0x7F7FFFFFu;
        return fromBits(b);
    }

    // FMAC results (the caller runs with round-toward-zero, like the VU). A result well inside the
    // normal range is final; otherwise the exact value decides, as VU1Interpreter's
    // normalizeFmacExactResult does: zero keeps the exact sign, underflow flushes, overflow clamps.
    [[gnu::noinline]] float fixResult(float r, double exact)
    {
        const double mag = std::fabs(exact);
        const uint32_t sign = std::signbit(exact) ? 0x80000000u : 0u;
        if (mag == 0.0 || mag < static_cast<double>(std::numeric_limits<float>::min()))
            return fromBits(sign);
        if (mag > static_cast<double>(std::numeric_limits<float>::max()))
            return fromBits(sign | 0x7F7FFFFFu);
        return r;
    }

    inline bool plain(float r)
    {
        return ((bitsOf(r) >> 23) & 0xFFu) - 2u <= 251u;
    }

    inline float mul(float a, float b)
    {
        const float r = a * b;
        return plain(r) ? r : fixResult(r, static_cast<double>(a) * static_cast<double>(b));
    }

    inline float madd(float acc, float a, float b)
    {
        const float r = acc + a * b;
        return plain(r) ? r : fixResult(r, static_cast<double>(acc) + static_cast<double>(a) * static_cast<double>(b));
    }

    inline float vmax(float x, float y) { return x > y ? x : y; }
    inline float vmini(float x, float y) { return x < y ? x : y; }

    // FTOIn: the integer's bits, as stored in a VF register.
    inline float vftoi(float value, double scale)
    {
        const double scaled = static_cast<double>(value) * scale;
        int32_t r;
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            r = std::numeric_limits<int32_t>::max();
        else if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            r = std::numeric_limits<int32_t>::min();
        else
            r = static_cast<int32_t>(scaled);
        float out;
        std::memcpy(&out, &r, 4);
        return out;
    }

    // normalizeResult without the flags.
    inline float clampResult(float value)
    {
        uint32_t b = bitsOf(value);
        const uint32_t e = (b >> 23) & 0xFFu;
        if (e == 0u)
            b &= 0x80000000u;
        else if (e == 0xFFu)
            b = (b & 0x80000000u) | 0x7F7FFFFFu;
        return fromBits(b);
    }

    // DIV Q, vf0w, w
    inline float reciprocal(float w)
    {
        const float den = vnrm(w);
        if (den == 0.0f)
            return std::signbit(den) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        return clampResult(1.0f / den);
    }

    // ERLENG P, v
    inline float erleng(const float *v)
    {
        const float x = vnrm(v[0]), y = vnrm(v[1]), z = vnrm(v[2]);
        const float len = std::sqrt(x * x + y * y + z * z);
        return clampResult(len != 0.0f ? 1.0f / len : len);
    }

    // CLIPw.xyz v, vw
    inline uint32_t clipFlags(const float *v, float w)
    {
        const uint32_t wb = bitsOf(w);
        const int32_t limit = (wb & 0x7F800000u) != 0u ? static_cast<int32_t>(wb & 0x7FFFFFFFu) : 0x007FFFFF;
        uint32_t flags = 0;
        for (int c = 0; c < 3; ++c)
        {
            const uint32_t b = bitsOf(v[c]);
            if (static_cast<int32_t>(b) > limit)
                flags |= 1u << (2 * c);
            if (static_cast<int32_t>(b ^ 0x80000000u) > limit)
                flags |= 2u << (2 * c);
        }
        return flags;
    }

    // Overlap of two qword ranges on the 1024-qword data memory ring.
    inline bool overlaps(uint32_t a, uint32_t alen, uint32_t b, uint32_t blen, uint32_t qmask)
    {
        return ((b - a) & qmask) < alen || ((a - b) & qmask) < blen;
    }

    struct VertexOut
    {
        float slot0[4];   // clip-space position
        float slot1[4];   // rotated (normalized) normal, Q
        uint16_t flags;   // slot2.w: clip flags
        float slot3[4];   // colour before FTOI0
        float copy[4];    // +96: the input colour, raw
        float stq[4], rgba[4], xyzf[4];
    };
}

int VU1Lifted::runNative(VU1Interpreter &vu, LiftedRun &run)
{
    static const Settings s_settings;
    if (!s_settings.on)
        return kLiftNotLifted;

    uint32_t patch = 0;
    std::memcpy(&patch, run.code + 0x1B70u, 4);
    if ((patch >> 25) != 0x20u)
        return kLiftNotLifted;
    const uint32_t mode = (0x1B78u + static_cast<uint32_t>((static_cast<int32_t>(patch << 21) >> 21) * 8)) & 0x3FFFu;
    const bool lit = mode == 0x1B80u || mode == 0x1E10u;
    const bool normalize = mode == 0x1E10u || mode == 0x2280u;
    if (!lit && mode != 0x2108u && mode != 0x2280u)
        return kLiftNotLifted;

    uint8_t *const mem = run.mem;
    const uint32_t qmask = run.dataMask >> 4;
    auto qw = [&](uint32_t q) { return mem + (q & qmask) * 16u; };
    auto load = [&](uint32_t q, float *v) { std::memcpy(v, qw(q), 16); };
    auto lo16 = [&](uint32_t q, int lane)
    {
        uint32_t w;
        std::memcpy(&w, qw(q) + lane * 4, 4);
        return static_cast<int16_t>(w & 0xFFFFu);
    };
    auto storeU16 = [&](uint32_t q, int lane, int32_t value)
    {
        const uint32_t w = static_cast<uint16_t>(value);
        std::memcpy(qw(q) + lane * 4, &w, 4);
    };

    const uint32_t top = vu.m_state.top & 0x3FFu;
    const uint32_t n = vu.m_state.itop & 0x3FFu;
    const int16_t outBase = lo16(26, 0);
    const uint32_t base = static_cast<uint16_t>(outBase);
    if (n == 0u || n > kMaxVertices)
    {
        ++g_stats.other;
        return kLiftNotLifted;
    }
    // Everything read or written must be disjoint, so computing first and storing afterwards is
    // the same as the microprogram's interleaving.
    const uint32_t inLen = 4u * n + 4u, outLen = 3u * n + 3u;
    if (overlaps(top, inLen, base, outLen, qmask) || overlaps(top + 96u, 4u * n, base, outLen, qmask))
    {
        ++g_stats.other;
        return kLiftNotLifted;
    }
    for (uint32_t c : {25u, 26u, 28u, 29u, 42u, 202u})
        if (overlaps(c, 1u, top, inLen, qmask) || overlaps(c, 1u, top + 96u, 4u * n, qmask) ||
            overlaps(c, 1u, base, outLen, qmask))
        {
            ++g_stats.other;
            return kLiftNotLifted;
        }

    float(&vf)[32][4] = vu.m_state.vf;
    float m1[4], m2[4], m3[4], m4[4], r8[4], r9[4], r10[4], s11[4], s12[4];
    float l13[4], l14[4], l15[4], c16[4], c17[4], c18[4], c19[4], emissive[4];
    for (int c = 0; c < 4; ++c)
    {
        m1[c] = vnrm(vf[1][c]);
        m2[c] = vnrm(vf[2][c]);
        m3[c] = vnrm(vf[3][c]);
        m4[c] = vnrm(vf[4][c]);
        r8[c] = vnrm(vf[8][c]);
        r9[c] = vnrm(vf[9][c]);
        r10[c] = vnrm(vf[10][c]);
        s11[c] = vnrm(vf[11][c]);
        s12[c] = vnrm(vf[12][c]);
        l13[c] = vnrm(vf[13][c]);
        l14[c] = vnrm(vf[14][c]);
        l15[c] = vnrm(vf[15][c]);
        c16[c] = vnrm(vf[16][c]);
        c17[c] = vnrm(vf[17][c]);
        c18[c] = vnrm(vf[18][c]);
        c19[c] = vnrm(vf[19][c]);
    }
    float raw25[4];
    load(25, raw25);
    for (int c = 0; c < 4; ++c)
        emissive[c] = vnrm(raw25[c]);

    auto rotate = [&](const float *nv, float *out)
    {
        for (int c = 0; c < 3; ++c)
            out[c] = madd(madd(mul(r8[c], nv[0]), r9[c], nv[1]), r10[c], nv[2]);
    };

    // The STQ register's w lane is never written by the program, so packets carry whatever it held.
    const float stqW = lit ? vf[28][3] : vf[26][3];

    VertexOut vo[kMaxVertices];
    uint32_t clipOr = 0, clipAnd = 63;
    float lastQ = 0.0f, lastP = 0.0f, lastAcc[4] = {}, lastSp[4] = {}, lastCol[4] = {};
    float lastRaw[3][4] = {}; // position, STQ, RGBA of the last vertex as loaded
    float lastL3 = 0.0f, lastLc[3] = {};
    uint32_t lastFlags = 0;
    for (uint32_t k = 0; k < n; ++k)
    {
        const uint32_t a = top + 4u * k;
        float P[4], N[4], T[4], C[4], Pn[4], Nn[4], Tn[4], Cn[4];
        load(a, P);
        load(a + 1u, N);
        load(a + 2u, T);
        load(a + 3u, C);
        for (int c = 0; c < 4; ++c)
        {
            Pn[c] = vnrm(P[c]);
            Nn[c] = vnrm(N[c]);
            Tn[c] = vnrm(T[c]);
            Cn[c] = vnrm(C[c]);
        }
        VertexOut &o = vo[k];
        std::memcpy(lastRaw[0], P, 16);
        std::memcpy(lastRaw[1], T, 16);
        std::memcpy(lastRaw[2], C, 16);

        float nr[3], nrmP = 0.0f;
        rotate(Nn, nr);
        if (normalize)
        {
            nrmP = erleng(nr);
            for (int c = 0; c < 3; ++c)
                nr[c] = mul(nr[c], nrmP);
            lastP = nrmP;
        }

        float clip[4];
        for (int c = 0; c < 4; ++c)
            clip[c] = madd(madd(madd(mul(m1[c], Pn[0]), m2[c], Pn[1]), m3[c], Pn[2]), m4[c], Pn[3]);
        const uint32_t flags = clipFlags(clip, clip[3]);
        const float Q = reciprocal(clip[3]);
        const float Q1 = mul(1.0f, Q);

        float colour[4];
        if (lit)
        {
            float L[4], lc[3], lightAmt[4], col[4];
            for (int c = 0; c < 4; ++c)
                L[c] = madd(madd(mul(l13[c], nr[0]), l14[c], nr[1]), l15[c], nr[2]);
            for (int c = 0; c < 3; ++c)
                lc[c] = madd(madd(madd(mul(c16[c], L[0]), c17[c], L[1]), c18[c], L[2]), c19[c], L[3]);
            for (int c = 0; c < 3; ++c)
                lightAmt[c] = vmax(lc[c], 0.0f);
            lightAmt[3] = r10[3];
            for (int c = 0; c < 4; ++c)
            {
                col[c] = madd(mul(Cn[c], lightAmt[c]), emissive[c], 1.0f);
                colour[c] = vmini(col[c], r8[3]);
                lastCol[c] = col[c];
            }
            lastL3 = L[3];
            for (int c = 0; c < 3; ++c)
                lastLc[c] = lc[c];
        }
        else
        {
            for (int c = 0; c < 3; ++c)
                colour[c] = mul(Cn[c], r9[3]);
            colour[3] = mul(Cn[3], r10[3]);
        }

        float sp[4], scr[4];
        for (int c = 0; c < 3; ++c)
            sp[c] = mul(clip[c], Q);
        sp[3] = lit ? clip[3] : mul(1.0f, clip[3]);
        for (int c = 0; c < 4; ++c)
        {
            lastAcc[c] = mul(sp[c], s11[c]);
            scr[c] = madd(lastAcc[c], s12[c], 1.0f);
        }

        for (int c = 0; c < 4; ++c)
        {
            o.slot0[c] = clip[c];
            o.slot3[c] = colour[c];
            o.copy[c] = C[c];
            o.rgba[c] = vftoi(colour[c], 1.0);
            o.xyzf[c] = vftoi(scr[c], 16.0);
            lastSp[c] = sp[c];
        }
        for (int c = 0; c < 3; ++c)
        {
            o.slot1[c] = nr[c];
            o.stq[c] = mul(Tn[c], Q);
        }
        o.slot1[3] = Q1;
        o.stq[3] = stqW;
        o.flags = static_cast<uint16_t>(flags);
        clipOr |= flags;
        clipAnd &= flags;
        lastQ = Q;
        lastFlags = flags;
    }

    const int16_t extraPass = lo16(202, 0);
    const int16_t next = lo16(29, 0);
    const bool culled = clipAnd != 0u;
    if (!culled)
    {
        if (extraPass != 0)
        {
            ++g_stats.extraPass;
            return kLiftNotLifted;
        }
        if (clipOr != 0u)
        {
            ++g_stats.clipped;
            return kLiftNotLifted;
        }
        if (static_cast<uint16_t>(next) * 8u != 0x27C0u)
        {
            ++g_stats.other;
            return kLiftNotLifted;
        }
    }

    // ---- commit: VU1 data memory -------------------------------------------------------------------
    if (s_settings.verbose && ((g_stats.drawn + g_stats.culled) & 0xFFFFFu) == 0u)
        g_stats.print();
    if (s_settings.corrupt)
        vo[0].rgba[0] = fromBits(bitsOf(vo[0].rgba[0]) ^ 1u);
    vu.flushPipelines();
    for (uint32_t k = 0; k < n; ++k)
    {
        const uint32_t a = top + 4u * k;
        const VertexOut &o = vo[k];
        std::memcpy(qw(a), o.slot0, 16);
        std::memcpy(qw(a + 1u), o.slot1, 16);
        storeU16(a + 2u, 3, o.flags);
        std::memcpy(qw(a + 3u), o.slot3, 16);
        std::memcpy(qw(a + 96u), o.copy, 16);
    }
    // The pipelined loop stores one vertex late; its first three stores are the registers' old
    // contents (the third is overwritten by the GIF tag unless the batch is culled).
    float garbage[3][4];
    if (lit)
    {
        std::memcpy(garbage[0], vf[28], 16);
        std::memcpy(garbage[1], vf[27], 16);
        std::memcpy(garbage[2], vf[26], 16);
    }
    else
    {
        std::memcpy(garbage[0], vf[26], 16);
        std::memcpy(garbage[1], vf[25], 16);
        for (int c = 0; c < 4; ++c)
            garbage[2][c] = vftoi(vnrm(vf[24][c]), 16.0);
    }
    for (int j = 0; j < 3; ++j)
        std::memcpy(qw(base + j), garbage[j], 16);
    for (uint32_t k = 0; k < n; ++k)
    {
        std::memcpy(qw(base + 3u + 3u * k), vo[k].stq, 16);
        std::memcpy(qw(base + 4u + 3u * k), vo[k].rgba, 16);
        std::memcpy(qw(base + 5u + 3u * k), vo[k].xyzf, 16);
    }

    // ---- registers the program leaves behind -------------------------------------------------------
    const VertexOut &last = vo[n - 1u];
    const uint32_t end = top + 4u * n;
    float tag[4];
    load(42, tag);
    const int16_t tagX = lo16(42, 0);
    float rawEnd[4][4];
    for (int j = 0; j < 4; ++j)
        load(end + j, rawEnd[j]);

    auto copy4 = [](float *dst, const float *src) { std::memcpy(dst, src, 16); };
    if (lit)
    {
        copy4(vf[20], lastRaw[0]);
        copy4(vf[24], lastRaw[1]);
        copy4(vf[29], lastRaw[2]);
        if (mode == 0x1B80u)
            copy4(vf[21], rawEnd[1]);
        else
        {
            vf[21][0] = lastLc[0];
            vf[21][1] = lastLc[1];
            vf[21][2] = lastLc[2];
            vf[21][3] = lastL3;
            // The pipeline already rotated (and started ERLENG on) the normal after the batch.
            float g[4], gr[3];
            for (int c = 0; c < 4; ++c)
                g[c] = vnrm(rawEnd[1][c]);
            rotate(g, gr);
            vf[30][0] = gr[0];
            vf[30][1] = gr[1];
            vf[30][2] = gr[2];
            vf[30][3] = rawEnd[1][3];
            lastP = erleng(gr);
        }
        copy4(vf[22], lastSp);
        copy4(vf[23], lastCol);
        vf[25][3] = last.slot1[3];
        copy4(vf[26], last.xyzf);
        copy4(vf[27], last.rgba);
        for (int c = 0; c < 3; ++c)
            vf[28][c] = last.stq[c];
        copy4(vf[31], tag);
    }
    else
    {
        copy4(vf[20], rawEnd[0]);
        copy4(vf[21], rawEnd[1]);
        copy4(vf[23], rawEnd[2]);
        copy4(vf[22], rawEnd[3]);
        copy4(vf[24], last.xyzf);
        copy4(vf[25], last.rgba);
        for (int c = 0; c < 3; ++c)
            vf[26][c] = last.stq[c];
        vf[27][3] = last.slot1[3];
        if (!culled)
            copy4(vf[31], tag);
    }
    copy4(vu.m_state.acc, lastAcc);
    vu.m_state.q = lastQ;
    if (normalize)
        vu.m_state.p = lastP;
    vu.m_state.clip = lastFlags;
    vu.m_workingClip = lastFlags;

    int32_t *vi = vu.m_state.vi;
    vi[1] = outBase;
    vi[9] = next;
    vi[10] = static_cast<int16_t>(n);
    vi[11] = static_cast<int16_t>(end);
    vi[12] = static_cast<int16_t>(clipOr);
    vi[14] = static_cast<int16_t>(outBase + 3 * static_cast<int32_t>(n + 1u));
    vi[15] = extraPass;

    vu.m_cycle += (normalize ? 40u : 30u) + n * (normalize ? 32u : 26u);
    if (culled)
    {
        ++g_stats.culled;
        vi[2] = tagX;
        vi[13] = static_cast<int16_t>(clipAnd);
        vu.m_state.pc = 0x0020u;
        return kLiftEnd;
    }

    // GIF tag (NLOOP += n), kick, swap the output buffers.
    const int16_t kickAt = static_cast<int16_t>(outBase + 2);
    storeU16(base + 2u, 0, static_cast<int16_t>(tagX + static_cast<int32_t>(n)));
    std::memcpy(qw(base + 2u) + 4, &tag[1], 12);
    vu.startXgkick(static_cast<uint16_t>(kickAt));
    vu.m_xgkick.cycleCredit = 0xFFFFFFF0u;
    vu.progressXgkick();
    const int16_t other = lo16(26, 1);
    storeU16(26, 1, outBase);
    storeU16(26, 0, other);
    vi[2] = other;
    vi[13] = kickAt;
    vu.m_state.pc = 0x27E8u;
    ++g_stats.drawn;
    return kLiftEnd;
}
