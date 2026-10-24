#include "gs_gpu_backend.h"

#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_memory.h"

#include "external/glad.h" // raylib's loader: its GL entry points are loaded once the window exists
#include "raylib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

namespace
{
    constexpr uint32_t kPageCount = 512u; // 4 MiB of GS memory in 8 KiB pages
    constexpr double kDepthScale = 1.0 / 4294967296.0;

    bool isDepthPsm(uint32_t psm) { return psm == GS_PSM_Z32 || psm == GS_PSM_Z24 || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S; }
    bool isSixteenBitPsm(uint32_t psm) { return psm == GS_PSM_CT16 || psm == GS_PSM_CT16S || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S; }
    bool isIndexedPsm(uint32_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H || psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    // Page dimensions in pixels for each storage format.
    void pageSize(uint32_t psm, uint32_t &w, uint32_t &h)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            w = 64u, h = 64u;
            break;
        case GS_PSM_T8:
            w = 128u, h = 64u;
            break;
        case GS_PSM_T4:
            w = 128u, h = 128u;
            break;
        default: // 32-bit layouts, including T8H/T4HL/T4HH
            w = 64u, h = 32u;
            break;
        }
    }

    // Pages a w x h rectangle at (x, y) of a buffer at block `bp` with width `bw` (64-pixel units)
    // can touch. Conservative: whole page rows, plus one page when the base is not page aligned.
    void pageSpan(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t y, uint32_t h, uint32_t &first, uint32_t &count)
    {
        uint32_t pw, ph;
        pageSize(psm, pw, ph);
        const uint32_t perRow = std::max<uint32_t>(1u, (std::max<uint32_t>(bw, 1u) * 64u + pw - 1u) / pw);
        const uint32_t rowFirst = y / ph;
        const uint32_t rowLast = (y + std::max<uint32_t>(h, 1u) - 1u) / ph;
        first = bp / 32u + rowFirst * perRow;
        count = (rowLast - rowFirst + 1u) * perRow + ((bp % 32u) ? 1u : 0u);
    }

    uint32_t rgba5551To8888(uint32_t c)
    {
        const uint32_t r = (c & 0x1Fu) << 3, g = ((c >> 5) & 0x1Fu) << 3, b = ((c >> 10) & 0x1Fu) << 3;
        const uint32_t a = (c & 0x8000u) ? 0x80u : 0u;
        return r | (g << 8) | (b << 16) | (a << 24);
    }

    uint32_t rgba8888To5551(uint32_t c)
    {
        return ((c & 0xFFu) >> 3) | ((((c >> 8) & 0xFFu) >> 3) << 5) | ((((c >> 16) & 0xFFu) >> 3) << 10) |
               (((c >> 31) & 1u) << 15);
    }

    const char *kVertexShader = R"(#version 330
layout(location = 0) in vec3 aPos;        // GS window x, y; z as depth in [0, 1]
layout(location = 1) in vec4 aColor;      // 0..255
layout(location = 2) in vec3 aStq;        // S T Q, or U V (1/16 texels) and 1
layout(location = 3) in vec2 aFogZconst;  // fog 0..255, 1 when the primitive has constant Z
uniform vec2 uTargetSize;
noperspective out vec4 vColor;
noperspective out vec3 vStq;
noperspective out float vFog;
noperspective out float vDepth;
flat out float vDepthFlat;
flat out float vDepthConst;
void main()
{
    gl_Position = vec4(aPos.x / uTargetSize.x * 2.0 - 1.0, aPos.y / uTargetSize.y * 2.0 - 1.0, 0.0, 1.0);
    vColor = aColor;
    vStq = aStq;
    vFog = aFogZconst.x;
    vDepth = aPos.z;
    vDepthFlat = aPos.z;
    vDepthConst = aFogZconst.y;
}
)";

    // Mirrors GSCpuBackend's per-pixel pipeline: texture fetch (point or 4-tap bilinear with the
    // GS wrap modes, CLUT in a palette texture), texture function, fog, alpha test, FBA. Blending is
    // left to GL with the factors set up per draw (oBlend carries the blend factor).
    const char *kFragmentShader = R"(#version 330
noperspective in vec4 vColor;
noperspective in vec3 vStq;
noperspective in float vFog;
noperspective in float vDepth;
flat in float vDepthFlat;
flat in float vDepthConst;
uniform int uTme;
uniform usampler2D uVram;       // GS memory as 1024 x 1024 32-bit words
uniform usampler2D uSwizzle;    // page table of the texture's format: [base block % 32][y][x]
uniform sampler2D uPalette;
uniform uint uTbp;              // TBP0 (blocks)
uniform uint uTbw;              // TBW (64-pixel units)
uniform ivec2 uPageSize;
uniform uint uUnpackedBits;     // 32, 16, 8 or 4
uniform uint uBitOffset;        // 24/28 for T8H, T4HL, T4HH
uniform uint uPackedBytes;      // bytes a read covers: 4, 2 or 1
uniform int uNibble;            // 4-bit indices
uniform int uTexFormat;         // 0 raw 32, 1 CT24, 2 CT16, 3 indexed, 4 Z24, 5 Z16
uniform ivec3 uTexa;            // TA0, TA1, AEM
uniform ivec2 uTexSize;
uniform int uFst;
uniform int uSprite;
uniform int uLinear;
uniform ivec2 uWrapMode;
uniform ivec4 uRegion;          // minU maxU minV maxV
uniform int uTfx;
uniform int uTcc;
uniform int uFge;
uniform vec3 uFogColor;
uniform int uAte;
uniform int uAtst;
uniform float uAref;
uniform int uFba;
uniform int uBlendC;            // 0 As, 1 Ad (unsupported, treated as As), 2 FIX
uniform float uBlendFix;
uniform float uSrcScale;        // multiplier for the source colour (|cs| of the blend), < 0 = use alpha factor
uniform int uSrcScaleMode;      // 0: 1, 1: f, 2: 1 - f, 3: 1 + f
layout(location = 0, index = 0) out vec4 oColor;
layout(location = 0, index = 1) out vec4 oBlend;

int wrapCoord(int c, int size, int mode, int lo, int hi)
{
    if (mode == 0) return c & (size - 1);
    if (mode == 1) return clamp(c, 0, size - 1);
    if (mode == 2) return clamp(c, lo, hi);
    return (c & lo) | hi;
}

// GSMem::PixelStorageTraits<psm>::Read, on the GPU copy of GS memory.
uint readRaw(int x, int y)
{
    uint ux = uint(x), uy = uint(y);
    uint pw = uint(uPageSize.x), ph = uint(uPageSize.y);
    uint page = uTbp / 32u + (uy / ph) * ((uTbw * 64u) / pw) + ux / pw;
    uint offs = texelFetch(uSwizzle, ivec2(int(ux % pw), int((uTbp % 32u) * ph + uy % ph)), 0).r;
    uint bits = (page * pw * ph + offs) * uUnpackedBits + uBitOffset;
    uint byteAddr = (bits / 8u) & (4194304u - uPackedBytes);
    uint wordIndex = byteAddr >> 2u;
    uint word = texelFetch(uVram, ivec2(int(wordIndex & 1023u), int(wordIndex >> 10u)), 0).r;
    uint shift = (byteAddr & 3u) * 8u;
    if (uPackedBytes == 4u)
        return (uTexFormat == 1 || uTexFormat == 4) ? (word & 0xFFFFFFu) : word;
    if (uPackedBytes == 2u)
        return (word >> shift) & 0xFFFFu;
    uint b = (word >> shift) & 0xFFu;
    return uNibble != 0 ? ((b >> (bits % 8u)) & 0xFu) : b;
}

// GSCpuBackend::FetchTexelUnlocked: format expansion, TEXA, CLUT.
vec4 fetchTexel(int u, int v)
{
    u = wrapCoord(u, uTexSize.x, uWrapMode.x, uRegion.x, uRegion.y);
    v = wrapCoord(v, uTexSize.y, uWrapMode.y, uRegion.z, uRegion.w);
    uint raw = readRaw(u, v);
    if (uTexFormat == 3)
        return floor(texelFetch(uPalette, ivec2(int(raw & 0xFFu), 0), 0) * 255.0 + 0.5);
    uint r, g, b, a;
    if (uTexFormat == 2 || uTexFormat == 5)
    {
        r = (raw & 0x1Fu) << 3;
        g = ((raw >> 5) & 0x1Fu) << 3;
        b = ((raw >> 10) & 0x1Fu) << 3;
        a = ((raw >> 15) & 1u) << 7;
    }
    else
    {
        r = raw & 0xFFu;
        g = (raw >> 8) & 0xFFu;
        b = (raw >> 16) & 0xFFu;
        a = raw >> 24;
    }
    bool rgbZero = r == 0u && g == 0u && b == 0u;
    if (uTexFormat == 1)
        a = (uTexa.z != 0 && rgbZero) ? 0u : uint(uTexa.x);
    else if (uTexFormat == 2)
        a = (a & 0x80u) != 0u ? uint(uTexa.y) : ((uTexa.z != 0 && rgbZero) ? 0u : uint(uTexa.x));
    return vec4(float(r), float(g), float(b), float(a));
}

vec4 sampleTexture()
{
    float tu, tv;
    if (uFst != 0)
    {
        float su = uSprite != 0 ? floor(vStq.x + 0.5) : floor(vStq.x + 1.0e-3);
        float sv = uSprite != 0 ? floor(vStq.y + 0.5) : floor(vStq.y + 1.0e-3);
        tu = su / 16.0;
        tv = sv / 16.0;
    }
    else
    {
        float q = abs(vStq.z) > 1.0e-8 ? vStq.z : 1.0;
        tu = vStq.x / q * float(uTexSize.x);
        tv = vStq.y / q * float(uTexSize.y);
    }
    if (uLinear == 0)
        return fetchTexel(int(tu), int(tv));
    float su = tu - 0.5, sv = tv - 0.5;
    int u0 = int(floor(su)), v0 = int(floor(sv));
    float fx = su - float(u0), fy = sv - float(v0);
    vec4 c00 = fetchTexel(u0, v0), c10 = fetchTexel(u0 + 1, v0);
    vec4 c01 = fetchTexel(u0, v0 + 1), c11 = fetchTexel(u0 + 1, v0 + 1);
    vec4 top = c00 + (c10 - c00) * fx;
    vec4 bottom = c01 + (c11 - c01) * fx;
    return floor(top + (bottom - top) * fy + 0.5);
}

void main()
{
    vec4 c = floor(vColor + 1.0e-3);
    if (uTme != 0)
    {
        vec4 t = sampleTexture();
        if (uTfx == 0)
            c = vec4(min(floor(t.rgb * c.rgb / 128.0), 255.0), uTcc != 0 ? min(floor(t.a * c.a / 128.0), 255.0) : c.a);
        else if (uTfx == 1)
            c = vec4(t.rgb, uTcc != 0 ? t.a : c.a);
        else if (uTfx == 2)
            c = vec4(min(floor(t.rgb * c.rgb / 128.0) + c.a, 255.0), uTcc != 0 ? min(t.a + c.a, 255.0) : c.a);
        else
            c = vec4(min(floor(t.rgb * c.rgb / 128.0) + c.a, 255.0), uTcc != 0 ? t.a : c.a);
    }
    if (uFge != 0)
    {
        float f = clamp(floor(vFog + 1.0e-3), 0.0, 255.0);
        c.rgb = floor(f * c.rgb / 256.0) + floor((255.0 - f) * uFogColor / 256.0);
    }
    if (uAte != 0)
    {
        bool pass = true;
        if (uAtst == 0) pass = false;
        else if (uAtst == 2) pass = c.a < uAref;
        else if (uAtst == 3) pass = c.a <= uAref;
        else if (uAtst == 4) pass = c.a == uAref;
        else if (uAtst == 5) pass = c.a >= uAref;
        else if (uAtst == 6) pass = c.a > uAref;
        else if (uAtst == 7) pass = c.a != uAref;
        if (!pass) discard;
    }
    float alpha = c.a;
    float f = (uBlendC == 2 ? uBlendFix : alpha) / 128.0;
    float s = uSrcScaleMode == 0 ? 1.0 : uSrcScaleMode == 1 ? f : uSrcScaleMode == 2 ? 1.0 - f : 1.0 + f;
    if (uFba != 0 && c.a < 128.0) alpha = c.a + 128.0;
    oColor = vec4(c.rgb * s / 255.0, alpha / 255.0);
    oBlend = vec4(clamp(f, 0.0, 1.0));
    gl_FragDepth = vDepthConst > 0.5 ? vDepthFlat : vDepth;
}
)";

    const char *kPresentVertexShader = R"(#version 330
void main()
{
    // Full-target triangle.
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

    // Builds the displayed picture from field snapshots (rows of the target are GS lines, row 0 first).
    // Mode 0 copies the current snapshot; mode 1 weaves the current and previous field (GS field
    // rendering: two half-height buffers a line apart) like the CPU FieldWeaver, for 1x; mode 2 is
    // for upscaled fields, which are whole pictures at more lines than a woven frame: the newest
    // field alone, the lower field moved up half a line so consecutive fields line up.
    const char *kPresentFragmentShader = R"(#version 330
uniform sampler2D uCur;
uniform sampler2D uPrev;
uniform int uMode;
uniform int uCurOnEven;
uniform int uScale;
uniform int uLines;
uniform float uShift;           // mode 2: rows to move the field up (half a line for the lower field)
out vec4 oColor;
void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (uMode == 0)
    {
        oColor = vec4(texelFetch(uCur, p, 0).rgb, 1.0);
        return;
    }
    if (uMode == 2)
    {
        vec2 size = vec2(textureSize(uCur, 0));
        oColor = vec4(texture(uCur, (gl_FragCoord.xy + vec2(0.0, uShift)) / size).rgb, 1.0);
        return;
    }
    int line = p.y / uScale;
    int sub = p.y - line * uScale;
    bool fromCur = ((line & 1) == 0) == (uCurOnEven != 0);
    vec4 c = fromCur ? texelFetch(uCur, ivec2(p.x, (line >> 1) * uScale + sub), 0)
                     : texelFetch(uPrev, ivec2(p.x, (line >> 1) * uScale + sub), 0);
    if (!fromCur)
    {
        int above = line > 0 ? line - 1 : line + 1;
        int below = line + 1 < 2 * uLines ? line + 1 : line - 1;
        vec4 a = texelFetch(uCur, ivec2(p.x, (above >> 1) * uScale + sub), 0);
        vec4 b = texelFetch(uCur, ivec2(p.x, (below >> 1) * uScale + sub), 0);
        vec3 lo = min(a.rgb, b.rgb) - 24.0 / 255.0;
        vec3 hi = max(a.rgb, b.rgb) + 24.0 / 255.0;
        if (any(lessThan(c.rgb, lo)) || any(greaterThan(c.rgb, hi)))
            c = (a + b) * 0.5;
    }
    oColor = vec4(c.rgb, 1.0);
}
)";

    GLuint compileShader(GLenum type, const char *source)
    {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char log[4096];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            std::cerr << "[gs-gpu] shader compile failed: " << log << std::endl;
        }
        return shader;
    }

    struct Vertex
    {
        float x, y, z;
        float r, g, b, a;
        float s, t, q;
        float fog, zconst;
    };
}

// ---------------------------------------------------------------------------------------------------
// GL renderer (host thread only)
// ---------------------------------------------------------------------------------------------------

class GSGpuRenderer
{
public:
    void SetVram(uint8_t *vram) { m_vram = vram; }

    explicit GSGpuRenderer(GSCpuBackend &cpu) : m_cpu(cpu)
    {
        m_mirrorVersion.fill(~0ull);
        if (const char *v = std::getenv("PS2X_GS_SCALE"))
            m_scale = static_cast<uint32_t>(std::clamp(std::atoi(v), 1, 8));
    }
    uint32_t Scale() const { return m_scale; }
    ~GSGpuRenderer() { Release(); }

    void EnsureGl();
    void Release();
    void BeginHostWork();
    void EndHostWork();

    void Draw(const GSPrimitiveBatch &batch);
    void FlushDraws();

    // Make GS memory current for pages [first, first + count) (or all of it).
    void DownloadOverlapping(uint32_t first, uint32_t count, const char *why = "other");
    void DownloadAll(const char *why = "other");
    // Colour buffers presentation may read (display buffers, the last draw targets).
    void DownloadForPresent(const GSPresentationRequest &request);
    // GS memory in [first, first + count) was written by someone else.
    void InvalidatePages(uint32_t first, uint32_t count);
    void InvalidateAll();
    void Reset();

    // Presentation straight from the GPU targets (false: the caller uses the CPU path).
    bool PresentFromGpu(const GSPresentationRequest &request);
    bool HasPresentation() const { return m_presentValid; }
    void InvalidatePresentation() { m_presentValid = false; }
    void DrawPresentation(float x, float y, float w, float h);

    // GS features the renderer does not reproduce exactly, by reason (reported by PS2X_GS_GPU_VERIFY
    // and at exit).
    std::map<std::string, uint64_t> unsupported;
    void Note(const char *reason) { ++unsupported[reason]; }
    std::string UnsupportedSummary() const
    {
        std::string out;
        for (const auto &[reason, n] : unsupported)
            out += " " + reason + "=" + std::to_string(n);
        return out.empty() ? " none" : out;
    }

private:
    struct Target
    {
        uint32_t fbp = 0, fbw = 0, psm = 0;
        bool depth = false;
        uint32_t width = 0, height = 0;
        uint32_t firstPage = 0, pageCount = 0;
        GLuint tex = 0;
        bool gpuNewer = false;
        bool vramNewer = true;
    };


    struct DrawKey
    {
        uint32_t colorFbp, colorFbw, colorPsm, zbp, zpsm;
        uint32_t depthMode; // 0 none, 1 test only, 2 test + write
        uint32_t ztst;
        uint32_t scissor[4];
        uint32_t colorMask;
        uint64_t alpha;
        uint32_t abe, colclamp, fba;
        uint32_t tme, fst, sprite, linear, tfx, tcc, wrap, fge, ate, atst, aref;
        uint64_t clamp;
        uint32_t fogColor;
        uint32_t tbp, tbw, texPsm, texa;
        GLuint palette;
        uint32_t texW, texH;
        bool operator==(const DrawKey &o) const { return std::memcmp(this, &o, sizeof(*this)) == 0; }
    };

    Target *FindOrCreateTarget(uint32_t fbp, uint32_t fbw, uint32_t psm, bool depth, uint32_t height);
    void RetireTarget(Target &t);
    void UploadTarget(Target &t);
    void DownloadTarget(Target &t);
    GLuint FramebufferFor(Target *color, Target *depth);
    // Texture reads come from the GPU copy of GS memory: write back targets drawn over the
    // texture's pages, then bring the copy up to date.
    void PrepareTextureSource(const GSDrawState &state);
    void SyncMirror();
    GLuint SwizzleTable(uint32_t psm, uint32_t &pw, uint32_t &ph);
    GLuint GetPalette(const GSDrawState &state);
    void ApplyKey(const DrawKey &key, Target *color, Target *depth);
    void BumpPages(uint32_t first, uint32_t count);
    // 1x staging texture of the given kind, at least w x h.
    GLuint Staging(bool depth, uint32_t w, uint32_t h);
    // Nearest-filtered copy of a whole texture onto another (both colour or both depth).
    void Blit(GLuint src, uint32_t sw, uint32_t sh, GLuint dst, uint32_t dw, uint32_t dh, bool depth);
    void BlitRect(GLuint src, int sx, int sy, int sw, int sh, GLuint dst, int dx, int dy, int dw, int dh, bool depth);
    uint64_t MaxVersion(uint32_t first, uint32_t count) const;

    GSCpuBackend &m_cpu;
    bool m_glReady = false;
    GLuint m_program = 0, m_vao = 0, m_vbo = 0;
    std::unordered_map<std::string, GLint> m_uniforms;
    GLint U(const char *name);

    std::vector<std::unique_ptr<Target>> m_targets;
    std::map<std::pair<GLuint, GLuint>, GLuint> m_fbos;
    uint8_t *m_vram = nullptr;
    GLuint m_mirrorTex = 0;
    std::array<uint64_t, kPageCount> m_mirrorVersion{};
    bool m_mirrorDirty = true;
    std::map<const uint16_t *, GLuint> m_swizzleTables;
    std::map<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>, GLuint> m_palettes;
    std::array<uint64_t, kPageCount> m_pageVersion{};
    uint64_t m_versionCounter = 0;
    uint64_t m_useCounter = 0;

    std::vector<Vertex> m_vertices;
    DrawKey m_key{};
    bool m_haveKey = false;
    Target *m_keyColor = nullptr;
    Target *m_keyDepth = nullptr;
    GLint m_savedViewport[4]{};
    uint32_t m_scale = 1;
    GLuint m_staging[2] = {0, 0};
    uint32_t m_stagingSize[2][2] = {{0, 0}, {0, 0}};
    GLuint m_blitFbo[2] = {0, 0};

    struct FieldSnapshot
    {
        GLuint tex = 0;
        uint32_t w = 0, h = 0, fbp = 0;
        uint64_t tick = 0;
        bool valid = false;
    };
    void CopyToSnapshot(FieldSnapshot &snap, const Target &t, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h, uint32_t fbp,
                        uint64_t tick);
    void RenderOutput(int mode, uint32_t w, uint32_t lines);
    bool DecideFieldOrder();
    void ReadSnapshot(const FieldSnapshot &snap, std::vector<uint8_t> &out);
    void DumpOutput(uint64_t tick);
    FieldSnapshot m_cur, m_prev;
    bool m_curOnEven = true;
    uint32_t m_newFields = 0;
    GLuint m_outputTex = 0, m_outputFbo = 0;
    uint32_t m_outW = 0, m_outH = 0;
    bool m_presentValid = false;
    GLuint m_presentProgram = 0, m_emptyVao = 0;
    uint64_t m_presentCount = 0;
    GLboolean m_cullWasEnabled = GL_FALSE;
    std::vector<uint32_t> m_scratch;
    std::vector<float> m_scratchF;
};

GLuint GSGpuRenderer::Staging(bool depth, uint32_t w, uint32_t h)
{
    const int k = depth ? 1 : 0;
    if (m_staging[k] && m_stagingSize[k][0] == w && m_stagingSize[k][1] == h)
        return m_staging[k];
    if (m_staging[k])
        glDeleteTextures(1, &m_staging[k]);
    glGenTextures(1, &m_staging[k]);
    glBindTexture(GL_TEXTURE_2D, m_staging[k]);
    if (depth)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    m_stagingSize[k][0] = w;
    m_stagingSize[k][1] = h;
    return m_staging[k];
}

void GSGpuRenderer::Blit(GLuint src, uint32_t sw, uint32_t sh, GLuint dst, uint32_t dw, uint32_t dh, bool depth)
{
    BlitRect(src, 0, 0, static_cast<int>(sw), static_cast<int>(sh), dst, 0, 0, static_cast<int>(dw), static_cast<int>(dh), depth);
}

void GSGpuRenderer::BlitRect(GLuint src, int sx, int sy, int sw, int sh, GLuint dst, int dx, int dy, int dw, int dh, bool depth)
{
    if (!m_blitFbo[0])
        glGenFramebuffers(2, m_blitFbo);
    const GLenum attachment = depth ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0;
    GLint drawFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFbo);
    GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_blitFbo[0]);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, attachment, GL_TEXTURE_2D, src, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_blitFbo[1]);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment, GL_TEXTURE_2D, dst, 0);
    if (depth)
    {
        glReadBuffer(GL_NONE);
        glDrawBuffer(GL_NONE);
    }
    else
    {
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
    }
    GLboolean depthMask = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
    glDepthMask(GL_TRUE);
    GLboolean colorMask[4];
    glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glBlitFramebuffer(sx, sy, sx + sw, sy + sh, dx, dy, dx + dw, dy + dh, depth ? GL_DEPTH_BUFFER_BIT : GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
    glDepthMask(depthMask);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(drawFbo));
    if (scissor)
        glEnable(GL_SCISSOR_TEST);
}

GLint GSGpuRenderer::U(const char *name)
{
    auto it = m_uniforms.find(name);
    if (it != m_uniforms.end())
        return it->second;
    const GLint loc = glGetUniformLocation(m_program, name);
    m_uniforms.emplace(name, loc);
    return loc;
}

void GSGpuRenderer::EnsureGl()
{
    if (m_glReady)
        return;
    m_glReady = true;
    const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    m_program = glCreateProgram();
    glAttachShader(m_program, vs);
    glAttachShader(m_program, fs);
    glLinkProgram(m_program);
    GLint ok = 0;
    glGetProgramiv(m_program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetProgramInfoLog(m_program, sizeof(log), nullptr, log);
        std::cerr << "[gs-gpu] program link failed: " << log << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, r)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, s)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, fog)));
    glBindVertexArray(0);

    const GLuint pvs = compileShader(GL_VERTEX_SHADER, kPresentVertexShader);
    const GLuint pfs = compileShader(GL_FRAGMENT_SHADER, kPresentFragmentShader);
    m_presentProgram = glCreateProgram();
    glAttachShader(m_presentProgram, pvs);
    glAttachShader(m_presentProgram, pfs);
    glLinkProgram(m_presentProgram);
    glDeleteShader(pvs);
    glDeleteShader(pfs);
    glGenVertexArrays(1, &m_emptyVao);

    glGenTextures(1, &m_mirrorTex);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_mirrorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, 1024, 1024, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);
    m_mirrorVersion.fill(~0ull);
    m_mirrorDirty = true;
    std::cerr << "[gs-gpu] internal resolution x" << m_scale << " (PS2X_GS_SCALE)" << std::endl;
    std::cerr << "[gs-gpu] OpenGL renderer: " << reinterpret_cast<const char *>(glGetString(GL_RENDERER)) << std::endl;
}

void GSGpuRenderer::Release()
{
    if (!m_glReady)
        return;
    for (auto &t : m_targets)
        glDeleteTextures(1, &t->tex);
    for (auto &[k, p] : m_palettes)
        glDeleteTextures(1, &p);
    for (auto &[k, f] : m_fbos)
        glDeleteFramebuffers(1, &f);
    for (auto &[k, t] : m_swizzleTables)
        glDeleteTextures(1, &t);
    if (m_mirrorTex)
        glDeleteTextures(1, &m_mirrorTex);
    m_mirrorTex = 0;
    m_targets.clear();
    m_palettes.clear();
    m_fbos.clear();
    m_swizzleTables.clear();
    glDeleteBuffers(1, &m_vbo);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteProgram(m_program);
    glDeleteProgram(m_presentProgram);
    glDeleteVertexArrays(1, &m_emptyVao);
    for (GLuint *tex : {&m_staging[0], &m_staging[1], &m_cur.tex, &m_prev.tex, &m_outputTex})
        if (*tex)
            glDeleteTextures(1, tex);
    if (m_blitFbo[0])
        glDeleteFramebuffers(2, m_blitFbo);
    if (m_outputFbo)
        glDeleteFramebuffers(1, &m_outputFbo);
    m_presentValid = false;
    m_glReady = false;
}

void GSGpuRenderer::BeginHostWork()
{
    EnsureGl();
    glGetIntegerv(GL_VIEWPORT, m_savedViewport);
    m_cullWasEnabled = glIsEnabled(GL_CULL_FACE);
    glDisable(GL_CULL_FACE); // the GS draws both windings; raylib enables back-face culling
    m_haveKey = false;       // raylib changed the GL state since the last pump
}

// Leave GL the way raylib expects it (raylib caches its own state).
void GSGpuRenderer::EndHostWork()
{
    FlushDraws();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(m_savedViewport[0], m_savedViewport[1], m_savedViewport[2], m_savedViewport[3]);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(0);
    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    if (m_cullWasEnabled)
        glEnable(GL_CULL_FACE);
}

void GSGpuRenderer::BumpPages(uint32_t first, uint32_t count)
{
    m_mirrorDirty = true;
    const uint64_t v = ++m_versionCounter;
    for (uint32_t i = 0; i < count; ++i)
        m_pageVersion[(first + i) % kPageCount] = v;
}

uint64_t GSGpuRenderer::MaxVersion(uint32_t first, uint32_t count) const
{
    uint64_t v = 0;
    for (uint32_t i = 0; i < count; ++i)
        v = std::max(v, m_pageVersion[(first + i) % kPageCount]);
    return v;
}

static bool rangesOverlap(uint32_t a, uint32_t an, uint32_t b, uint32_t bn)
{
    return a < b + bn && b < a + an;
}

void GSGpuRenderer::DownloadOverlapping(uint32_t first, uint32_t count, const char *why)
{
    for (auto &t : m_targets)
        if (t->gpuNewer && rangesOverlap(t->firstPage, t->pageCount, first, count))
        {
            ++unsupported[std::string("download:") + why];
            DownloadTarget(*t);
        }
}

void GSGpuRenderer::DownloadAll(const char *why)
{
    FlushDraws();
    for (auto &t : m_targets)
        if (t->gpuNewer)
        {
            ++unsupported[std::string("download:") + why];
            DownloadTarget(*t);
        }
}

void GSGpuRenderer::DownloadForPresent(const GSPresentationRequest &request)
{
    FlushDraws();
    auto downloadFrame = [&](uint32_t fbp, uint32_t fbw)
    {
        for (auto &t : m_targets)
            if (!t->depth && t->gpuNewer && t->fbp == fbp && (fbw == 0u || t->fbw == fbw))
                DownloadTarget(*t);
    };
    downloadFrame(static_cast<uint32_t>(request.dispfb1 & 0x1FFu), static_cast<uint32_t>((request.dispfb1 >> 9) & 0x3Fu));
    downloadFrame(static_cast<uint32_t>(request.dispfb2 & 0x1FFu), static_cast<uint32_t>((request.dispfb2 >> 9) & 0x3Fu));
    for (const GSFrameReg &frame : request.contextFrames)
        downloadFrame(frame.fbp, 0u);
    if (request.hasPreferredSource)
        downloadFrame(request.preferredSource.fbp, 0u);
}

void GSGpuRenderer::InvalidatePages(uint32_t first, uint32_t count)
{
    BumpPages(first, count);
    for (auto &t : m_targets)
        if (rangesOverlap(t->firstPage, t->pageCount, first, count))
            t->vramNewer = true;
}

void GSGpuRenderer::InvalidateAll()
{
    InvalidatePages(0, kPageCount);
}

void GSGpuRenderer::Reset()
{
    m_vertices.clear();
    m_haveKey = false;
    if (m_glReady)
    {
        for (auto &t : m_targets)
            glDeleteTextures(1, &t->tex);
        for (auto &[k, f] : m_fbos)
            glDeleteFramebuffers(1, &f);
    }
    m_targets.clear();
    m_fbos.clear();
    BumpPages(0, kPageCount);
}

void GSGpuRenderer::RetireTarget(Target &t)
{
    if (t.gpuNewer)
        DownloadTarget(t);
    if (&t == m_keyColor || &t == m_keyDepth)
    {
        FlushDraws();
        m_haveKey = false;
        m_keyColor = m_keyDepth = nullptr;
    }
    for (auto it = m_fbos.begin(); it != m_fbos.end();)
    {
        if (it->first.first == t.tex || it->first.second == t.tex)
        {
            glDeleteFramebuffers(1, &it->second);
            it = m_fbos.erase(it);
        }
        else
            ++it;
    }
    glDeleteTextures(1, &t.tex);
    t.tex = 0;
}

GSGpuRenderer::Target *GSGpuRenderer::FindOrCreateTarget(uint32_t fbp, uint32_t fbw, uint32_t psm, bool depth, uint32_t height)
{
    uint32_t pw, ph;
    pageSize(psm, pw, ph);
    height = std::min<uint32_t>(2048u, ((std::max<uint32_t>(height, 1u) + ph - 1u) / ph) * ph);
    const uint32_t width = std::max<uint32_t>(fbw, 1u) * 64u;
    const bool sixteen = isSixteenBitPsm(psm);

    Target *match = nullptr;
    for (auto &t : m_targets)
        if (t->fbp == fbp && t->fbw == fbw && t->depth == depth && isSixteenBitPsm(t->psm) == sixteen)
            match = t.get();
    if (match && match->height >= height)
    {
        match->psm = psm;
        return match;
    }

    uint32_t first, count;
    pageSpan(psm, fbp * 32u, fbw, 0u, height, first, count);
    // Anything else on these pages becomes stale: write it back and drop it.
    FlushDraws();
    for (auto it = m_targets.begin(); it != m_targets.end();)
    {
        Target &t = **it;
        if (&t == match || rangesOverlap(t.firstPage, t.pageCount, first, count))
        {
            RetireTarget(t);
            it = m_targets.erase(it);
        }
        else
            ++it;
    }

    auto t = std::make_unique<Target>();
    t->fbp = fbp;
    t->fbw = fbw;
    t->psm = psm;
    t->depth = depth;
    t->width = width;
    t->height = height;
    t->firstPage = first;
    t->pageCount = count;
    glGenTextures(1, &t->tex);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, t->tex);
    if (depth)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width * m_scale, height * m_scale, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width * m_scale, height * m_scale, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);
    t->vramNewer = true;
    m_targets.push_back(std::move(t));
    return m_targets.back().get();
}

void GSGpuRenderer::UploadTarget(Target &t)
{
    Note("upload");
    const uint32_t n = t.width * t.height;
    m_scratch.resize(n);
    m_cpu.ReadRect(t.psm, t.fbp * 32u, t.fbw, 0, 0, t.width, t.height, m_scratch.data());
    glActiveTexture(GL_TEXTURE3);
    // At 1x the data goes straight into the target; scaled targets get it through a 1x staging
    // texture and a nearest blit.
    const GLuint dst = m_scale == 1u ? t.tex : Staging(t.depth, t.width, t.height);
    glBindTexture(GL_TEXTURE_2D, dst);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    if (t.depth)
    {
        m_scratchF.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            m_scratchF[i] = static_cast<float>(static_cast<double>(m_scratch[i]) * kDepthScale);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, t.width, t.height, GL_DEPTH_COMPONENT, GL_FLOAT, m_scratchF.data());
    }
    else
    {
        if (isSixteenBitPsm(t.psm))
            for (uint32_t &c : m_scratch)
                c = rgba5551To8888(c);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, t.width, t.height, GL_RGBA, GL_UNSIGNED_BYTE, m_scratch.data());
    }
    glActiveTexture(GL_TEXTURE0);
    if (m_scale != 1u)
        Blit(dst, t.width, t.height, t.tex, t.width * m_scale, t.height * m_scale, t.depth);
    t.vramNewer = false;
}

void GSGpuRenderer::DownloadTarget(Target &t)
{
    if (m_keyColor == &t || m_keyDepth == &t)
        FlushDraws();
    const uint32_t n = t.width * t.height;
    m_scratch.resize(n);
    GLuint src = t.tex;
    if (m_scale != 1u)
    {
        src = Staging(t.depth, t.width, t.height);
        Blit(t.tex, t.width * m_scale, t.height * m_scale, src, t.width, t.height, t.depth);
    }
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, t.depth ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    if (t.depth)
    {
        glReadBuffer(GL_NONE);
        m_scratchF.resize(n);
        glReadPixels(0, 0, t.width, t.height, GL_DEPTH_COMPONENT, GL_FLOAT, m_scratchF.data());
        for (uint32_t i = 0; i < n; ++i)
        {
            const double z = std::floor(static_cast<double>(m_scratchF[i]) / kDepthScale + 0.5);
            m_scratch[i] = static_cast<uint32_t>(std::min(std::max(z, 0.0), 4294967295.0));
        }
    }
    else
    {
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, t.width, t.height, GL_RGBA, GL_UNSIGNED_BYTE, m_scratch.data());
        if (isSixteenBitPsm(t.psm))
            for (uint32_t &c : m_scratch)
                c = rgba8888To5551(c);
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    m_cpu.WriteRect(t.psm, t.fbp * 32u, t.fbw, 0, 0, t.width, t.height, m_scratch.data());
    BumpPages(t.firstPage, t.pageCount);
    t.gpuNewer = false;
}

GLuint GSGpuRenderer::FramebufferFor(Target *color, Target *depth)
{
    const auto key = std::make_pair(color ? color->tex : 0u, depth ? depth->tex : 0u);
    auto it = m_fbos.find(key);
    if (it != m_fbos.end())
        return it->second;
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    if (color)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color->tex, 0);
    if (depth)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth->tex, 0);
    if (!color)
    {
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
    }
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "[gs-gpu] incomplete framebuffer 0x" << std::hex << status << std::dec << " color="
                  << (color ? std::to_string(color->fbp) + "/" + std::to_string(color->width) + "x" + std::to_string(color->height) +
                                  " psm " + std::to_string(color->psm) + " tex " + std::to_string(color->tex)
                            : std::string("none"))
                  << " depth="
                  << (depth ? std::to_string(depth->fbp) + "/" + std::to_string(depth->width) + "x" + std::to_string(depth->height) +
                                  " psm " + std::to_string(depth->psm) + " tex " + std::to_string(depth->tex)
                            : std::string("none"))
                  << std::endl;
    m_fbos.emplace(key, fbo);
    return fbo;
}

void GSGpuRenderer::PrepareTextureSource(const GSDrawState &state)
{
    const GSTex0Reg &tex0 = state.context.tex0;
    uint32_t first, count;
    pageSpan(tex0.psm, tex0.tbp0, tex0.tbw, 0u, state.textureHeight, first, count);
    DownloadOverlapping(first, count, "texture");
    SyncMirror();
}

void GSGpuRenderer::SyncMirror()
{
    if (!m_mirrorDirty || !m_vram)
        return;
    m_mirrorDirty = false;
    bool flushed = false;
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, m_mirrorTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    for (uint32_t page = 0; page < kPageCount;)
    {
        if (m_mirrorVersion[page] == m_pageVersion[page])
        {
            ++page;
            continue;
        }
        uint32_t end = page;
        while (end < kPageCount && m_mirrorVersion[end] != m_pageVersion[end])
        {
            m_mirrorVersion[end] = m_pageVersion[end];
            ++end;
        }
        if (!flushed)
        {
            FlushDraws(); // batched draws read the old contents
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, m_mirrorTex);
            flushed = true;
        }
        // A page is 8 KiB: two rows of the 1024-word mirror.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, static_cast<GLint>(page * 2u), 1024, static_cast<GLsizei>((end - page) * 2u),
                        GL_RED_INTEGER, GL_UNSIGNED_INT, m_vram + static_cast<size_t>(page) * 8192u);
        page = end;
    }
    glActiveTexture(GL_TEXTURE0);
}

GLuint GSGpuRenderer::SwizzleTable(uint32_t psm, uint32_t &pw, uint32_t &ph)
{
    const uint16_t *table = GSMem::PageTableData(psm, pw, ph);
    if (!table)
    {
        Note("texture-format");
        table = GSMem::PageTableData(GS_PSM_CT32, pw, ph);
    }
    auto it = m_swizzleTables.find(table);
    if (it != m_swizzleTables.end())
        return it->second;
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, pw, 32 * ph, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, table);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);
    m_swizzleTables.emplace(table, tex);
    return tex;
}

GLuint GSGpuRenderer::GetPalette(const GSDrawState &state)
{
    const GSTex0Reg &tex0 = state.context.tex0;
    const uint64_t texa = static_cast<uint64_t>(state.texa.ta0) | (static_cast<uint64_t>(state.texa.ta1) << 8) |
                          (static_cast<uint64_t>(state.texa.aem) << 16);
    const auto key = std::make_tuple(m_cpu.ClutVersion(), static_cast<uint32_t>(tex0.cpsm), static_cast<uint32_t>(tex0.csm),
                                     static_cast<uint32_t>(tex0.csa), static_cast<uint32_t>(tex0.psm), texa);
    auto it = m_palettes.find(key);
    if (it != m_palettes.end())
        return it->second;
    if (m_palettes.size() >= 256u)
    {
        for (auto &[k, p] : m_palettes)
            glDeleteTextures(1, &p);
        m_palettes.clear();
    }
    uint32_t entries[256];
    for (uint32_t i = 0; i < 256u; ++i)
        entries[i] = m_cpu.PaletteEntry(state, static_cast<uint8_t>(i));
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, entries);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);
    m_palettes.emplace(key, tex);
    return tex;
}

void GSGpuRenderer::Draw(const GSPrimitiveBatch &batch)
{
    const GSDrawState &st = batch.state;
    const GSContext &ctx = st.context;
    const uint32_t ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    if (ztst == 0u)
        return; // NEVER

    DrawKey key{};
    key.colorFbp = ctx.frame.fbp;
    key.colorFbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
    key.colorPsm = ctx.frame.psm;
    // ZBUF never set up (not a Z format): no depth buffer at all.
    const bool zbufValid = isDepthPsm(ctx.zbuf.psm);
    if (!zbufValid && (ztst > 1u || !ctx.zbuf.zmask))
        Note("zbuf-not-z");
    const bool depthTest = zbufValid && ztst != 1u;
    const bool depthWrite = zbufValid && !ctx.zbuf.zmask;
    key.depthMode = depthTest ? (depthWrite ? 2u : 1u) : (depthWrite ? 3u : 0u);
    key.zbp = key.depthMode ? ctx.zbuf.zbp : 0u;
    key.zpsm = key.depthMode ? ctx.zbuf.psm : 0u;
    key.ztst = ztst;
    key.scissor[0] = ctx.scissor.x0;
    key.scissor[1] = ctx.scissor.y0;
    key.scissor[2] = ctx.scissor.x1;
    key.scissor[3] = ctx.scissor.y1;
    uint32_t mask = 0;
    for (uint32_t c = 0; c < 4u; ++c)
    {
        const uint32_t byte = (ctx.frame.fbmsk >> (8u * c)) & 0xFFu;
        if (byte == 0u)
            mask |= 1u << c;
        else if (byte != 0xFFu)
        {
            mask |= 1u << c;
            Note("partial-fbmsk");
        }
    }
    if (ctx.frame.psm == GS_PSM_CT24)
        mask &= 7u;
    key.colorMask = mask;
    key.abe = st.prim.abe ? 1u : 0u;
    key.alpha = st.prim.abe ? ctx.alpha : 0u;
    key.colclamp = static_cast<uint32_t>(st.colclamp & 1u);
    key.fba = static_cast<uint32_t>(ctx.fba & 1u);
    key.tme = st.prim.tme ? 1u : 0u;
    key.fst = st.prim.fst ? 1u : 0u;
    key.sprite = st.prim.type == GS_PRIM_SPRITE ? 1u : 0u;
    key.fge = st.prim.fge ? 1u : 0u;
    key.fogColor = static_cast<uint32_t>(st.fogR) | (static_cast<uint32_t>(st.fogG) << 8) | (static_cast<uint32_t>(st.fogB) << 16);
    key.ate = static_cast<uint32_t>(ctx.test & 1u);
    key.atst = static_cast<uint32_t>((ctx.test >> 1) & 7u);
    key.aref = static_cast<uint32_t>((ctx.test >> 4) & 0xFFu);
    if (key.ate && ((ctx.test >> 12) & 3u) != 0u)
        Note("afail");
    if ((ctx.test >> 14) & 1u)
        Note("date");
    if (st.pabe)
        Note("pabe");

    if (st.prim.tme)
    {
        PrepareTextureSource(st);
        key.tbp = ctx.tex0.tbp0;
        key.tbw = ctx.tex0.tbw;
        key.texPsm = ctx.tex0.psm;
        key.texa = static_cast<uint32_t>(st.texa.ta0) | (static_cast<uint32_t>(st.texa.ta1) << 8) |
                   (static_cast<uint32_t>(st.texa.aem) << 16);
        key.palette = isIndexedPsm(ctx.tex0.psm) ? GetPalette(st) : 0u;
        key.texW = st.textureWidth;
        key.texH = st.textureHeight;
        key.linear = st.linearFilter ? 1u : 0u;
        key.tfx = ctx.tex0.tfx;
        key.tcc = ctx.tex0.tcc;
        key.clamp = ctx.clamp;
    }

    // Size the targets to what the scissor allows to be drawn.
    const uint32_t height = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
    if (!m_haveKey || !(key == m_key))
    {
        FlushDraws();
        if (key.depthMode && mask != 0u)
        {
            uint32_t cf, cc, zf, zc;
            pageSpan(key.colorPsm, key.colorFbp * 32u, key.colorFbw, 0u, height, cf, cc);
            pageSpan(key.zpsm, key.zbp * 32u, key.colorFbw, 0u, height, zf, zc);
            if (rangesOverlap(cf, cc, zf, zc))
            {
                Note("color-depth-overlap");
                return;
            }
        }
        Target *depth = key.depthMode ? FindOrCreateTarget(key.zbp, key.colorFbw, key.zpsm, true, height) : nullptr;
        Target *color = (mask != 0u) ? FindOrCreateTarget(key.colorFbp, key.colorFbw, key.colorPsm, false, height) : nullptr;
        if (color && color->vramNewer)
            UploadTarget(*color);
        if (depth && depth->vramNewer)
            UploadTarget(*depth);
        // Creating the depth target may have retired the colour one (same pages) and vice versa.
        if (!color && !depth)
            return;
        m_key = key;
        m_haveKey = true;
        m_keyColor = color;
        m_keyDepth = depth;
        ApplyKey(key, color, depth);
    }

    const int ofx = ctx.xyoffset.ofx >> 4;
    const int ofy = ctx.xyoffset.ofy >> 4;
    auto makeVertex = [&](const GSVertex &v, const GSVertex &colorFrom, bool zconst, float zvalue)
    {
        Vertex out{};
        out.x = v.x - static_cast<float>(ofx);
        out.y = v.y - static_cast<float>(ofy);
        out.z = zvalue;
        const GSVertex &c = st.prim.iip ? v : colorFrom;
        out.r = c.r;
        out.g = c.g;
        out.b = c.b;
        out.a = c.a;
        if (st.prim.fst)
        {
            out.s = static_cast<float>(v.u);
            out.t = static_cast<float>(v.v);
            out.q = 1.0f;
        }
        else
        {
            out.s = v.s;
            out.t = v.t;
            out.q = v.q;
        }
        out.fog = v.fog;
        out.zconst = zconst ? 1.0f : 0.0f;
        return out;
    };

    switch (st.prim.type)
    {
    case GS_PRIM_SPRITE:
    {
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        // GSCpuBackend::DrawSprite: integer corners, colour/Z/fog of the second vertex,
        // texture coordinates linear across the rectangle.
        float x0 = std::trunc(v0.x) - static_cast<float>(ofx), y0 = std::trunc(v0.y) - static_cast<float>(ofy);
        float x1 = std::trunc(v1.x) - static_cast<float>(ofx), y1 = std::trunc(v1.y) - static_cast<float>(ofy);
        float s0, t0, s1, t1;
        if (st.prim.fst)
        {
            s0 = static_cast<float>(v0.u >> 4) * 16.0f;
            t0 = static_cast<float>(v0.v >> 4) * 16.0f;
            s1 = static_cast<float>(v1.u >> 4) * 16.0f;
            t1 = static_cast<float>(v1.v >> 4) * 16.0f;
        }
        else
        {
            const float q0 = std::fabs(v0.q) > 1.0e-8f ? v0.q : 1.0f;
            const float q1 = std::fabs(v1.q) > 1.0e-8f ? v1.q : 1.0f;
            s0 = v0.s / q0, t0 = v0.t / q0, s1 = v1.s / q1, t1 = v1.t / q1;
        }
        if (x0 > x1)
        {
            std::swap(x0, x1);
            std::swap(s0, s1);
        }
        if (y0 > y1)
        {
            std::swap(y0, y1);
            std::swap(t0, t1);
        }
        if (x1 - x0 < 1.0f)
            x1 = x0 + 1.0f;
        if (y1 - y0 < 1.0f)
            y1 = y0 + 1.0f;
        const float z = static_cast<float>(static_cast<double>(static_cast<uint32_t>(v1.z)) * kDepthScale);
        Vertex corner{};
        corner.z = z;
        corner.r = v1.r, corner.g = v1.g, corner.b = v1.b, corner.a = v1.a;
        corner.q = 1.0f;
        corner.fog = v1.fog;
        corner.zconst = 1.0f;
        auto at = [&](float x, float y, float s, float t)
        {
            Vertex v = corner;
            v.x = x, v.y = y, v.s = s, v.t = t;
            m_vertices.push_back(v);
        };
        at(x0, y0, s0, t0);
        at(x1, y0, s1, t0);
        at(x0, y1, s0, t1);
        at(x1, y0, s1, t0);
        at(x1, y1, s1, t1);
        at(x0, y1, s0, t1);
        break;
    }
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
    {
        const GSVertex &v0 = batch.vertices[0], &v1 = batch.vertices[1], &v2 = batch.vertices[2];
        const bool zconst = v0.z == v1.z && v1.z == v2.z;
        auto depthOf = [](const GSVertex &v)
        { return static_cast<float>(std::floor(v.z + 0.5) * kDepthScale); };
        m_vertices.push_back(makeVertex(v0, v2, zconst, depthOf(v0)));
        m_vertices.push_back(makeVertex(v1, v2, zconst, depthOf(v1)));
        m_vertices.push_back(makeVertex(v2, v2, zconst, depthOf(v2)));
        break;
    }
    default:
        Note(st.prim.type == GS_PRIM_POINT ? "points" : "lines");
        break;
    }
}

void GSGpuRenderer::ApplyKey(const DrawKey &key, Target *color, Target *depth)
{
    const uint32_t width = color ? color->width : depth->width;
    const uint32_t height = color ? color->height : depth->height;
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferFor(color, depth));
    glViewport(0, 0, width * m_scale, height * m_scale);
    glEnable(GL_SCISSOR_TEST);
    const uint32_t sx0 = std::min(key.scissor[0], width), sy0 = std::min(key.scissor[1], height);
    const uint32_t sx1 = std::min(key.scissor[2] + 1u, width), sy1 = std::min(key.scissor[3] + 1u, height);
    glScissor(sx0 * m_scale, sy0 * m_scale, sx1 > sx0 ? (sx1 - sx0) * m_scale : 0, sy1 > sy0 ? (sy1 - sy0) * m_scale : 0);
    glColorMask(color && (key.colorMask & 1u), color && (key.colorMask & 2u), color && (key.colorMask & 4u),
                color && (key.colorMask & 8u));
    if (depth)
    {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(key.ztst == 1u ? GL_ALWAYS : key.ztst == 2u ? GL_GEQUAL : GL_GREATER);
        glDepthMask((key.depthMode == 2u || key.depthMode == 3u) ? GL_TRUE : GL_FALSE);
    }
    else
    {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
    }

    glUseProgram(m_program);
    glUniform2f(U("uTargetSize"), static_cast<float>(width), static_cast<float>(height));
    glUniform1i(U("uTme"), static_cast<GLint>(key.tme));
    GLuint swizzle = 0;
    if (key.tme)
    {
        uint32_t pw = 64, ph = 32;
        swizzle = SwizzleTable(key.texPsm, pw, ph);
        const uint32_t psm = key.texPsm;
        const bool nibble = psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
        const uint32_t unpacked = (psm == GS_PSM_T4) ? 4u : (psm == GS_PSM_T8) ? 8u : isSixteenBitPsm(psm) ? 16u : 32u;
        const uint32_t bitOffset = (psm == GS_PSM_T8H || psm == GS_PSM_T4HL) ? 24u : (psm == GS_PSM_T4HH) ? 28u : 0u;
        const uint32_t packed = (psm == GS_PSM_T4 || psm == GS_PSM_T8 || psm == GS_PSM_T8H || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH) ? 1u
                                : isSixteenBitPsm(psm) ? 2u : 4u;
        const int format = isIndexedPsm(psm) ? 3 : psm == GS_PSM_CT24 ? 1 : (psm == GS_PSM_CT16 || psm == GS_PSM_CT16S) ? 2
                           : psm == GS_PSM_Z24 ? 4 : (psm == GS_PSM_Z16 || psm == GS_PSM_Z16S) ? 5 : 0;
        glUniform1ui(U("uTbp"), key.tbp);
        glUniform1ui(U("uTbw"), key.tbw);
        glUniform2i(U("uPageSize"), static_cast<GLint>(pw), static_cast<GLint>(ph));
        glUniform1ui(U("uUnpackedBits"), unpacked);
        glUniform1ui(U("uBitOffset"), bitOffset);
        glUniform1ui(U("uPackedBytes"), packed);
        glUniform1i(U("uNibble"), nibble ? 1 : 0);
        glUniform1i(U("uTexFormat"), format);
        glUniform3i(U("uTexa"), static_cast<GLint>(key.texa & 0xFFu), static_cast<GLint>((key.texa >> 8) & 0xFFu),
                    static_cast<GLint>((key.texa >> 16) & 1u));
    }
    glUniform1i(U("uFst"), static_cast<GLint>(key.fst));
    glUniform1i(U("uSprite"), static_cast<GLint>(key.sprite));
    glUniform1i(U("uLinear"), static_cast<GLint>(key.linear));
    glUniform2i(U("uTexSize"), static_cast<GLint>(std::max<uint32_t>(key.texW, 1u)), static_cast<GLint>(std::max<uint32_t>(key.texH, 1u)));
    glUniform2i(U("uWrapMode"), static_cast<GLint>(key.clamp & 3u), static_cast<GLint>((key.clamp >> 2) & 3u));
    glUniform4i(U("uRegion"), static_cast<GLint>((key.clamp >> 4) & 0x3FFu), static_cast<GLint>((key.clamp >> 14) & 0x3FFu),
                static_cast<GLint>((key.clamp >> 24) & 0x3FFu), static_cast<GLint>((key.clamp >> 34) & 0x3FFu));
    glUniform1i(U("uTfx"), static_cast<GLint>(key.tfx));
    glUniform1i(U("uTcc"), static_cast<GLint>(key.tcc));
    glUniform1i(U("uFge"), static_cast<GLint>(key.fge));
    glUniform3f(U("uFogColor"), static_cast<float>(key.fogColor & 0xFFu), static_cast<float>((key.fogColor >> 8) & 0xFFu),
                static_cast<float>((key.fogColor >> 16) & 0xFFu));
    glUniform1i(U("uAte"), static_cast<GLint>(key.ate));
    glUniform1i(U("uAtst"), static_cast<GLint>(key.atst));
    glUniform1f(U("uAref"), static_cast<float>(key.aref));
    glUniform1i(U("uFba"), static_cast<GLint>(key.fba));
    glUniform1i(U("uVram"), 0);
    glUniform1i(U("uSwizzle"), 1);
    glUniform1i(U("uPalette"), 2);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, key.tme ? m_mirrorTex : 0u);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, swizzle);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, key.palette);
    glActiveTexture(GL_TEXTURE0);

    // Blending: (A - B) * C / 128 + D with A, B, D in {Cs, Cd, 0} and C in {As, Ad, FIX}, as
    // cs * Cs + cd * Cd. The shader outputs Cs * |cs| and the blend factor f = C / 128; GL applies
    // the sign (equation) and |cd| (destination factor).
    if (!key.abe)
    {
        glDisable(GL_BLEND);
        glUniform1i(U("uSrcScaleMode"), 0);
        glUniform1i(U("uBlendC"), 0);
        glUniform1f(U("uBlendFix"), 128.0f);
        return;
    }
    const uint32_t a = key.alpha & 3u, b = (key.alpha >> 2) & 3u, c = (key.alpha >> 4) & 3u, d = (key.alpha >> 6) & 3u;
    const float fix = static_cast<float>((key.alpha >> 32) & 0xFFu);
    if (c == 1u)
        Note("blend-dest-alpha");
    // Coefficients in terms of f: value = k0 + k1 * f.
    auto coeff = [&](uint32_t which, int &k0, int &k1)
    {
        k0 = (d == which) ? 1 : 0;
        k1 = (a == which ? 1 : 0) - (b == which ? 1 : 0);
    };
    int s0, s1, d0, d1;
    coeff(0u, s0, s1);
    coeff(1u, d0, d1);
    // Source scale mode from (s0, s1): 1, f, 1-f, 1+f, or their negations.
    int srcSign = 1, srcMode = 0;
    if (s0 == 0 && s1 == 0)
        srcMode = -1;
    else if (s0 == 1 && s1 == 0)
        srcMode = 0;
    else if (s0 == 0)
        srcMode = 1, srcSign = s1;
    else if (s1 == -1)
        srcMode = 2;
    else
        srcMode = 3;
    int dstSign = 1;
    GLenum dstFactor = GL_ZERO;
    if (d0 == 0 && d1 == 0)
        dstFactor = GL_ZERO;
    else if (d0 == 1 && d1 == 0)
        dstFactor = GL_ONE;
    else if (d0 == 0)
        dstFactor = GL_SRC1_ALPHA, dstSign = d1;
    else if (d1 == -1)
        dstFactor = GL_ONE_MINUS_SRC1_ALPHA;
    else
    {
        dstFactor = GL_ONE; // 1 + f cannot be expressed
        Note("blend-1+f");
    }
    GLenum equation = GL_FUNC_ADD;
    GLenum srcFactor = srcMode < 0 ? GL_ZERO : GL_ONE;
    if (srcSign < 0 && dstSign > 0)
        equation = GL_FUNC_REVERSE_SUBTRACT; // dst - src
    else if (srcSign > 0 && dstSign < 0)
        equation = GL_FUNC_SUBTRACT;         // src - dst
    else if (srcSign < 0 && dstSign < 0)
    {
        srcFactor = GL_ZERO; // everything negative clamps to 0
        dstFactor = GL_ZERO;
    }
    glEnable(GL_BLEND);
    glBlendEquationSeparate(equation, GL_FUNC_ADD);
    glBlendFuncSeparate(srcFactor, dstFactor, GL_ONE, GL_ZERO);
    glUniform1i(U("uSrcScaleMode"), srcMode < 0 ? 0 : srcMode);
    glUniform1i(U("uBlendC"), static_cast<GLint>(c));
    glUniform1f(U("uBlendFix"), fix);
}

void GSGpuRenderer::FlushDraws()
{
    if (m_vertices.empty())
        return;
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(m_vertices.size() * sizeof(Vertex)), m_vertices.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_vertices.size()));
    glBindVertexArray(0);
    if (m_keyColor)
        m_keyColor->gpuNewer = true;
    if (m_keyDepth && (m_key.depthMode == 2u || m_key.depthMode == 3u))
        m_keyDepth->gpuNewer = true;
    m_vertices.clear();
}

void GSGpuRenderer::CopyToSnapshot(FieldSnapshot &snap, const Target &t, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h,
                                   uint32_t fbp, uint64_t tick)
{
    const uint32_t sw = w * m_scale, sh = h * m_scale;
    if (!snap.tex || snap.w != w || snap.h != h)
    {
        if (snap.tex)
            glDeleteTextures(1, &snap.tex);
        glGenTextures(1, &snap.tex);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, snap.tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, sw, sh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glActiveTexture(GL_TEXTURE0);
    }
    BlitRect(t.tex, static_cast<int>(ox * m_scale), static_cast<int>(oy * m_scale), static_cast<int>(sw), static_cast<int>(sh), snap.tex, 0, 0,
             static_cast<int>(sw), static_cast<int>(sh), false);
    snap.w = w;
    snap.h = h;
    snap.fbp = fbp;
    snap.tick = tick;
    snap.valid = true;
}

void GSGpuRenderer::ReadSnapshot(const FieldSnapshot &snap, std::vector<uint8_t> &out)
{
    const GLuint staging = Staging(false, snap.w, snap.h);
    Blit(snap.tex, snap.w * m_scale, snap.h * m_scale, staging, snap.w, snap.h, false);
    out.resize(static_cast<size_t>(snap.w) * snap.h * 4u);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_blitFbo[1]);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, snap.w, snap.h, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

// FieldWeaver::CombEnergy on both orders (green channel, every fourth pixel of odd lines).
bool GSGpuRenderer::DecideFieldOrder()
{
    std::vector<uint8_t> cur, prev;
    ReadSnapshot(m_cur, cur);
    ReadSnapshot(m_prev, prev);
    const uint32_t w = m_cur.w, h = m_cur.h;
    const size_t rowBytes = static_cast<size_t>(w) * 4u;
    auto comb = [&](const std::vector<uint8_t> &top, const std::vector<uint8_t> &bottom)
    {
        uint64_t energy = 0;
        for (uint32_t y = 1; y + 1u < h; y += 2u)
        {
            const uint8_t *a = top.data() + y * rowBytes;
            const uint8_t *b = bottom.data() + y * rowBytes;
            const uint8_t *c = top.data() + (y + 1u) * rowBytes;
            for (uint32_t x = 1; x < w * 4u; x += 16u)
                energy += static_cast<uint64_t>(std::abs(2 * b[x] - a[x] - c[x]));
        }
        return energy;
    };
    return comb(cur, prev) <= comb(prev, cur);
}

void GSGpuRenderer::RenderOutput(int mode, uint32_t w, uint32_t lines)
{
    const uint32_t outW = w * m_scale, outH = (mode == 1 ? 2u * lines : lines) * m_scale;
    if (!m_outputTex || m_outW != outW || m_outH != outH)
    {
        if (m_outputTex)
            glDeleteTextures(1, &m_outputTex);
        if (!m_outputFbo)
            glGenFramebuffers(1, &m_outputFbo);
        glGenTextures(1, &m_outputTex);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, m_outputTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, outW, outH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glActiveTexture(GL_TEXTURE0);
        glBindFramebuffer(GL_FRAMEBUFFER, m_outputFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_outputTex, 0);
        m_outW = outW;
        m_outH = outH;
    }
    m_haveKey = false;
    glBindFramebuffer(GL_FRAMEBUFFER, m_outputFbo);
    glViewport(0, 0, outW, outH);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glUseProgram(m_presentProgram);
    glUniform1i(glGetUniformLocation(m_presentProgram, "uCur"), 0);
    glUniform1i(glGetUniformLocation(m_presentProgram, "uPrev"), 1);
    glUniform1i(glGetUniformLocation(m_presentProgram, "uMode"), mode);
    glUniform1i(glGetUniformLocation(m_presentProgram, "uCurOnEven"), m_curOnEven ? 1 : 0);
    glUniform1i(glGetUniformLocation(m_presentProgram, "uScale"), static_cast<GLint>(m_scale));
    glUniform1i(glGetUniformLocation(m_presentProgram, "uLines"), static_cast<GLint>(lines));
    glUniform1f(glGetUniformLocation(m_presentProgram, "uShift"), m_curOnEven ? 0.0f : 0.5f * static_cast<float>(m_scale));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_cur.tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mode == 1 ? m_prev.tex : m_cur.tex);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_emptyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GSGpuRenderer::DumpOutput(uint64_t tick)
{
    static const char *dumpDir = std::getenv("PS2X_DUMP_FRAMES");
    if (!dumpDir || !*dumpDir)
        return;
    static const uint32_t dumpEvery = []()
    {
        const char *every = std::getenv("PS2X_DUMP_EVERY");
        const long value = every ? std::strtol(every, nullptr, 10) : 60;
        return static_cast<uint32_t>(value > 0 ? value : 60);
    }();
    if ((m_presentCount % dumpEvery) != 0u)
        return;
    std::vector<uint8_t> pixels(static_cast<size_t>(m_outW) * m_outH * 4u);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_outputFbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, m_outW, m_outH, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    Image image{};
    image.data = pixels.data();
    image.width = static_cast<int>(m_outW);
    image.height = static_cast<int>(m_outH);
    image.mipmaps = 1;
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/frame_%06llu.png", dumpDir, static_cast<unsigned long long>(tick));
    ExportImage(image, path);
}

bool GSGpuRenderer::PresentFromGpu(const GSPresentationRequest &request)
{
    const bool en1 = (request.pmode & 1u) != 0u, en2 = (request.pmode & 2u) != 0u;
    const bool same = request.dispfb1 == request.dispfb2 && request.display1 == request.display2;
    if (en1 == en2 && !(en1 && same))
    {
        Note("present-two-circuits");
        return false;
    }
    const uint64_t dispfb = en1 ? request.dispfb1 : request.dispfb2;
    const uint64_t display = en1 ? request.display1 : request.display2;
    const uint32_t fbp = static_cast<uint32_t>(dispfb & 0x1FFu), fbw = static_cast<uint32_t>((dispfb >> 9) & 0x3Fu);
    const uint32_t psm = static_cast<uint32_t>((dispfb >> 15) & 0x1Fu);
    const uint32_t ox = static_cast<uint32_t>((dispfb >> 32) & 0x7FFu), oy = static_cast<uint32_t>((dispfb >> 43) & 0x7FFu);
    // decodeDisplaySize (gs_cpu_backend.cpp)
    const uint32_t dw = static_cast<uint32_t>((display >> 32) & 0x0FFFu), dh = static_cast<uint32_t>((display >> 44) & 0x07FFu);
    const uint32_t magh = static_cast<uint32_t>((display >> 23) & 0x0Fu);
    uint32_t width = (dw + 1u) / (magh + 1u), height = dh + 1u;
    if (width < 64u || height < 64u)
        width = 640u, height = 448u;
    width = std::min<uint32_t>(width, 640u);
    height = std::min<uint32_t>(height, 512u);
    if (psm != GS_PSM_CT32 && psm != GS_PSM_CT24)
    {
        Note("present-psm");
        return false;
    }
    if (request.hasPreferredSource && request.preferredDestFbp == fbp)
    {
        Note("present-preferred-source");
        return false;
    }
    Target *target = nullptr;
    for (auto &t : m_targets)
        if (!t->depth && t->fbp == fbp && t->fbw == fbw)
            target = t.get();
    if (!target || ox + width > target->width || oy + height > target->height)
        return false; // not drawn on the GPU (movies, uploads): the CPU path reads GS memory
    FlushDraws();
    if (target->vramNewer)
        UploadTarget(*target);

    static const bool s_deinterlace = []()
    {
        const char *v = std::getenv("PS2X_DEINTERLACE");
        return !(v && v[0] == '0');
    }();
    const uint64_t tick = request.vsyncTick;
    if (!s_deinterlace || height > 256u)
    {
        CopyToSnapshot(m_cur, *target, ox, oy, width, height, fbp, tick);
        m_prev.valid = false;
        RenderOutput(0, width, height);
    }
    else
    {
        const bool sameShape = m_cur.valid && m_cur.w == width && m_cur.h == height;
        if (!sameShape || fbp != m_cur.fbp)
        {
            if (sameShape)
                std::swap(m_cur, m_prev);
            else
                m_prev.valid = false;
            CopyToSnapshot(m_cur, *target, ox, oy, width, height, fbp, tick);
            if (m_prev.valid && (m_newFields++ % 16u) == 0u)
                m_curOnEven = DecideFieldOrder();
        }
        else
        {
            const uint64_t heldSince = m_cur.tick;
            CopyToSnapshot(m_cur, *target, ox, oy, width, height, fbp, heldSince);
        }
        const bool pair = m_prev.valid && tick - m_cur.tick <= 8u;
        RenderOutput(pair ? (m_scale == 1u ? 1 : 2) : 0, width, height);
    }
    ++m_presentCount;
    DumpOutput(tick);
    m_presentValid = true;
    return true;
}

void GSGpuRenderer::DrawPresentation(float x, float y, float w, float h)
{
    if (!m_presentValid || !m_outputTex)
        return;
    Texture2D tex{};
    tex.id = m_outputTex;
    tex.width = static_cast<int>(m_outW);
    tex.height = static_cast<int>(m_outH);
    tex.mipmaps = 1;
    tex.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    DrawTexturePro(tex, Rectangle{0.0f, 0.0f, static_cast<float>(m_outW), static_cast<float>(m_outH)}, Rectangle{x, y, w, h},
                   Vector2{0.0f, 0.0f}, 0.0f, WHITE);
}

// ---------------------------------------------------------------------------------------------------
// GSGpuBackend: command queue (guest thread) and execution (host thread)
// ---------------------------------------------------------------------------------------------------

namespace
{
    bool envFlag(const char *name)
    {
        const char *v = std::getenv(name);
        return v && *v && std::strcmp(v, "0") != 0;
    }

    struct VerifyStats
    {
        uint64_t frames = 0, badFrames = 0;
        double sumMeanAbs = 0.0;
        std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    } g_verify;
}

bool GSGpuBackend::Requested()
{
    const char *v = std::getenv("PS2X_GS_BACKEND");
    return v && std::strcmp(v, "gpu") == 0;
}

GSGpuBackend::GSGpuBackend()
{
    m_renderer = std::make_unique<GSGpuRenderer>(m_cpu);
    if (envFlag("PS2X_GS_GPU_VERIFY"))
    {
        m_shadow = std::make_unique<GSCpuBackend>();
        std::cerr << "[gs-gpu] verifying against the CPU rasterizer (PS2X_GS_GPU_VERIFY)" << std::endl;
    }
    std::cerr << "[gs-gpu] GPU GS backend selected (PS2X_GS_BACKEND=gpu)" << std::endl;
}

GSGpuBackend::~GSGpuBackend() = default;

bool GSGpuBackend::OnHostThread() const
{
    return !m_hostStarted.load(std::memory_order_acquire) || std::this_thread::get_id() == m_hostThread;
}

void GSGpuBackend::PublishLocked()
{
    if (m_pending.empty())
        return;
    m_ready.push_back(std::move(m_pending));
    m_pending.clear();
    m_pending.reserve(1024);
    m_workCv.notify_one();
}

void GSGpuBackend::Enqueue(Command &&command)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    const bool publish = command.type != CommandType::Submit;
    m_pending.push_back(std::move(command));
    ++m_queued;
    if (publish || m_pending.size() >= 512u)
    {
        PublishLocked();
        if (m_ready.size() > kMaxReadyBatches && m_hostStarted.load(std::memory_order_acquire) && !OnHostThread())
            m_doneCv.wait(lock, [&]() { return m_ready.size() <= kMaxReadyBatches / 2u || m_stopped; });
    }
}

void GSGpuBackend::RunOnHost(const std::function<void()> &fn) const
{
    auto *self = const_cast<GSGpuBackend *>(this);
    if (!m_hostStarted.load(std::memory_order_acquire))
    {
        // Start-up, before the host loop runs: nothing is queued yet, so run in place.
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_queued == m_completed && m_pending.empty())
        {
            lock.unlock();
            fn();
            return;
        }
    }
    if (std::this_thread::get_id() == m_hostThread)
    {
        // Drain everything queued so far in order, then run.
        for (;;)
        {
            std::vector<Command> batch;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                self->PublishLocked();
                if (m_ready.empty())
                    break;
                batch = std::move(m_ready.front());
                m_ready.pop_front();
            }
            self->ExecuteBatch(batch);
        }
        fn();
        return;
    }
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_stopped)
        return;
    Command command;
    command.type = CommandType::Call;
    command.extra = std::make_unique<CommandExtra>();
    command.extra->call = fn;
    self->m_pending.push_back(std::move(command));
    const uint64_t target = ++self->m_queued;
    self->PublishLocked();
    m_doneCv.wait(lock, [&]() { return m_completed >= target || m_stopped; });
}

void GSGpuBackend::ExecuteBatch(std::vector<Command> &commands)
{
    for (Command &command : commands)
        Execute(command);
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_completed += commands.size();
    }
    m_doneCv.notify_all();
}

void GSGpuBackend::PumpHostThread(std::chrono::steady_clock::time_point deadline)
{
    if (!m_hostStarted.load(std::memory_order_acquire))
    {
        m_hostThread = std::this_thread::get_id();
        m_hostStarted.store(true, std::memory_order_release);
    }
    m_renderer->BeginHostWork();
    for (;;)
    {
        std::vector<Command> batch;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_stopped)
                break;
            if (m_ready.empty())
                m_workCv.wait_until(lock, deadline, [&]() { return !m_ready.empty() || m_stopped; });
            if (m_ready.empty() || m_stopped)
                break;
            batch = std::move(m_ready.front());
            m_ready.pop_front();
        }
        ExecuteBatch(batch);
        if (std::chrono::steady_clock::now() >= deadline)
            break;
    }
    m_renderer->EndHostWork();
    static auto s_lastSummary = std::chrono::steady_clock::now();
    static const bool s_stats = envFlag("PS2X_GS_GPU_STATS");
    if (s_stats && std::chrono::steady_clock::now() - s_lastSummary > std::chrono::seconds(10))
    {
        s_lastSummary = std::chrono::steady_clock::now();
        std::cerr << "[gs-gpu] counts:" << m_renderer->UnsupportedSummary() << std::endl;
    }
}

bool GSGpuBackend::HasPresentation() const
{
    return m_renderer->HasPresentation();
}

void GSGpuBackend::DrawPresentation(float x, float y, float w, float h)
{
    m_renderer->DrawPresentation(x, y, w, h);
}

void GSGpuBackend::ShutdownHostThread()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopped = true;
    }
    m_doneCv.notify_all();
    m_workCv.notify_all();
    std::cerr << "[gs-gpu] not reproduced exactly:" << m_renderer->UnsupportedSummary() << std::endl;
    m_renderer->Release();
}

void GSGpuBackend::Execute(Command &command)
{
    GSGpuRenderer &r = *m_renderer;
    switch (command.type)
    {
    case CommandType::Submit:
        r.Draw(command.batch);
        if (m_shadow)
            m_shadow->Submit(command.batch);
        break;
    case CommandType::LoadClut:
    {
        uint32_t first, count;
        pageSpan(command.extra->tex0.cpsm, command.extra->tex0.cbp, 1u, 0u, 16u, first, count);
        r.DownloadOverlapping(first, count, "clut");
        m_cpu.LoadClut(command.extra->tex0, command.extra->texclut);
        if (m_shadow)
            m_shadow->LoadClut(command.extra->tex0, command.extra->texclut);
        break;
    }
    case CommandType::BeginTransfer:
    {
        const GSTransferCommand &t = command.extra->transfer;
        r.FlushDraws();
        if (t.direction == 1u || t.direction == 2u)
        {
            uint32_t first, count;
            pageSpan(t.bitbltbuf.spsm, t.bitbltbuf.sbp, t.bitbltbuf.sbw, t.trxpos.ssay, t.trxreg.rrh, first, count);
            r.DownloadOverlapping(first, count, "transfer-source");
        }
        uint32_t first, count;
        pageSpan(t.bitbltbuf.dpsm, t.bitbltbuf.dbp, t.bitbltbuf.dbw, t.trxpos.dsay, t.trxreg.rrh, first, count);
        if (t.direction == 0u || t.direction == 2u)
            r.DownloadOverlapping(first, count, "transfer-dest"); // partial writes keep the rest of the pages
        m_cpu.BeginTransfer(t);
        if (t.direction == 2u)
            r.InvalidatePages(first, count);
        command.extra->args[0] = first;
        command.extra->args[1] = count;
        m_transferFirstPage = first;
        m_transferPageCount = count;
        if (m_shadow)
            m_shadow->BeginTransfer(t);
        break;
    }
    case CommandType::UploadImage:
        m_cpu.UploadImage(command.extra->data.data(), static_cast<uint32_t>(command.extra->data.size()));
        r.InvalidatePages(m_transferFirstPage, m_transferPageCount);
        if (m_shadow)
            m_shadow->UploadImage(command.extra->data.data(), static_cast<uint32_t>(command.extra->data.size()));
        break;
    case CommandType::TextureFlush:
        m_cpu.TextureFlush();
        if (m_shadow)
            m_shadow->TextureFlush();
        break;
    case CommandType::WriteVram:
    {
        r.DownloadAll();
        m_cpu.WriteVram(command.extra->args[0], command.extra->args[1], command.extra->args[2], command.extra->args[3], command.extra->args[4], command.extra->args[5]);
        r.InvalidateAll();
        if (m_shadow)
            m_shadow->WriteVram(command.extra->args[0], command.extra->args[1], command.extra->args[2], command.extra->args[3], command.extra->args[4], command.extra->args[5]);
        break;
    }
    case CommandType::Present:
    {
        static const bool s_gpuPresent = []()
        {
            const char *v = std::getenv("PS2X_GS_GPU_PRESENT");
            return !(v && v[0] == '0');
        }();
        if (s_gpuPresent && !m_shadow && r.PresentFromGpu(command.extra->presentRequest))
        {
            m_presentPending.store(false, std::memory_order_release);
            if (command.extra->presentCallback)
                command.extra->presentCallback(PresentationFrame{});
            break;
        }
        r.InvalidatePresentation();
        r.DownloadForPresent(command.extra->presentRequest);
        PresentationFrame frame = m_cpu.Present(command.extra->presentRequest);
        if (m_shadow)
        {
            PresentationFrame reference = m_shadow->Present(command.extra->presentRequest);
            if (frame && reference && frame.width == reference.width && frame.height == reference.height)
            {
                uint64_t sum = 0, bad = 0;
                const size_t n = std::min(frame.pixels.size(), reference.pixels.size());
                for (size_t i = 0; i < n; ++i)
                {
                    if ((i & 3u) == 3u)
                        continue;
                    const int d = std::abs(static_cast<int>(frame.pixels[i]) - static_cast<int>(reference.pixels[i]));
                    sum += static_cast<uint64_t>(d);
                    bad += d > 16 ? 1u : 0u;
                }
                const double meanAbs = static_cast<double>(sum) / static_cast<double>(std::max<size_t>(1u, n / 4u * 3u));
                ++g_verify.frames;
                g_verify.sumMeanAbs += meanAbs;
                if (bad > n / 400u)
                    ++g_verify.badFrames;
                const auto now = std::chrono::steady_clock::now();
                static const long s_dumpTick = []()
                {
                    const char *v = std::getenv("PS2X_GS_GPU_VERIFY_TICK");
                    return v ? std::strtol(v, nullptr, 0) : -1L;
                }();
                const bool dumpThis = s_dumpTick >= 0 && command.extra->presentRequest.vsyncTick >= static_cast<uint64_t>(s_dumpTick) &&
                                      command.extra->presentRequest.vsyncTick < static_cast<uint64_t>(s_dumpTick) + 4u;
                if (dumpThis || now - g_verify.lastReport > std::chrono::seconds(5))
                {
                    g_verify.lastReport = now;
                    std::cerr << "[gs-gpu-verify] frames=" << g_verify.frames << " mean|diff|="
                              << (g_verify.sumMeanAbs / static_cast<double>(g_verify.frames)) << " frames>0.25%bad="
                              << g_verify.badFrames << " last=" << meanAbs << " unsupported:" << r.UnsupportedSummary() << std::endl;
                    static const char *dumpDir = std::getenv("PS2X_DUMP_FRAMES");
                    if (dumpDir && *dumpDir)
                    {
                        // Side by side: GPU | CPU | amplified difference, as a binary PPM.
                        char path[512];
                        std::snprintf(path, sizeof(path), "%s/gpu_verify_%06llu.ppm", dumpDir,
                                      static_cast<unsigned long long>(command.extra->presentRequest.vsyncTick));
                        if (FILE *f = std::fopen(path, "wb"))
                        {
                            std::fprintf(f, "P6\n%u %u\n255\n", frame.width * 3u, frame.height);
                            for (uint32_t y = 0; y < frame.height; ++y)
                                for (uint32_t part = 0; part < 3u; ++part)
                                    for (uint32_t x = 0; x < frame.width; ++x)
                                    {
                                        const size_t o = (static_cast<size_t>(y) * frame.width + x) * 4u;
                                        for (uint32_t c = 0; c < 3u; ++c)
                                        {
                                            int v = part == 0 ? frame.pixels[o + c] : part == 1 ? reference.pixels[o + c]
                                                                                             : std::min(255, 8 * std::abs(int(frame.pixels[o + c]) - int(reference.pixels[o + c])));
                                            std::fputc(v, f);
                                        }
                                    }
                            std::fclose(f);
                        }
                    }
                }
            }
        }
        m_presentPending.store(false, std::memory_order_release);
        if (command.extra->presentCallback)
            command.extra->presentCallback(std::move(frame));
        break;
    }
    case CommandType::Call:
        if (command.extra->call)
            command.extra->call();
        break;
    }
}

void GSGpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    m_vram = vram;
    m_vramSize = vramSize;
    m_cpu.Initialize(vram, vramSize);
    m_renderer->SetVram(vram);
    if (m_shadow)
    {
        m_shadowVram.assign(vram, vram + vramSize);
        m_shadow->Initialize(m_shadowVram.data(), vramSize);
    }
}

void GSGpuBackend::Reset()
{
    RunOnHost([this]()
              {
                  m_renderer->Reset();
                  m_cpu.Reset();
                  if (m_shadow)
                  {
                      m_shadow->Reset();
                      if (m_vram)
                          std::memcpy(m_shadowVram.data(), m_vram, std::min<size_t>(m_shadowVram.size(), m_vramSize));
                  } });
}

void GSGpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    Command command;
    command.type = CommandType::Submit;
    command.batch = batch;
    Enqueue(std::move(command));
}

void GSGpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    Command command;
    command.type = CommandType::LoadClut;
    command.extra = std::make_unique<CommandExtra>();
    command.extra->tex0 = tex0;
    command.extra->texclut = texclut;
    Enqueue(std::move(command));
}

void GSGpuBackend::BeginTransfer(const GSTransferCommand &transferCommand)
{
    Command command;
    command.type = CommandType::BeginTransfer;
    command.extra = std::make_unique<CommandExtra>();
    command.extra->transfer = transferCommand;
    Enqueue(std::move(command));
}

void GSGpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    Command command;
    command.type = CommandType::UploadImage;
    command.extra = std::make_unique<CommandExtra>();
    command.extra->data.assign(data, data + sizeBytes);
    Enqueue(std::move(command));
}

void GSGpuBackend::Flush()
{
}

void GSGpuBackend::TextureFlush()
{
    Command command;
    command.type = CommandType::TextureFlush;
    Enqueue(std::move(command));
}

void GSGpuBackend::Sync(GSSyncReason reason)
{
    if (reason == GSSyncReason::Finish)
        return;
    RunOnHost([this, reason]()
              {
                  m_renderer->DownloadAll();
                  m_cpu.Sync(reason); });
}

PresentationFrame GSGpuBackend::Present(const GSPresentationRequest &request)
{
    PresentationFrame frame;
    RunOnHost([&]()
              {
                  m_renderer->DownloadAll();
                  frame = m_cpu.Present(request); });
    return frame;
}

bool GSGpuBackend::PresentAsync(const GSPresentationRequest &request, PresentCallback callback)
{
    if (m_presentPending.exchange(true, std::memory_order_acq_rel))
        return false;
    Command command;
    command.type = CommandType::Present;
    command.extra = std::make_unique<CommandExtra>();
    command.extra->presentRequest = request;
    command.extra->presentCallback = std::move(callback);
    Enqueue(std::move(command));
    return true;
}

bool GSGpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    bool result = false;
    RunOnHost([&]()
              {
                  m_renderer->DownloadAll();
                  result = m_cpu.ClearFramebuffer(context, rgba);
                  m_renderer->InvalidateAll();
                  if (m_shadow)
                      m_shadow->ClearFramebuffer(context, rgba); });
    return result;
}

uint32_t GSGpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    uint32_t result = 0;
    RunOnHost([&]()
              {
                  result = m_cpu.ConsumeLocalToHostBytes(dst, maxBytes);
                  if (m_shadow)
                  {
                      std::vector<uint8_t> discard(maxBytes);
                      m_shadow->ConsumeLocalToHostBytes(discard.data(), maxBytes);
                  } });
    return result;
}

uint32_t GSGpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    uint32_t result = 0;
    RunOnHost([&]()
              {
                  m_renderer->DownloadAll();
                  result = m_cpu.ReadVram(psm, base, bw, x, y); });
    return result;
}

void GSGpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    Command command;
    command.type = CommandType::WriteVram;
    command.extra = std::make_unique<CommandExtra>();
    command.extra->args[0] = psm;
    command.extra->args[1] = base;
    command.extra->args[2] = bw;
    command.extra->args[3] = x;
    command.extra->args[4] = y;
    command.extra->args[5] = value;
    Enqueue(std::move(command));
}

void GSGpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    RunOnHost([&]()
              {
                  m_renderer->DownloadAll();
                  m_cpu.SnapshotVram(out); });
}

GSTransferSnapshot GSGpuBackend::GetTransferSnapshot() const
{
    GSTransferSnapshot result{};
    RunOnHost([&]()
              { result = m_cpu.GetTransferSnapshot(); });
    return result;
}
