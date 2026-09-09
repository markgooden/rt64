//
// RT64
//

#pragma once

#include <cstdint>
#include <vector>

#include "rt64_render_target.h"
#include "rt64_render_worker.h"

namespace RT64 {
    // Copies a render target's resolved texture into a CPU side RGBA8 buffer.
    //
    // This exists because the traced image never reaches RDRAM. dlreplay hashes an RDRAM
    // range, which State::fullSync fills by copying each framebuffer pair's colour target
    // back (rt64_state.cpp:1458, :1479), and the tracer composites into a different target
    // entirely - measured with PDRT64_RT_DUMPTARGET, the two sets of pointers are disjoint,
    // and the composite is recorded after that frame's writeback besides. So an RT change
    // that alters only pixels had no automated check at all.
    //
    // Reading the target directly keeps that out of the raster path. The alternative,
    // resolving the composited target into the pair's colour target before writeback, would
    // reach into code invariant 3 guards, to serve a debug readback.
    //
    // Rows come back top down and tightly packed, four bytes per pixel. The row pitch a
    // texture to buffer copy requires is an alignment detail of the copy, not of the result,
    // so it is undone here rather than handed to every caller.
    bool rtReadbackTargetRgba(RenderDevice *device, RenderTarget *target, std::vector<uint8_t> &rgba,
        uint32_t &outWidth, uint32_t &outHeight);
};
