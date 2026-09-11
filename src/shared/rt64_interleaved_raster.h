#pragma once

#include "shared/rt64_hlsl.h"

#define MAX_INTERLEAVED_RASTERS     8

#ifdef HLSL_CPU
namespace interop {
#endif
    // gDiffuse's alpha when the pixel is an interleaved raster layer rather than a traced
    // surface. Compose emits those pixels as they are: the raster path has already shaded and
    // tonemapped them, so lighting them again would count the shading twice.
    //
    // A sentinel in the alpha rather than a buffer of its own because compose already reads
    // gDiffuse.a to tell a hit from a miss, and this is a third answer to that same question.
    // Any value above 1 works; the buffer is float.
#ifndef HLSL_CPU
    static const float RasterLayerAlpha = 2.0f;
#endif

    struct InterleavedRaster {
        uint rasterSceneIndex;
        uint firstInstanceIndex;
        uint colorTextureIndex;
        uint depthTextureIndex;
    };
#ifdef HLSL_CPU
};
#endif