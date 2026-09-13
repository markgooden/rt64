//
// RT64
//

#pragma once

#include "rt64_sampler_library.h"

namespace RT64 {
    struct ShaderRecord {
        std::unique_ptr<RenderPipeline> pipeline;
        std::unique_ptr<RenderPipelineLayout> pipelineLayout;
    };

    struct ShaderLibrary {
        SamplerLibrary samplerLibrary;
        bool usesHDR = false;
        bool usesHardwareResolve = false;

        // All shaders.
        ShaderRecord bicubicScaling;
        ShaderRecord boxFilter;
        ShaderRecord compose;
        ShaderRecord debug;
        ShaderRecord fbChangesClear;
        ShaderRecord fbChangesDrawColor;
        ShaderRecord fbChangesDrawDepth;
        ShaderRecord fbReadAnyChanges;
        ShaderRecord fbReadAnyFull;
        ShaderRecord fbReinterpret;
        ShaderRecord fbWriteColor;
        ShaderRecord fbWriteDepth;
        ShaderRecord fbWriteDepthMS;
        ShaderRecord gaussianFilterRGB3x3;
        ShaderRecord rtEdgeFilter;

        // Writes the traced surface's depth into the raster path's depth buffer, so raster
        // draws ordered after an RT scene still have their occluders. See RtDepthWritePS.hlsl.
        ShaderRecord rtDepthWrite;
        ShaderRecord histogramAverage;
        ShaderRecord histogramClear;
        ShaderRecord histogramSet;
        ShaderRecord idle;
        ShaderRecord im3dLine;
        ShaderRecord im3dPoint;
        ShaderRecord im3dTriangle;
        ShaderRecord luminanceHistogram;
        ShaderRecord postProcess;
        ShaderRecord rspModify;
        ShaderRecord rspProcess;
        ShaderRecord rspSmoothNormal;
        ShaderRecord rspVertexTestZ;
        ShaderRecord rspVertexTestZMS;
        ShaderRecord rspWorld;
        ShaderRecord rtCopyColorToDepth;
        ShaderRecord rtCopyDepthToColor;
        ShaderRecord rtCopyColorToDepthMS;
        ShaderRecord rtCopyDepthToColorMS;
        ShaderRecord textureDecode;
        ShaderRecord textureCopy;
        ShaderRecord textureResolve;
        ShaderRecord videoInterfaceLinear;
        ShaderRecord videoInterfaceNearest;
        ShaderRecord videoInterfacePixel;

        ShaderLibrary(bool usesHDR, bool usesHardwareResolve);
        ~ShaderLibrary();
        void setupCommonShaders(RenderInterface *rhi, RenderDevice *device);
        void setupMultisamplingShaders(RenderInterface *rhi, RenderDevice *device, const RenderMultisampling &multisampling);
    };
};
