//
// RT64
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/rt64_common.h"
#include "common/rt64_plume.h"

#include "rt64_shader_common.h"
#include "rt64_shader_compiler.h"
#include "rt64_shader_library.h"
#include "rt64_upscaler.h"

#include "shared/rt64_raytracing_params.h"

namespace RT64 {
    // Hash of the ubershader's ShaderDescription. The frame graph looks this up directly in
    // RaytracingState::shaderProgramsMap: specialized per-material RT shaders are deferred
    // upstream (rt64_framebuffer_renderer.cpp:1578-1580), so this is the only entry that
    // must exist for a frame to be traced.
    static const uint64_t UberShaderHash = 0;

    // The knobs the debugger's Raytracing tab writes and the workload queue forwards.
    // Field types are fixed by the ImGui calls in rt64_state.cpp:2484-2535.
    struct RaytracingConfiguration {
        int diSamples = 1;
        int giSamples = 1;
        int maxLights = 12;
        int maxReflections = 2;
        float motionBlurStrength = 0.0f;
        int motionBlurSamples = 0;
        interop::VisualizationMode visualizationMode = interop::VisualizationMode::Final;
        UpscaleMode upscalerMode = UpscaleMode::Bilinear;
        Upscaler::QualityMode upscalerQualityMode = Upscaler::QualityMode::Auto;
        float upscalerSharpness = 0.0f;
        bool upscalerResolutionOverride = false;
        bool upscalerReactiveMask = false;
        bool upscalerLockMask = false;
        float resolutionScale = 1.0f;
        bool denoiserEnabled = true;
    };

    // One hit group pair. Both members are indices into the RT pipeline's program list,
    // pushed straight into FramebufferRenderer::hitGroupVector, which is a
    // std::vector<RenderPipelineProgram> declared unguarded at
    // rt64_framebuffer_renderer.h:70.
    struct RaytracingShaderPrograms {
        RenderPipelineProgram surface;
        RenderPipelineProgram shadow;
    };

    // One compiled RT pipeline plus the hit groups it contains. The cache holds several so a
    // newly compiled shader set can be swapped in between frames without stalling the one in
    // flight, exactly as RasterShaderCache does for the raster path.
    struct RaytracingState {
        std::unique_ptr<RenderPipeline> pipeline;
        std::unordered_map<uint64_t, RaytracingShaderPrograms> shaderProgramsMap;

        // The ray generation and miss programs the shader binding table is built from.
        // rayGenPrograms is ordered: the frame graph picks one by writing a fixed index into
        // groups.rayGen.startIndex (rt64_framebuffer_renderer.cpp:808, 833, 838, 848, 861),
        // so position in this vector is what those numbers mean.
        std::vector<RenderPipelineProgram> rayGenPrograms;
        std::vector<RenderPipelineProgram> missPrograms;

        const RaytracingShaderPrograms &getShaderPrograms(const ShaderDescription &desc) const;
    };

    // Modelled on RasterShaderCache (rt64_raster_shader_cache.h), which has the same
    // submit / state-swap shape. Only the ubershader path is required; see UberShaderHash.
    struct RaytracingShaderCache {
        static const uint32_t StateCount = 2;

        RenderDevice *device = nullptr;
        RenderShaderFormat shaderFormat = RenderShaderFormat::UNKNOWN;
        const ShaderLibrary *shaderLibrary = nullptr;
        std::unique_ptr<ShaderCompiler> shaderCompiler;
        // Bound on the command list and given to the pipeline, both. plume attaches it to
        // the state object as the local root signature and bakes its descriptor tables into
        // every shader record (plume_d3d12.cpp:3348-3358, :4083-4093), while the state
        // object's global root signature is a dummy plume builds itself.
        //
        // Binding an empty layout to match that dummy looks more correct and is not:
        // setDescriptorSet indexes setViewRootIndices by set number, and an empty layout
        // has none, so the bind walks off the end (:2515-2524). A standalone test against
        // plume crashed exactly there, and dispatches 120 frames clean with this
        // arrangement.
        std::unique_ptr<RenderPipelineLayout> pipelineLayout;
        std::vector<RaytracingState> states;
        std::atomic<int32_t> activeState = { 0 };
        std::mutex submissionMutex;
        std::queue<ShaderDescription> descQueue;
        std::unordered_map<uint64_t, bool> shaderHashes;
        bool setupDone = false;

        RaytracingShaderCache(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary);
        ~RaytracingShaderCache();

        // Compiles the RT pipeline and its layout. Called once, lazily, from the workload
        // queue the first time an RT frame is configured (rt64_workload_queue.cpp:244-246).
        void setup();
        bool isSetup() const;

        // Queues a draw call's material for a specialized RT shader. A no-op while only the
        // ubershader exists, but the frame graph calls it per draw call, so it must be cheap.
        void submit(const ShaderDescription &desc);

        // Advances to the next state at the end of a frame (rt64_workload_queue.cpp:696).
        void setNextState();
        int32_t getActiveState() const;
    };
};
