//
// RT64
//

#pragma once

#include <memory>
#include <vector>

#include "common/rt64_common.h"
#include "common/rt64_plume.h"
#include "common/rt64_user_configuration.h"

#include "rt64_buffer_uploader.h"
#include "rt64_descriptor_sets.h"
#include "rt64_framebuffer_renderer_call.h"
#include "rt64_raytracing_shader_cache.h"
#include "rt64_render_target.h"
#include "rt64_render_target_manager.h"
#include "rt64_render_worker.h"
#include "rt64_shader_library.h"
#include "rt64_upscaler.h"

#include "preset/rt64_preset_scene.h"
#include "shared/rt64_interleaved_raster.h"
#include "shared/rt64_point_light.h"
#include "shared/rt64_raytracing_params.h"

namespace RT64 {
    // One perspective projection's worth of traceable scene, gathered during the draw-call
    // walk and consumed by FramebufferRenderer::updateRaytracingScene / submitRaytracingScene.
    //
    // Nothing in the pinned tree defines this type — rt64_framebuffer_renderer.h:52 declares
    // a std::vector<RaytracingScene> and every use is guarded, so it went missing with the
    // rest of the RT implementation. It lives here rather than in the renderer because
    // RaytracingResources::updateLightsBuffer takes it, and the renderer includes this header.
    // Every field below is fixed by a use in rt64_framebuffer_renderer.cpp:1414-1744.
    struct RaytracingScene {
        // Instances belonging to this scene, indices into FramebufferRenderer's draw calls.
        std::vector<uint32_t> instanceIndices;

        // Rasterized views composited into the traced image. Must hold at least one element
        // by the time it is uploaded (:1807-1809).
        std::vector<interop::InterleavedRaster> interleavedRasters;

        // The projection this scene was captured under, and the previous frame's, for
        // reprojection. A scene only accepts draw calls whose matrices match its own within
        // a threshold (:1467-1471).
        hlslpp::float4x4 curViewMatrix;
        hlslpp::float4x4 curProjMatrix;
        hlslpp::float4x4 prevViewMatrix;
        hlslpp::float4x4 prevProjMatrix;

        RenderViewport viewport;
        RenderRect scissor;

        PresetScene presetScene;
        int screenWidth = 0;
        int screenHeight = 0;

        // Borrowed from the projection, which outlives the scene within a frame.
        const interop::PointLight *pointLights = nullptr;
        uint32_t lightCount = 0;

        float deltaTime = 0.0f;
    };

    // Owns every GPU resource the path tracer needs that is not already owned by the raster
    // path: the acceleration structures, the G-buffer and accumulation textures, the hit
    // query buffers, the shader binding table, and the descriptor sets for the compute and
    // raster stages of the frame graph.
    //
    // Every member below is named by a call site in rt64_framebuffer_renderer.cpp's
    // RT_ENABLED code, which is what fixes its type. Nothing here is speculative.
    struct RaytracingResources {
        // Hit queries recorded per pixel by the any-hit shaders. The frame graph sizes the
        // four hit buffers as textureWidth * textureHeight * MaxHitQueries
        // (rt64_framebuffer_renderer.cpp:360), and binds them with that same compile-time
        // constant, so it cannot be clamped at allocation time without the views running
        // off the end of the buffers.
        //
        // It costs 30 bytes per query per pixel across the four buffers, which is the
        // whole reason this number is small. RT64 renders at the window resolution, not
        // the N64's: at 2880x1980 the original 16 asked for 2.6 GB and killed the driver.
        // Four keeps that under 700 MB, and four layers of transparency is the usual
        // choice for the any-hit path this exists to serve - which is not implemented yet,
        // so nothing reads these buffers at all today.
        static const uint32_t MaxHitQueries = 4;

        // Auto-exposure histogram size, fixed by the shaders that share it
        // (shaders/LuminanceHistogramCS.hlsl:10, shaders/HistogramAverageCS.hlsl:11).
        static const uint32_t HistogramBins = 64;

        // One BLAS and the buffers backing it. The draw-call walk adds one of these per RT
        // draw call, so the k-th entry corresponds to the k-th index in the scene's
        // instanceIndices — see updateTopLevelASResources.
        //
        // bufferSize and scratchSize record what was allocated, not what the current build
        // needs, so a frame whose geometry fits the existing allocation reuses it.
        struct BottomLevelAS {
            std::unique_ptr<RenderAccelerationStructure> accelerationStructure;
            std::unique_ptr<RenderBuffer> buffer;
            std::unique_ptr<RenderBuffer> scratchBuffer;
            std::vector<RenderBottomLevelASMesh> meshes;
            RenderBottomLevelASBuildInfo buildInfo;
            uint64_t bufferSize = 0;
            uint64_t scratchSize = 0;
        };

        RenderWorker *worker = nullptr;
        UserConfiguration::GraphicsAPI graphicsAPI = UserConfiguration::GraphicsAPI::OptionCount;

        // Acceleration structures. The frame graph tests bottomLevelASVector.empty() to
        // decide whether an RT frame is worth submitting (rt64_framebuffer_renderer.cpp:1212),
        // so it must hold this frame's entries and no more. Entries retired by
        // resetBottomLevelAS move to the pool rather than being destroyed, which is what
        // keeps a steady stream of frames from reallocating every buffer every frame.
        std::vector<BottomLevelAS> bottomLevelASVector;
        std::vector<BottomLevelAS> bottomLevelASPool;
        std::unique_ptr<RenderAccelerationStructure> topLevelAS;
        std::unique_ptr<RenderBuffer> topLevelASBuffer;
        std::unique_ptr<RenderBuffer> topLevelASScratchBuffer;
        std::unique_ptr<RenderBuffer> topLevelASInstancesBuffer;
        RenderTopLevelASBuildInfo topLevelASBuildInfo;
        std::vector<RenderTopLevelASInstance> topLevelASInstances;
        uint64_t topLevelASBufferSize = 0;
        uint64_t topLevelASScratchSize = 0;
        uint64_t topLevelASInstancesSize = 0;

        // Shader binding table. The frame graph rewrites groups.rayGen.startIndex between
        // dispatches to select which ray generation program runs
        // (rt64_framebuffer_renderer.cpp:808, 833, 838, 848, 861).
        std::unique_ptr<RenderBuffer> shaderBindingTableBuffer;
        RenderShaderBindingTableInfo shaderBindingTableInfo;
        uint64_t shaderBindingTableSize = 0;

        // Ray generation inputs and the shading G-buffer.
        std::unique_ptr<RenderTexture> viewDirectionTexture;
        std::unique_ptr<RenderTexture> shadingPositionTexture;
        std::unique_ptr<RenderTexture> shadingNormalTexture;
        std::unique_ptr<RenderTexture> shadingSpecularTexture;
        std::unique_ptr<RenderTexture> diffuseTexture;
        std::unique_ptr<RenderTexture> instanceIdTexture;
        std::unique_ptr<RenderTexture> reflectionTexture;
        std::unique_ptr<RenderTexture> refractionTexture;
        std::unique_ptr<RenderTexture> transparentTexture;
        std::unique_ptr<RenderTexture> flowTexture;
        std::unique_ptr<RenderTexture> reactiveMaskTexture;
        std::unique_ptr<RenderTexture> lockMaskTexture;

        // Double buffered across frames for reprojection; swapBuffers picks the current half.
        std::unique_ptr<RenderTexture> directLightTexture[2];
        std::unique_ptr<RenderTexture> indirectLightTexture[2];
        std::unique_ptr<RenderTexture> normalRoughnessTexture[2];
        std::unique_ptr<RenderTexture> depthTexture[2];
        std::unique_ptr<RenderTexture> filteredDirectLightTexture[2];
        std::unique_ptr<RenderTexture> filteredIndirectLightTexture[2];
        std::unique_ptr<RenderTexture> outputTexture[2];
        std::unique_ptr<RenderFramebuffer> outputFramebuffer[2];

        // Auto exposure and upscaling.
        std::unique_ptr<RenderTexture> downscaledOutputTexture;
        std::unique_ptr<RenderTexture> lumaAverageTexture;
        std::unique_ptr<RenderTexture> upscaledOutputTexture;

        // Hit queries, written by the any-hit shaders and read by the shading passes.
        std::unique_ptr<RenderBuffer> hitVelocityDistanceBuffer;
        std::unique_ptr<RenderBuffer> hitColorBuffer;
        std::unique_ptr<RenderBuffer> hitNormalFogBuffer;
        std::unique_ptr<RenderBuffer> hitInstanceIdBuffer;
        std::unique_ptr<RenderBufferFormattedView> hitVelocityDistanceBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitColorBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitNormalFogBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitInstanceIdBufferView;

        // Uploaded through BufferUploader, so both are BufferPairs rather than raw buffers
        // (rt64_framebuffer_renderer.cpp:1811 takes the address of rtParamsBuffer).
        BufferPair rtParamsBuffer;
        BufferPair lightsBuffer;

        // 64 bins of auto-exposure histogram, shared by the four luminance stages
        // (shaders/LuminanceHistogramCS.hlsl:10). Never named by the frame graph, which only
        // binds the descriptor sets, so it is owned here.
        std::unique_ptr<RenderBuffer> luminanceHistogramBuffer;

        // Descriptor sets for the non-RT stages of the frame graph. Every pipeline these
        // bind to is already built unconditionally in ShaderLibrary.
        std::unique_ptr<RaytracingComposeDescriptorSet> composeSet;
        std::unique_ptr<GaussianFilterDescriptorSet> indirectFilterSets[2];
        std::unique_ptr<BicubicScalingDescriptorSet> downscaleSet;
        std::unique_ptr<LuminanceHistogramDescriptorSet> lumaSet;
        std::unique_ptr<HistogramAverageDescriptorSet> lumaAvgSet;
        std::unique_ptr<HistogramClearDescriptorSet> lumaClearSet;
        std::unique_ptr<HistogramSetDescriptorSet> lumaSetSet;
        std::unique_ptr<PostProcessDescriptorSet> postProcessSet;

        // Rasterized views composited into the traced image (mirrors, screens). Their
        // multisampling and HDR settings track the framebuffer they will be composited into,
        // which is why updateInterleavedRenderTargets is handed both.
        std::vector<std::unique_ptr<RenderTarget>> interleavedColorTargetVector;
        std::vector<std::unique_ptr<RenderTarget>> interleavedDepthTargetVector;
        std::vector<std::unique_ptr<RenderFramebufferStorage>> interleavedFramebufferStorageVector;
        RenderMultisampling interleavedMultisampling;
        bool interleavedUsesHDR = false;

        // Parameters uploaded to the RT pipeline each frame.
        interop::RaytracingParams rtParams;

        uint32_t textureWidth = 0;
        uint32_t textureHeight = 0;
        int maxReflections = 2;
        UpscaleMode upscalerMode = UpscaleMode::Bilinear;
        bool denoiserEnabled = true;
        bool upscaleActive = false;
        bool upscalerReactiveMask = false;
        bool upscalerLockMask = false;
        bool swapBuffers = false;
        bool skipReprojection = true;
        bool transitionOutputBuffers = false;
        bool updateOutputBuffers = true;

        RaytracingResources(RenderWorker *worker, UserConfiguration::GraphicsAPI graphicsAPI);
        ~RaytracingResources();

        // Acceleration structures. resetBottomLevelAS clears the mesh accumulation at the
        // start of a frame; addBottomLevelASMesh is called once per RT draw call during the
        // draw-call walk (rt64_framebuffer_renderer.cpp:1596); the update calls size and
        // create the resources; the submit calls record the builds.
        void resetBottomLevelAS();
        void addBottomLevelASMesh(const RenderBottomLevelASMesh &mesh);
        void updateBottomLevelASResources(RenderWorker *worker);
        void submitBottomLevelASCreation(RenderWorker *worker);
        void updateTopLevelASResources(RenderWorker *worker, const std::vector<InstanceDrawCall> &instanceDrawCalls, const std::vector<uint32_t> &instanceIndices);
        void submitTopLevelASCreation(RenderWorker *worker);

        // Shader binding table, rebuilt whenever the hit groups or descriptor sets change.
        void createShaderBindingTable(RenderWorker *worker, const RaytracingState *rtState, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount, const std::vector<RenderPipelineProgram> &hitGroups);

        // Output resources. createOutputBuffers allocates at the scene's resolution;
        // updateOutputBuffers is a flag the frame graph sets to request that.
        void createOutputBuffers(RenderWorker *worker, int width, int height);
        void updateInterleavedRenderTargets(RenderWorker *worker, int width, int height, uint32_t targetCount, const RenderMultisampling &multisampling, bool usesHDR);
        void updateMultisampling();
        void updateShaderSets(RenderWorker *worker, const ShaderLibrary *shaderLibrary);
        void updateLightsBuffer(RenderWorker *worker, const RaytracingScene &rtScene);

        void setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged);
        Upscaler *getUpscaler(UpscaleMode mode) const;
    };
};
