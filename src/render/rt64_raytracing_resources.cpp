//
// RT64
//

#include "rt64_raytracing_resources.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace RT64 {
    static uint64_t roundUp(uint64_t value, uint64_t powerOf2Alignment) {
        return (value + powerOf2Alignment - 1) & ~(powerOf2Alignment - 1);
    }

    // Acceleration structure buffers are allocated with headroom so a frame whose geometry
    // grows slightly does not force a reallocation, the same way BufferUploader sizes the
    // vertex and index streams (rt64_buffer_uploader.cpp:93-96).
    static const uint64_t ASBlockAlignment = 256;

    static uint64_t allocationForSize(uint64_t requiredSize) {
        return roundUp(std::max((requiredSize * 3) / 2, ASBlockAlignment), ASBlockAlignment);
    }

    // Recreates a buffer only when what exists is too small, and reports whether it had to.
    // allocatedSize records what was allocated rather than what was asked for, so growth
    // headroom is not forgotten between frames.
    static bool ensureBuffer(RenderDevice *device, std::unique_ptr<RenderBuffer> &buffer, uint64_t &allocatedSize, uint64_t requiredSize, const RenderBufferDesc &desc) {
        if ((buffer != nullptr) && (allocatedSize >= requiredSize)) {
            return false;
        }

        allocatedSize = desc.size;
        buffer = device->createBuffer(desc);
        return true;
    }

    RaytracingResources::RaytracingResources(RenderWorker *worker, UserConfiguration::GraphicsAPI graphicsAPI) {
        assert(worker != nullptr);

        this->worker = worker;
        this->graphicsAPI = graphicsAPI;
    }

    RaytracingResources::~RaytracingResources() { }

    // Bottom level acceleration structures.
    //
    // Every RT draw call gets its own BLAS. That is what makes the k-th BLAS the k-th
    // instance in updateTopLevelASResources, and it is not just an implementation
    // convenience: the draw-call walk adds a mesh and pushes an instance index in the same
    // iteration (rt64_framebuffer_renderer.cpp:1596 and :1744), so the two lists are
    // parallel by construction.
    //
    // They are rebuilt every frame rather than refitted, because the vertices they are built
    // from are rewritten every frame by the RSP world-space compute pass. This model has no
    // static geometry to refit against.

    void RaytracingResources::resetBottomLevelAS() {
        // Retire this frame's entries into the pool rather than destroying them, so their
        // buffers survive into the next frame and get reused at the same sizes. The vector
        // itself must end up empty, because the frame graph tests it to decide whether an
        // RT frame is worth submitting (rt64_framebuffer_renderer.cpp:1212).
        for (BottomLevelAS &blas : bottomLevelASVector) {
            bottomLevelASPool.emplace_back(std::move(blas));
        }

        bottomLevelASVector.clear();
    }

    void RaytracingResources::addBottomLevelASMesh(const RenderBottomLevelASMesh &mesh) {
        BottomLevelAS blas;
        if (!bottomLevelASPool.empty()) {
            blas = std::move(bottomLevelASPool.back());
            bottomLevelASPool.pop_back();
        }

        blas.meshes.clear();
        blas.meshes.emplace_back(mesh);
        bottomLevelASVector.emplace_back(std::move(blas));
    }

    void RaytracingResources::updateBottomLevelASResources(RenderWorker *worker) {
        assert(worker != nullptr);

        for (BottomLevelAS &blas : bottomLevelASVector) {
            if (blas.meshes.empty()) {
                continue;
            }

            // preferFastTrace over preferFastBuild: each structure is traced several times
            // per frame - primary, direct, indirect, refraction and one pass per reflection
            // (rt64_framebuffer_renderer.cpp:815-862) - against a single build, so trace
            // cost dominates.
            worker->device->setBottomLevelASBuildInfo(blas.buildInfo, blas.meshes.data(), uint32_t(blas.meshes.size()), false, true);

            const uint64_t bufferSize = allocationForSize(blas.buildInfo.accelerationStructureSize);
            const bool bufferChanged = ensureBuffer(worker->device, blas.buffer, blas.bufferSize, blas.buildInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(bufferSize));
            const uint64_t scratchSize = allocationForSize(blas.buildInfo.scratchSize);
            ensureBuffer(worker->device, blas.scratchBuffer, blas.scratchSize, blas.buildInfo.scratchSize, RenderBufferDesc::DefaultBuffer(scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));

            // The structure is a view onto its buffer, so it only needs recreating when the
            // buffer moved underneath it.
            if (bufferChanged || (blas.accelerationStructure == nullptr)) {
                blas.accelerationStructure = worker->device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, blas.buffer->at(0), blas.bufferSize));
            }
        }
    }

    void RaytracingResources::submitBottomLevelASCreation(RenderWorker *worker) {
        assert(worker != nullptr);

        thread_local std::vector<RenderBufferBarrier> afterBuildBarriers;
        afterBuildBarriers.clear();

        for (BottomLevelAS &blas : bottomLevelASVector) {
            if ((blas.accelerationStructure == nullptr) || blas.meshes.empty()) {
                continue;
            }

            worker->commandList->buildBottomLevelAS(blas.accelerationStructure.get(), blas.scratchBuffer->at(0), blas.buildInfo);
            afterBuildBarriers.emplace_back(RenderBufferBarrier(blas.buffer.get(), RenderBufferAccess::READ));
        }

        // The top level build reads every bottom level structure, so all of them must have
        // landed before it is recorded.
        if (!afterBuildBarriers.empty()) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterBuildBarriers);
        }
    }

    // Top level acceleration structure.

    void RaytracingResources::updateTopLevelASResources(RenderWorker *worker, const std::vector<InstanceDrawCall> &instanceDrawCalls, const std::vector<uint32_t> &instanceIndices) {
        assert(worker != nullptr);

        // The scene's instances and this frame's bottom level structures come out of the
        // same walk, so they are parallel lists. Anything else means the walk changed and
        // the mapping below stopped being meaningful.
        assert(instanceIndices.size() == bottomLevelASVector.size());

        topLevelASInstances.clear();
        topLevelASInstances.reserve(instanceIndices.size());

        for (size_t i = 0; i < instanceIndices.size(); i++) {
            const BottomLevelAS &blas = bottomLevelASVector[i];
            if (blas.accelerationStructure == nullptr) {
                continue;
            }

            const InstanceDrawCall &drawCall = instanceDrawCalls[instanceIndices[i]];
            assert(drawCall.type == InstanceDrawCall::Type::Raytracing);

            RenderTopLevelASInstance instance;
            instance.bottomLevelAS = blas.buffer->at(0);
            instance.instanceID = uint32_t(i);
            instance.instanceMask = drawCall.raytracing.queryMask;
            instance.instanceContributionToHitGroupIndex = drawCall.raytracing.hitGroupIndex;
            instance.cullDisable = drawCall.raytracing.cullDisable;

            // The transform is left as the default identity. The vertices these structures
            // are built from were already written in world space by the RSP world compute
            // pass, which is the reason for building from that buffer in the first place.
            topLevelASInstances.emplace_back(instance);
        }

        worker->device->setTopLevelASBuildInfo(topLevelASBuildInfo, topLevelASInstances.data(), uint32_t(topLevelASInstances.size()), false, true);

        const uint64_t bufferSize = allocationForSize(topLevelASBuildInfo.accelerationStructureSize);
        const bool bufferChanged = ensureBuffer(worker->device, topLevelASBuffer, topLevelASBufferSize, topLevelASBuildInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(bufferSize));
        const uint64_t scratchSize = allocationForSize(topLevelASBuildInfo.scratchSize);
        ensureBuffer(worker->device, topLevelASScratchBuffer, topLevelASScratchSize, topLevelASBuildInfo.scratchSize, RenderBufferDesc::DefaultBuffer(scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));

        // plume hands the instance descriptors back already in the backend's own layout and
        // leaves the upload to us. They go straight into an upload heap rather than being
        // staged, because the data is small and rewritten in full every frame.
        const uint64_t instancesSize = uint64_t(topLevelASBuildInfo.instancesBufferData.size());
        if (instancesSize > 0) {
            ensureBuffer(worker->device, topLevelASInstancesBuffer, topLevelASInstancesSize, instancesSize, RenderBufferDesc::UploadBuffer(allocationForSize(instancesSize), RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));

            const RenderRange writtenRange(0, instancesSize);
            void *dstData = topLevelASInstancesBuffer->map();
            memcpy(dstData, topLevelASBuildInfo.instancesBufferData.data(), instancesSize);
            topLevelASInstancesBuffer->unmap(0, &writtenRange);
        }

        if (bufferChanged || (topLevelAS == nullptr)) {
            topLevelAS = worker->device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::TOP_LEVEL, topLevelASBuffer->at(0), topLevelASBufferSize));
        }
    }

    void RaytracingResources::submitTopLevelASCreation(RenderWorker *worker) {
        assert(worker != nullptr);

        if ((topLevelAS == nullptr) || topLevelASInstances.empty()) {
            return;
        }

        worker->commandList->buildTopLevelAS(topLevelAS.get(), topLevelASScratchBuffer->at(0), topLevelASInstancesBuffer->at(0), topLevelASBuildInfo);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(topLevelASBuffer.get(), RenderBufferAccess::READ));
    }

    // Shader binding table.

    void RaytracingResources::createShaderBindingTable(RenderWorker *worker, const RaytracingState *rtState, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount, const std::vector<RenderPipelineProgram> &hitGroups) {
        assert(worker != nullptr);
        assert(rtState != nullptr);
        assert(rtState->pipeline != nullptr);

        // The frame graph selects a ray generation program by index rather than by rebuilding
        // the table, so all five have to be in it, in order
        // (rt64_framebuffer_renderer.cpp:808, 833, 838, 848, 861).
        RenderShaderBindingGroups groups;
        groups.rayGen = RenderShaderBindingGroup(rtState->rayGenPrograms.data(), uint32_t(rtState->rayGenPrograms.size()));
        groups.miss = RenderShaderBindingGroup(rtState->missPrograms.data(), uint32_t(rtState->missPrograms.size()));
        groups.hitGroup = RenderShaderBindingGroup(hitGroups.data(), uint32_t(hitGroups.size()));

        worker->device->setShaderBindingTableInfo(shaderBindingTableInfo, groups, rtState->pipeline.get(), descriptorSets, descriptorSetCount);

        // As with the top level instances, plume fills the table's bytes and leaves the
        // upload to us. It is rewritten whenever the descriptor sets change, which is every
        // frame, so it lives in an upload heap rather than being staged.
        const uint64_t tableSize = uint64_t(shaderBindingTableInfo.tableBufferData.size());
        if (tableSize == 0) {
            return;
        }

        if ((shaderBindingTableBuffer == nullptr) || (shaderBindingTableSize < tableSize)) {
            shaderBindingTableSize = allocationForSize(tableSize);
            shaderBindingTableBuffer = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(shaderBindingTableSize, RenderBufferFlag::SHADER_BINDING_TABLE));
        }

        const RenderRange writtenRange(0, tableSize);
        void *dstData = shaderBindingTableBuffer->map();
        memcpy(dstData, shaderBindingTableInfo.tableBufferData.data(), tableSize);
        shaderBindingTableBuffer->unmap(0, &writtenRange);
    }

    // Output resources.
    //
    // Formats are not free choices. The component counts come from the UAV declarations in
    // shaders/FbRendererRT.hlsli:30-56, and the hit buffer element sizes come from the byte
    // strides the frame graph itself computes when it binds them
    // (rt64_framebuffer_renderer.cpp:360-364): 16, 4, 8 and 2 bytes per query.

    static std::unique_ptr<RenderTexture> createStorageTexture(RenderDevice *device, uint32_t width, uint32_t height, RenderFormat format) {
        return device->createTexture(RenderTextureDesc::Texture2D(width, height, 1, format, RenderTextureFlag::UNORDERED_ACCESS | RenderTextureFlag::STORAGE));
    }

    void RaytracingResources::createOutputBuffers(RenderWorker *worker, int width, int height) {
        assert(worker != nullptr);
        assert((width > 0) && (height > 0));

        RenderDevice *device = worker->device;
        textureWidth = uint32_t(width);
        textureHeight = uint32_t(height);

        // World position needs full float precision; everything else that carries colour or
        // a normal is fine at half.
        const RenderFormat HDR = RenderFormat::R16G16B16A16_FLOAT;
        shadingPositionTexture = createStorageTexture(device, textureWidth, textureHeight, RenderFormat::R32G32B32A32_FLOAT);
        viewDirectionTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);
        shadingNormalTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);
        shadingSpecularTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);
        diffuseTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);
        reflectionTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);
        refractionTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);
        transparentTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);

        // gInstanceId is RWTexture2D<int>, and -1 has to survive as "nothing was hit".
        instanceIdTexture = createStorageTexture(device, textureWidth, textureHeight, RenderFormat::R32_SINT);

        // gFlow is float2, the two mask textures are scalar. The masks are half float rather
        // than 8-bit unorm because they are written through a UAV, and typed UAV writes to
        // 8-bit formats are not guaranteed without checking format support first.
        flowTexture = createStorageTexture(device, textureWidth, textureHeight, RenderFormat::R16G16_FLOAT);
        reactiveMaskTexture = createStorageTexture(device, textureWidth, textureHeight, RenderFormat::R16_FLOAT);
        lockMaskTexture = createStorageTexture(device, textureWidth, textureHeight, RenderFormat::R16_FLOAT);

        for (uint32_t i = 0; i < 2; i++) {
            directLightTexture[i] = createStorageTexture(device, textureWidth, textureHeight, HDR);
            indirectLightTexture[i] = createStorageTexture(device, textureWidth, textureHeight, HDR);
            normalRoughnessTexture[i] = createStorageTexture(device, textureWidth, textureHeight, HDR);
            filteredDirectLightTexture[i] = createStorageTexture(device, textureWidth, textureHeight, HDR);
            filteredIndirectLightTexture[i] = createStorageTexture(device, textureWidth, textureHeight, HDR);

            // Depth is compared against the previous frame's for reprojection, so it keeps
            // full precision.
            depthTexture[i] = createStorageTexture(device, textureWidth, textureHeight, RenderFormat::R32_FLOAT);

            // The compose pass draws into these, so they are render targets as well as
            // sampled inputs to post-process.
            outputTexture[i] = device->createTexture(RenderTextureDesc::ColorTarget(textureWidth, textureHeight, HDR));

            const RenderTexture *colorAttachment = outputTexture[i].get();
            outputFramebuffer[i] = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1));
        }

        // Auto exposure works off a fixed eighth-resolution copy of the composed image
        // (rt64_framebuffer_renderer.cpp:999-1003 dispatches over width/8 by height/8).
        downscaledOutputTexture = createStorageTexture(device, std::max(textureWidth / 8, 1U), std::max(textureHeight / 8, 1U), HDR);
        lumaAverageTexture = createStorageTexture(device, 1, 1, RenderFormat::R32_FLOAT);
        upscaledOutputTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);

        const uint32_t hitBufferPixelCount = textureWidth * textureHeight * MaxHitQueries;
        hitVelocityDistanceBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 16, RenderBufferFlag::STORAGE | RenderBufferFlag::FORMATTED));
        hitColorBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 4, RenderBufferFlag::STORAGE | RenderBufferFlag::FORMATTED));
        hitNormalFogBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 8, RenderBufferFlag::STORAGE | RenderBufferFlag::FORMATTED));
        hitInstanceIdBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 2, RenderBufferFlag::STORAGE | RenderBufferFlag::FORMATTED));
        hitVelocityDistanceBufferView = hitVelocityDistanceBuffer->createBufferFormattedView(RenderFormat::R32G32B32A32_FLOAT);
        hitColorBufferView = hitColorBuffer->createBufferFormattedView(RenderFormat::R8G8B8A8_UNORM);
        hitNormalFogBufferView = hitNormalFogBuffer->createBufferFormattedView(RenderFormat::R16G16B16A16_FLOAT);
        hitInstanceIdBufferView = hitInstanceIdBuffer->createBufferFormattedView(RenderFormat::R16_UINT);

        if (luminanceHistogramBuffer == nullptr) {
            luminanceHistogramBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(HistogramBins * sizeof(uint32_t), RenderBufferFlag::STORAGE));
        }

        rtParams.resolution = { float(textureWidth), float(textureHeight), 1.0f / float(textureWidth), 1.0f / float(textureHeight) };

        // Everything above was just created, so it is in whatever layout the backend starts
        // resources in. The frame graph does the transition itself the first time it submits
        // (rt64_framebuffer_renderer.cpp:766-787); this only tells it that it must.
        transitionOutputBuffers = true;

        // Nothing accumulated at the previous size can be reprojected into the new one.
        skipReprojection = true;
    }

    void RaytracingResources::updateInterleavedRenderTargets(RenderWorker *worker, int width, int height, uint32_t targetCount, const RenderMultisampling &multisampling, bool usesHDR) {
        assert(worker != nullptr);

        // A change in either setting invalidates the targets rather than resizing them,
        // because both are baked into the RenderTarget at construction.
        const bool settingsChanged = (multisampling.sampleCount != interleavedMultisampling.sampleCount) || (usesHDR != interleavedUsesHDR);
        if (settingsChanged) {
            interleavedFramebufferStorageVector.clear();
            interleavedColorTargetVector.clear();
            interleavedDepthTargetVector.clear();
            interleavedMultisampling = multisampling;
            interleavedUsesHDR = usesHDR;
        }

        while (interleavedColorTargetVector.size() < targetCount) {
            // The address is only used to name the target for the debugger, and these are
            // not backed by RDRAM at all, so the index stands in for one.
            const uint32_t addressForName = uint32_t(interleavedColorTargetVector.size());
            interleavedColorTargetVector.emplace_back(std::make_unique<RenderTarget>(addressForName, Framebuffer::Type::Color, interleavedMultisampling, interleavedUsesHDR));
            interleavedDepthTargetVector.emplace_back(std::make_unique<RenderTarget>(addressForName, Framebuffer::Type::Depth, interleavedMultisampling, interleavedUsesHDR));
            interleavedFramebufferStorageVector.emplace_back(std::make_unique<RenderFramebufferStorage>());
        }

        for (uint32_t i = 0; i < targetCount; i++) {
            RenderTarget *colorTarget = interleavedColorTargetVector[i].get();
            RenderTarget *depthTarget = interleavedDepthTargetVector[i].get();
            colorTarget->setupColor(worker, uint32_t(width), uint32_t(height));
            colorTarget->setupColorFramebuffer(worker);
            depthTarget->setupDepth(worker, uint32_t(width), uint32_t(height));
            depthTarget->setupDepthFramebuffer(worker);

            RenderFramebufferKey framebufferKey;
            framebufferKey.modifierKey = i;
            interleavedFramebufferStorageVector[i]->setup(worker->device, framebufferKey, colorTarget, depthTarget);
        }
    }

    void RaytracingResources::updateMultisampling() {
        // Multisampling is fixed when a RenderTarget is constructed, so the interleaved
        // targets cannot be adjusted in place. Dropping them makes the next
        // updateInterleavedRenderTargets rebuild them at the new setting; nothing else here
        // is multisampled.
        interleavedFramebufferStorageVector.clear();
        interleavedColorTargetVector.clear();
        interleavedDepthTargetVector.clear();
    }

    void RaytracingResources::updateShaderSets(RenderWorker *worker, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);
        assert(shaderLibrary != nullptr);

        // Nothing to bind until the output resources exist.
        if (outputTexture[0] == nullptr) {
            return;
        }

        RenderDevice *device = worker->device;
        const SamplerLibrary &samplerLibrary = shaderLibrary->samplerLibrary;
        if (composeSet == nullptr) {
            composeSet = std::make_unique<RaytracingComposeDescriptorSet>(samplerLibrary, device);
            indirectFilterSets[0] = std::make_unique<GaussianFilterDescriptorSet>(samplerLibrary, device);
            indirectFilterSets[1] = std::make_unique<GaussianFilterDescriptorSet>(samplerLibrary, device);
            downscaleSet = std::make_unique<BicubicScalingDescriptorSet>(samplerLibrary, device);
            lumaSet = std::make_unique<LuminanceHistogramDescriptorSet>(device);
            lumaAvgSet = std::make_unique<HistogramAverageDescriptorSet>(device);
            lumaClearSet = std::make_unique<HistogramClearDescriptorSet>(device);
            lumaSetSet = std::make_unique<HistogramSetDescriptorSet>(device);
            postProcessSet = std::make_unique<PostProcessDescriptorSet>(samplerLibrary, device);
        }

        // The half the current frame writes. swapBuffers only flips in advanceFrame
        // (rt64_framebuffer_renderer.cpp:1828), after this and after the ray dispatch, so
        // both see the same value.
        const uint32_t cur = swapBuffers ? 1 : 0;

        // Compose reads the filtered accumulation buffers, which is why the frame graph
        // copies the raw ones into [1] before it draws (:878-919).
        composeSet->setTexture(composeSet->gFlow, flowTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gDiffuse, diffuseTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gDirectLight, filteredDirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gIndirectLight, filteredIndirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gReflection, reflectionTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gRefraction, refractionTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gTransparent, transparentTexture.get(), RenderTextureLayout::SHADER_READ);

        // The GI blur ping-pongs between the two filtered buffers. Set k reads [k] and
        // writes [1 - k], which is what makes the frame graph's alternating barriers at
        // :938-943 line up, and leaves the result in [1] after its five iterations.
        for (uint32_t k = 0; k < 2; k++) {
            indirectFilterSets[k]->setTexture(indirectFilterSets[k]->gInput, filteredIndirectLightTexture[k].get(), RenderTextureLayout::SHADER_READ);
            indirectFilterSets[k]->setTexture(indirectFilterSets[k]->gOutput, filteredIndirectLightTexture[1 - k].get(), RenderTextureLayout::GENERAL);
        }

        downscaleSet->setTexture(downscaleSet->gInput, outputTexture[cur].get(), RenderTextureLayout::SHADER_READ);
        downscaleSet->setTexture(downscaleSet->gOutput, downscaledOutputTexture.get(), RenderTextureLayout::GENERAL);

        lumaSet->setTexture(lumaSet->HDRTexture, downscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ);
        lumaSet->setBuffer(lumaSet->LuminanceHistogram, luminanceHistogramBuffer.get(), HistogramBins * sizeof(uint32_t));
        lumaAvgSet->setBuffer(lumaAvgSet->LuminanceHistogram, luminanceHistogramBuffer.get(), HistogramBins * sizeof(uint32_t));
        lumaAvgSet->setTexture(lumaAvgSet->LuminanceOutput, lumaAverageTexture.get(), RenderTextureLayout::GENERAL);
        lumaClearSet->setBuffer(lumaClearSet->LuminanceHistogram, luminanceHistogramBuffer.get(), HistogramBins * sizeof(uint32_t));
        lumaSetSet->setTexture(lumaSetSet->LuminanceOutput, lumaAverageTexture.get(), RenderTextureLayout::GENERAL);

        // Post-process reads the upscaled image when an upscaler ran and the composed one
        // otherwise. getUpscaler returns nullptr in this tree, so this is always the
        // composed image today, but the condition is the one the frame graph applies.
        const bool upscalerActive = upscaleActive && (getUpscaler(upscalerMode) != nullptr);
        RenderTexture *postProcessInput = upscalerActive ? upscaledOutputTexture.get() : outputTexture[cur].get();
        postProcessSet->setBuffer(postProcessSet->RtParams, rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        postProcessSet->setTexture(postProcessSet->gInput, postProcessInput, RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gFlow, flowTexture.get(), RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gLumaAvg, lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ);
    }

    void RaytracingResources::updateLightsBuffer(RenderWorker *worker, const RaytracingScene &rtScene) {
        assert(worker != nullptr);

        rtParams.lightsCount = rtScene.lightCount;

        // The shader indexes SceneLights unconditionally, and the frame graph sizes the
        // binding with std::max(lightsCount, 1) (rt64_framebuffer_renderer.cpp:366), so the
        // buffer must exist even for a scene with no lights.
        const uint32_t lightCount = std::max(rtScene.lightCount, 1U);
        const uint64_t requiredSize = uint64_t(lightCount) * sizeof(interop::PointLight);
        if ((lightsBuffer.defaultBuffer == nullptr) || (lightsBuffer.allocatedSize < requiredSize)) {
            lightsBuffer.allocatedSize = allocationForSize(requiredSize);
            lightsBuffer.uploadBuffer = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(lightsBuffer.allocatedSize));
            lightsBuffer.defaultBuffer = worker->device->createBuffer(RenderBufferDesc::DefaultBuffer(lightsBuffer.allocatedSize, RenderBufferFlag::STORAGE));
        }

        if (rtScene.lightCount > 0) {
            const uint64_t copySize = uint64_t(rtScene.lightCount) * sizeof(interop::PointLight);
            const RenderRange writtenRange(0, copySize);
            void *dstData = lightsBuffer.uploadBuffer->map();
            memcpy(dstData, rtScene.pointLights, copySize);
            lightsBuffer.uploadBuffer->unmap(0, &writtenRange);
            worker->commandList->copyBufferRegion(lightsBuffer.defaultBuffer->at(0), lightsBuffer.uploadBuffer->at(0), copySize);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(lightsBuffer.defaultBuffer.get(), RenderBufferAccess::READ));
        }
    }

    // Configuration.

    void RaytracingResources::setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged) {
        maxReflections = rtConfig.maxReflections;
        denoiserEnabled = rtConfig.denoiserEnabled;
        upscalerMode = rtConfig.upscalerMode;
        upscalerReactiveMask = rtConfig.upscalerReactiveMask;
        upscalerLockMask = rtConfig.upscalerLockMask;

        rtParams.diSamples = uint32_t(std::max(rtConfig.diSamples, 0));
        rtParams.giSamples = uint32_t(std::max(rtConfig.giSamples, 0));
        rtParams.maxLights = uint32_t(std::max(rtConfig.maxLights, 0));
        rtParams.motionBlurStrength = rtConfig.motionBlurStrength;
        rtParams.motionBlurSamples = uint32_t(std::max(rtConfig.motionBlurSamples, 0));
        rtParams.visualizationMode = rtConfig.visualizationMode;

        if (resolutionChanged) {
            // The output resources are sized from the scene, which the frame graph only
            // knows once it has walked the draw calls, so this can only request the work.
            updateOutputBuffers = true;

            // Nothing accumulated at the old resolution can be reprojected into the new one.
            skipReprojection = true;
        }
    }

    Upscaler *RaytracingResources::getUpscaler(UpscaleMode mode) const {
        // No upscaler backend exists in the tree: rt64_upscaler.h is an abstract interface
        // with no DLSS, FSR or XeSS implementation behind it. Every caller is null-guarded
        // and skips jitter and the upscale pass, so this is correct behaviour rather than
        // an unfinished stub.
        (void)(mode);
        return nullptr;
    }
};
