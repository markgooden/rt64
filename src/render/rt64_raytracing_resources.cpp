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
