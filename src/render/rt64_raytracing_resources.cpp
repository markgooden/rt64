//
// RT64
//

#include "rt64_raytracing_resources.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

#ifdef _WIN32
#   include <d3d12.h>
#   include "plume_d3d12.h"
#endif

namespace RT64 {
#ifdef _WIN32
    // Drains the debug layer's message queue into our own log. The messages otherwise go
    // to the debugger output, where a run launched from a script never sees them - and an
    // invalid call is what precedes nearly every removed device, so it is the one thing
    // worth having that DRED has not provided here.
    //
    // Silent unless the debug layer is on, since the info queue does not exist otherwise.
    static void drainDebugMessages(RenderDevice *device, UserConfiguration::GraphicsAPI graphicsAPI) {
        if (graphicsAPI != UserConfiguration::GraphicsAPI::D3D12) {
            return;
        }

        plume::D3D12Device *d3d12Device = static_cast<plume::D3D12Device *>(device);
        if ((d3d12Device == nullptr) || (d3d12Device->d3d == nullptr)) {
            return;
        }

        ID3D12InfoQueue *infoQueue = nullptr;
        if (FAILED(d3d12Device->d3d->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            return;
        }

        const UINT64 count = infoQueue->GetNumStoredMessages();
        for (UINT64 i = 0; i < count; i++) {
            SIZE_T length = 0;
            if (FAILED(infoQueue->GetMessage(i, nullptr, &length)) || (length == 0)) {
                continue;
            }

            std::vector<uint8_t> storage(length);
            D3D12_MESSAGE *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
            if (FAILED(infoQueue->GetMessage(i, message, &length))) {
                continue;
            }

            // Warnings and below are noise from the raster path; errors are what matter.
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                fprintf(stderr, "rt64: D3D12 %s - %.*s\n",
                    (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) ? "CORRUPTION" : "ERROR",
                    int(message->DescriptionByteLength), message->pDescription);
            }
        }

        if (count > 0) {
            infoQueue->ClearStoredMessages();
            fflush(stderr);
        }

        infoQueue->Release();
    }

    // A removed device turns every later call into a failure that reports nothing useful:
    // buffer creation returns an object whose resource is null, and the next map() faults.
    // The first sign of it in this path was a crash three calls downstream of the actual
    // cause, so the state is checked before the RT work each frame and reported once.
    //
    // Reaching the D3D12 device does not mean editing plume: D3D12Device is a public type
    // in plume_d3d12.h with its ID3D12Device8 in the open.
    //
    // DRED is what makes the report worth having. Without it a removal says only
    // DXGI_ERROR_DEVICE_HUNG or _REMOVED; with it the breadcrumbs name the last GPU
    // operations that completed, so a fault inside an acceleration structure build looks
    // different from one inside a dispatch.
    // A removed device takes other threads down with it: they get nullptr back from creates
    // they do not null-check and fault, and a report written line by line is cut off
    // mid-sentence when they do. The report is therefore accumulated here and written once,
    // which is as close to atomic as this can get without a lock the crashing thread would
    // not take anyway.
    struct DredReport {
        char text[8192];
        size_t used = 0;

        void addf(const char *format, ...) {
            if (used >= (sizeof(text) - 1)) {
                return;
            }

            va_list args;
            va_start(args, format);
            const int wrote = vsnprintf(text + used, sizeof(text) - used, format, args);
            va_end(args);
            if (wrote > 0) {
                used += size_t(wrote);
                if (used > (sizeof(text) - 1)) {
                    used = sizeof(text) - 1;
                }
            }
        }

        void flush() const {
            fwrite(text, 1, used, stderr);
            fflush(stderr);
        }
    };

    // DRED's strings and history arrays are written by the driver into memory this process
    // does not own, and after a DXGI_ERROR_DRIVER_INTERNAL_ERROR they are not always
    // readable: printing pCommandListDebugNameW with %ls faulted here, killing the process
    // in the middle of the one report that would have explained the removal. Every read of
    // DRED memory therefore goes through these, which fall back to a placeholder rather than
    // taking the process down. No C++ objects live in either, because __try forbids unwinding.
    static void copyDredName(const wchar_t *src, char *dst, size_t cap) {
        if (src == nullptr) {
            strncpy_s(dst, cap, "(unnamed)", _TRUNCATE);
            return;
        }

        __try {
            size_t i = 0;
            for (; (i + 1) < cap; i++) {
                const wchar_t w = src[i];
                if (w == L'\0') {
                    break;
                }

                dst[i] = ((w >= 0x20) && (w < 0x7F)) ? char(w) : '?';
            }

            dst[i] = '\0';
        }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            strncpy_s(dst, cap, "(unreadable)", _TRUNCATE);
        }
    }

    static bool readDredOp(const D3D12_AUTO_BREADCRUMB_OP *history, uint32_t index, int *out) {
        __try {
            *out = int(history[index]);
            return true;
        }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            return false;
        }
    }

    static bool reportDeviceRemoval(RenderDevice *device, UserConfiguration::GraphicsAPI graphicsAPI, const char *where) {
        if (graphicsAPI != UserConfiguration::GraphicsAPI::D3D12) {
            return false;
        }

        plume::D3D12Device *d3d12Device = static_cast<plume::D3D12Device *>(device);
        if ((d3d12Device == nullptr) || (d3d12Device->d3d == nullptr)) {
            return false;
        }

        const HRESULT reason = d3d12Device->d3d->GetDeviceRemovedReason();
        if (SUCCEEDED(reason)) {
            return false;
        }

        static bool reported = false;
        if (reported) {
            return true;
        }

        reported = true;

        DredReport report;
        report.addf("rt64: the D3D12 device was removed before %s, reason 0x%08lX\n", where, (unsigned long)reason);

        ID3D12DeviceRemovedExtendedData1 *dred = nullptr;
        if (FAILED(d3d12Device->d3d->QueryInterface(IID_PPV_ARGS(&dred)))) {
            report.addf("rt64: no DRED available; set PDRT64_RT_DRED=1 before launching to turn it on\n");
            report.flush();
            return true;
        }

        uint32_t nodeCount = 0;
        uint32_t incompleteCount = 0;
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 breadcrumbs = {};
        if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&breadcrumbs))) {
            for (const D3D12_AUTO_BREADCRUMB_NODE1 *node = breadcrumbs.pHeadAutoBreadcrumbNode; node != nullptr; node = node->pNext) {
                nodeCount++;
                const uint32_t done = (node->pLastBreadcrumbValue != nullptr) ? *node->pLastBreadcrumbValue : 0;
                if (done >= node->BreadcrumbCount) {
                    // Everything this list recorded finished, so it is not the one that died.
                    continue;
                }

                char name[128];
                copyDredName(node->pCommandListDebugNameW, name, sizeof(name));
                report.addf("rt64: DRED - command list '%s' stopped at operation %u of %u\n",
                    name, done, node->BreadcrumbCount);

                // The operation it stopped on is the one that faulted, and the few before it
                // are the context that makes it readable.
                const uint32_t first = (done > 4) ? (done - 4) : 0;
                for (uint32_t i = first; (i <= done) && (i < node->BreadcrumbCount) && (node->pCommandHistory != nullptr); i++) {
                    int op = 0;
                    if (!readDredOp(node->pCommandHistory, i, &op)) {
                        report.addf("rt64: DRED -   [%u] (history unreadable)\n", i);
                        break;
                    }

                    report.addf("rt64: DRED -   [%u] op %d%s\n", i, op, (i == done) ? "  <-- faulted here" : "");
                }

                incompleteCount++;
            }

            // Never silent. Every list having finished points at a timeout rather than a
            // fault, and those are different problems with different fixes.
            report.addf("rt64: DRED - %u command lists recorded, %u unfinished%s\n", nodeCount, incompleteCount,
                ((nodeCount > 0) && (incompleteCount == 0)) ? " (all completed - looks like a timeout, not a fault)" : "");
        }
        else {
            report.addf("rt64: DRED - no breadcrumbs were captured\n");
        }

        D3D12_DRED_PAGE_FAULT_OUTPUT1 pageFault = {};
        if (SUCCEEDED(dred->GetPageFaultAllocationOutput1(&pageFault))) {
            report.addf("rt64: DRED - page fault at GPU address 0x%llX\n", (unsigned long long)pageFault.PageFaultVA);
            for (const D3D12_DRED_ALLOCATION_NODE1 *node = pageFault.pHeadRecentFreedAllocationNode; node != nullptr; node = node->pNext) {
                char freedName[128];
                copyDredName(node->ObjectNameW, freedName, sizeof(freedName));
                report.addf("rt64: DRED -   recently freed: '%s'\n", freedName);
            }
        }

        dred->Release();
        report.flush();
        return true;
    }
#else
    static void drainDebugMessages(RenderDevice *, UserConfiguration::GraphicsAPI) { }
    static bool reportDeviceRemoval(RenderDevice *, UserConfiguration::GraphicsAPI, const char *) { return false; }
#endif

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
    static bool ensureBuffer(RenderDevice *device, std::unique_ptr<RenderBuffer> &buffer, uint64_t &allocatedSize, uint64_t requiredSize, const RenderBufferDesc &desc, std::vector<std::unique_ptr<RenderBuffer>> &retired) {
        if ((buffer != nullptr) && (allocatedSize >= requiredSize)) {
            return false;
        }

        // The buffer being replaced may still be referenced by a frame in flight, so it is
        // retired rather than released here. See retiredBuffers in the header.
        if (buffer != nullptr) {
            retired.emplace_back(std::move(buffer));
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
        // Once per frame. Everything the CPU writes and the GPU reads within a frame moves
        // to a different slot here, so a frame still in flight is never overwritten.
        frameSlot = (frameSlot + 1) % FrameSlots;

        // Anything retired the last time this slot was current was replaced FrameSlots frames
        // ago, so every frame that could still have been referencing it has retired too. This
        // is the only place either list is released.
        retiredBuffers[frameSlot].clear();
        retiredStructures[frameSlot].clear();

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

        drainDebugMessages(worker->device, graphicsAPI);

        if (reportDeviceRemoval(worker->device, graphicsAPI, "the bottom level acceleration structures")) {
            return;
        }

        for (BottomLevelAS &blas : bottomLevelASVector) {
            if (blas.meshes.empty()) {
                continue;
            }

            // preferFastBuild, not preferFastTrace. An earlier version chose fast trace on
            // the grounds that each structure is traced five or more times per frame
            // against a single build (rt64_framebuffer_renderer.cpp:815-862). That
            // reasoning ignored the other half: the build also happens every frame,
            // because the vertices are rewritten every frame, and there is one of these
            // per draw call. Fast trace is for geometry built once and traced for many
            // frames; fully dynamic geometry is the case fast build exists for.
            worker->device->setBottomLevelASBuildInfo(blas.buildInfo, blas.meshes.data(), uint32_t(blas.meshes.size()), true, false);

            const uint64_t bufferSize = allocationForSize(blas.buildInfo.accelerationStructureSize);
            const bool bufferChanged = ensureBuffer(worker->device, blas.buffer, blas.bufferSize, blas.buildInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(bufferSize), retiredBuffers[frameSlot]);
            const uint64_t scratchSize = allocationForSize(blas.buildInfo.scratchSize);
            ensureBuffer(worker->device, blas.scratchBuffer, blas.scratchSize, blas.buildInfo.scratchSize, RenderBufferDesc::DefaultBuffer(scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH), retiredBuffers[frameSlot]);

            // The structure is a view onto its buffer, so it only needs recreating when the
            // buffer moved underneath it.
            if (bufferChanged || (blas.accelerationStructure == nullptr)) {
                if (blas.accelerationStructure != nullptr) {
                    retiredStructures[frameSlot].emplace_back(std::move(blas.accelerationStructure));
                }

                blas.accelerationStructure = worker->device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, blas.buffer->at(0), blas.bufferSize));
            }
        }
    }

    void RaytracingResources::submitBottomLevelASCreation(RenderWorker *worker) {
        assert(worker != nullptr);

        // Reported whenever a frame needs meaningfully more structures than any frame
        // before it, rather than once. One structure is built per RT draw call, so this
        // is the per-frame acceleration structure cost - and a one-shot version of this
        // reported "1 structure" from a loading screen and then stayed quiet through
        // every real frame that followed, which is the opposite of useful.
        //
        // The pool size comes along because it is the part that persists across frames:
        // retired entries keep their buffers so a steady stream of frames stops
        // reallocating, and it is worth seeing if that ever stops being bounded.
        static size_t reportedPeak = 0;
        if (bottomLevelASVector.size() > (reportedPeak + reportedPeak / 2)) {
            reportedPeak = bottomLevelASVector.size();
            uint64_t totalTriangles = 0;
            uint64_t totalScratch = 0;
            for (const BottomLevelAS &blas : bottomLevelASVector) {
                for (const RenderBottomLevelASMesh &mesh : blas.meshes) {
                    totalTriangles += mesh.indexCount / 3;
                }
                totalScratch += blas.scratchSize;
            }

            fprintf(stdout, "rt64: building %zu bottom level structures, %llu triangles, %llu KB scratch, pool %zu\n",
                bottomLevelASVector.size(), (unsigned long long)totalTriangles,
                (unsigned long long)(totalScratch / 1024), bottomLevelASPool.size());
            fflush(stdout);
        }

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

    void RaytracingResources::updateTopLevelASResources(RenderWorker *worker, const std::vector<InstanceDrawCall> &instanceDrawCalls, const std::vector<RenderAffineTransform> &instanceTransforms, const std::vector<uint32_t> &instanceIndices) {
        assert(worker != nullptr);

        // Map each instance to its structure by counting, not by position.
        //
        // The two lists are not parallel, which an earlier version of this assumed and an
        // assert claimed. Structures are added once per raytraced draw call across the
        // whole frame (rt64_framebuffer_renderer.cpp:1596), while instanceIndices belongs
        // to one scene, and a frame can produce several. A frame here reported ten
        // structures against a single instance: nine of them belonged to other draw calls,
        // so instance zero was pointing at whichever structure happened to be first. The
        // top level structure then referenced geometry that was never built for it, which
        // the debug layer cannot see - every call is valid - and the driver reports as an
        // internal error a few frames later.
        //
        // What is reliable is the order: structures are added in the same order the draw
        // calls are appended, and only for raytraced ones. So the structure for a draw call
        // is the number of raytraced draw calls that precede it.
        thread_local std::vector<uint32_t> blasIndexByDrawCall;
        blasIndexByDrawCall.assign(instanceDrawCalls.size(), UINT32_MAX);

        uint32_t raytracedSoFar = 0;
        for (size_t i = 0; i < instanceDrawCalls.size(); i++) {
            if (instanceDrawCalls[i].type == InstanceDrawCall::Type::Raytracing) {
                blasIndexByDrawCall[i] = raytracedSoFar++;
            }
        }

        if (reportDeviceRemoval(worker->device, graphicsAPI, "the top level acceleration structure")) {
            return;
        }

        topLevelASInstances.clear();
        topLevelASInstances.reserve(instanceIndices.size());

        for (size_t i = 0; i < instanceIndices.size(); i++) {
            const uint32_t drawCallIndex = instanceIndices[i];
            if (drawCallIndex >= blasIndexByDrawCall.size()) {
                continue;
            }

            const uint32_t blasIndex = blasIndexByDrawCall[drawCallIndex];
            if (blasIndex >= bottomLevelASVector.size()) {
                // Either the draw call is not raytraced, or it produced no structure. Both
                // mean there is nothing to place in the top level structure for it, and
                // both used to be silently indexed past instead.
                continue;
            }

            const BottomLevelAS &blas = bottomLevelASVector[blasIndex];
            if (blas.accelerationStructure == nullptr) {
                continue;
            }

            const InstanceDrawCall &drawCall = instanceDrawCalls[drawCallIndex];
            assert(drawCall.type == InstanceDrawCall::Type::Raytracing);

            RenderTopLevelASInstance instance;
            instance.bottomLevelAS = blas.buffer->at(0);
            // The draw call index, not the position in this scene's instance list. A hit
            // shader needs its draw call's RenderIndices entry to find where that call's
            // triangles start in the shared index buffer, and instanceRenderIndices is
            // indexed by draw call - the two vectors are filled in the same loop, one entry
            // each per game call (rt64_framebuffer_renderer.cpp:1834-1840). The value is only
            // ever read back through InstanceID(), and nothing depends on it being dense.
            instance.instanceID = drawCallIndex;
            instance.instanceMask = drawCall.raytracing.queryMask;
            instance.instanceContributionToHitGroupIndex = drawCall.raytracing.hitGroupIndex;
            instance.cullDisable = drawCall.raytracing.cullDisable;

            // The affine that puts this call's geometry into the scene's space, which is
            // identity for every call whose projection shares the scene's view matrix.
            // The vertices these structures are built from were written in world space by
            // the RSP world compute pass, and for a projection with its own view matrix
            // that world is the projection's own - the room's, in Perfect Dark, which is
            // why those projections are a different space and were being refused entry
            // to the scene rather than placed in it.
            instance.transform = (drawCallIndex < instanceTransforms.size()) ? instanceTransforms[drawCallIndex] : RenderAffineTransform();
            topLevelASInstances.emplace_back(instance);
        }

        // PDRT64_RT_DUMPCAM: the masks the instances carry. TraceRay is issued with 0xFF, so
        // an instance whose mask is 0 is invisible to every ray no matter where it is.
        const char *dumpCam = getenv("PDRT64_RT_DUMPCAM");
        if (dumpCam != nullptr) {
            // The value is the frame interval, so a short driven run can be sampled
            // finely without a rebuild. PDRT64_RT_DUMPCAM=1 alone keeps the old cadence.
            const int dumpInstEvery = std::max(1, atoi(dumpCam));
            static uint32_t instCalls = 0;
            instCalls++;
            if ((instCalls < (dumpInstEvery * 12)) && ((instCalls % dumpInstEvery) == 0) && !topLevelASInstances.empty()) {
                fprintf(stderr, "rt64: --- instance call %u ---\n", instCalls);
                fprintf(stderr, "rt64: %zu top level instances\n", topLevelASInstances.size());
                for (size_t d = 0; (d < topLevelASInstances.size()) && (d < 8); d++) {
                    fprintf(stderr, "rt64:   instance %zu mask 0x%02X hitGroup %u cullDisable %d\n",
                        d, topLevelASInstances[d].instanceMask,
                        topLevelASInstances[d].instanceContributionToHitGroupIndex,
                        int(topLevelASInstances[d].cullDisable));
                }

                fflush(stderr);
            }
        }

        worker->device->setTopLevelASBuildInfo(topLevelASBuildInfo, topLevelASInstances.data(), uint32_t(topLevelASInstances.size()), false, true);

        const uint64_t bufferSize = allocationForSize(topLevelASBuildInfo.accelerationStructureSize);
        const bool bufferChanged = ensureBuffer(worker->device, topLevelASBuffer, topLevelASBufferSize, topLevelASBuildInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(bufferSize), retiredBuffers[frameSlot]);
        const uint64_t scratchSize = allocationForSize(topLevelASBuildInfo.scratchSize);
        ensureBuffer(worker->device, topLevelASScratchBuffer, topLevelASScratchSize, topLevelASBuildInfo.scratchSize, RenderBufferDesc::DefaultBuffer(scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH), retiredBuffers[frameSlot]);

        // plume hands the instance descriptors back already in the backend's own layout and
        // leaves the upload to us. They go straight into an upload heap rather than being
        // staged, because the data is small and rewritten in full every frame.
        const uint64_t instancesSize = uint64_t(topLevelASBuildInfo.instancesBufferData.size());
        if (instancesSize > 0) {
            ensureBuffer(worker->device, topLevelASInstancesSlots[frameSlot], topLevelASInstancesSlotSizes[frameSlot], instancesSize,
                RenderBufferDesc::UploadBuffer(allocationForSize(instancesSize), RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT), retiredBuffers[frameSlot]);
            topLevelASInstancesBuffer = topLevelASInstancesSlots[frameSlot].get();

            const RenderRange writtenRange(0, instancesSize);
            void *dstData = topLevelASInstancesBuffer->map();
            memcpy(dstData, topLevelASBuildInfo.instancesBufferData.data(), instancesSize);
            topLevelASInstancesBuffer->unmap(0, &writtenRange);
        }

        if (bufferChanged || (topLevelAS == nullptr)) {
            if (topLevelAS != nullptr) {
                retiredStructures[frameSlot].emplace_back(std::move(topLevelAS));
            }

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

        // DispatchRays takes a single ray generation *record*, not a table of them, and
        // D3D12 reads everything inside RayGenerationShaderRecord.SizeInBytes as that one
        // record's local root arguments.
        //
        // plume sizes every group as stride * programCount and leaves startIndex at 0
        // (plume_d3d12.cpp:4058-4068), then builds the dispatch as
        //
        //     StartAddress = table + offset + startIndex * stride
        //     SizeInBytes  = size
        //
        // (plume_d3d12.cpp:2001-2002). That is right for the miss and hit group tables,
        // which really are tables the shaders index into, and wrong for this one as soon as
        // the frame graph selects a program: with five 64 byte records, startIndex 4 declares
        // a 320 byte record starting at the last one, so 256 bytes of the miss and hit group
        // records are read as local root arguments. The device hangs intermittently, in a
        // later raster draw rather than in the dispatch, and no validation layer sees it -
        // every API call involved is legal.
        //
        // The size the raygen group should carry is therefore one record. Only traceRays
        // reads it; the table's bytes come from tableBufferData.
        shaderBindingTableInfo.groups.rayGen.size = shaderBindingTableInfo.groups.rayGen.stride;

        // As with the top level instances, plume fills the table's bytes and leaves the
        // upload to us. It is rewritten whenever the descriptor sets change, which is every
        // frame, so it lives in an upload heap rather than being staged.
        const uint64_t tableSize = uint64_t(shaderBindingTableInfo.tableBufferData.size());

        // One-time dump of what plume actually built. A malformed table is the remaining
        // explanation for a device that dies only when traceRays runs: every call involved
        // is valid, so nothing else can see it. Each record starts with a 32 byte shader
        // identifier, and an all-zero one means the dispatch jumps to nothing.
        {
            static bool dumped = false;
            if (!dumped && (tableSize > 0)) {
                dumped = true;
                const auto &g = shaderBindingTableInfo.groups;
                fprintf(stdout, "rt64: binding table %llu bytes\n", (unsigned long long)tableSize);
                const struct { const char *name; const RenderShaderBindingGroupInfo *info; uint32_t count; } groups[] = {
                    { "rayGen", &g.rayGen, uint32_t(rtState->rayGenPrograms.size()) },
                    { "miss", &g.miss, uint32_t(rtState->missPrograms.size()) },
                    { "hitGroup", &g.hitGroup, uint32_t(hitGroups.size()) }
                };

                for (const auto &group : groups) {
                    fprintf(stdout, "rt64:   %-8s offset %llu stride %llu size %llu, %u records\n", group.name,
                        (unsigned long long)group.info->offset, (unsigned long long)group.info->stride,
                        (unsigned long long)group.info->size, group.count);

                    for (uint32_t r = 0; r < group.count; r++) {
                        const uint64_t at = group.info->offset + uint64_t(r) * group.info->stride;
                        if ((at + 32) > tableSize) {
                            fprintf(stdout, "rt64:     [%u] runs past the end of the table\n", r);
                            continue;
                        }

                        const uint8_t *id = shaderBindingTableInfo.tableBufferData.data() + at;
                        bool allZero = true;
                        for (uint32_t b = 0; b < 32; b++) {
                            allZero = allZero && (id[b] == 0);
                        }

                        fprintf(stdout, "rt64:     [%u] at %llu: %02x%02x%02x%02x%02x%02x%02x%02x%s\n", r, (unsigned long long)at,
                            id[0], id[1], id[2], id[3], id[4], id[5], id[6], id[7],
                            allZero ? "  <-- ALL ZERO, this record dispatches to nothing" : "");
                    }
                }

                fflush(stdout);
            }
        }
        if (tableSize == 0) {
            return;
        }

        ensureBuffer(worker->device, shaderBindingTableSlots[frameSlot], shaderBindingTableSlotSizes[frameSlot], tableSize,
            RenderBufferDesc::UploadBuffer(allocationForSize(tableSize), RenderBufferFlag::SHADER_BINDING_TABLE), retiredBuffers[frameSlot]);
        shaderBindingTableBuffer = shaderBindingTableSlots[frameSlot].get();

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
        // Every recreation of these replaces resources a frame in flight may still be reading,
        // so when they happen matters. Gated with the rest of the acceleration structure
        // diagnostics.
        if (getenv("PDRT64_RT_CHECKAS") != nullptr) {
            fprintf(stderr, "rt64: RT output buffers (re)created at %dx%d\n", width, height);
            fflush(stderr);
        }

        assert(worker != nullptr);
        assert((width > 0) && (height > 0));

        // Every ray generation dispatch, every filter iteration and every one of these
        // buffers is sized from this, and the frame graph hands over the display
        // resolution - which at a window-sized 2880x1980 is 5.7 million pixels traced six
        // times per frame. Measured at 110 ms of GPU time per frame there, against 21 ms
        // for the same frame with the path tracer off, which is what eventually trips the
        // two second timeout the driver enforces.
        //
        // RaytracingConfiguration carries a resolutionScale for exactly this, and it was
        // being read in setRaytracingConfig and then ignored. Tracing below the display
        // resolution and letting compose sample the result back up is the normal
        // arrangement for a path tracer, not a workaround.
        //
        // Clamped rather than trusted: the value reaches here from a debugger slider.
        const float scale = std::clamp(resolutionScale, 0.125f, 1.0f);
        width = std::max(int(lround(width * scale)), 1);
        height = std::max(int(lround(height * scale)), 1);

        // Nothing to do when the size has not moved, and doing it anyway is actively
        // dangerous rather than merely wasteful.
        //
        // The frame graph asks for this whenever the raytracing configuration changes,
        // which includes a swap chain resize (rt64_workload_queue.cpp:242-248), and it
        // asks with the size it already has. Every assignment below drops the previous
        // texture, and a texture the previous frame's command lists still reference is
        // not ours to drop - freeing one out from under work in flight is a
        // use-after-free on the GPU, which surfaces as a removed device seconds later
        // and a crash somewhere else entirely.
        //
        // Around 1.5 GB is reallocated here at a window-sized resolution, so the repeat
        // case is also the expensive one.
        if ((textureWidth == uint32_t(width)) && (textureHeight == uint32_t(height)) && (outputTexture[0] != nullptr)) {
            return;
        }

        // A real resize does still replace textures that earlier frames may have
        // referenced, so the queue is drained first. This runs only when the size
        // actually changes, which is rare enough that the stall does not matter and
        // cheap compared to the reallocation that follows it.
        if (outputTexture[0] != nullptr) {
            worker->wait();
        }

        RenderDevice *device = worker->device;
        textureWidth = uint32_t(width);
        textureHeight = uint32_t(height);

        {
            const uint64_t queries = uint64_t(textureWidth) * textureHeight * MaxHitQueries;
            fprintf(stdout, "rt64: creating output buffers at %ux%u, %llu hit queries, %llu MB of hit buffers\n",
                textureWidth, textureHeight, (unsigned long long)queries, (unsigned long long)((queries * 30) / (1024 * 1024)));
            fflush(stdout);
        }

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
            // sampled inputs to post-process - and their format is not a free choice. The
            // compose pipeline declares R32G32B32A32_FLOAT for render target 0
            // (rt64_shader_library.cpp:297), and D3D12 rejects a draw whose render target
            // format differs from the pipeline's. This was HDR like everything else here,
            // which the debug layer reported as "the render target format in slot 0 does
            // not match that specified by the current pipeline state" on every compose.
            outputTexture[i] = device->createTexture(RenderTextureDesc::ColorTarget(textureWidth, textureHeight, RenderFormat::R32G32B32A32_FLOAT));

            const RenderTexture *colorAttachment = outputTexture[i].get();
            outputFramebuffer[i] = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1));
        }

        // Auto exposure works off a fixed eighth-resolution copy of the composed image
        // (rt64_framebuffer_renderer.cpp:999-1003 dispatches over width/8 by height/8).
        downscaledOutputTexture = createStorageTexture(device, std::max(textureWidth / 8, 1U), std::max(textureHeight / 8, 1U), HDR);
        lumaAverageTexture = createStorageTexture(device, 1, 1, RenderFormat::R32_FLOAT);
        upscaledOutputTexture = createStorageTexture(device, textureWidth, textureHeight, HDR);

        const uint32_t hitBufferPixelCount = textureWidth * textureHeight * MaxHitQueries;
        // UNORDERED_ACCESS as well as STORAGE, and the distinction is not cosmetic. They
        // are separate flags (contrib/plume/plume_render_interface_types.h:414-424) and only
        // UNORDERED_ACCESS reaches the mask that sets
        // D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS (plume_d3d12.cpp:2740). All four are
        // bound as RWBuffers (shaders/FbRendererRT.hlsli:30-33), so a UAV is created over
        // them, and creating one over a resource that does not allow it removes the device.
        // RT64's own buffer helper spells both out for this reason (rt64_workload.cpp:233).
        const RenderBufferFlags HitBufferFlags = RenderBufferFlag::STORAGE | RenderBufferFlag::FORMATTED | RenderBufferFlag::UNORDERED_ACCESS;
        hitVelocityDistanceBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 16, HitBufferFlags));
        hitColorBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 4, HitBufferFlags));
        hitNormalFogBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 8, HitBufferFlags));
        hitInstanceIdBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(uint64_t(hitBufferPixelCount) * 2, HitBufferFlags));
        hitVelocityDistanceBufferView = hitVelocityDistanceBuffer->createBufferFormattedView(RenderFormat::R32G32B32A32_FLOAT);
        hitColorBufferView = hitColorBuffer->createBufferFormattedView(RenderFormat::R8G8B8A8_UNORM);
        hitNormalFogBufferView = hitNormalFogBuffer->createBufferFormattedView(RenderFormat::R16G16B16A16_FLOAT);
        hitInstanceIdBufferView = hitInstanceIdBuffer->createBufferFormattedView(RenderFormat::R16_UINT);

        if (luminanceHistogramBuffer == nullptr) {
            luminanceHistogramBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(HistogramBins * sizeof(uint32_t), RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS));
        }

        rtParams.resolution = { float(textureWidth), float(textureHeight), 1.0f / float(textureWidth), 1.0f / float(textureHeight) };

        // Everything above was just created, so it is in whatever layout the backend starts
        // resources in. The frame graph does the transition itself the first time it submits
        // (rt64_framebuffer_renderer.cpp:766-787); this only tells it that it must.
        transitionOutputBuffers = true;

        // Nothing accumulated at the previous size can be reprojected into the new one.
        skipReprojection = true;

        reportDeviceRemoval(device, graphicsAPI, "the output buffers were created");
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
            // One depth target for every layer, not one each. The layers are interleaved
            // between RT scenes but among themselves they are ordinary draws sharing the
            // game's single depth buffer, and a per-layer depth buffer cleared per layer
            // throws that away: nothing then orders one layer against another, and the
            // composite cannot tell a layer in front from one behind. Measured
            // 2026-09-12 as black drawn over the console, which is layer 0's opening
            // winning on depth against a layer 1 that holds the console itself.
            RenderTarget *depthTarget = interleavedDepthTargetVector[0].get();

            // KNOWN DEFECT, not yet fixed. setupColor and setupDepth release the existing
            // texture and allocate a new one every time they are called
            // (rt64_render_target.cpp:80,89), and this runs every frame, so a
            // full-resolution colour and depth target is destroyed and rebuilt per frame
            // while earlier frames' command lists may still reference them.
            //
            // Measured: at 2880x1980 the device is removed after 36 frames, where a quarter
            // of that resolution survives 187 - per-frame churn proportional to target size.
            //
            // Two attempts at fixing it were both worse and are recorded so they are not
            // repeated. RenderTarget::resize skips the framebuffer setup that has to follow
            // it and died at frame 6. Guarding the whole block on a size change died at
            // frame 5. Something else in this path depends on the setup running, and that
            // dependency needs to be understood before it is removed.
            colorTarget->setupColor(worker, uint32_t(width), uint32_t(height));
            colorTarget->setupColorFramebuffer(worker);
            if (i == 0) {
                depthTarget->setupDepth(worker, uint32_t(width), uint32_t(height));
                depthTarget->setupDepthFramebuffer(worker);
            }

            RenderFramebufferKey framebufferKey;
            framebufferKey.modifierKey = i;
            interleavedFramebufferStorageVector[i]->setup(worker->device, framebufferKey, colorTarget, depthTarget);
        }

        reportDeviceRemoval(worker->device, graphicsAPI, "the interleaved render targets were set up");
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

        reportDeviceRemoval(device, graphicsAPI, "the shader descriptor sets were updated");
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

        // A change of scale changes the size of every RT resource, so it counts as a
        // resolution change even when the display resolution has not moved.
        const bool scaleChanged = (resolutionScale != rtConfig.resolutionScale);
        resolutionScale = rtConfig.resolutionScale;

        rtParams.diSamples = uint32_t(std::max(rtConfig.diSamples, 0));
        rtParams.giSamples = uint32_t(std::max(rtConfig.giSamples, 0));
        rtParams.maxLights = uint32_t(std::max(rtConfig.maxLights, 0));
        // PDRT64_RT_REFLECT: a floor under every draw call's own reflectionFactor. Read
        // once; this runs whenever the configuration changes rather than per frame.
        {
            static const float overrideValue = []() {
                const char *env = getenv("PDRT64_RT_REFLECT");
                return (env != nullptr) ? float(atof(env)) : 0.0f;
            }();
            rtParams.reflectionOverride = std::clamp(overrideValue, 0.0f, 1.0f);
        }

        rtParams.motionBlurStrength = rtConfig.motionBlurStrength;
        rtParams.motionBlurSamples = uint32_t(std::max(rtConfig.motionBlurSamples, 0));
        rtParams.visualizationMode = rtConfig.visualizationMode;

        if (resolutionChanged || scaleChanged) {
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
