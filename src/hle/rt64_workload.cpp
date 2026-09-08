//
// RT64
//

#include "rt64_workload.h"

namespace RT64 {
    // Common functions.

    static uint64_t roundUp(uint64_t value, uint64_t powerOf2Alignment) {
        return (value + powerOf2Alignment - 1) & ~(powerOf2Alignment - 1);
    }

    // Workload

    void Workload::reset() {
        submissionFrame = 0;
        fbPairCount = 0;
        fbPairSubmitted = 0;
        gameCallCount = 0;
        viOriginalRate = 0;
        workloadId = 0;
        extended.testZIndexCount = 0;
        extended.ditherNoiseStrength = 1.0f;

        commandWarnings.clear();
        spriteCommands.clear();
        pointLights.clear();
        fbChangePool.reset();
        fbStorage.reset();
        physicalAddressTransformMap.clear();
        transformIdMap.clear();

        // Always make sure there's at least one framebuffer pair, even if it's not configured yet, since a game can technically send
        // information to the RSP before it draws anything to a color image.
        adjustVector(fbPairs, 1);
        fbPairs[0].reset();

        resetDrawData();
        resetDrawDataRanges();
        resetRSPOutputBuffers();
        resetWorldOutputBuffers();
    }

    void Workload::resetDrawData() {
        drawData.posFloats.clear();
        drawData.velFloats.clear();
        drawData.tcFloats.clear();
        drawData.tcVelFloats.clear();
        drawData.normColBytes.clear();
        drawData.viewProjIndices.clear();
        drawData.worldIndices.clear();
        drawData.fogIndices.clear();
        drawData.lightIndices.clear();
        drawData.lightCounts.clear();
        drawData.lookAtIndices.clear();
        drawData.faceIndices.clear();
        drawData.modifyPosUints.clear();
        drawData.rdpParams.clear();
        drawData.extraParams.clear();
        drawData.renderParams.clear();
        drawData.posTransformed.clear();
        drawData.posScreen.clear();
        drawData.rdpTiles.clear();
        drawData.lerpRdpTiles.clear();
        drawData.gpuTiles.clear();
        drawData.callTiles.clear();
        drawData.rspViewports.clear();
        drawData.viewportClipRatios.clear();
        drawData.viewportOrigins.clear();
        drawData.rspFog.clear();
        drawData.rspLights.clear();
        drawData.rspLookAt.clear();
        drawData.lerpRspLookAt.clear();
        drawData.loadOperations.clear();
        drawData.worldTransforms.clear();
        drawData.viewTransforms.clear();
        drawData.projTransforms.clear();
        drawData.viewProjTransforms.clear();
        drawData.modViewTransforms.clear();
        drawData.modProjTransforms.clear();
        drawData.modViewProjTransforms.clear();
        drawData.prevViewTransforms.clear();
        drawData.prevProjTransforms.clear();
        drawData.prevViewProjTransforms.clear();
        drawData.lerpWorldTransforms.clear();
        drawData.prevWorldTransforms.clear();
        drawData.invTWorldTransforms.clear();
        drawData.triPosFloats.clear();
        drawData.triTcFloats.clear();
        drawData.triColorFloats.clear();
        drawData.transformGroups.clear();
        drawData.worldTransformGroups.clear();
        drawData.viewProjTransformGroups.clear();
        drawData.worldTransformSegmentedAddresses.clear();
        drawData.worldTransformPhysicalAddresses.clear();
        drawData.worldTransformVertexIndices.clear();
        drawData.viewportClipRatios.push_back(1);
        drawData.viewportClipRatios.push_back(1);
        drawData.viewportClipRatios.push_back(-1);
        drawData.viewportClipRatios.push_back(-1);

        // Push an identity matrix into the transforms by default so rects can use them.
        drawData.rspViewports.push_back(interop::RSPViewport::identity());
        drawData.viewTransforms.push_back(interop::float4x4::identity());
        drawData.projTransforms.push_back(interop::float4x4::identity());
        drawData.viewProjTransforms.push_back(interop::float4x4::identity());
        drawData.worldTransforms.push_back(interop::float4x4::identity());
        drawData.transformGroups.push_back(TransformGroup());
        drawData.worldTransformGroups.push_back(0);
        drawData.viewProjTransformGroups.push_back(0);
        drawData.worldTransformSegmentedAddresses.push_back(0);
        drawData.worldTransformPhysicalAddresses.push_back(0);
        drawData.worldTransformVertexIndices.push_back(0);
        drawData.viewportOrigins.push_back(0);
    }

    void Workload::resetDrawDataRanges() {
        auto &r = drawRanges;
        r.posFloats = { 0, 0 };
        r.velFloats = { 0, 0 };
        r.tcFloats = { 0, 0 };
        r.tcVelFloats = { 0, 0 };
        r.normColBytes = { 0, 0 };
        r.viewProjIndices = { 0, 0 };
        r.worldIndices = { 0, 0 };
        r.fogIndices = { 0, 0 };
        r.lightIndices = { 0, 0 };
        r.lightCounts = { 0, 0 };
        r.lookAtIndices = { 0, 0 };
        r.faceIndices = { 0, 0 };
        r.modifyPosUints = { 0, 0 };
        r.rdpParams = { 0, 0 };
        r.extraParams = { 0, 0 };
        r.renderParams = { 0, 0 };
        r.viewProjTransforms = { 0, 0 };
        r.worldTransforms = { 0, 0 };
        r.rdpTiles = { 0, 0 };
        r.gpuTiles = { 0, 0 };
        r.callTiles = { 0, 0 };
        r.rspViewports = { 0, 0 };
        r.rspFog = { 0, 0 };
        r.rspLights = { 0, 0 };
        r.rspLookAt = { 0, 0 };
        r.loadOperations = { 0, 0 };
        r.triPosFloats = { 0, 0 };
        r.triTcFloats = { 0, 0 };
        r.triColorFloats = { 0, 0 };
    }

    void Workload::resetRSPOutputBuffers() {
        outputBuffers.screenPosBuffer.computedSize = 0;
        outputBuffers.genTexCoordBuffer.computedSize = 0;
        outputBuffers.shadedColBuffer.computedSize = 0;
    }

    void Workload::resetWorldOutputBuffers() {
        outputBuffers.worldPosBuffer.computedSize = 0;
        outputBuffers.worldNormBuffer.computedSize = 0;
        outputBuffers.worldVelBuffer.computedSize = 0;
    }

    void Workload::updateDrawDataRanges() {
        auto &r = drawRanges;
        r.posFloats.second = drawData.posFloats.size();
        r.velFloats.second = drawData.velFloats.size();
        r.tcFloats.second = drawData.tcFloats.size();
        r.tcVelFloats.second = drawData.tcVelFloats.size();
        r.normColBytes.second = drawData.normColBytes.size();
        r.viewProjIndices.second = drawData.viewProjIndices.size();
        r.worldIndices.second = drawData.worldIndices.size();
        r.fogIndices.second = drawData.fogIndices.size();
        r.lightIndices.second = drawData.lightIndices.size();
        r.lightCounts.second = drawData.lightCounts.size();
        r.lookAtIndices.second = drawData.lookAtIndices.size();
        r.faceIndices.second = drawData.faceIndices.size();
        r.modifyPosUints.second = drawData.modifyPosUints.size();
        r.rdpParams.second = drawData.rdpParams.size();
        r.extraParams.second = drawData.extraParams.size();
        r.renderParams.second = drawData.renderParams.size();
        r.viewProjTransforms.second = drawData.viewProjTransforms.size();
        r.worldTransforms.second = drawData.worldTransforms.size();
        r.rdpTiles.second = drawData.rdpTiles.size();
        r.gpuTiles.second = drawData.gpuTiles.size();
        r.callTiles.second = drawData.callTiles.size();
        r.rspViewports.second = drawData.rspViewports.size();
        r.rspFog.second = drawData.rspFog.size();
        r.rspLights.second = drawData.rspLights.size();
        r.rspLookAt.second = drawData.rspLookAt.size();
        r.loadOperations.second = drawData.loadOperations.size();
        r.triPosFloats.second = drawData.triPosFloats.size();
        r.triTcFloats.second = drawData.triTcFloats.size();
        r.triColorFloats.second = drawData.triColorFloats.size();
    }
    
    void Workload::uploadDrawData(RenderWorker *worker, BufferUploader *bufferUploader) {
        // PDRT64_RT_DUMPCAM: where the geometry actually is. The primary rays start at the
        // camera the projection processor recovered, so the object space extent and the
        // transforms applied to it are what say whether a ray could reach it at all.
        const char *dumpCam = getenv("PDRT64_RT_DUMPCAM");
        if (dumpCam != nullptr) {
            // The value is the frame interval, so a short driven run can be sampled
            // finely without a rebuild. PDRT64_RT_DUMPCAM=1 alone keeps the old cadence.
            const int dumpGeomEvery = std::max(1, atoi(dumpCam));
            static uint32_t geomCalls = 0;
            geomCalls++;
            if ((geomCalls < (dumpGeomEvery * 12)) && ((geomCalls % dumpGeomEvery) == 0) && !drawData.posFloats.empty()) {
                float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
                const size_t vertexCount = drawData.posFloats.size() / 3;
                for (size_t v = 0; v < vertexCount; v++) {
                    for (int a = 0; a < 3; a++) {
                        const float c = drawData.posFloats[v * 3 + a];
                        lo[a] = std::min(lo[a], c);
                        hi[a] = std::max(hi[a], c);
                    }
                }

                fprintf(stderr, "rt64: geom call %u: %zu verts, object aabb (%.1f %.1f %.1f) - (%.1f %.1f %.1f)\n",
                    geomCalls, vertexCount, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
                fprintf(stderr, "rt64: %zu world transforms, %zu viewProj transforms\n",
                    drawData.worldTransforms.size(), drawData.viewProjTransforms.size());

                // The centroid and extent of every world transform's translation, rather
                // than a few by index: indices are per-frame draw order and do not name the
                // same object twice, but the cloud as a whole is stable. If the view is
                // folded into these transforms the whole cloud moves rigidly with the
                // camera; if they are true world space it stays put while the camera moves.
                // Only a moving camera tells those apart, which is what tools/rtdrive.ps1
                // is for.
                if (!drawData.worldTransforms.empty()) {
                    double sum[3] = { 0.0, 0.0, 0.0 };
                    float wlo[3] = { 1e30f, 1e30f, 1e30f }, whi[3] = { -1e30f, -1e30f, -1e30f };
                    for (const interop::float4x4 &m : drawData.worldTransforms) {
                        for (int a = 0; a < 3; a++) {
                            sum[a] += m[3][a];
                            wlo[a] = std::min(wlo[a], m[3][a]);
                            whi[a] = std::max(whi[a], m[3][a]);
                        }
                    }

                    const double n = double(drawData.worldTransforms.size());
                    fprintf(stderr, "rt64:   world translation centroid %.1f %.1f %.1f\n",
                        sum[0] / n, sum[1] / n, sum[2] / n);
                    fprintf(stderr, "rt64:   world translation aabb (%.1f %.1f %.1f) - (%.1f %.1f %.1f)\n",
                        wlo[0], wlo[1], wlo[2], whi[0], whi[1], whi[2]);
                }

                // The game's own view-projection, as a witness that the camera actually
                // moved. This is the matrix that puts world vertices on screen, so if the
                // camera lives in the projection stack this row changes as the player moves,
                // and if it lives in the modelview it does not. Reading it next to the world
                // translation cloud above is what separates the two: exactly one of them can
                // hold the camera.
                // Every perspective view-projection, not just the first: index 0 is the 2D
                // projection the HUD is drawn with and is always identity, which says nothing
                // about the camera. A perspective matrix is the one with [3][3] == 0.
                for (size_t v = 0; v < drawData.viewProjTransforms.size(); v++) {
                    const interop::float4x4 &vp = drawData.viewProjTransforms[v];
                    if (vp[3][3] != 0.0f) {
                        continue;
                    }

                    fprintf(stderr, "rt64:   viewProj[%zu] row3 %.2f %.2f %.2f  row2 %.2f %.2f %.2f\n",
                        v, vp[3][0], vp[3][1], vp[3][2], vp[2][0], vp[2][1], vp[2][2]);
                }

                fflush(stderr);
            }
        }

        const RenderBufferFlags rtInputFlag = worker->device->getCapabilities().raytracing ? RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT : RenderBufferFlag::NONE;
        bufferUploader->submit(worker, {
            { drawData.posFloats.data(), drawRanges.posFloats, sizeof(float), RenderBufferFlag::FORMATTED, { RenderFormat::R32_FLOAT }, &drawBuffers.positionBuffer },
            { drawData.velFloats.data(), drawRanges.velFloats, sizeof(float), RenderBufferFlag::FORMATTED, { RenderFormat::R32_FLOAT }, &drawBuffers.velocityBuffer },
            { drawData.tcFloats.data(), drawRanges.tcFloats, sizeof(float), RenderBufferFlag::FORMATTED, { RenderFormat::R32_FLOAT }, &drawBuffers.texcoordBuffer },
            { drawData.tcVelFloats.data(), drawRanges.tcVelFloats, sizeof(float), RenderBufferFlag::FORMATTED, { RenderFormat::R32_FLOAT }, &drawBuffers.texcoordVelocityBuffer },
            { drawData.normColBytes.data(), drawRanges.normColBytes, sizeof(uint8_t), RenderBufferFlag::FORMATTED | RenderBufferFlag::STORAGE, { RenderFormat::R8_UINT, RenderFormat::R8_SINT }, &drawBuffers.normalColorBuffer },
            { drawData.viewProjIndices.data(), drawRanges.viewProjIndices, sizeof(uint16_t), RenderBufferFlag::FORMATTED | RenderBufferFlag::STORAGE, { RenderFormat::R16_UINT }, &drawBuffers.viewProjIndicesBuffer },
            { drawData.worldIndices.data(), drawRanges.worldIndices, sizeof(uint16_t), RenderBufferFlag::FORMATTED | RenderBufferFlag::STORAGE, { RenderFormat::R16_UINT }, &drawBuffers.worldIndicesBuffer },
            { drawData.fogIndices.data(), drawRanges.fogIndices, sizeof(uint16_t), RenderBufferFlag::FORMATTED | RenderBufferFlag::STORAGE, { RenderFormat::R16_UINT }, &drawBuffers.fogIndicesBuffer },
            { drawData.lightIndices.data(), drawRanges.lightIndices, sizeof(uint16_t), RenderBufferFlag::FORMATTED | RenderBufferFlag::STORAGE, { RenderFormat::R16_UINT }, &drawBuffers.lightIndicesBuffer },
            { drawData.lightCounts.data(), drawRanges.lightCounts, sizeof(uint8_t), RenderBufferFlag::FORMATTED | RenderBufferFlag::STORAGE, { RenderFormat::R8_UINT }, &drawBuffers.lightCountsBuffer },
            { drawData.lookAtIndices.data(), drawRanges.lookAtIndices, sizeof(uint16_t), RenderBufferFlag::FORMATTED | RenderBufferFlag::STORAGE, { RenderFormat::R16_UINT }, &drawBuffers.lookAtIndicesBuffer },
            { drawData.faceIndices.data(), drawRanges.faceIndices, sizeof(uint32_t), RenderBufferFlag::INDEX | RenderBufferFlag::STORAGE | rtInputFlag, { }, &drawBuffers.faceIndicesBuffer },
            { drawData.modifyPosUints.data(), drawRanges.modifyPosUints, sizeof(uint32_t), RenderBufferFlag::FORMATTED, { RenderFormat::R32_UINT }, &drawBuffers.modifyPosUintsBuffer },
            { drawData.rdpParams.data(), drawRanges.rdpParams, sizeof(interop::RDPParams), RenderBufferFlag::STORAGE, { }, &drawBuffers.rdpParamsBuffer },
            { drawData.renderParams.data(), drawRanges.renderParams, sizeof(interop::RenderParams), RenderBufferFlag::STORAGE, { }, &drawBuffers.renderParamsBuffer },
            { drawData.rdpTiles.data(), drawRanges.rdpTiles, sizeof(interop::RDPTile), RenderBufferFlag::STORAGE, {}, &drawBuffers.rdpTilesBuffer },
            { drawData.rspViewports.data(), drawRanges.rspViewports, sizeof(interop::RSPViewport), RenderBufferFlag::STORAGE, { }, &drawBuffers.rspViewportsBuffer },
            { drawData.rspFog.data(), drawRanges.rspFog, sizeof(interop::RSPFog), RenderBufferFlag::STORAGE, { }, &drawBuffers.rspFogBuffer },
            { drawData.rspLights.data(), drawRanges.rspLights, sizeof(interop::RSPLight), RenderBufferFlag::STORAGE, { }, &drawBuffers.rspLightsBuffer },
            { drawData.rspLookAt.data(), drawRanges.rspLookAt, sizeof(interop::RSPLookAt), RenderBufferFlag::STORAGE, { }, &drawBuffers.rspLookAtBuffer },
            { drawData.triPosFloats.data(), drawRanges.triPosFloats, sizeof(float), RenderBufferFlag::VERTEX, { }, &drawBuffers.triPosBuffer },
            { drawData.triTcFloats.data(), drawRanges.triTcFloats, sizeof(float), RenderBufferFlag::VERTEX, { }, &drawBuffers.triTcBuffer },
            { drawData.triColorFloats.data(), drawRanges.triColorFloats, sizeof(float), RenderBufferFlag::VERTEX, { }, &drawBuffers.triColorBuffer }
        });
    }

    void updateOutputBuffer(RenderWorker *worker, ComputedBuffer &computedBuffer, uint64_t requiredSize, RenderBufferFlags flags = RenderBufferFlag::NONE) {
        if (computedBuffer.allocatedSize >= requiredSize) {
            return;
        }

        // Recreate the buffer.
        computedBuffer.allocatedSize = (requiredSize * 3) / 2;
        computedBuffer.allocatedSize = roundUp(computedBuffer.allocatedSize, 256);
        computedBuffer.buffer = worker->device->createBuffer(RenderBufferDesc::DefaultBuffer(computedBuffer.allocatedSize, flags | RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS));

        // Set the computed size to 0 if it was recreated.
        computedBuffer.computedSize = 0;
    }

    void Workload::updateOutputBuffers(RenderWorker *worker) {
        const uint32_t vertexCount = drawData.vertexCount();
        const RenderBufferFlags rtInputFlag = worker->device->getCapabilities().raytracing ? RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT : RenderBufferFlag::NONE;
        updateOutputBuffer(worker, outputBuffers.screenPosBuffer, vertexCount * sizeof(float) * 4, RenderBufferFlag::VERTEX);
        updateOutputBuffer(worker, outputBuffers.genTexCoordBuffer, vertexCount * sizeof(float) * 2, RenderBufferFlag::VERTEX);
        updateOutputBuffer(worker, outputBuffers.shadedColBuffer, vertexCount * sizeof(float) * 4, RenderBufferFlag::VERTEX);
        updateOutputBuffer(worker, outputBuffers.worldPosBuffer, vertexCount * sizeof(float) * 4, rtInputFlag);
        updateOutputBuffer(worker, outputBuffers.worldNormBuffer, vertexCount * sizeof(float) * 4);
        updateOutputBuffer(worker, outputBuffers.worldVelBuffer, vertexCount * sizeof(float) * 4);
        updateOutputBuffer(worker, outputBuffers.testZIndexBuffer, extended.testZIndexCount * sizeof(uint32_t), RenderBufferFlag::INDEX | RenderBufferFlag::STORAGE);
    }

    void nextDrawDataRange(DrawRanges::Range &range) {
        range.first = range.second;
    }

    void Workload::nextDrawDataRanges() {
        auto &r = drawRanges;
        nextDrawDataRange(r.posFloats);
        nextDrawDataRange(r.velFloats);
        nextDrawDataRange(r.tcFloats);
        nextDrawDataRange(r.tcVelFloats);
        nextDrawDataRange(r.normColBytes);
        nextDrawDataRange(r.viewProjIndices);
        nextDrawDataRange(r.worldIndices);
        nextDrawDataRange(r.fogIndices);
        nextDrawDataRange(r.lightIndices);
        nextDrawDataRange(r.lightCounts);
        nextDrawDataRange(r.lookAtIndices);
        nextDrawDataRange(r.faceIndices);
        nextDrawDataRange(r.modifyPosUints);
        nextDrawDataRange(r.rdpParams);
        nextDrawDataRange(r.extraParams);
        nextDrawDataRange(r.renderParams);
        nextDrawDataRange(r.viewProjTransforms);
        nextDrawDataRange(r.worldTransforms);
        nextDrawDataRange(r.rdpTiles);
        nextDrawDataRange(r.gpuTiles);
        nextDrawDataRange(r.callTiles);
        nextDrawDataRange(r.rspViewports);
        nextDrawDataRange(r.rspFog);
        nextDrawDataRange(r.rspLights);
        nextDrawDataRange(r.rspLookAt);
        nextDrawDataRange(r.loadOperations);
        nextDrawDataRange(r.triPosFloats);
        nextDrawDataRange(r.triTcFloats);
        nextDrawDataRange(r.triColorFloats);
    }

    void Workload::begin(uint64_t submissionFrame) {
        reset();

        this->submissionFrame = submissionFrame;
    }

    bool Workload::addFramebufferPair(uint32_t colorAddress, uint8_t colorFmt, uint8_t colorSiz, uint16_t colorWidth, uint32_t depthAddress) {
        uint32_t fbPairIndex;
        bool addedPair = false;
        if ((fbPairCount == 0) || !fbPairs[fbPairCount - 1].isEmpty()) {
            fbPairIndex = fbPairCount++;
            adjustVector(fbPairs, fbPairCount);
            addedPair = true;
        }
        else {
            fbPairIndex = fbPairCount - 1;
            addedPair = false;
        }

        auto &fbPair = fbPairs[fbPairIndex];
        fbPair.reset();
        fbPair.colorImage.address = colorAddress;
        fbPair.colorImage.fmt = colorFmt;
        fbPair.colorImage.siz = colorSiz;
        fbPair.colorImage.width = colorWidth;
        fbPair.depthImage.address = depthAddress;

        return addedPair;
    }

    int Workload::currentFramebufferPairIndex() const {
        if (fbPairCount > 0) {
            return fbPairCount - 1;
        }
        else {
            return 0;
        }
    }
};