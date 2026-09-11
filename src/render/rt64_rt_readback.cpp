//
// RT64
//

#include "rt64_rt_readback.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace RT64 {
    // D3D12 requires each row of a texture to buffer copy to start on a 256 byte boundary.
    static const uint32_t RowAlignment = 256;

    static uint32_t alignUp(uint32_t value, uint32_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // A readback that declines is a normal answer - nothing traced yet, a target mid-resize -
    // so it cannot report by failing loudly. Under PDRT64_RT_DUMPTARGET it says which reason
    // applied, because "none this frame" on every frame of a run that visibly traced is
    // otherwise indistinguishable from the hook never being called.
    static void declined(const char *reason) {
        if (getenv("PDRT64_RT_DUMPTARGET") != nullptr) {
            fprintf(stderr, "rt64: readback declined - %s\n", reason);
            fflush(stderr);
        }
    }

    // Progress trace. An access violation leaves nothing this computes alive to be printed,
    // so the only way to see how far it got is to say so as it goes.
    static void step(const char *what) {
        static const bool tracing = (getenv("PDRT64_RT_READBACK_TRACE") != nullptr);
        if (tracing) {
            fprintf(stderr, "rt64: readback step - %s\n", what);
            fflush(stderr);
        }
    }

    bool RtReadback::enabled() {
        static const bool on = (getenv("PDRT64_RT_READBACK") != nullptr);
        return on;
    }

    void RtReadback::record(RenderWorker *worker, RenderTarget *target) {
        if (!enabled() || (worker == nullptr)) {
            return;
        }

        // Drain the previous frame's copy first. No fence is needed: the render thread
        // executes and waits on its command list within each frame, so a copy recorded on the
        // last frame's list has completed before this frame is being recorded.
        step("record entered");
        if (pending && (buffer != nullptr)) {
            pending = false;

            const uint8_t *mapped = reinterpret_cast<const uint8_t *>(buffer->map());
            if (mapped == nullptr) {
                declined("could not map the readback buffer");
            }
            else {
                // Both colour formats a target can carry. The tracer composites into the HDR
                // one - measured, it is what this declined on first - and the raster path uses
                // the 8 bit one, so handling only the format that seemed obvious would have
                // read back nothing on exactly the path this exists to see.
                const bool isHdr = (pendingFormat == RenderTarget::colorBufferFormat(true));
                const uint32_t outRowBytes = pendingWidth * 4;

                std::lock_guard<std::mutex> lock(resultMutex);
                resultRgba.resize(size_t(outRowBytes) * pendingHeight);
                for (uint32_t y = 0; y < pendingHeight; y++) {
                    const uint8_t *srcRow = mapped + size_t(y) * pendingAlignedRowBytes;
                    uint8_t *dstRow = resultRgba.data() + size_t(y) * outRowBytes;
                    if (isHdr) {
                        // Both formats are UNORM, so the sixteen bit one is the same value at
                        // more precision and the top byte is it. No tonemapping: the point is
                        // to see what the tracer produced, not what a display would make of it.
                        const uint16_t *src16 = reinterpret_cast<const uint16_t *>(srcRow);
                        for (uint32_t x = 0; x < pendingWidth * 4; x++) {
                            dstRow[x] = uint8_t(src16[x] >> 8);
                        }
                    }
                    else {
                        memcpy(dstRow, srcRow, outRowBytes);
                    }
                }

                resultWidth = pendingWidth;
                resultHeight = pendingHeight;
                resultValid = true;
                buffer->unmap();
            }
        }

        if (target == nullptr) {
            return;
        }

        // The texture, not getResolvedTexture(): this runs immediately after the post process
        // draw, so the live texture is the one carrying the composed image, and the resolved
        // one is only written when something calls resolveTarget. MSAA would need that resolve
        // first and is not handled - the harness this serves runs without it.
        step("reading target");
        RenderTexture *texture = target->texture.get();
        if ((texture == nullptr) || (target->width == 0) || (target->height == 0)) {
            declined("target has no texture or no size");
            return;
        }

        const bool isHdr = (target->format == RenderTarget::colorBufferFormat(true));
        const bool isLdr = (target->format == RenderTarget::colorBufferFormat(false));
        if (!isHdr && !isLdr) {
            declined("target format is neither colour format");
            return;
        }

        const uint32_t width = target->width;
        const uint32_t height = target->height;
        const uint32_t formatSize = RenderFormatSize(target->format);
        const uint32_t alignedRowBytes = alignUp(width * formatSize, RowAlignment);
        const uint64_t neededSize = uint64_t(alignedRowBytes) * height;

        // The target is resized as the resolution scale settles, so the buffer is grown to fit
        // rather than allocated once. Reusing a buffer that is merely large enough keeps this
        // from reallocating every frame.
        step("sizing buffer");
        if ((buffer == nullptr) || (bufferSize < neededSize)) {
            buffer = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(neededSize));
            bufferSize = (buffer != nullptr) ? neededSize : 0;
        }

        if (buffer == nullptr) {
            declined("could not create the readback buffer");
            return;
        }

        // COLOR_WRITE is where the post process draw leaves the target. The transition back is
        // to that same layout, so this is invisible to whatever the frame does next - the
        // caller's own barriers still see the state they expect.
        step("barrier to copy source");
        worker->commandList->barriers(RenderBarrierStage::COPY,
            RenderTextureBarrier(texture, RenderTextureLayout::COPY_SOURCE));

        // PlacedFootprint() leaves RenderTextureCopyLocation::texture null, and
        // D3D12CommandList::copyTextureRegion calls setSamplePositions(dstLocation.texture)
        // before it ever looks at the location's type (contrib/plume/plume_d3d12.cpp:2315).
        // setSamplePositions asserts non-null and then dereferences (:2483-2486), and asserts
        // are compiled out in release - so a texture to buffer copy access violates on its
        // destination location, with the debug layer and GPU based validation both silent
        // because the fault is on the CPU and never reaches the GPU.
        //
        // Every copyTextureRegion in RT64 itself is an upload with a texture destination
        // (rt64_texture_cache.cpp:564, :820), so the readback direction is unexercised
        // upstream. Plume is a submodule we may not edit (invariant 3), so instead the
        // destination carries the source texture in the one field setSamplePositions reads.
        // toD3D12 builds a PLACED_FOOTPRINT location from buffer and placedFootprint only and
        // never reads texture (:692-709), so this changes nothing about the copy - and the
        // sample positions it applies are the source's, which is what a texture to texture
        // copy of the same surface would have used anyway.
        step("copyTextureRegion");
        RenderTextureCopyLocation dstLocation =
            RenderTextureCopyLocation::PlacedFootprint(buffer.get(), target->format, width, height, 1, alignedRowBytes / formatSize);
        dstLocation.texture = texture;

        worker->commandList->copyTextureRegion(dstLocation, RenderTextureCopyLocation::Subresource(texture));

        step("barrier back");
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS,
            RenderTextureBarrier(texture, RenderTextureLayout::COLOR_WRITE));

        step("recorded");
        pendingWidth = width;
        pendingHeight = height;
        pendingAlignedRowBytes = alignedRowBytes;
        pendingFormat = target->format;
        pending = true;
    }

    bool RtReadback::take(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height) {
        std::lock_guard<std::mutex> lock(resultMutex);
        if (!resultValid || resultRgba.empty()) {
            return false;
        }

        rgba = resultRgba;
        width = resultWidth;
        height = resultHeight;
        return true;
    }
};
