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

    // A readback that declines is a normal answer - raytracing off, nothing traced yet - so
    // it cannot report by failing loudly. Under PDRT64_RT_DUMPTARGET it says which of the
    // reasons applied, because "none this frame" on every frame of a run that visibly traced
    // is otherwise indistinguishable from the hook never being called.
    static bool declined(const char *reason) {
        if (getenv("PDRT64_RT_DUMPTARGET") != nullptr) {
            fprintf(stderr, "rt64: readback declined - %s\n", reason);
            fflush(stderr);
        }

        return false;
    }

    bool rtReadbackTargetRgba(RenderDevice *device, RenderTarget *target, std::vector<uint8_t> &rgba,
        uint32_t &outWidth, uint32_t &outHeight)
    {
        if (device == nullptr) {
            return declined("no device");
        }

        if (target == nullptr) {
            return declined("no composited target");
        }

        // A worker of its own, rather than borrowing one of the application's. Those are
        // driven by the render threads, and opening a command list on one from here crashed
        // outright - two writers on the same list. One is built and torn down per call, which
        // costs nothing that matters for a debug readback taken once a frame in a harness.
        RenderWorker readbackWorker(device, "RT64 RT Readback", RenderCommandListType::DIRECT);
        RenderWorker *worker = &readbackWorker;

        RenderTexture *texture = target->getResolvedTexture();
        if ((texture == nullptr) || (target->width == 0) || (target->height == 0)) {
            return declined("target has no resolved texture or no size");
        }

        // Both colour formats a target can carry. The tracer composites into the HDR one -
        // measured, it is what this declined on first - and the raster path uses the 8 bit
        // one, so handling only the format that seemed obvious would have read back nothing
        // on exactly the path this exists to see.
        const bool isHdr = (target->format == RenderTarget::colorBufferFormat(true));
        const bool isLdr = (target->format == RenderTarget::colorBufferFormat(false));
        if (!isHdr && !isLdr) {
            if (getenv("PDRT64_RT_DUMPTARGET") != nullptr) {
                fprintf(stderr, "rt64: readback declined - format %d is neither colour format (%d, %d)\n",
                    int(target->format), int(RenderTarget::colorBufferFormat(false)),
                    int(RenderTarget::colorBufferFormat(true)));
                fflush(stderr);
            }

            return false;
        }

        const uint32_t width = target->width;
        const uint32_t height = target->height;
        const uint32_t formatSize = RenderFormatSize(target->format);
        const uint32_t rowBytes = width * formatSize;
        const uint32_t alignedRowBytes = alignUp(rowBytes, RowAlignment);
        const uint64_t bufferSize = uint64_t(alignedRowBytes) * height;

        std::unique_ptr<RenderBuffer> readbackBuffer =
            device->createBuffer(RenderBufferDesc::ReadbackBuffer(bufferSize));
        if (readbackBuffer == nullptr) {
            return declined("could not create the readback buffer");
        }

        {
            // The target is left in SHADER_READ, which is where the compose pass leaves it and
            // where the rest of the frame expects to find it. Putting it back is not optional:
            // this runs mid-frame, between other work on the same target.
            RenderWorkerExecution execution(worker);
            worker->commandList->barriers(RenderBarrierStage::COPY,
                RenderTextureBarrier(texture, RenderTextureLayout::COPY_SOURCE));

            worker->commandList->copyTextureRegion(
                RenderTextureCopyLocation::PlacedFootprint(readbackBuffer.get(), target->format, width, height, 1, alignedRowBytes / formatSize),
                RenderTextureCopyLocation::Subresource(texture));

            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE,
                RenderTextureBarrier(texture, RenderTextureLayout::SHADER_READ));
        }

        worker->wait();

        const uint8_t *mapped = reinterpret_cast<const uint8_t *>(readbackBuffer->map());
        if (mapped == nullptr) {
            return declined("could not map the readback buffer");
        }

        // Drop the copy's row padding, so a caller gets an image rather than a footprint, and
        // narrow an HDR target to eight bits a channel on the way out. Both formats are UNORM,
        // so the sixteen bit one is the same value at more precision and the top byte is it -
        // no tonemapping is wanted here, since the point is to see what the tracer produced
        // rather than what a display would make of it.
        const uint32_t outRowBytes = width * 4;
        rgba.resize(size_t(outRowBytes) * height);
        for (uint32_t y = 0; y < height; y++) {
            const uint8_t *srcRow = mapped + size_t(y) * alignedRowBytes;
            uint8_t *dstRow = rgba.data() + size_t(y) * outRowBytes;
            if (isHdr) {
                const uint16_t *src16 = reinterpret_cast<const uint16_t *>(srcRow);
                for (uint32_t x = 0; x < width * 4; x++) {
                    dstRow[x] = uint8_t(src16[x] >> 8);
                }
            }
            else {
                memcpy(dstRow, srcRow, outRowBytes);
            }
        }

        readbackBuffer->unmap();

        outWidth = width;
        outHeight = height;
        return true;
    }
};
