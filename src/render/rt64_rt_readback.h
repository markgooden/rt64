//
// RT64
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "rt64_render_target.h"
#include "rt64_render_worker.h"

namespace RT64 {
    // A CPU side copy of the target the path tracer composites into.
    //
    // This exists because no hash of RDRAM can ever show the traced image. RT64 renders each
    // frame twice: State::fullSync (rt64_state.cpp:1349) calls endFramebuffers with
    // raytracingEnabled hardcoded false and copies its colour targets back to RDRAM, and
    // WorkloadQueue::renderFrame (rt64_workload_queue.cpp:756) passes the real flag and
    // renders for presentation only. The pass that fills RDRAM has raytracing switched off at
    // the call site, so the traced image has to be read from the presentation pass directly.
    //
    // The first attempt did the whole copy from the calling thread on a worker of its own and
    // faulted with an access violation inside the copy. The reason is ownership, not the copy:
    // the composite target belongs to the render thread, which resizes it between frames - it
    // was measured going 640x440 then 960x660 on the same address - so a texture read from
    // another thread can be released underneath the copy that is using it.
    //
    // So the copy is recorded on the render thread's own command list, in the frame that drew
    // it, immediately after the compose and post process draws. Ordering then comes from the
    // command list rather than from hope, and nothing touches the target off-thread.
    //
    // The result is one frame behind. record() drains the previous frame's copy before
    // starting a new one, which is safe without a fence because the render thread executes and
    // waits on its command list within each frame (rt64_workload_queue.cpp:894-895), so by the
    // time a frame is being recorded the previous frame's copy has completed. A harness that
    // wants frame N's image should therefore run N+1 frames.
    struct RtReadback {
        // Written and read only on the render thread.
        std::unique_ptr<RenderBuffer> buffer;
        uint64_t bufferSize = 0;
        uint32_t pendingWidth = 0;
        uint32_t pendingHeight = 0;
        uint32_t pendingAlignedRowBytes = 0;
        RenderFormat pendingFormat = RenderFormat::UNKNOWN;
        bool pending = false;

        // Handed across threads, so everything below is under the mutex.
        std::mutex resultMutex;
        std::vector<uint8_t> resultRgba;
        uint32_t resultWidth = 0;
        uint32_t resultHeight = 0;
        bool resultValid = false;

        // True when PDRT64_RT_READBACK is set. Off by default: this costs a full target sized
        // copy every frame, which is worth it for a harness and not for playing the game.
        static bool enabled();

        // Render thread, immediately after the post process draw. Drains the previous frame's
        // copy into the result, then records this frame's.
        void record(RenderWorker *worker, RenderTarget *target);

        // Any thread. Returns false until a frame has actually been traced and drained.
        bool take(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height);
    };
};
