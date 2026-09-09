//
// RT64
//

#pragma once

#include <vector>

#include "shared/rt64_point_light.h"

namespace RT64 {
    // Lights supplied from outside the display list.
    //
    // Perfect Dark bakes its level lighting into vertex colours and barely uses the RSP
    // light path - 32 of 5878 vertices on an in-level frame - so a path tracer cannot get
    // its lights from the stream. It gets them from the game's own room lights, gathered
    // port side and handed over here (port/rt64/rt64_lights.h).
    //
    // A frame's worth, replaced whole each time rather than accumulated: the set is small,
    // it changes as rooms come on screen, and a light that has been shot out has to stop
    // existing rather than linger.
    struct ExternalLights {
        static void set(const interop::PointLight *lights, uint32_t count);
        static const std::vector<interop::PointLight> &get();
    };
};
