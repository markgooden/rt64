//
// RT64
//

#include "rt64_external_lights.h"

namespace RT64 {
    // One frame's lights. The producer writes this between frames and the frame graph reads
    // it while building the scene, both on the same thread that drives the backend.
    static std::vector<interop::PointLight> g_externalLights;

    void ExternalLights::set(const interop::PointLight *lights, uint32_t count) {
        g_externalLights.clear();
        if ((lights != nullptr) && (count > 0)) {
            g_externalLights.assign(lights, lights + count);
        }
    }

    const std::vector<interop::PointLight> &ExternalLights::get() {
        return g_externalLights;
    }
};
