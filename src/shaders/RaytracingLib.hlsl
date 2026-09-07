//
// RT64
//

// The ray tracing pipeline's shader library. Every entry point the frame graph dispatches
// lives here and is exported by name; RaytracingShaderCache looks each one up on the
// created pipeline (rt64_raytracing_shader_cache.cpp).
//
// The frame graph dispatches five ray generation programs by index, in this order, by
// rewriting shaderBindingTableInfo.groups.rayGen.startIndex between traceRays calls
// (rt64_framebuffer_renderer.cpp:808, 833, 838, 848, 861):
//
//   0 primary   1 direct   2 indirect   3 reflection   4 refraction
//
// Only primary visibility is implemented. The other four are declared and clear the buffer
// they own, so the dispatch order, the binding table layout and the compose pass all stay
// intact and the shading passes can be filled in one at a time without touching the frame
// graph. Anything else would mean either changing the guarded driver or leaving stale
// contents in the accumulation buffers for compose to read.

#include "FbRendererCommon.hlsli"
#include "FbRendererRT.hlsli"

struct SurfacePayload {
    float3 normal;
    float t;
    int instanceId;
};

// Matches RenderRaytracingPipelineDesc::maxAttributeSize, which defaults to two floats for
// the built-in triangle intersection.
struct TriangleAttributes {
    float2 barycentrics;
};

// Builds the primary ray for a pixel from the pinhole camera vectors the renderer computes
// each frame (rt64_framebuffer_renderer.cpp:724-737). cameraU, cameraV and cameraW are
// already scaled by the focal distance and field of view there, so this only has to place
// the pixel on the image plane.
static RayDesc primaryRayForPixel(uint2 pixel, uint2 dimensions) {
    const float2 pixelCenter = float2(pixel) + 0.5f + RtParams.pixelJitter;
    float2 screen = (pixelCenter / float2(dimensions)) * 2.0f - 1.0f;

    // Screen space runs downwards while the camera's V vector runs up.
    screen.y = -screen.y;

    RayDesc ray;
    ray.Origin = float3(RtParams.viewI[3][0], RtParams.viewI[3][1], RtParams.viewI[3][2]);
    ray.Direction = normalize(screen.x * RtParams.cameraU.xyz + screen.y * RtParams.cameraV.xyz + RtParams.cameraW.xyz);
    ray.TMin = RtParams.nearDist;
    ray.TMax = RtParams.farDist;
    return ray;
}

[shader("raygeneration")]
void PrimaryRayGen() {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 dimensions = DispatchRaysDimensions().xy;
    const RayDesc ray = primaryRayForPixel(pixel, dimensions);

    SurfacePayload payload;
    payload.normal = float3(0.0f, 0.0f, 0.0f);
    payload.t = -1.0f;
    payload.instanceId = -1;

    // Both query masks are traced: a draw call is tagged with one or the other depending on
    // whether it writes or tests depth (rt64_framebuffer_renderer.cpp:1590), and primary
    // visibility wants everything that is in the scene at all.
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    gInstanceId[pixel] = payload.instanceId;
    gViewDirection[pixel] = float4(ray.Direction, 0.0f);
    gShadingNormal[pixel] = float4(payload.normal, 0.0f);

    if (payload.instanceId >= 0) {
        gShadingPosition[pixel] = float4(ray.Origin + ray.Direction * payload.t, 1.0f);
        gDepth[pixel] = payload.t;
    }
    else {
        gShadingPosition[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        gDepth[pixel] = RtParams.farDist;
    }

    // Cleared rather than left stale: compose reads all of these unconditionally
    // (shaders/ComposePS.hlsl:19-25), so whatever the shading passes have not written yet
    // must read as nothing rather than as last frame's contents.
    gShadingSpecular[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    gDiffuse[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    gFlow[pixel] = float2(0.0f, 0.0f);
    gReactiveMask[pixel] = 0.0f;
    gLockMask[pixel] = 0.0f;
    gNormalRoughness[pixel] = float4(payload.normal, 1.0f);
}

[shader("raygeneration")]
void DirectRayGen() {
    const uint2 pixel = DispatchRaysIndex().xy;
    gDirectLightAccum[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

[shader("raygeneration")]
void IndirectRayGen() {
    const uint2 pixel = DispatchRaysIndex().xy;
    gIndirectLightAccum[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

[shader("raygeneration")]
void ReflectionRayGen() {
    const uint2 pixel = DispatchRaysIndex().xy;
    gReflection[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

[shader("raygeneration")]
void RefractionRayGen() {
    const uint2 pixel = DispatchRaysIndex().xy;
    gRefraction[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    gTransparent[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

[shader("closesthit")]
void SurfaceClosestHit(inout SurfacePayload payload, in TriangleAttributes attributes) {
    payload.instanceId = int(InstanceID());
    payload.t = RayTCurrent();

    // A geometric normal from the triangle's own vertices would need this draw call's offset
    // into the shared index buffer, which the instance does not carry yet. Until the vertex
    // fetch convention is settled, the normal faces the ray, which is enough for the
    // instance ID and shading position views to be read honestly and visibly wrong for the
    // normal view rather than plausibly wrong.
    payload.normal = -WorldRayDirection();
}

[shader("miss")]
void SurfaceMiss(inout SurfacePayload payload) {
    payload.instanceId = -1;
    payload.t = -1.0f;
    payload.normal = float3(0.0f, 0.0f, 0.0f);
}

[shader("closesthit")]
void ShadowClosestHit(inout SurfacePayload payload, in TriangleAttributes attributes) {
    payload.instanceId = int(InstanceID());
    payload.t = RayTCurrent();
}

[shader("miss")]
void ShadowMiss(inout SurfacePayload payload) {
    payload.instanceId = -1;
    payload.t = -1.0f;
}
