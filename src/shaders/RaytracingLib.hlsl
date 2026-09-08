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
#include "TextureSampler.hlsli"

struct SurfacePayload {
    float3 normal;
    float3 albedo;
    float t;
    int instanceId;
};

// Matches RenderRaytracingPipelineDesc::maxAttributeSize, which defaults to two floats for
// the built-in triangle intersection.
struct TriangleAttributes {
    float2 barycentrics;
};

// Builds the primary ray for a pixel from the pinhole camera vectors the renderer computes
// each frame (rt64_framebuffer_renderer.cpp:846-863). cameraU, cameraV and cameraW are
// already scaled by the focal distance and field of view there, so this only has to place
// the pixel on the image plane.
static RayDesc primaryRayForPixel(uint2 pixel, uint2 dimensions) {
    const float2 pixelCenter = float2(pixel) + 0.5f + RtParams.pixelJitter;
    float2 screen = (pixelCenter / float2(dimensions)) * 2.0f - 1.0f;

    // Screen space runs downwards while the camera's V vector runs up.
    screen.y = -screen.y;

    RayDesc ray;

    // RtParams arrives transposed, so the camera position is viewI's column 3 and not its
    // row 3. The CPU stores a float4x4 as row-major memory (shared/rt64_hlsl.h:222) and DXC
    // packs cbuffer matrices column-major, since nothing passes -Zpr. RT64's own shaders
    // depend on that: RSPWorldCS.hlsl:39 is mul(worldMat, float4(pos, 1.0f)), which only
    // translates when the translation sits in the HLSL matrix's column 3. Reading row 3
    // returned (0, 0, 0) for any affine view matrix, so every primary ray started at the
    // world origin and missed the scene - gDepth came back saturated at farDist over the
    // whole frame, which is what the Depth debug view showed.
    ray.Origin = float3(RtParams.viewI[0][3], RtParams.viewI[1][3], RtParams.viewI[2][3]);
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
    payload.albedo = float3(0.0f, 0.0f, 0.0f);
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

    // The surface colour where the ray landed. Nothing lights it yet - the direct and
    // indirect passes are still stubs - so this is the albedo on its own, which is what the
    // Diffuse debug view is for.
    gDiffuse[pixel] = float4(payload.albedo, 1.0f);
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

// Fetches the surface where the ray struck it. The instance carries its draw call index
// (rt64_raytracing_resources.cpp:468), so instanceRenderIndices gives where this draw
// call's triangles begin in the shared index buffer, and PrimitiveIndex() counts triangles
// from exactly there - the bottom level structure was built over the index buffer starting
// at that offset (rt64_framebuffer_renderer.cpp:1892) with the vertex buffer whole from
// zero, so the indices it reads are already global.
[shader("closesthit")]
void SurfaceClosestHit(inout SurfacePayload payload, in TriangleAttributes attributes) {
    payload.instanceId = int(InstanceID());
    payload.t = RayTCurrent();

    const RenderIndices renderIndices = instanceRenderIndices[InstanceID()];
    const uint indexStart = renderIndices.faceIndicesStart + PrimitiveIndex() * 3;
    const uint i0 = indexBuffer.Load(indexStart * 4);
    const uint i1 = indexBuffer.Load((indexStart + 1) * 4);
    const uint i2 = indexBuffer.Load((indexStart + 2) * 4);

    // posBuffer is the world space position the RSP world compute pass writes, four floats
    // per vertex, so the normal comes out in world space with no further transform.
    const float3 p0 = asfloat(posBuffer.Load3(i0 * 16));
    const float3 p1 = asfloat(posBuffer.Load3(i1 * 16));
    const float3 p2 = asfloat(posBuffer.Load3(i2 * 16));
    const float3 geometricNormal = normalize(cross(p1 - p0, p2 - p0));

    // Turned to face the ray. The N64 draws plenty of geometry double sided and the winding
    // of a back face would otherwise light it from behind.
    payload.normal = (dot(geometricNormal, WorldRayDirection()) > 0.0f) ? -geometricNormal : geometricNormal;

    // The shaded vertex colour the RSP pass already computed, interpolated across the
    // triangle. This is the game's own vertex lighting rather than a material albedo, and it
    // stands in as one until textures and the colour combiner are read here: it is what the
    // raster path would have started from for the same triangle.
    const float3 barycentrics = float3(1.0f - attributes.barycentrics.x - attributes.barycentrics.y,
        attributes.barycentrics.x, attributes.barycentrics.y);
    const float3 c0 = asfloat(shadedColBuffer.Load4(i0 * 16)).rgb;
    const float3 c1 = asfloat(shadedColBuffer.Load4(i1 * 16)).rgb;
    const float3 c2 = asfloat(shadedColBuffer.Load4(i2 * 16)).rgb;
    const float3 shadeColor = c0 * barycentrics.x + c1 * barycentrics.y + c2 * barycentrics.z;

    // The draw call's render state, reached through its instance index the same way the
    // raster pixel shader reaches it (RasterPS.hlsl:51).
    const RenderParams rp = DynamicRenderParams[renderIndices.instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };

    float3 texelColor = float3(1.0f, 1.0f, 1.0f);
    if (renderFlagUsesTexture0(rp.flags)) {
        // genTexCoordBuffer holds the texture coordinates the RSP pass generated, two
        // floats per vertex.
        const float2 t0 = asfloat(genTexCoordBuffer.Load2(i0 * 8));
        const float2 t1 = asfloat(genTexCoordBuffer.Load2(i1 * 8));
        const float2 t2 = asfloat(genTexCoordBuffer.Load2(i2 * 8));
        const float2 vertexUV = t0 * barycentrics.x + t1 * barycentrics.y + t2 * barycentrics.z;

        const uint globalTileIndex = renderIndices.rdpTileIndex;
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        const GPUTile gpuTile = GPUTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = renderCMS0(rp.flags);
            rdpTile.cmt = renderCMT0(rp.flags);
            rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
        }

        // The same coordinate transform sampleTexture applies before sampling
        // (TextureSampler.hlsli:218-245), minus the parts that only make sense for a
        // rasterized rect. It is reproduced rather than called because sampleTexture picks a
        // mip level from screen space derivatives, and a ray has none.
        const bool texturePerspective = (otherMode.textPersp() == G_TP_PERSP);
        const float perspCorrectionMod = texturePerspective ? 1.0f : 0.5f;
        float2 uvCoord = vertexUV * perspCorrectionMod * float2(rdpTile.shifts, rdpTile.shiftt);
        if (gpuTileFlagShiftedByHalf(gpuTile.flags)) {
            uvCoord += float2(0.5f, 0.5f);
        }

        uvCoord *= gpuTile.tcScale;
        uvCoord -= (float2(rdpTile.uls, rdpTile.ult) * gpuTile.ulScale) / 4.0f;

        const uint filter = otherMode.textFilt();
        const bool filterBilerp = (filter != G_TF_POINT) && (otherMode.cycleType() != G_CYC_COPY);
        const bool filterAverage = (filter == G_TF_AVERAGE);

        // Mip zero always. Choosing a level needs the ray's footprint, which is ray
        // differentials or a cone width carried in the payload - worth doing, and a separate
        // piece of work from getting the right texel on the surface at all.
        const float4 sampled = sampleTextureLevel(rdpTile, gpuTile, filterBilerp, filterAverage,
            renderFlagLinearFiltering(rp.flags), uvCoord, otherMode.textLUT(),
            renderFlagCanDecodeTMEM(rp.flags), 0, renderFlagUsesHDR(rp.flags));

        texelColor = sampled.rgb;
    }

    // Texture times shade, which is what the great majority of Perfect Dark's combiner
    // settings amount to for the first cycle. The real colour combiner is the next piece;
    // this is the one term that gets the surface looking like itself.
    payload.albedo = texelColor * shadeColor;
}

[shader("miss")]
void SurfaceMiss(inout SurfacePayload payload) {
    payload.instanceId = -1;
    payload.t = -1.0f;
    payload.normal = float3(0.0f, 0.0f, 0.0f);
    payload.albedo = float3(0.0f, 0.0f, 0.0f);
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
