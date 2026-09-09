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

#include "shared/rt64_color_combiner.h"

#include "FbRendererCommon.hlsli"
#include "FbRendererRT.hlsli"
#include "Random.hlsli"
#include "TextureSampler.hlsli"

// Pushes a shadow ray off the surface it starts on. The geometry is N64 scale - a level's
// world coordinates run to the thousands - so this is in those units, not in metres.
static const float ShadowRayBias = 0.5f;

struct SurfacePayload {
    float3 normal;
    float3 albedo;
    float3 ambient;
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
    payload.ambient = float3(0.0f, 0.0f, 0.0f);
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

    // The surface without its baked lighting. The compose pass multiplies this by the light
    // buffers (ComposePS.hlsl:28), so anything left in here that is really light would be
    // counted twice the moment there is any.
    gDiffuse[pixel] = float4(payload.albedo, (payload.instanceId >= 0) ? 1.0f : 0.0f);

    // The baked lighting, seeded into the direct light buffer for DirectRayGen to add to.
    // With no lights in range this reconstructs exactly what the game draws - albedo times
    // shade - and a light contributes on top of it rather than instead of it.
    gDirectLightAccum[pixel] = float4(payload.ambient, 1.0f);
    gFlow[pixel] = float2(0.0f, 0.0f);
    gReactiveMask[pixel] = 0.0f;
    gLockMask[pixel] = 0.0f;
    gNormalRoughness[pixel] = float4(payload.normal, 1.0f);
}

// Direct lighting, with ray traced shadows.
//
// The lights are the game's own: the RSP light set the vertex pass would have used, taken
// per pixel instead of per vertex and with a shadow ray between the surface and each one.
// Their contributions use the shared microcode-matching helpers where the maths is the
// same, but computed in world space rather than object space - computePosLight and
// computeDirLight transform the light by the object's world matrix because the RSP works
// on object space normals, and a hit here already has a world space normal and position.
//
// This writes the lighting term on its own. It is deliberately not folded into the
// composed image yet: Perfect Dark bakes this same lighting into its vertex colours, which
// is what the albedo is built from, so multiplying the two would count it twice. Separating
// them is the next question, and the DirectLightRaw debug view is where this can be read
// honestly in the meantime.
[shader("raygeneration")]
void DirectRayGen() {
    const uint2 pixel = DispatchRaysIndex().xy;

    // Seeded by PrimaryRayGen with the surface's baked lighting, so every early return here
    // leaves that in place. Clearing first would throw away the only light most of Perfect
    // Dark has: the rooms the game lights are the minority.
    const int instanceId = gInstanceId[pixel];
    if (instanceId < 0) {
        return;
    }

    const float4 shadingPosition = gShadingPosition[pixel];
    if (shadingPosition.w <= 0.0f) {
        return;
    }

    const float3 surfacePosition = shadingPosition.xyz;
    const float3 surfaceNormal = gShadingNormal[pixel].xyz;

    // SceneLights carries the game's own room lights, gathered port side and handed over
    // each frame (port/rt64/rt64_lights.h). They are not in the display list: Perfect Dark
    // bakes its level lighting into vertex colours and its RSP light path covered 32 of
    // 5878 vertices on an in-level frame, so the RSP set this shader first read was empty
    // for all but a fraction of a percent of the geometry.
    const uint lightCount = RtParams.lightsCount;
    if (lightCount == 0) {
        return;
    }

    float3 accumulated = gDirectLightAccum[pixel].rgb;

    for (uint i = 0; i < lightCount; i++) {
        const PointLight light = SceneLights[i];

        float3 toLight = light.position - surfacePosition;
        const float lightDistance = length(toLight);
        if (lightDistance <= 0.0f) {
            continue;
        }

        toLight /= lightDistance;

        // Outside the light's reach. attenuationRadius is derived from the fitting's own
        // size where the lights are built (rt64_host.cpp), because the game records an
        // extent but never a range.
        if (lightDistance >= light.attenuationRadius) {
            continue;
        }

        const float nDotL = dot(surfaceNormal, toLight);
        if (nDotL <= 0.0f) {
            continue;
        }

        // The game's lights point somewhere: a ceiling fitting faces down, and lighting the
        // ceiling above it with the same strength would be wrong. A light with no direction
        // recorded is treated as omnidirectional rather than as facing along zero.
        float directionalFalloff = 1.0f;
        const float directionLength = length(light.direction);
        if (directionLength > 0.0f) {
            const float3 lightForward = light.direction / directionLength;
            directionalFalloff = saturate(dot(lightForward, -toLight));
        }

        if (directionalFalloff <= 0.0f) {
            continue;
        }

        // Inverse square, softened near the source so a surface touching the fitting does
        // not blow out, and faded to nothing at the edge of the reach so a light does not
        // end in a visible ring.
        const float normalizedDistance = lightDistance / light.attenuationRadius;
        const float radius = max(light.pointRadius, 1.0f);
        const float falloff = saturate(1.0f - normalizedDistance * normalizedDistance);
        const float attenuation = falloff * falloff / (1.0f + pow(lightDistance / radius, 2.0f));

        const float3 contribution = light.diffuseColor * (nDotL * directionalFalloff * attenuation);
        if (all(contribution <= 0.0f)) {
            continue;
        }

        RayDesc shadowRay;
        shadowRay.Origin = surfacePosition + surfaceNormal * ShadowRayBias;
        shadowRay.Direction = toLight;
        shadowRay.TMin = 0.0f;
        shadowRay.TMax = lightDistance - ShadowRayBias;

        SurfacePayload shadowPayload;
        shadowPayload.normal = float3(0.0f, 0.0f, 0.0f);
        shadowPayload.albedo = float3(0.0f, 0.0f, 0.0f);
        shadowPayload.ambient = float3(0.0f, 0.0f, 0.0f);
        shadowPayload.t = -1.0f;
        shadowPayload.instanceId = -1;

        // The shadow hit group and the shadow miss shader are the second entries in their
        // tables (rt64_raytracing_shader_cache.cpp:30-33, :143-144), which is what the two
        // ones select. ACCEPT_FIRST_HIT_AND_END_SEARCH because any blocker will do - the
        // shadow closest hit exists only to say that something was in the way.
        TraceRay(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1, 0, 1, shadowRay, shadowPayload);
        if (shadowPayload.instanceId < 0) {
            accumulated += contribution;
        }
    }

    gDirectLightAccum[pixel] = float4(accumulated, 1.0f);
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
// Samples one of a draw call's tiles at the given texture coordinate.
//
// The coordinate transform sampleTexture applies before sampling
// (TextureSampler.hlsli:218-245) is reproduced rather than called, because sampleTexture
// picks a mip level from screen space derivatives and a ray has none. Only the parts that
// mean something for a traced hit are kept: the perspective correction, the half-texel
// shift, the tile scale and the upper-left offset. The parts that exist for a rasterized
// rect - the next pixel bug, the low precision coordinate rounding - are left out.
//
// Mip zero always. Choosing a level needs the ray's footprint, either ray differentials or
// a cone width carried in the payload, and that is a separate piece of work.
static float4 sampleTileAtUV(uint globalTileIndex, float2 vertexUV, OtherMode otherMode, RenderFlags renderFlags,
    uint cms, uint cmt, uint nativeSampler)
{
    RDPTile rdpTile = RDPTiles[globalTileIndex];
    const GPUTile gpuTile = GPUTiles[globalTileIndex];
    if (!renderFlagDynamicTiles(renderFlags)) {
        rdpTile.cms = cms;
        rdpTile.cmt = cmt;
        rdpTile.nativeSampler = nativeSampler;
    }

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
    return sampleTextureLevel(rdpTile, gpuTile, filterBilerp, filterAverage,
        renderFlagLinearFiltering(renderFlags), uvCoord, otherMode.textLUT(),
        renderFlagCanDecodeTMEM(renderFlags), 0, renderFlagUsesHDR(renderFlags));
}

// Everything the combiner needs about one hit, so a closest hit and an any hit agree by
// construction rather than by two copies staying in step.
struct SurfaceShading {
    // The surface's own colour, with the shade term taken out of it.
    float3 albedo;

    // The shade term on its own: the vertex colour the RSP pass computed, which in Perfect
    // Dark is where the level's baked lighting lives. It is light, not material, so it
    // belongs with the lighting rather than multiplied into the albedo.
    float3 ambient;
    float alphaCompareValue;
};

static SurfaceShading shadeSurface(RenderIndices renderIndices, uint i0, uint i1, uint i2, float3 barycentrics) {
    const float4 c0 = asfloat(shadedColBuffer.Load4(i0 * 16));
    const float4 c1 = asfloat(shadedColBuffer.Load4(i1 * 16));
    const float4 c2 = asfloat(shadedColBuffer.Load4(i2 * 16));
    const float4 shadeColor = c0 * barycentrics.x + c1 * barycentrics.y + c2 * barycentrics.z;

    // The draw call's render state, reached through its instance index the same way the
    // raster pixel shader reaches it (RasterPS.hlsl:51-60).
    const uint instanceIndex = renderIndices.instanceIndex;
    const RenderParams rp = DynamicRenderParams[instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };

    const float2 t0 = asfloat(genTexCoordBuffer.Load2(i0 * 8));
    const float2 t1 = asfloat(genTexCoordBuffer.Load2(i1 * 8));
    const float2 t2 = asfloat(genTexCoordBuffer.Load2(i2 * 8));
    const float2 vertexUV = t0 * barycentrics.x + t1 * barycentrics.y + t2 * barycentrics.z;

    // Both tiles, because the combiner can reference either. The raster path picks the pair
    // by LOD (RasterPS.hlsl:131-161); with no LOD here they are tile zero and tile one of
    // the draw call's range.
    float4 texVal0 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 texVal1 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (renderFlagUsesTexture0(rp.flags)) {
        texVal0 = sampleTileAtUV(renderIndices.rdpTileIndex, vertexUV, otherMode, rp.flags,
            renderCMS0(rp.flags), renderCMT0(rp.flags), renderFlagNativeSampler0(rp.flags));
    }

    if (renderFlagUsesTexture1(rp.flags)) {
        const uint tile1 = (renderIndices.rdpTileCount > 1) ? 1 : 0;
        texVal1 = sampleTileAtUV(renderIndices.rdpTileIndex + tile1, vertexUV, otherMode, rp.flags,
            renderCMS1(rp.flags), renderCMT1(rp.flags), renderFlagNativeSampler1(rp.flags));
    }

    uint randomSeed = initRand(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x,
        asuint(RtParams.nearDist), 16);

    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texVal0;
    ccInputs.texVal1 = texVal1;
    ccInputs.primColor = instanceRDPParams[instanceIndex].primColor;
    ccInputs.shadeColor = shadeColor;
    ccInputs.envColor = instanceRDPParams[instanceIndex].envColor;
    ccInputs.keyCenter = instanceRDPParams[instanceIndex].keyCenter;
    ccInputs.keyScale = instanceRDPParams[instanceIndex].keyScale;

    // No LOD fraction: that is the same missing ray footprint that pins sampling to mip zero.
    ccInputs.lodFraction = 0.0f;
    ccInputs.primLodFrac = instanceRDPParams[instanceIndex].primLOD.x;
    ccInputs.noise = nextRand(randomSeed);
    ccInputs.K4 = (instanceRDPParams[instanceIndex].convertK[4] / 255.0f);
    ccInputs.K5 = (instanceRDPParams[instanceIndex].convertK[5] / 255.0f);

    float4 combinerColor;
    float alphaCompareValue;
    colorCombiner.run(ccInputs, combinerColor, alphaCompareValue);

    // Again with the shade term white, which is what separates the material from the light
    // baked into it. The combiner is a general expression and not always a product, so the
    // shade cannot simply be divided back out; running it a second time asks the combiner
    // itself what the surface looks like unlit. The textures are sampled once and both runs
    // share them, so this costs an evaluation and no bandwidth.
    //
    // The first run keeps the alpha, because alpha compare is a property of the surface as
    // the game draws it and has nothing to do with the split.
    ColorCombiner::Inputs albedoInputs = ccInputs;
    albedoInputs.shadeColor = float4(1.0f, 1.0f, 1.0f, shadeColor.a);

    float4 albedoColor;
    float albedoAlphaUnused;
    colorCombiner.run(albedoInputs, albedoColor, albedoAlphaUnused);

    SurfaceShading shading;
    shading.albedo = albedoColor.rgb;
    shading.ambient = shadeColor.rgb;
    shading.alphaCompareValue = alphaCompareValue;
    return shading;
}

// The alpha test the raster path applies to a pixel (RasterPS.hlsl:203-213). A rasterized
// pixel that fails is simply not written; a ray that fails has to leave the hit
// unregistered, which is what an any hit shader is for.
static bool alphaTestFails(RenderIndices renderIndices, float alphaCompareValue) {
    const uint instanceIndex = renderIndices.instanceIndex;
    const RenderParams rp = DynamicRenderParams[instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };
    const uint alphaCompare = otherMode.alphaCompare();
    if (alphaCompare == G_AC_DITHER) {
        uint randomSeed = initRand(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x,
            asuint(RtParams.farDist), 16);
        return alphaCompareValue < nextRand(randomSeed);
    }

    if (alphaCompare == G_AC_THRESHOLD) {
        return alphaCompareValue < instanceRDPParams[instanceIndex].blendColor.a;
    }

    return false;
}

// Reads the triangle a hit landed on. The instance carries its draw call index
// (rt64_raytracing_resources.cpp:468), so instanceRenderIndices gives where this draw
// call's triangles begin in the shared index buffer, and PrimitiveIndex() counts triangles
// from exactly there - the bottom level structure was built over the index buffer starting
// at that offset (rt64_framebuffer_renderer.cpp:1892) with the vertex buffer whole from
// zero, so the indices it reads are already global.
static RenderIndices fetchTriangle(out uint i0, out uint i1, out uint i2) {
    const RenderIndices renderIndices = instanceRenderIndices[InstanceID()];
    const uint indexStart = renderIndices.faceIndicesStart + PrimitiveIndex() * 3;
    i0 = indexBuffer.Load(indexStart * 4);
    i1 = indexBuffer.Load((indexStart + 1) * 4);
    i2 = indexBuffer.Load((indexStart + 2) * 4);
    return renderIndices;
}

static float3 barycentricsOf(TriangleAttributes attributes) {
    return float3(1.0f - attributes.barycentrics.x - attributes.barycentrics.y,
        attributes.barycentrics.x, attributes.barycentrics.y);
}

[shader("closesthit")]
void SurfaceClosestHit(inout SurfacePayload payload, in TriangleAttributes attributes) {
    payload.instanceId = int(InstanceID());
    payload.t = RayTCurrent();

    uint i0, i1, i2;
    const RenderIndices renderIndices = fetchTriangle(i0, i1, i2);

    // posBuffer is the world space position the RSP world compute pass writes, four floats
    // per vertex, so the normal comes out in world space with no further transform.
    const float3 p0 = asfloat(posBuffer.Load3(i0 * 16));
    const float3 p1 = asfloat(posBuffer.Load3(i1 * 16));
    const float3 p2 = asfloat(posBuffer.Load3(i2 * 16));
    const float3 geometricNormal = normalize(cross(p1 - p0, p2 - p0));

    // Turned to face the ray. The N64 draws plenty of geometry double sided and the winding
    // of a back face would otherwise light it from behind.
    payload.normal = (dot(geometricNormal, WorldRayDirection()) > 0.0f) ? -geometricNormal : geometricNormal;

    const SurfaceShading shading = shadeSurface(renderIndices, i0, i1, i2, barycentricsOf(attributes));
    payload.albedo = shading.albedo;
    payload.ambient = shading.ambient;
}

// Alpha compare, as a hit that never happened.
//
// Every mesh is built non-opaque (rt64_framebuffer_renderer.cpp:1892), so this runs for
// each candidate hit before it is accepted. Without it a texture's cut-out texels trace as
// solid, and grates, foliage and railings are sheets rather than shapes - and a shadow ray
// through a railing would be stopped by the whole quad, which is why the shadow hit group
// gets the same test.
[shader("anyhit")]
void SurfaceAnyHit(inout SurfacePayload payload, in TriangleAttributes attributes) {
    uint i0, i1, i2;
    const RenderIndices renderIndices = fetchTriangle(i0, i1, i2);
    const SurfaceShading shading = shadeSurface(renderIndices, i0, i1, i2, barycentricsOf(attributes));
    if (alphaTestFails(renderIndices, shading.alphaCompareValue)) {
        IgnoreHit();
    }
}

[shader("miss")]
void SurfaceMiss(inout SurfacePayload payload) {
    payload.instanceId = -1;
    payload.t = -1.0f;
    payload.normal = float3(0.0f, 0.0f, 0.0f);
    payload.albedo = float3(0.0f, 0.0f, 0.0f);
    payload.ambient = float3(0.0f, 0.0f, 0.0f);
}

[shader("closesthit")]
void ShadowClosestHit(inout SurfacePayload payload, in TriangleAttributes attributes) {
    payload.instanceId = int(InstanceID());
    payload.t = RayTCurrent();
}

[shader("anyhit")]
void ShadowAnyHit(inout SurfacePayload payload, in TriangleAttributes attributes) {
    uint i0, i1, i2;
    const RenderIndices renderIndices = fetchTriangle(i0, i1, i2);
    const SurfaceShading shading = shadeSurface(renderIndices, i0, i1, i2, barycentricsOf(attributes));
    if (alphaTestFails(renderIndices, shading.alphaCompareValue)) {
        IgnoreHit();
    }
}

[shader("miss")]
void ShadowMiss(inout SurfacePayload payload) {
    payload.instanceId = -1;
    payload.t = -1.0f;
}
