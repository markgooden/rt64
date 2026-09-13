//
// RT64
//

#include "Color.hlsli"
#include "Constants.hlsli"
#include "Math.hlsli"

#include "shared/rt64_interleaved_raster.h"

SamplerState gSampler : register(s0);
Texture2D<float4> gFlow : register(t1);
Texture2D<float4> gDiffuse : register(t2);
Texture2D<float4> gDirectLight : register(t3);
Texture2D<float4> gIndirectLight : register(t4);
Texture2D<float4> gReflection : register(t5);
Texture2D<float4> gRefraction : register(t6);
Texture2D<float4> gTransparent : register(t7);

// What the raster path already drew into this framebuffer's colour target. Bound by
// submitRaytracingScene, which resolves the target and transitions it for the duration of
// this draw - compose writes to the output texture, not to the colour target, so reading it
// here is reading a different resource.
Texture2D<float4> gBackgroundColor : register(t8);

// The game's baked vertex shade, on its own and unfiltered. Primary visibility writes it
// apart from the traced light so a denoiser can smooth one without smearing the other.
// Its alpha carries the traced surface's depth, in the 0..1 the raster path's depth buffer
// holds - PrimaryRayGen puts it there because this shader has no constant buffer and so
// cannot derive it.
Texture2D<float4> gBakedLight : register(t9);

// The depth the raster path left behind gBackgroundColor. Bound since 2026-09-13 and unread
// until now, which was the bug: without it a traced pixel wins over a rastered one
// unconditionally, so any raster surface in front of traced geometry is simply discarded.
//
// A grille wall in Investigation is drawn by the raster path while the room behind it is
// traced. Every pixel where the trace hit something returned the traced shading and dropped
// the grille, and every pixel where it missed returned the grille - so the wall came out
// striped and see-through, with guards visible through it. That is the "clipping through
// walls" a playthrough reported on 2026-09-13.
#ifdef MULTISAMPLING
Texture2DMS<float> gBackgroundDepth : register(t10);

float loadBackgroundDepth(int2 pixelPos) {
    // Sample zero only. This is an ordering test, not a shading one: averaging depth across
    // samples invents a value between two surfaces and belongs to neither.
    return gBackgroundDepth.Load(pixelPos, 0);
}
#else
Texture2D<float> gBackgroundDepth : register(t10);

float loadBackgroundDepth(int2 pixelPos) {
    return gBackgroundDepth.Load(int3(pixelPos, 0));
}
#endif

// Whether the raster path drew something nearer here than the traced surface.
//
// A load rather than a filtered sample, and by normalised coordinate because the depth target
// is not the size the trace ran at - it carries the resolution scale and the dispatch does not
// (measured 960x660 against 640x440).
bool rasterIsInFront(float2 uv, float tracedZ) {
    uint width = 0;
    uint height = 0;
    uint samples = 0;
#ifdef MULTISAMPLING
    gBackgroundDepth.GetDimensions(width, height, samples);
#else
    uint levels = 0;
    gBackgroundDepth.GetDimensions(0, width, height, levels);
#endif

    // Cleared to 1.0, so a pixel the raster path never drew loses this comparison on its own.
    const float rasterZ = loadBackgroundDepth(int2(uv * float2(width, height)));
    return rasterZ < tracedZ;
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    float4 diffuse = gDiffuse.SampleLevel(gSampler, uv, 0);

    // An interleaved raster layer resolved in front of the traced surface
    // (RaytracingLib.hlsl's PrimaryRayGen). The raster path has already shaded it, so it is
    // emitted as it stands - lighting it here would count that shading twice, and running it
    // through LinearToSrgb would brighten it against the traced half of the image.
    if (diffuse.a >= RasterLayerAlpha) {
        return float4(diffuse.rgb, 1.0f);
    }

    // Read before the branch: a blended surface can be in front of a traced hit or in front
    // of nothing, and both cases need it.
    const float4 transparent = gTransparent.SampleLevel(gSampler, uv, 0);

    // What the raster path drew, and how far away it was. Read before the branch for the same
    // reason the transparent layer is: a rastered surface can be in front of a traced hit or
    // in front of nothing.
    const float3 background = gBackgroundColor.SampleLevel(gSampler, uv, 0).rgb;

    if (diffuse.a > EPSILON) {
        const float4 baked = gBakedLight.SampleLevel(gSampler, uv, 0);

        // The raster path wins where it drew something nearer. Its pixel is already shaded, so
        // it is emitted as it stands - the same reasoning the interleaved layer branch above
        // rests on - with the blended layer still going over it.
        if (rasterIsInFront(uv, baked.a)) {
            return float4(background * (1.0f - transparent.a) + transparent.rgb, 1.0f);
        }

        // Traced light plus the baked shade. They were one buffer until 2026-09-13, and
        // the sum is identical - only what may be filtered has changed.
        const float3 directLight = gDirectLight.SampleLevel(gSampler, uv, 0).rgb + baked.rgb;
        const float3 indirectLight = gIndirectLight.SampleLevel(gSampler, uv, 0).rgb;
        const float4 reflection = gReflection.SampleLevel(gSampler, uv, 0);
        float3 refraction = gRefraction.SampleLevel(gSampler, uv, 0).rgb;

        // We intentionally mix the HDR buffer that will be upscaled in sRGB space to preserve the color of effects like fog and such.
        // No LinearToSrgb. Measured 2026-09-12 against the OpenGL reference over 25612
        // pixels: the traced image fits `reference encoded to sRGB` with an rms of 18.4/255
        // where the identity fits at 81.9, and the best pure gamma is 0.38 - which is the
        // sRGB encode exponent. The albedo this shader receives is the colour combiner's
        // output built from N64 texels and vertex colours, and those are display space
        // values already; encoding them again is the washed-out look the whole frame has
        // had. gBackgroundColor two branches down was already exempted for the same
        // reason, on the same reasoning, for the raster half of the image.
        float3 result = lerp(diffuse.rgb, diffuse.rgb * (directLight + indirectLight), diffuse.a);
        // Blended over by its own strength, not added: a mirror replaces what is under it
        // in proportion to how much of a mirror it is, and adding would leave the diffuse
        // surface at full brightness with the reflection on top of it.
        result = result * (1.0f - reflection.a) + reflection.rgb;
        result += refraction;

        // Blended over, not added. RefractionRayGen writes the nearest blended surface
        // premultiplied with its combiner alpha, so this is the standard over operator and it
        // darkens where the surface is dark - adding it could only ever brighten, which is
        // wrong for every blended surface the game uses to tint rather than to glow.
        result = result * (1.0f - transparent.a) + transparent.rgb;
        return float4(result, 1.0f);
    }
    else {
        // Nothing was traced here, so what belongs at this pixel is whatever the raster path
        // drew. This used to return diffuse.rgb, which PrimaryRayGen leaves at zero on a miss,
        // so the composed image was black everywhere the tracer did not cover and the post
        // process pass painted that over the raster draws underneath it.
        //
        // Not run through LinearToSrgb: the value read back is what the raster path wrote to
        // the colour target, already in the space the final image is in, where diffuse.rgb is
        // the tracer's linear albedo. Converting it would brighten the untraced half of the
        // image relative to the traced half.
        // The blended surface goes over the raster background too. A pane of glass in front
        // of geometry the tracer did not reach is still in front of it, and returning the
        // background alone dropped it.
        return float4(background * (1.0f - transparent.a) + transparent.rgb, 1.0f);
    }
}
