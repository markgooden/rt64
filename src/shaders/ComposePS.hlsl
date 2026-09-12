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

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    float4 diffuse = gDiffuse.SampleLevel(gSampler, uv, 0);

    // An interleaved raster layer resolved in front of the traced surface
    // (RaytracingLib.hlsl's PrimaryRayGen). The raster path has already shaded it, so it is
    // emitted as it stands - lighting it here would count that shading twice, and running it
    // through LinearToSrgb would brighten it against the traced half of the image.
    if (diffuse.a >= RasterLayerAlpha) {
        return float4(diffuse.rgb, 1.0f);
    }

    if (diffuse.a > EPSILON) {
        float3 directLight = gDirectLight.SampleLevel(gSampler, uv, 0).rgb;
        float3 indirectLight = gIndirectLight.SampleLevel(gSampler, uv, 0).rgb;
        float3 reflection = gReflection.SampleLevel(gSampler, uv, 0).rgb;
        float3 refraction = gRefraction.SampleLevel(gSampler, uv, 0).rgb;
        float3 transparent = gTransparent.SampleLevel(gSampler, uv, 0).rgb;

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
        result += reflection;
        result += refraction;
        result += transparent;
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
        return float4(gBackgroundColor.SampleLevel(gSampler, uv, 0).rgb, 1.0f);
    }
}
