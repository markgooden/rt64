//
// RT64
//

// An edge aware blur for the lighting buffers, in the shape of an a-trous wavelet pass: a
// fixed 5x5 kernel whose taps are spread further apart on each successive pass, so a few
// passes cover a wide radius without a wide kernel.
//
// The alternative already in the tree is GaussianFilterRGB3x3CS, which the frame graph runs
// five times over the indirect light. It knows nothing about the scene, so it blurs a shadow
// edge and a silhouette exactly as happily as it blurs noise. This weights every tap by how
// much the surface under it agrees with the surface under the centre pixel - normal and view
// depth, both of which primary visibility already writes - and so stops at the edges the
// noise does not cross either.
//
// Written for the traced light, which is the only thing here that is noisy. The game's baked
// vertex shade is in a buffer of its own and is deliberately not passed through this.

#define BLOCK_SIZE 8

struct EdgeFilterCB {
    uint2 TextureSize;
    float2 TexelSize;

    // Tap spacing in pixels. 1, 2, 4, ... across successive passes is what makes this a-trous
    // rather than a plain box.
    int StepSize;

    // How quickly a tap loses its weight as it disagrees with the centre. Higher is stricter.
    float NormalSigma;
    float DepthSigma;
    float Padding;
};

[[vk::push_constant]] ConstantBuffer<EdgeFilterCB> gConstants : register(b0);
Texture2D<float4> gInput : register(t1);
Texture2D<float4> gNormal : register(t2);
Texture2D<float> gDepth : register(t3);
RWTexture2D<float4> gOutput : register(u4);

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void CSMain(uint2 DTid : SV_DispatchThreadID) {
    if ((DTid.x >= gConstants.TextureSize.x) || (DTid.y >= gConstants.TextureSize.y)) {
        return;
    }

    const float3 centerNormal = gNormal[DTid].xyz;
    const float centerDepth = gDepth[DTid];

    // A pixel with no surface under it - the tracer missed - has nothing to filter and nothing
    // to say about its neighbours. Leaving it alone keeps the background out of the blur.
    if (dot(centerNormal, centerNormal) <= 0.0f) {
        gOutput[DTid] = gInput[DTid];
        return;
    }

    // The 5x5 B3 spline kernel the a-trous papers use, as a separable 1D row.
    const float kernelWeights[5] = { 1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f };

    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            const int2 tap = int2(DTid) + int2(dx, dy) * gConstants.StepSize;
            if ((tap.x < 0) || (tap.y < 0) || (tap.x >= int(gConstants.TextureSize.x)) || (tap.y >= int(gConstants.TextureSize.y))) {
                continue;
            }

            const float3 tapNormal = gNormal[tap].xyz;
            if (dot(tapNormal, tapNormal) <= 0.0f) {
                continue;
            }

            // Normal agreement, raised to a power rather than exponentiated: cheaper, and the
            // shape that matters is "falls away quickly once the surfaces disagree".
            const float normalWeight = pow(saturate(dot(centerNormal, tapNormal)), gConstants.NormalSigma);

            // Depth agreement, relative to the centre's own distance, so a tolerance that
            // works up close does not reject everything across a room. This game's world
            // coordinates run to the thousands, which is why nothing here is in metres.
            const float tapDepth = gDepth[tap];
            const float depthDelta = abs(tapDepth - centerDepth) / max(abs(centerDepth), 1.0f);
            const float depthWeight = exp(-depthDelta * gConstants.DepthSigma);

            const float weight = kernelWeights[dx + 2] * kernelWeights[dy + 2] * normalWeight * depthWeight;
            sum += gInput[tap] * weight;
            weightSum += weight;
        }
    }

    gOutput[DTid] = (weightSum > 0.0f) ? (sum / weightSum) : gInput[DTid];
}
