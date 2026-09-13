//
// RT64
//

// Writes the traced surface's depth back into the raster path's depth buffer.
//
// A draw call the tracer takes is removed from the raster path, and with it goes the depth it
// would have written. Every raster draw ordered after the RT scene then depth tests against a
// buffer with holes in it exactly where the traced geometry is - and a draw that tests depth
// without writing it (zCmp 1, zUpd 0, which is every blended surface the game draws) has lost
// its occluders entirely, so it draws over things it is behind.
//
// Measured on level.0000: draw calls 214, 217 and 218 are blended quads ordered after the RT
// scene, and without this pass they paint the lit console over Joanna's body. It is the same
// defect a playthrough reported on 2026-09-13 as surfaces visible through other surfaces.
//
// The depth comes from gBakedLight's alpha, where PrimaryRayGen puts the traced hit's z in the
// same 0..1 the raster depth buffer holds (RaytracingLib.hlsl). Nothing else has to be bound:
// this pass reuses the compose descriptor set whole.

Texture2D<float4> gBakedLight : register(t9);

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0, out float resultDepth : SV_DEPTH) : SV_TARGET {
    // A load, not a filtered sample. Interpolating depth across a silhouette invents a surface
    // that is in neither of the two it sits between - and the buffer is not the size this pass
    // runs at anyway (the trace dispatches at the scene's resolution, the depth target carries
    // the resolution scale), so the normalised coordinate is what maps between them.
    uint width = 0;
    uint height = 0;
    gBakedLight.GetDimensions(width, height);

    const float z = gBakedLight.Load(uint3(uint2(uv * float2(width, height)), 0)).a;

    // 1.0 is what PrimaryRayGen leaves where it hit nothing. Writing it would push the far
    // plane over whatever the raster path had already put there.
    if (z >= 1.0f) {
        discard;
    }

    resultDepth = z;

    // The colour write mask is zero on this pipeline, so this goes nowhere. The depth is the
    // whole output; SV_TARGET is here because a pixel shader needs one.
    return 0.0f;
}
