//
// RT64
//

#include "rt64_raytracing_shader_cache.h"

#include <cassert>

#include "rt64_descriptor_sets.h"

#ifdef _WIN32
#   include "shaders/RaytracingLib.hlsl.dxil.h"
#endif

#include "shaders/RaytracingLib.hlsl.spirv.h"

namespace RT64 {
    // Export names in the shader library. The five ray generation programs must stay in this
    // order: the frame graph selects one by rewriting groups.rayGen.startIndex to a fixed
    // index between dispatches (rt64_framebuffer_renderer.cpp:808, 833, 838, 848, 861), so
    // the order here is what those numbers mean.
    static const char *RayGenNames[] = {
        "PrimaryRayGen",
        "DirectRayGen",
        "IndirectRayGen",
        "ReflectionRayGen",
        "RefractionRayGen"
    };

    static const char *MissNames[] = {
        "SurfaceMiss",
        "ShadowMiss"
    };

    static const char *SurfaceHitGroupName = "SurfaceHitGroup";
    static const char *ShadowHitGroupName = "ShadowHitGroup";

    const RaytracingShaderPrograms &RaytracingState::getShaderPrograms(const ShaderDescription &desc) const {
        // Specialized per-material programs are not built: the frame graph's own call site
        // for this is commented out in favour of the ubershader
        // (rt64_framebuffer_renderer.cpp:1578-1580). Until they are, every material resolves
        // to the same pair.
        (void)(desc);

        auto it = shaderProgramsMap.find(UberShaderHash);
        assert((it != shaderProgramsMap.end()) && "The ubershader programs must exist before any frame is traced.");
        return it->second;
    }

    RaytracingShaderCache::RaytracingShaderCache(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary) {
        assert(device != nullptr);
        assert(shaderLibrary != nullptr);

        this->device = device;
        this->shaderFormat = shaderFormat;
        this->shaderLibrary = shaderLibrary;
    }

    RaytracingShaderCache::~RaytracingShaderCache() { }

    bool RaytracingShaderCache::isSetup() const {
        return setupDone;
    }

    void RaytracingShaderCache::setup() {
        assert(!setupDone);

        const void *libraryBlob = nullptr;
        uint64_t libraryBlobSize = 0;
        switch (shaderFormat) {
#   ifdef _WIN32
        case RenderShaderFormat::DXIL:
            libraryBlob = RaytracingLibBlobDXIL;
            libraryBlobSize = uint64_t(std::size(RaytracingLibBlobDXIL));
            break;
#   endif
        case RenderShaderFormat::SPIRV:
            libraryBlob = RaytracingLibBlobSPIRV;
            libraryBlobSize = uint64_t(std::size(RaytracingLibBlobSPIRV));
            break;
        default:
            assert(false && "Unknown shader format.");
            return;
        }

        // The pipeline layout must match what the frame graph binds around a traceRays call:
        // the common set, the texture set twice and the framebuffer set
        // (rt64_framebuffer_renderer.cpp:811-814), which is the same shape the raster path
        // builds at rt64_raster_shader.cpp:450-458 minus its push constants.
        FramebufferRendererDescriptorCommonSet descriptorCommonSet(shaderLibrary->samplerLibrary, device->getCapabilities().raytracing);
        FramebufferRendererDescriptorTextureSet descriptorTextureSet;
        FramebufferRendererDescriptorFramebufferSet descriptorFramebufferSet;
        RenderPipelineLayoutBuilder layoutBuilder;

        // begin(isLocal, allowInputLayout).
        //
        // isLocal=true because that is the role plume gives this layout. It is
        // attached to the state object as D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE
        // and associated with every export (contrib/plume/plume_d3d12.cpp:3348-3358);
        // the global root signature is a dummy plume creates itself (:3360-3363).
        // The reason is visible in setShaderBindingTableInfo, which writes one
        // descriptor table handle per root parameter into every shader record
        // (:4066, :4083-4093) - the bindings travel in the binding table, which is
        // what a local root signature means. D3D12 rejects a state object whose
        // local root signature was not created with
        // D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE, which is the E_INVALIDARG
        // this cost.
        //
        // allowInputLayout=false. The raster path passes true
        // (rt64_raster_shader.cpp:451) because it has a vertex layout; ray tracing
        // has no input assembler, and D3D12 rejects that flag here too.
        layoutBuilder.begin(true, false);
        layoutBuilder.addDescriptorSet(descriptorCommonSet);
        layoutBuilder.addDescriptorSet(descriptorTextureSet);
        layoutBuilder.addDescriptorSet(descriptorTextureSet);
        layoutBuilder.addDescriptorSet(descriptorFramebufferSet);
        layoutBuilder.end();
        pipelineLayout = layoutBuilder.create(device);

        std::unique_ptr<RenderShader> libraryShader = device->createShader(libraryBlob, libraryBlobSize, "", shaderFormat);

        std::vector<RenderRaytracingPipelineLibrarySymbol> symbols;
        for (const char *name : RayGenNames) {
            symbols.emplace_back(RenderRaytracingPipelineLibrarySymbol(name, RenderRaytracingPipelineLibrarySymbolType::RAYGEN));
        }

        for (const char *name : MissNames) {
            symbols.emplace_back(RenderRaytracingPipelineLibrarySymbol(name, RenderRaytracingPipelineLibrarySymbolType::MISS));
        }

        symbols.emplace_back(RenderRaytracingPipelineLibrarySymbol("SurfaceClosestHit", RenderRaytracingPipelineLibrarySymbolType::CLOSEST_HIT));
        symbols.emplace_back(RenderRaytracingPipelineLibrarySymbol("ShadowClosestHit", RenderRaytracingPipelineLibrarySymbolType::CLOSEST_HIT));

        // Both hit groups get an any hit shader, which is where alpha compare happens: a
        // rasterized pixel that fails the test is simply not written, while a ray has to
        // leave the hit unregistered. The shadow group needs it as much as the surface one,
        // or a shadow ray through a railing is stopped by the whole quad.
        symbols.emplace_back(RenderRaytracingPipelineLibrarySymbol("SurfaceAnyHit", RenderRaytracingPipelineLibrarySymbolType::ANY_HIT));
        symbols.emplace_back(RenderRaytracingPipelineLibrarySymbol("ShadowAnyHit", RenderRaytracingPipelineLibrarySymbolType::ANY_HIT));

        const RenderRaytracingPipelineLibrary library(libraryShader.get(), symbols.data(), uint32_t(symbols.size()));
        const RenderRaytracingPipelineHitGroup hitGroups[] = {
            RenderRaytracingPipelineHitGroup(SurfaceHitGroupName, "SurfaceClosestHit", "SurfaceAnyHit"),
            RenderRaytracingPipelineHitGroup(ShadowHitGroupName, "ShadowClosestHit", "ShadowAnyHit")
        };

        RenderRaytracingPipelineDesc pipelineDesc;
        pipelineDesc.libraries = &library;
        pipelineDesc.librariesCount = 1;
        pipelineDesc.hitGroups = hitGroups;
        pipelineDesc.hitGroupsCount = uint32_t(std::size(hitGroups));
        pipelineDesc.pipelineLayout = pipelineLayout.get();

        // float3 normal, float t, int instanceId.
        // normal, albedo, t and instance id. Six floats fitted the placeholder payload that
        // carried no surface colour; a closest hit that reads the vertex attributes needs
        // eight.
        pipelineDesc.maxPayloadSize = 8 * sizeof(float);

        // Primary visibility only, so no ray is cast from inside a hit shader yet.
        pipelineDesc.maxRecursionDepth = 1;

        // State update is deliberately off. Only D3D12 reports the capability at all -
        // plume's Vulkan backend hardcodes raytracingStateUpdate to false
        // (contrib/plume/plume_vulkan.cpp:4125) - and pipeline creation fails outright if it
        // is requested without support, so enabling it would make this D3D12-only for no
        // gain while there is a single pipeline to update.
        pipelineDesc.stateUpdateEnabled = false;

        states.resize(StateCount);
        for (RaytracingState &state : states) {
            state.pipeline = device->createRaytracingPipeline(pipelineDesc);

            // There is deliberately no null check here, because there is nothing to
            // check: createRaytracingPipeline always returns a non-null object
            // (contrib/plume/plume_d3d12.cpp:3916-3918), and a failed
            // CreateStateObject leaves that object with a null state object and an
            // empty program map, having only printed to stderr. getProgram then
            // dereferences an end() iterator (:3439-3442) with its assert compiled
            // out of a release build, which is undefined behaviour rather than an
            // error anyone can act on.
            //
            // An earlier version of this loop did check for null, which read as
            // defensive and was in fact dead code - it is what let an invalid
            // pipeline layout reach traceRays and crash. plume needs a way to
            // report this; that is an upstream proposal, not something to work
            // around in the fork by editing contrib. Until then the only real
            // defence is not to hand plume a descriptor it will reject, so the
            // layout above is built to the rules the state object imposes.
            RaytracingShaderPrograms programs;
            programs.surface = state.pipeline->getProgram(SurfaceHitGroupName);
            programs.shadow = state.pipeline->getProgram(ShadowHitGroupName);
            state.shaderProgramsMap[UberShaderHash] = programs;

            for (const char *name : RayGenNames) {
                state.rayGenPrograms.emplace_back(state.pipeline->getProgram(name));
            }

            for (const char *name : MissNames) {
                state.missPrograms.emplace_back(state.pipeline->getProgram(name));
            }
        }

        setupDone = true;
    }

    void RaytracingShaderCache::submit(const ShaderDescription &desc) {
        // Queues a material for a specialized pipeline. Nothing consumes the queue yet
        // because only the ubershader is built, but the frame graph calls this once per RT
        // draw call (rt64_workload_queue.cpp:525), so it stays cheap: the hash set makes a
        // repeat submission of the same material free.
        const uint64_t hash = desc.hash();
        std::scoped_lock<std::mutex> lock(submissionMutex);
        if (shaderHashes.find(hash) != shaderHashes.end()) {
            return;
        }

        shaderHashes[hash] = true;
        descQueue.push(desc);
    }

    void RaytracingShaderCache::setNextState() {
        // Rotates to the next pipeline at the end of a frame so a state being rebuilt is
        // never the one a frame in flight is tracing against. With one pipeline built from
        // one library every state is identical today, which makes this a no-op in effect but
        // keeps the rotation the frame graph expects.
        activeState = (activeState + 1) % int32_t(states.size());
    }

    int32_t RaytracingShaderCache::getActiveState() const {
        return activeState;
    }
};
