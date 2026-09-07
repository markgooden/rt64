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
        layoutBuilder.begin(false, true);
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

        const RenderRaytracingPipelineLibrary library(libraryShader.get(), symbols.data(), uint32_t(symbols.size()));
        const RenderRaytracingPipelineHitGroup hitGroups[] = {
            RenderRaytracingPipelineHitGroup(SurfaceHitGroupName, "SurfaceClosestHit"),
            RenderRaytracingPipelineHitGroup(ShadowHitGroupName, "ShadowClosestHit")
        };

        RenderRaytracingPipelineDesc pipelineDesc;
        pipelineDesc.libraries = &library;
        pipelineDesc.librariesCount = 1;
        pipelineDesc.hitGroups = hitGroups;
        pipelineDesc.hitGroupsCount = uint32_t(std::size(hitGroups));
        pipelineDesc.pipelineLayout = pipelineLayout.get();

        // float3 normal, float t, int instanceId.
        pipelineDesc.maxPayloadSize = 6 * sizeof(float);

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
            if (state.pipeline == nullptr) {
                assert(false && "Failed to create the raytracing pipeline.");
                return;
            }

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
