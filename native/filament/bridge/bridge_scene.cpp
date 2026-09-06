#include "bridge_scene.h"
#include "bridge_internal.h"
#include "bridge_controller.h"
#include "bridge_material.h"

void bridge_scene_destroy(FilamentBridge* bridge) {
    if (!bridge) return;
    const bool has_scene_resources =
            bridge->asset != nullptr || !bridge->glb_bytes.empty() ||
            !bridge->brightness.scene_materials.empty() ||
            !bridge->brightness.skybox_materials.empty() ||
            !bridge->brightness.skybox_entities.empty();
    if (!has_scene_resources) return;
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=destroy begin bridge=%p\n",
            static_cast<void*>(bridge));
    std::fflush(stderr);
    if (bridge->asset && bridge->scene) {
        bridge->scene->removeEntities(
                bridge->asset->getEntities(), bridge->asset->getEntityCount());
    }
    if (bridge->asset && bridge->asset_loader) {
        bridge->asset_loader->destroyAsset(bridge->asset);
    }
    bridge->asset = nullptr;
    bridge->brightness.scene_materials.clear();
    bridge->brightness.skybox_materials.clear();
    bridge->brightness.skybox_entities.clear();
    bridge->glb_bytes.clear();
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=destroy done\n");
    std::fflush(stderr);
}

int bridge_scene_load_glb(FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count) {
    if (!bridge || !bridge->engine || !bridge->asset_loader || !bytes || !byte_count) return 0;
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=start bytes=%u bridge=%p engine=%p\n",
            byte_count, static_cast<void*>(bridge), static_cast<void*>(bridge->engine));
    std::fflush(stderr);
    bridge_scene_destroy(bridge);
    bridge->last_error.clear();
    bridge->glb_bytes.assign(bytes, bytes + byte_count);
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=createAsset begin\n");
    std::fflush(stderr);
    bridge->asset = bridge->asset_loader->createAsset(bridge->glb_bytes.data(), byte_count);
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=createAsset done asset=%p\n",
            static_cast<void*>(bridge->asset));
    std::fflush(stderr);
    if (!bridge->asset) {
        bridge_set_error(bridge, "Filament could not parse GLB");
        return 0;
    }
    filament::gltfio::ResourceConfiguration config{bridge->engine, nullptr, true};
    filament::gltfio::ResourceLoader resources(config);
    resources.addTextureProvider("image/png", bridge->texture_provider);
    resources.addTextureProvider("image/jpeg", bridge->texture_provider);
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=loadResources begin\n");
    std::fflush(stderr);
    if (!resources.loadResources(bridge->asset)) {
        bridge_scene_destroy(bridge);
        bridge_set_error(bridge, "Filament could not load GLB resources");
        return 0;
    }
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=loadResources done\n");
    std::fflush(stderr);
    bridge->scene->addEntities(
            bridge->asset->getEntities(), bridge->asset->getEntityCount());
    // The legacy controller head/top lights are isolated from environment geometry.
    bridge_material_collect_brightness(bridge, false);
    bridge->asset->releaseSourceData();
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=flush begin\n");
    std::fflush(stderr);
    bridge->engine->flushAndWait();
    std::fprintf(stderr, "[FilamentBridge] load_glb phase=flush done\n");
    std::fflush(stderr);
    bridge->glb_bytes.clear();
    return 1;
}

int bridge_scene_apply_animations(FilamentBridge* bridge, double time_seconds) {
    if (!bridge) return 0;
    for (auto& controller : bridge->controllers) {
        bridge_controller_update_animations(bridge, controller);
    }
    if (!bridge->asset) return 1;
    auto* animator = bridge->asset->getInstance()->getAnimator();
    if (!animator) return 1;
    const size_t count = animator->getAnimationCount();
    for (size_t index = 0; index < count; ++index) {
        const float duration = animator->getAnimationDuration(index);
        const float time = duration > 0.0f
                ? std::fmod(static_cast<float>(time_seconds), duration)
                : 0.0f;
        animator->applyAnimation(index, std::max(0.0f, time));
    }
    animator->updateBoneMatrices();
    return 1;
}

uint32_t bridge_scene_animation_count(const FilamentBridge* bridge) {
    if (!bridge || !bridge->asset) return 0;
    auto* animator = bridge->asset->getInstance()->getAnimator();
    return animator ? static_cast<uint32_t>(animator->getAnimationCount()) : 0;
}

float bridge_scene_animation_duration(const FilamentBridge* bridge, uint32_t animation_index) {
    if (!bridge || !bridge->asset) return 0.0f;
    auto* animator = bridge->asset->getInstance()->getAnimator();
    if (!animator || animation_index >= animator->getAnimationCount()) return 0.0f;
    return animator->getAnimationDuration(animation_index);
}
