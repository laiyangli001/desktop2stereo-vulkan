#include "preview_bridge.h"
#include "bridge_internal.h"
#include "bridge_material.h"

void preview_bridge_set_error(FilamentPreview* preview, const char* message) {
    if (preview) {
        preview->last_error = message;
    }
}

void preview_bridge_destroy_asset(FilamentPreview* preview) {
    if (preview->asset && preview->scene) {
        preview->scene->removeEntities(
                preview->asset->getEntities(), preview->asset->getEntityCount());
    }
    if (preview->asset && preview->asset_loader) {
        preview->asset_loader->destroyAsset(preview->asset);
    }
    preview->asset = nullptr;
    preview->brightness.scene_materials.clear();
    preview->brightness.skybox_materials.clear();
    preview->glb_bytes.clear();
}

void preview_bridge_destroy_screen(FilamentPreview* preview) {
    if (!preview || !preview->engine) return;
    if (preview->scene && !preview->screen_entity.isNull()) {
        preview->scene->remove(preview->screen_entity);
    }
    if (!preview->screen_entity.isNull()) {
        preview->engine->destroy(preview->screen_entity);
        preview->screen_entity = {};
    }
    if (preview->screen_vertex_buffer) {
        preview->engine->destroy(preview->screen_vertex_buffer);
        preview->screen_vertex_buffer = nullptr;
    }
    if (preview->screen_index_buffer) {
        preview->engine->destroy(preview->screen_index_buffer);
        preview->screen_index_buffer = nullptr;
    }
    if (preview->screen_material_instance) {
        preview->engine->destroy(preview->screen_material_instance);
        preview->screen_material_instance = nullptr;
    }
    if (preview->screen_material) {
        preview->engine->destroy(preview->screen_material);
        preview->screen_material = nullptr;
    }
    preview->screen_vertices.clear();
    preview->screen_indices.clear();
}

int preview_bridge_update_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees) {
    if (!preview || !preview->engine || !preview->screen_vertex_buffer ||
            !std::isfinite(position_x) || !std::isfinite(position_y) ||
            !std::isfinite(position_z) || !std::isfinite(width) ||
            !std::isfinite(height) || width <= 0.0f || height <= 0.0f ||
            !std::isfinite(rotation_x_degrees) || !std::isfinite(rotation_y_degrees) ||
            !std::isfinite(rotation_z_degrees)) return 0;
    constexpr float kPi = 3.14159265358979323846f;
    const float yaw = rotation_x_degrees * kPi / 180.0f;
    const float pitch = rotation_y_degrees * kPi / 180.0f;
    const float roll = rotation_z_degrees * kPi / 180.0f;
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cr = std::cos(roll), sr = std::sin(roll);
    const filament::math::float3 right{
            cy * cr + sy * sp * sr, sr * cp, -sy * cr + cy * sp * sr};
    const filament::math::float3 up{
            -cy * sr + sy * sp * cr, cr * cp, sr * sy + cy * sp * cr};
    const filament::math::float3 center{position_x, position_y, position_z};
    const filament::math::float3 half_right = right * (width * 0.5f);
    const filament::math::float3 half_up = up * (height * 0.5f);
    preview->screen_vertices = {
            {center - half_right - half_up, {0.0f, 0.0f}},
            {center + half_right - half_up, {1.0f, 0.0f}},
            {center - half_right + half_up, {0.0f, 1.0f}},
            {center + half_right + half_up, {1.0f, 1.0f}},
    };
    preview->screen_vertex_buffer->setBufferAt(*preview->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    preview->screen_vertices.data(),
                    preview->screen_vertices.size() * sizeof(PreviewScreenVertex), nullptr));
    return 1;
}

int preview_bridge_create_screen(FilamentPreview* preview) {
    if (!preview || !preview->engine || !preview->scene) return 0;
    const char* shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            float2 uv = getUV0();
            float2 grid_uv = abs(fract(uv * float2(16.0, 9.0)) - 0.5);
            float line = step(0.47, max(grid_uv.x, grid_uv.y));
            float3 base = float3(0.1, 0.45, 1.0);
            float3 grid = mix(base, float3(0.72, 0.88, 1.0), line * 0.35);
            material.baseColor = float4(grid, 0.72);
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S Preview Screen")
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::TRANSPARENT)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false)
            // The preview screen is a virtual display layer; do not let the
            // environment depth buffer hide it.
            .depthCulling(false)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    const filamat::Package package = builder.build(preview->engine->getJobSystem());
    if (!package.isValid()) {
        preview_bridge_set_error(preview, "Filament could not build preview screen material");
        return 0;
    }
    preview->screen_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*preview->engine);
    if (!preview->screen_material) {
        preview_bridge_set_error(preview, "Filament could not create preview screen material");
        return 0;
    }
    preview->screen_material_instance = preview->screen_material->createInstance();
    if (!preview->screen_material_instance) {
        preview_bridge_set_error(preview, "Filament could not create preview screen material instance");
        return 0;
    }
    preview->screen_vertices.resize(4);
    preview->screen_indices = {0, 1, 2, 1, 3, 2};
    preview->screen_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(4)
            .bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(PreviewScreenVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(PreviewScreenVertex))
            .build(*preview->engine);
    preview->screen_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(static_cast<uint32_t>(preview->screen_indices.size()))
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*preview->engine);
    if (!preview->screen_vertex_buffer || !preview->screen_index_buffer) {
        preview_bridge_set_error(preview, "Filament could not create preview screen geometry");
        return 0;
    }
    preview->screen_index_buffer->setBuffer(*preview->engine,
            filament::IndexBuffer::BufferDescriptor(
                    preview->screen_indices.data(),
                    preview->screen_indices.size() * sizeof(uint16_t), nullptr));
    preview->screen_entity = utils::EntityManager::get().create();
    const auto result = filament::RenderableManager::Builder(1)
            .boundingBox({{-20000.0f, -20000.0f, -20000.0f}, {20000.0f, 20000.0f, 20000.0f}})
            .material(0, preview->screen_material_instance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    preview->screen_vertex_buffer, preview->screen_index_buffer,
                    0, static_cast<uint32_t>(preview->screen_indices.size()))
            .priority(7)
            .culling(false)
            .castShadows(false)
            .receiveShadows(false)
            .build(*preview->engine, preview->screen_entity);
    if (result != filament::RenderableManager::Builder::Success) {
        preview_bridge_set_error(preview, "Filament could not create preview screen renderable");
        return 0;
    }
    preview->scene->addEntity(preview->screen_entity);
    return 1;
}

FilamentPreview* preview_bridge_create(void* native_window, uint32_t width, uint32_t height) {
    auto preview = std::make_unique<FilamentPreview>();
    if (!native_window || width == 0 || height == 0) {
        preview_bridge_set_error(preview.get(), "Preview window or dimensions are invalid");
        return preview.release();
    }
    preview->engine = filament::Engine::Builder()
            .backend(filament::Engine::Backend::DEFAULT)
            .build();
    if (!preview->engine) {
        preview_bridge_set_error(preview.get(), "Filament preview Engine creation failed");
        return preview.release();
    }
    preview->renderer = preview->engine->createRenderer();
    preview->scene = preview->engine->createScene();
    preview->view = preview->engine->createView();
    preview->camera = preview->engine->createCamera(
            utils::EntityManager::get().create());
    preview->materials = filament::gltfio::createJitShaderProvider(preview->engine);
    preview->texture_provider = filament::gltfio::createStbProvider(preview->engine);
    // Keep the desktop preview target aligned with the validated OpenXR sRGB
    // swapchain path and the shared Rec709 color-grading output.
    preview->swapchain = preview->engine->createSwapChain(
            native_window, filament::SwapChain::CONFIG_SRGB_COLORSPACE);
    if (!preview->renderer || !preview->scene || !preview->view || !preview->camera ||
            !preview->materials || !preview->texture_provider || !preview->swapchain) {
        preview_bridge_set_error(preview.get(), "Filament preview resource creation failed");
        return preview.release();
    }
    // Desktop swapchain images and depth buffers are reused across frames.
    // Clear both explicitly so preview rendering cannot accumulate trails.
    filament::Renderer::ClearOptions clear_options;
    clear_options.clearColor = filament::math::double4{0.0, 0.0, 0.0, 1.0};
    clear_options.clear = true;
    clear_options.discard = true;
    preview->renderer->setClearOptions(clear_options);
    preview->camera->lookAt(
            filament::math::float3{0.0f, 0.0f, 3.0f},
            filament::math::float3{0.0f, 0.0f, 0.0f},
            filament::math::float3{0.0f, 1.0f, 0.0f});
    preview->view->setScene(preview->scene);
    preview->view->setCamera(preview->camera);
    // Imported GLB renderables use Filament's default render channel 2.
    preview->view->setChannelDepthClearEnabled(2, true);
    if (!bridge_material_configure_color_pipeline(preview.get())) {
        preview_bridge_set_error(preview.get(), "Filament preview color pipeline creation failed");
        return preview.release();
    }
    preview->view->setViewport(filament::Viewport{0, 0, width, height});
    filament::gltfio::AssetConfiguration config{preview->engine, preview->materials};
    preview->asset_loader = filament::gltfio::AssetLoader::create(config);
    if (!preview->asset_loader) {
        preview_bridge_set_error(preview.get(), "Filament preview AssetLoader creation failed");
    }
    return preview.release();
}

void preview_bridge_destroy(FilamentPreview* preview) {
    if (!preview) return;
    preview_bridge_destroy_asset(preview);
    preview_bridge_destroy_screen(preview);
    if (preview->scene && !preview->fill_light.isNull()) {
        preview->scene->remove(preview->fill_light);
    }
    if (!preview->fill_light.isNull() && preview->engine) {
        preview->engine->destroy(preview->fill_light);
    }
    if (preview->indirect_light && preview->engine) {
        preview->scene->setIndirectLight(nullptr);
        preview->engine->destroy(preview->indirect_light);
    }
    if (preview->swapchain && preview->engine) preview->engine->destroy(preview->swapchain);
    if (preview->color_grading && preview->engine) {
        preview->view->setColorGrading(nullptr);
        preview->engine->destroy(preview->color_grading);
    }
    if (preview->view && preview->engine) preview->engine->destroy(preview->view);
    if (preview->camera && preview->engine) preview->engine->destroy(preview->camera->getEntity());
    if (preview->scene && preview->engine) preview->engine->destroy(preview->scene);
    if (preview->renderer && preview->engine) preview->engine->destroy(preview->renderer);
    if (preview->asset_loader) filament::gltfio::AssetLoader::destroy(&preview->asset_loader);
    if (preview->materials) {
        preview->materials->destroyMaterials();
        delete preview->materials;
    }
    delete preview->texture_provider;
    if (preview->engine) filament::Engine::destroy(&preview->engine);
    delete preview;
}

int preview_bridge_load_glb(FilamentPreview* preview, const uint8_t* bytes, uint32_t byte_count) {
    if (!preview || !preview->engine || !preview->asset_loader || !bytes || !byte_count) return 0;
    preview_bridge_destroy_asset(preview);
    preview->last_error.clear();
    preview->glb_bytes.assign(bytes, bytes + byte_count);
    preview->asset = preview->asset_loader->createAsset(preview->glb_bytes.data(), byte_count);
    if (!preview->asset) {
        preview_bridge_set_error(preview, "Filament preview could not parse GLB");
        return 0;
    }
    filament::gltfio::ResourceConfiguration config{};
    config.engine = preview->engine;
    config.normalizeSkinningWeights = true;
    filament::gltfio::ResourceLoader resources(config);
    resources.addTextureProvider("image/png", preview->texture_provider);
    resources.addTextureProvider("image/jpeg", preview->texture_provider);
    if (!resources.loadResources(preview->asset)) {
        preview_bridge_destroy_asset(preview);
        preview_bridge_set_error(preview, "Filament preview could not load GLB resources");
        return 0;
    }
    preview->scene->addEntities(
            preview->asset->getEntities(), preview->asset->getEntityCount());
    bridge_material_collect_brightness(preview, true);
    if (!preview->screen_material_instance && !preview_bridge_create_screen(preview)) {
        preview_bridge_destroy_asset(preview);
        return 0;
    }
    preview->asset->releaseSourceData();
    preview->engine->flushAndWait();
    preview->glb_bytes.clear();
    return 1;
}

int preview_bridge_apply_animations(FilamentPreview* preview, double time_seconds) {
    if (!preview || !preview->asset || !std::isfinite(time_seconds)) return 0;
    auto* animator = preview->asset->getInstance()->getAnimator();
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

int preview_bridge_set_camera(
        FilamentPreview* preview,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    if (!preview || !preview->camera) return 0;
    preview->camera->lookAt(
            filament::math::float3{eye_x, eye_y, eye_z},
            filament::math::float3{center_x, center_y, center_z},
            filament::math::float3{up_x, up_y, up_z});
    return 1;
}

int preview_bridge_set_projection(
        FilamentPreview* preview,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    if (!preview || !preview->camera || vertical_fov_degrees <= 0.0 ||
            aspect <= 0.0 || near_plane <= 0.0 || far_plane <= near_plane) return 0;
    preview->camera->setProjection(
            vertical_fov_degrees, aspect, near_plane, far_plane);
    return 1;
}

int preview_bridge_set_viewport(FilamentPreview* preview, uint32_t width, uint32_t height) {
    if (!preview || !preview->view || width == 0 || height == 0) return 0;
    preview->view->setViewport(filament::Viewport{0, 0, width, height});
    return 1;
}

int preview_bridge_set_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees) {
    return preview_bridge_update_screen(preview, position_x, position_y, position_z,
            width, height, rotation_x_degrees, rotation_y_degrees, rotation_z_degrees);
}

int preview_bridge_render(FilamentPreview* preview) {
    if (!preview || !preview->renderer || !preview->swapchain || !preview->view) return 0;
    if (!preview->renderer->beginFrame(preview->swapchain)) return 1;
    preview->renderer->render(preview->view);
    preview->renderer->endFrame();
    return 1;
}

const char* preview_bridge_last_error(const FilamentPreview* preview) {
    return preview ? preview->last_error.c_str() : "preview is null";
}
