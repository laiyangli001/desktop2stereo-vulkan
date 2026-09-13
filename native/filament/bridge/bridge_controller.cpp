#include "bridge_controller.h"
#include "bridge_internal.h"

#include <cstdlib>
#include <cstring>

int bridge_controller_set_material_override(
        FilamentBridge* bridge, float roughness_factor, float metallic_factor,
        float specular_red, float specular_green, float specular_blue) {
    if (!bridge || !bridge->engine ||
            !std::isfinite(roughness_factor) || !std::isfinite(metallic_factor) ||
            !std::isfinite(specular_red) || !std::isfinite(specular_green) ||
            !std::isfinite(specular_blue)) {
        return 0;
    }
    const bool set_roughness = roughness_factor >= 0.0f;
    const bool set_metallic = metallic_factor >= 0.0f;
    const bool set_specular = specular_red >= 0.0f &&
            specular_green >= 0.0f && specular_blue >= 0.0f;
    auto& renderables = bridge->engine->getRenderableManager();
    for (auto& controller : bridge->controllers) {
        if (!controller.asset) continue;
        for (size_t index = 0;
                index < controller.asset->getRenderableEntityCount(); ++index) {
            const auto entity = controller.asset->getRenderableEntities()[index];
            const auto instance = renderables.getInstance(entity);
            if (!instance.isValid()) continue;
            for (size_t primitive = 0;
                    primitive < renderables.getPrimitiveCount(instance); ++primitive) {
                auto* material = renderables.getMaterialInstanceAt(instance, primitive);
                if (!material) continue;
                const auto* definition = material->getMaterial();
                if (set_roughness && definition->hasParameter("roughnessFactor")) {
                    material->setParameter("roughnessFactor",
                            std::clamp(roughness_factor, 0.0f, 1.0f));
                }
                if (set_metallic && definition->hasParameter("metallicFactor")) {
                    material->setParameter("metallicFactor",
                            std::clamp(metallic_factor, 0.0f, 1.0f));
                }
                if (set_specular && definition->hasParameter("specularColorFactor")) {
                    material->setParameter("specularColorFactor", filament::math::float3{
                            specular_red, specular_green, specular_blue});
                }
            }
        }
    }
    return 1;
}

void bridge_controller_destroy(FilamentBridge* bridge, ControllerAsset& controller) {
    if (controller.asset && bridge->scene) {
        bridge->scene->removeEntities(
                controller.asset->getEntities(), controller.asset->getEntityCount());
    }
    if (controller.asset && bridge->foreground_scene) {
        bridge->foreground_scene->removeEntities(
                controller.asset->getEntities(), controller.asset->getEntityCount());
    }
    if (controller.asset && bridge->asset_loader) {
        bridge->asset_loader->destroyAsset(controller.asset);
    }
    controller = {};
}

void bridge_controller_eye_diagnostic_destroy(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    if (bridge->controller_eye_diagnostic_material_instance) {
        bridge->engine->destroy(bridge->controller_eye_diagnostic_material_instance);
        bridge->controller_eye_diagnostic_material_instance = nullptr;
    }
    if (bridge->controller_eye_diagnostic_material) {
        bridge->engine->destroy(bridge->controller_eye_diagnostic_material);
        bridge->controller_eye_diagnostic_material = nullptr;
    }
}

std::string bridge_controller_semantic(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    const bool touched = name.find("touched") != std::string::npos ||
            name.find("thumbrest") != std::string::npos;
    if (touched && (name.find("thumbstick_xaxis") != std::string::npos ||
            name.find("touchpad_xaxis") != std::string::npos)) return "joystick_x_touched";
    if (touched && (name.find("thumbstick_yaxis") != std::string::npos ||
            name.find("touchpad_yaxis") != std::string::npos)) return "joystick_y_touched";
    if (touched && (name.find("thumbstick") != std::string::npos ||
            name.find("touchpad") != std::string::npos ||
            name.find("thumbrest") != std::string::npos)) return "joystick_touched";
    if (name.find("thumbstick_xaxis") != std::string::npos ||
            name.find("touchpad_xaxis") != std::string::npos) return "joystick_x";
    if (name.find("thumbstick_yaxis") != std::string::npos ||
            name.find("touchpad_yaxis") != std::string::npos) return "joystick_y";
    if (name.find("thumbstick") != std::string::npos ||
            name.find("touchpad") != std::string::npos) return "joystick";
    if (name.find("trigger") != std::string::npos) return "trigger";
    if (name.find("squeeze") != std::string::npos ||
            name.find("grip") != std::string::npos ||
            name.find("grasp") != std::string::npos) return "grip";
    if (name.find("a_button") != std::string::npos ||
            name.find("abutton") != std::string::npos) return "a_button";
    if (name.find("b_button") != std::string::npos ||
            name.find("bbutton") != std::string::npos) return "b_button";
    if (name.find("x_button") != std::string::npos ||
            name.find("xbutton") != std::string::npos) return "x_button";
    if (name.find("y_button") != std::string::npos ||
            name.find("ybutton") != std::string::npos) return "y_button";
    if (name.find("menu") != std::string::npos) return "menu_button";
    if (name.find("photo_button") != std::string::npos ||
            name.find("home_button") != std::string::npos ||
            name.find("homebutton") != std::string::npos ||
            name.find("app_button") != std::string::npos ||
            name.find("pico") != std::string::npos) return "home_button";
    return {};
}

namespace {

bool controller_eye_diagnostic_requested() {
    const char* value = std::getenv("D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC");
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool ensure_controller_eye_diagnostic_material(FilamentBridge* bridge) {
    if (bridge->controller_eye_diagnostic_material_instance) return true;
    const char* fragment_shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            material.baseColor = variable_d2sEyeIndex.x < 0.5
                    ? float4(1.0, 0.0, 0.0, 1.0)
                    : float4(0.0, 1.0, 0.0, 1.0);
        }
    )FILAMENT";
    const char* vertex_shader = R"FILAMENT(
        void materialVertex(inout MaterialVertexInputs material) {
            material.d2sEyeIndex = float4(float(getEyeIndex()), 0.0, 0.0, 0.0);
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S Controller Eye Diagnostic")
            .material(fragment_shader)
            .materialVertex(vertex_shader)
            .variable(filamat::MaterialBuilder::Variable::CUSTOM0, "d2sEyeIndex")
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::OPAQUE)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(true)
            .depthCulling(true)
            .stereoscopicType(filamat::MaterialBuilder::StereoscopicType::MULTIVIEW)
            .stereoscopicEyeCount(2)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    const filamat::Package package = builder.build(bridge->engine->getJobSystem());
    if (!package.isValid()) {
        bridge_set_error(bridge, "Filament could not build controller eye diagnostic material");
        return false;
    }
    bridge->controller_eye_diagnostic_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*bridge->engine);
    if (!bridge->controller_eye_diagnostic_material) {
        bridge_set_error(bridge, "Filament could not create controller eye diagnostic material");
        return false;
    }
    bridge->controller_eye_diagnostic_material_instance =
            bridge->controller_eye_diagnostic_material->createInstance();
    return bridge->controller_eye_diagnostic_material_instance != nullptr;
}

struct ControllerQuaternion {
    float x;
    float y;
    float z;
    float w;
};

ControllerQuaternion controller_quaternion_from_matrix(
        const filament::math::mat4f& matrix) {
    const auto component = [&matrix](int row, int column) {
        const float x = matrix[column][0];
        const float y = matrix[column][1];
        const float z = matrix[column][2];
        const float length = std::sqrt(x * x + y * y + z * z);
        return length > 1.0e-8f ? matrix[column][row] / length : (row == column ? 1.0f : 0.0f);
    };
    const float m00 = component(0, 0);
    const float m01 = component(0, 1);
    const float m02 = component(0, 2);
    const float m10 = component(1, 0);
    const float m11 = component(1, 1);
    const float m12 = component(1, 2);
    const float m20 = component(2, 0);
    const float m21 = component(2, 1);
    const float m22 = component(2, 2);
    ControllerQuaternion result{};
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        const float scale = std::sqrt(trace + 1.0f) * 2.0f;
        result = {(m21 - m12) / scale, (m02 - m20) / scale,
                (m10 - m01) / scale, 0.25f * scale};
    } else if (m00 > m11 && m00 > m22) {
        const float scale = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        result = {0.25f * scale, (m01 + m10) / scale,
                (m02 + m20) / scale, (m21 - m12) / scale};
    } else if (m11 > m22) {
        const float scale = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        result = {(m01 + m10) / scale, 0.25f * scale,
                (m12 + m21) / scale, (m02 - m20) / scale};
    } else {
        const float scale = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        result = {(m02 + m20) / scale, (m12 + m21) / scale,
                0.25f * scale, (m10 - m01) / scale};
    }
    const float length = std::sqrt(result.x * result.x + result.y * result.y +
            result.z * result.z + result.w * result.w);
    if (length > 1.0e-8f) {
        result.x /= length;
        result.y /= length;
        result.z /= length;
        result.w /= length;
    }
    return result;
}

ControllerQuaternion controller_quaternion_slerp(
        ControllerQuaternion first, ControllerQuaternion second, float amount) {
    float dot = first.x * second.x + first.y * second.y +
            first.z * second.z + first.w * second.w;
    if (dot < 0.0f) {
        second = {-second.x, -second.y, -second.z, -second.w};
        dot = -dot;
    }
    if (dot > 0.9995f) {
        ControllerQuaternion result{
                first.x + amount * (second.x - first.x),
                first.y + amount * (second.y - first.y),
                first.z + amount * (second.z - first.z),
                first.w + amount * (second.w - first.w)};
        const float length = std::sqrt(result.x * result.x + result.y * result.y +
                result.z * result.z + result.w * result.w);
        return {result.x / length, result.y / length, result.z / length, result.w / length};
    }
    const float theta_zero = std::acos(std::clamp(dot, -1.0f, 1.0f));
    const float theta = theta_zero * amount;
    const float sin_theta_zero = std::sin(theta_zero);
    const float second_scale = std::sin(theta) / sin_theta_zero;
    const float first_scale = std::cos(theta) - dot * second_scale;
    return {
            first.x * first_scale + second.x * second_scale,
            first.y * first_scale + second.y * second_scale,
            first.z * first_scale + second.z * second_scale,
            first.w * first_scale + second.w * second_scale};
}

}  // namespace

filament::math::mat4f bridge_controller_interpolate_transform(
        const filament::math::mat4f& value,
        const filament::math::mat4f& minimum,
        const filament::math::mat4f& maximum,
        float amount) {
    const float t = std::clamp(std::abs(amount), 0.0f, 1.0f);
    const auto& target = amount < 0.0f ? minimum : maximum;
    filament::math::mat4f result = value;
    const auto rotation = controller_quaternion_slerp(
            controller_quaternion_from_matrix(value),
            controller_quaternion_from_matrix(target), t);
    const float xx = rotation.x * rotation.x;
    const float yy = rotation.y * rotation.y;
    const float zz = rotation.z * rotation.z;
    const float xy = rotation.x * rotation.y;
    const float xz = rotation.x * rotation.z;
    const float yz = rotation.y * rotation.z;
    const float wx = rotation.w * rotation.x;
    const float wy = rotation.w * rotation.y;
    const float wz = rotation.w * rotation.z;
    const float rotation_rows[3][3] = {
            {1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy)},
            {2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx)},
            {2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy)}};
    for (int column = 0; column < 3; ++column) {
        const auto column_length = [](const filament::math::mat4f& matrix, int index) {
            const float x = matrix[index][0];
            const float y = matrix[index][1];
            const float z = matrix[index][2];
            return std::sqrt(x * x + y * y + z * z);
        };
        const float value_scale = column_length(value, column);
        const float target_scale = column_length(target, column);
        const float scale = value_scale + (target_scale - value_scale) * t;
        for (int row = 0; row < 3; ++row) {
            result[column][row] = rotation_rows[row][column] * scale;
        }
    }
    for (int row = 0; row < 3; ++row) {
        result[3][row] = value[3][row] + (target[3][row] - value[3][row]) * t;
    }
    return result;
}

float bridge_controller_animation_amount(
        const ControllerAsset& controller, const std::string& semantic) {
    if (semantic == "trigger") return controller.trigger;
    if (semantic == "grip") return controller.grip;
    if (semantic == "joystick_x") return controller.joystick_x;
    if (semantic == "joystick_y") return controller.joystick_y;
    if (semantic == "joystick") return controller.button_values[5];
    if (semantic == "joystick_x_touched") {
        return controller.button_values[6] * controller.joystick_x;
    }
    if (semantic == "joystick_y_touched") {
        return controller.button_values[6] * controller.joystick_y;
    }
    if (semantic == "joystick_touched") return controller.button_values[6];
    if (semantic == "a_button") return controller.button_values[0];
    if (semantic == "b_button") return controller.button_values[1];
    if (semantic == "x_button") return controller.button_values[2];
    if (semantic == "y_button") return controller.button_values[3];
    if (semantic == "menu_button") return controller.button_values[4];
    if (semantic == "home_button") return 0.0f;
    return 0.0f;
}

namespace {

void bridge_controller_add_animation(
        FilamentBridge* bridge, ControllerAsset& controller,
        const std::string& value_name, utils::Entity value_entity) {
    if (!bridge || !controller.asset || value_entity.isNull()) return;
    const std::string suffix = "_value";
    if (value_name.size() <= suffix.size() ||
            value_name.compare(value_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return;
    }
    const std::string prefix = value_name.substr(0, value_name.size() - suffix.size());
    const auto min_entity = controller.asset->getFirstEntityByName(
            (prefix + "_min").c_str());
    const auto max_entity = controller.asset->getFirstEntityByName(
            (prefix + "_max").c_str());
    const auto semantic = bridge_controller_semantic(value_name);
    auto& transforms = bridge->engine->getTransformManager();
    const auto value_instance = transforms.getInstance(value_entity);
    const auto min_instance = transforms.getInstance(min_entity);
    const auto max_instance = transforms.getInstance(max_entity);
    if (semantic.empty() || min_entity.isNull() || max_entity.isNull() ||
            !value_instance.isValid() || !min_instance.isValid() || !max_instance.isValid()) {
        return;
    }
    if (std::any_of(controller.animations.begin(), controller.animations.end(),
            [&value_entity](const ControllerAnimation& animation) {
                return animation.value_entity == value_entity;
            })) {
        return;
    }
    controller.animations.push_back({
            value_entity, min_entity, max_entity,
            transforms.getTransform(value_instance),
            transforms.getTransform(min_instance),
            transforms.getTransform(max_instance),
            semantic, value_name});
}

}  // namespace

void bridge_controller_update_animations(
        FilamentBridge* bridge, ControllerAsset& controller) {
    if (!controller.asset || !bridge->engine) return;
    auto& transforms = bridge->engine->getTransformManager();
    for (const auto& animation : controller.animations) {
        const float amount = bridge_controller_animation_amount(controller, animation.semantic);
        if (!transforms.hasComponent(animation.value_entity)) continue;
        transforms.setTransform(
                transforms.getInstance(animation.value_entity),
                bridge_controller_interpolate_transform(
                        animation.value_transform,
                        animation.min_transform,
                        animation.max_transform,
                        amount));
    }
}

int bridge_controller_load(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count) {
    if (!bridge || !bridge->engine || !bridge->asset_loader ||
            hand > 1 || !bytes || !byte_count) {
        return 0;
    }
    auto& controller = bridge->controllers[hand];
    bridge_controller_destroy(bridge, controller);
    controller.bytes.assign(bytes, bytes + byte_count);
    controller.asset = bridge->asset_loader->createAsset(
            controller.bytes.data(), byte_count);
    if (!controller.asset) {
        bridge_set_error(bridge, "Filament could not parse controller GLB");
        controller = {};
        return 0;
    }
    filament::gltfio::ResourceConfiguration config{};
    config.engine = bridge->engine;
    config.normalizeSkinningWeights = true;
    filament::gltfio::ResourceLoader resources(config);
    resources.addTextureProvider("image/png", bridge->texture_provider);
    resources.addTextureProvider("image/jpeg", bridge->texture_provider);
    if (!resources.loadResources(controller.asset)) {
        bridge_controller_destroy(bridge, controller);
        bridge_set_error(bridge, "Filament could not load controller GLB resources");
        return 0;
    }
    bridge->foreground_scene->addEntities(
            controller.asset->getEntities(), controller.asset->getEntityCount());
    const bool eye_diagnostic = controller_eye_diagnostic_requested();
    if (eye_diagnostic && !ensure_controller_eye_diagnostic_material(bridge)) {
        bridge_controller_destroy(bridge, controller);
        return 0;
    }
    // Keep controllers in front of the virtual screen and below the laser.
    // Both are rendered in the same Filament View, so priority is the
    // ordering contract for the foreground objects.
    auto& renderables = bridge->engine->getRenderableManager();
    for (size_t index = 0; index < controller.asset->getRenderableEntityCount(); ++index) {
        const auto entity = controller.asset->getRenderableEntities()[index];
        const auto instance = renderables.getInstance(entity);
        if (!instance.isValid()) continue;
        renderables.setPriority(instance, 6);
        renderables.setLightChannel(instance, 0, false);
        renderables.setLightChannel(instance, 1, true);
        renderables.setLayerMask(instance, 0xff, 0x01);
        if (eye_diagnostic) {
            const size_t primitive_count = renderables.getPrimitiveCount(instance);
            for (size_t primitive = 0; primitive < primitive_count; ++primitive) {
                renderables.setMaterialInstanceAt(
                        instance, primitive,
                        bridge->controller_eye_diagnostic_material_instance);
            }
        }
    }
    for (size_t index = 0; index < controller.asset->getEntityCount(); ++index) {
        const auto entity = controller.asset->getEntities()[index];
        const char* raw_name = controller.asset->getName(entity);
        if (!raw_name) continue;
        const std::string value_name(raw_name);
        const std::string suffix = "_value";
        if (value_name.size() <= suffix.size() ||
                value_name.compare(value_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        bridge_controller_add_animation(bridge, controller, value_name, entity);
    }
    // Always complete discovery from the legacy node contract. Some Filament
    // SDK builds omit non-renderable nodes from getEntities(), and others may
    // expose only a subset. bridge_controller_add_animation deduplicates them.
    constexpr const char* kControllerValues[] = {
            "xr_standard_trigger_pressed_value",
            "xr_standard_squeeze_pressed_value",
            "xr_standard_thumbstick_pressed_value",
            "xr_standard_thumbstick_xaxis_pressed_value",
            "xr_standard_thumbstick_yaxis_pressed_value",
            "xr_standard_touchpad_pressed_value",
            "xr_standard_touchpad_xaxis_pressed_value",
            "xr_standard_touchpad_yaxis_pressed_value",
            "xr_standard_touchpad_xaxis_touched_value",
            "xr_standard_touchpad_yaxis_touched_value",
            "thumbrest_pressed_value",
            "a_button_pressed_value", "b_button_pressed_value",
            "x_button_pressed_value", "y_button_pressed_value",
            "LMenu_pressed_value", "RMenu_value", "menu_pressed_value",
            "HomeButton_pressed_value", "LPico_value", "RPico_value",
    };
    for (const char* value_name : kControllerValues) {
        const auto entity = controller.asset->getFirstEntityByName(value_name);
        bridge_controller_add_animation(bridge, controller, value_name, entity);
    }
    if (controller.animations.empty()) {
        bridge_controller_destroy(bridge, controller);
        bridge_set_error(bridge, "Controller GLB exposes no _value/_min/_max animation triplets");
        return 0;
    }
    std::printf("[FilamentBridge] controller loaded hand=%u animations=%zu",
            hand, controller.animations.size());
    for (const auto& animation : controller.animations) {
        std::printf(" %s:%s", animation.value_name.c_str(), animation.semantic.c_str());
    }
    std::printf("\n");
    std::fflush(stdout);
    if (eye_diagnostic) {
        std::printf(
                "[FilamentBridge] controller eye diagnostic active hand=%u "
                "eye0=red eye1=green\n", hand);
        std::fflush(stdout);
    }
    controller.asset->releaseSourceData();
    controller.bytes.clear();
    bridge->engine->flushAndWait();
    bridge_controller_update_animations(bridge, controller);
    return 1;
}

int bridge_controller_set_pose(
        FilamentBridge* bridge, uint32_t hand, const float* matrix16) {
    if (!bridge || !bridge->engine || hand > 1 || !matrix16 ||
            !bridge->controllers[hand].asset) return 0;
    const auto root = bridge->controllers[hand].asset->getRoot();
    auto& transforms = bridge->engine->getTransformManager();
    const auto instance = transforms.getInstance(root);
    if (!instance.isValid()) return 0;
    const filament::math::mat4f matrix(
            matrix16[0], matrix16[1], matrix16[2], matrix16[3],
            matrix16[4], matrix16[5], matrix16[6], matrix16[7],
            matrix16[8], matrix16[9], matrix16[10], matrix16[11],
            matrix16[12], matrix16[13], matrix16[14], matrix16[15]);
    transforms.setTransform(instance, matrix);
    return 1;
}

int bridge_controller_set_inputs(
        FilamentBridge* bridge, uint32_t hand,
        float trigger, float grip,
        float joystick_x, float joystick_y,
        uint32_t button_mask) {
    if (!bridge || hand > 1 || !bridge->controllers[hand].asset) return 0;
    auto& controller = bridge->controllers[hand];
    const auto now = std::chrono::steady_clock::now();
    float alpha = 1.0f;
    if (controller.input_initialized) {
        const float delta_seconds = std::clamp(
                std::chrono::duration<float>(now - controller.last_input_time).count(),
                0.0f, 0.05f);
        alpha = std::min(1.0f, delta_seconds * 24.0f);
    }
    controller.last_input_time = now;
    controller.input_initialized = true;
    const auto smooth = [alpha](float current, float target) {
        return current + (target - current) * alpha;
    };
    controller.trigger = smooth(controller.trigger, std::clamp(trigger, 0.0f, 1.0f));
    controller.grip = smooth(controller.grip, std::clamp(grip, 0.0f, 1.0f));
    controller.joystick_x = smooth(
            controller.joystick_x, std::clamp(joystick_x, -1.0f, 1.0f));
    controller.joystick_y = smooth(
            controller.joystick_y, std::clamp(joystick_y, -1.0f, 1.0f));
    controller.button_mask = button_mask;
    for (uint32_t bit = 0; bit < controller.button_values.size(); ++bit) {
        const float target = (button_mask & (1u << bit)) ? 1.0f : 0.0f;
        controller.button_values[bit] = smooth(controller.button_values[bit], target);
    }
    bridge_controller_update_animations(bridge, controller);
    return 1;
}

int bridge_controller_set_visible(
        FilamentBridge* bridge, uint32_t hand, int visible) {
    if (!bridge || !bridge->engine || hand > 1 ||
            !bridge->controllers[hand].asset) return 0;
    auto& controller = bridge->controllers[hand];
    const bool next_visible = visible != 0;
    if (controller.visible == next_visible) return 1;
    auto& renderables = bridge->engine->getRenderableManager();
    for (size_t index = 0;
            index < controller.asset->getRenderableEntityCount(); ++index) {
        const auto entity = controller.asset->getRenderableEntities()[index];
        const auto instance = renderables.getInstance(entity);
        if (!instance.isValid()) continue;
        renderables.setLayerMask(instance, 0xff, next_visible ? 0x01 : 0x00);
    }
    controller.visible = next_visible;
    return 1;
}
