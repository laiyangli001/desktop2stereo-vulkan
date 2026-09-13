#!/usr/bin/env python3
"""Apply the pinned Filament 1.76 D2S Vulkan external-image extension."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one replacement in {path}: found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_d2s_vulkan_external_image.py FILAMENT_SOURCE")
    root = Path(sys.argv[1]).resolve()
    platform = root / "filament/backend/include/backend/Platform.h"
    vulkan_platform_h = root / "filament/backend/include/backend/platforms/VulkanPlatform.h"
    vulkan_platform_cpp = root / "filament/backend/src/vulkan/platform/VulkanPlatform.cpp"
    vulkan_driver_cpp = root / "filament/backend/src/vulkan/VulkanDriver.cpp"
    vulkan_swapchain_cpp = root / "filament/backend/src/vulkan/VulkanSwapChain.cpp"
    vulkan_texture_cpp = root / "filament/backend/src/vulkan/VulkanTexture.cpp"
    vulkan_handles_h = root / "filament/backend/src/vulkan/VulkanHandles.h"
    vulkan_handles_cpp = root / "filament/backend/src/vulkan/VulkanHandles.cpp"
    vulkan_async_handles_cpp = root / "filament/backend/src/vulkan/VulkanAsyncHandles.cpp"
    renderer_cpp = root / "filament/src/details/Renderer.cpp"
    post_process_manager_cpp = root / "filament/src/PostProcessManager.cpp"
    vulkan_fbo_cache_cpp = root / "filament/backend/src/vulkan/VulkanFboCache.cpp"

    # Filament 1.76 builds a multiview clearDepth package but always registers
    # the ordinary instanced package in PostProcessManager. Match the package
    # to the Engine stereo mode just like the default material and skybox do.
    replace_once(
        post_process_manager_cpp,
        '        { "clearDepth",                 MATERIAL(MATERIALS, CLEARDEPTH) },\n',
        "",
    )
    replace_once(
        post_process_manager_cpp,
        """        for (auto const& info: sMaterialList) {
            registerPostProcessMaterial(info.name, info);
        }
""",
        """        for (auto const& info: sMaterialList) {
            registerPostProcessMaterial(info.name, info);
        }
#ifdef FILAMENT_ENABLE_MULTIVIEW
        if (engine.getConfig().stereoscopicType == StereoscopicType::MULTIVIEW) {
            StaticMaterialInfo const clearDepthInfo = {
                    "clearDepth", MATERIAL(MATERIALS, CLEARDEPTH_MULTIVIEW) };
            registerPostProcessMaterial(clearDepthInfo.name, clearDepthInfo);
        } else
#endif
        {
            StaticMaterialInfo const clearDepthInfo = {
                    "clearDepth", MATERIAL(MATERIALS, CLEARDEPTH) };
            registerPostProcessMaterial(clearDepthInfo.name, clearDepthInfo);
        }
""",
    )

    replace_once(
        renderer_cpp,
        "#include <chrono>\n#include <limits>\n",
        "#include <chrono>\n#include <cstdio>\n#include <cstdlib>\n#include <limits>\n",
    )
    replace_once(
        renderer_cpp,
        """    variant.setShadowSampler2D(view.hasShadowing() && view.getShadowType() != ShadowType::PCF);
    variant.setStereo(view.hasStereo());
""",
        """    variant.setShadowSampler2D(view.hasShadowing() && view.getShadowType() != ShadowType::PCF);
    variant.setStereo(view.hasStereo());
    static bool d2sStereoTraceLogged[2] = {};
    const unsigned d2sStereoTraceIndex = view.hasStereo() ? 1u : 0u;
    if (!d2sStereoTraceLogged[d2sStereoTraceIndex] &&
            std::getenv("D2S_FILAMENT_EYE_DIAGNOSTIC")) {
        d2sStereoTraceLogged[d2sStereoTraceIndex] = true;
        std::fprintf(stderr,
                "[D2S stereo trace] renderer viewStereo=%u variantStereo=%u "
                "multiview=%u engineType=%u eyeCount=%u\\n",
                static_cast<unsigned>(view.hasStereo()),
                static_cast<unsigned>(variant.hasStereo()),
                static_cast<unsigned>(isRenderingMultiview),
                static_cast<unsigned>(engine.getConfig().stereoscopicType),
                static_cast<unsigned>(engine.getConfig().stereoscopicEyeCount));
        std::fflush(stderr);
    }
""",
    )
    replace_once(
        vulkan_fbo_cache_cpp,
        "#include <utils/Panic.h>\n",
        "#include <utils/Panic.h>\n\n#include <cstdio>\n#include <cstdlib>\n",
    )
    replace_once(
        vulkan_fbo_cache_cpp,
        """    if (config.viewCount > 1) {
        // Fill the multiview create info.
""",
        """    if (config.viewCount > 1) {
        static bool d2sStereoTraceLogged = false;
        if (!d2sStereoTraceLogged && std::getenv("D2S_FILAMENT_EYE_DIAGNOSTIC")) {
            d2sStereoTraceLogged = true;
            std::fprintf(stderr,
                    "[D2S stereo trace] renderPass viewCount=%u viewMask=0x%x\\n",
                    config.viewCount, subpassViewMask);
            std::fflush(stderr);
        }
        // Fill the multiview create info.
""",
    )
    replace_once(
        vulkan_async_handles_cpp,
        "#include <chrono>\n#include <cstdint>\n",
        """#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
""",
    )
    replace_once(
        vulkan_async_handles_cpp,
        """    static_assert(static_cast<ShaderStage>(0) == ShaderStage::VERTEX &&
            static_cast<ShaderStage>(1) == ShaderStage::FRAGMENT &&
            MAX_SHADER_MODULES == 2);

    for (size_t i = 0; i < MAX_SHADER_MODULES; i++) {
""",
        """    static_assert(static_cast<ShaderStage>(0) == ShaderStage::VERTEX &&
            static_cast<ShaderStage>(1) == ShaderStage::FRAGMENT &&
            MAX_SHADER_MODULES == 2);

    const char* const d2sProgramName = builder.getName().c_str_safe();
    const bool d2sTraceProgram = std::getenv("D2S_FILAMENT_EYE_DIAGNOSTIC") &&
            (std::strstr(d2sProgramName, "D2S") ||
                    std::strcmp(d2sProgramName, "clearDepth") == 0);
    bool d2sVertexHasViewIndex = false;
    bool d2sFragmentHasViewIndex = false;
    for (size_t i = 0; i < MAX_SHADER_MODULES; i++) {
""",
    )
    replace_once(
        vulkan_async_handles_cpp,
        """        Program::ShaderBlob const& blob = blobs[i];

        uint32_t* data = (uint32_t*) blob.data();
""",
        """        Program::ShaderBlob const& blob = blobs[i];
        const char* const d2sShaderDumpDir =
                std::getenv("D2S_FILAMENT_SHADER_DUMP_DIR");
        if (d2sShaderDumpDir && d2sShaderDumpDir[0] != '\\0' &&
                std::strcmp(d2sProgramName, "D2S Controller Eye Diagnostic") == 0) {
            char d2sShaderDumpPath[1024];
            std::snprintf(d2sShaderDumpPath, sizeof(d2sShaderDumpPath),
                    "%s/filament_controller_eye_diag.%s.spv", d2sShaderDumpDir,
                    i == 0 ? "vert" : "frag");
            if (std::FILE* const dump = std::fopen(d2sShaderDumpPath, "wb")) {
                std::fwrite(blob.data(), 1, blob.size(), dump);
                std::fclose(dump);
                std::fprintf(stderr,
                        "[D2S stereo trace] shader dump stage=%s bytes=%zu path=%s\\n",
                        i == 0 ? "vertex" : "fragment",
                        static_cast<size_t>(blob.size()),
                        d2sShaderDumpPath);
                std::fflush(stderr);
            }
        }
        if (d2sTraceProgram) {
            const auto* words = reinterpret_cast<const uint32_t*>(blob.data());
            for (size_t word = 0; word < blob.size() / sizeof(uint32_t); ++word) {
                if (i == 0) {
                    d2sVertexHasViewIndex |= words[word] == 4440u;
                } else {
                    d2sFragmentHasViewIndex |= words[word] == 4440u;
                }
            }
        }

        uint32_t* data = (uint32_t*) blob.data();
""",
    )
    replace_once(
        vulkan_async_handles_cpp,
        """#if FVK_ENABLED(FVK_DEBUG_SHADER_MODULE)
    FVK_LOGD << "Created VulkanProgram " << builder << ", shaders = (" << modules[0]
""",
        """    if (d2sTraceProgram) {
        std::fprintf(stderr,
                "[D2S stereo trace] program name=%s multiview=%u "
                "vertexViewIndex=%u fragmentViewIndex=%u %s\\n",
                d2sProgramName, static_cast<unsigned>(builder.isMultiview()),
                static_cast<unsigned>(d2sVertexHasViewIndex),
                static_cast<unsigned>(d2sFragmentHasViewIndex),
                programString.c_str_safe());
        std::fflush(stderr);
    }

#if FVK_ENABLED(FVK_DEBUG_SHADER_MODULE)
    FVK_LOGD << "Created VulkanProgram " << builder << ", shaders = (" << modules[0]
""",
    )
    replace_once(
        vulkan_driver_cpp,
        "#include <chrono>\n#include <mutex>\n",
        """#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
""",
    )
    replace_once(
        vulkan_driver_cpp,
        """    // Update the VK raster state.
    auto rt = mCurrentRenderPass.renderTarget;

    VulkanPipelineCache::RasterState const vulkanRasterState{
""",
        """    // Update the VK raster state.
    auto rt = mCurrentRenderPass.renderTarget;
    static unsigned d2sControllerDrawTraceCount = 0;
    if (d2sControllerDrawTraceCount < 8 &&
            std::getenv("D2S_FILAMENT_EYE_DIAGNOSTIC") &&
            std::strcmp(program->name.c_str(), "D2S Controller Eye Diagnostic") == 0) {
        ++d2sControllerDrawTraceCount;
        auto& d2sColor = rt->getColor(0);
        auto const& d2sPrimaryRange = d2sColor.texture->getPrimaryViewRange();
        std::fprintf(stderr,
                "[D2S stereo trace] controller draw program=%s viewCount=%u "
                "samples=%u resolveMask=0x%x fbLayers=%u colorLayers=%u "
                "primaryLayers=%u viewType=%u %s\\n",
                program->name.c_str(), rt->getRenderPassKey().viewCount,
                static_cast<unsigned>(rt->getSamples()),
                static_cast<unsigned>(rt->getRenderPassKey().needsResolveMask),
                static_cast<unsigned>(rt->getFboKey().layers),
                static_cast<unsigned>(d2sColor.layerCount),
                static_cast<unsigned>(d2sPrimaryRange.layerCount),
                static_cast<unsigned>(d2sColor.texture->getViewType()),
                program->programString.c_str_safe());
        std::fflush(stderr);
    }

    VulkanPipelineCache::RasterState const vulkanRasterState{
""",
    )
    replace_once(
        vulkan_texture_cpp,
        """              format, fvkutils::getViewType(SamplerType::SAMPLER_2D),
""",
        """              format, fvkutils::getViewType(getSamplerTypeFromDepth(depth)),
""",
    )

    replace_once(
        platform,
        "    class ExternalImageHandle;\n\n    class ExternalImage {\n",
        """    class ExternalImageHandle;

    // Metadata for a borrowed application-owned Vulkan image.
    struct VulkanExternalImageData {
        uint64_t image = 0;
        int32_t format = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    class ExternalImage {
""",
    )
    replace_once(
        platform,
        "    protected:\n        virtual ~ExternalImage() noexcept;\n",
        """    protected:
        ExternalImage() noexcept = default;
        virtual ~ExternalImage() noexcept;

    public:
        virtual bool getVulkanImageData(VulkanExternalImageData& out) const noexcept {
            out = {};
            return false;
        }
""",
    )
    replace_once(
        vulkan_platform_h,
        """    // Note that the image metadata might change per-frame, hence we need a method for extracting
    // it.
    virtual ExternalImageMetadata extractExternalImageMetadata(ExternalImageHandleRef image) const {
        return {};
    }
""",
        """    // Creates a handle for a borrowed application-owned VkImage.
    static ExternalImageHandle createExternalImageFromVkImage(
            VkImage image, VkFormat format, uint32_t width, uint32_t height) noexcept;

    // Note that the image metadata might change per-frame, hence we need a method for extracting
    // it.
    virtual ExternalImageMetadata extractExternalImageMetadata(ExternalImageHandleRef image) const;
""",
    )
    replace_once(
        vulkan_platform_h,
        """    virtual ImageData createVkImageFromExternal(ExternalImageHandleRef image,
            uint32_t logicalWidth, uint32_t logicalHeight) const {
        return {};
    }
""",
        """    virtual ImageData createVkImageFromExternal(ExternalImageHandleRef image,
            uint32_t logicalWidth, uint32_t logicalHeight) const;
""",
    )

    # Filament's Vulkan driver keeps a single default RenderTarget. With two
    # OpenXR eye SwapChains, isSwapchainBound() can return true for eye0 while
    # the driver is now making eye1 current, so eye1 renders into eye0's image
    # and the shared semaphores eventually trigger VK_ERROR_DEVICE_LOST. Track
    # which swapchain/image is actually bound and rebind on eye switches.
    replace_once(
        vulkan_handles_h,
        """    bool isSwapchainBound() const {
        return isSwapChain() && mInfo->colors[0];
    }
""",
        """    bool isSwapchainBound() const {
        return isSwapChain() && mInfo->colors[0];
    }

    bool isBoundToSwapChain(void const* swapchain, void const* image) const {
        return isSwapchainBound() &&
                mBoundSwapChain == swapchain &&
                mBoundSwapChainImage == image;
    }
""",
    )
    replace_once(
        vulkan_handles_h,
        """    bool mOffscreen;
    bool mProtected;

    std::unique_ptr<Auxiliary> mInfo;
""",
        """    bool mOffscreen;
    bool mProtected;
    void const* mBoundSwapChain = nullptr;
    void const* mBoundSwapChainImage = nullptr;

    std::unique_ptr<Auxiliary> mInfo;
""",
    )
    replace_once(
        vulkan_handles_h,
        """        std::swap(mInfo, target.mInfo);
""",
        """        std::swap(mInfo, target.mInfo);
        std::swap(mBoundSwapChain, target.mBoundSwapChain);
        std::swap(mBoundSwapChainImage, target.mBoundSwapChainImage);
""",
    )
    replace_once(
        vulkan_handles_cpp,
        """    mInfo->colors.set(0);
}

void VulkanRenderTarget::releaseSwapchain() {
    mInfo->colors = {};
    mInfo->attachments.clear();
}
""",
        """    mInfo->colors.set(0);
    mBoundSwapChain = swapchain.get();
    mBoundSwapChainImage = swapchain->getCurrentColor().get();
}

void VulkanRenderTarget::releaseSwapchain() {
    mInfo->colors = {};
    mInfo->attachments.clear();
    mBoundSwapChain = nullptr;
    mBoundSwapChainImage = nullptr;
}
""",
    )
    replace_once(
        vulkan_driver_cpp,
        """    // Swapchain has already been bound to the default render target.  We just return.
    if (mDefaultRenderTarget->isSwapchainBound()) {
        // true means that the rendertarget has the right images attached.
        return true;
    }

    auto const [acquired, backingChanged] = mCurrentSwapChain->acquire();
""",
        """    auto const [acquired, backingChanged] = mCurrentSwapChain->acquire();
""",
    )
    replace_once(
        vulkan_driver_cpp,
        """    if (acquired) {
        mDefaultRenderTarget->bindSwapChain(mCurrentSwapChain);
        return true;
    }
""",
        """    if (acquired) {
        void const* color = mCurrentSwapChain->getCurrentColor().get();
        if (!mDefaultRenderTarget->isBoundToSwapChain(
                mCurrentSwapChain.get(), color)) {
            mDefaultRenderTarget->releaseSwapchain();
            mDefaultRenderTarget->bindSwapChain(mCurrentSwapChain);
        }
        return true;
    }
""",
    )

    marker = "VulkanPlatform::VulkanPlatform() = default;\n"
    cpp = vulkan_platform_cpp.read_text(encoding="utf-8")
    if cpp.count(marker) != 1:
        raise RuntimeError("VulkanPlatform constructor marker changed")
    helper = """class D2SVulkanExternalImage final : public Platform::ExternalImage {
public:
    explicit D2SVulkanExternalImage(Platform::VulkanExternalImageData data) noexcept
            : mData(data) {}
    ~D2SVulkanExternalImage() override;

protected:
    bool getVulkanImageData(Platform::VulkanExternalImageData& out) const noexcept override {
        out = mData;
        return mData.image != 0 && mData.width != 0 && mData.height != 0;
    }

private:
    Platform::VulkanExternalImageData mData;
};

D2SVulkanExternalImage::~D2SVulkanExternalImage() = default;

Platform::ExternalImageHandle VulkanPlatform::createExternalImageFromVkImage(
        VkImage image, VkFormat format, uint32_t width, uint32_t height) noexcept {
    if (image == VK_NULL_HANDLE || width == 0 || height == 0) {
        return {};
    }
    return Platform::ExternalImageHandle(new D2SVulkanExternalImage({
            reinterpret_cast<uint64_t>(image), static_cast<int32_t>(format), width, height}));
}

VulkanPlatform::ExternalImageMetadata VulkanPlatform::extractExternalImageMetadata(
        ExternalImageHandleRef image) const {
    Platform::VulkanExternalImageData data;
    if (!image || !image->getVulkanImageData(data)) {
        return {};
    }
    const bool srgb = data.format == static_cast<int32_t>(VK_FORMAT_R8G8B8A8_SRGB);
    return {
            .filamentFormat = srgb ? TextureFormat::SRGB8_A8 : TextureFormat::RGBA8,
            .filamentUsage = TextureUsage::SAMPLEABLE,
            .width = data.width,
            .height = data.height,
            .layers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .format = static_cast<VkFormat>(data.format),
            .externalFormat = 0,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            .allocationSize = 0,
            .memoryTypeBits = 0,
            .ycbcrConversionComponents = {},
            .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
            .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
            .xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN,
            .yChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN,
            .isStagingRequired = false,
            .isChromaConversionRequired = false};
}

VulkanPlatform::ImageData VulkanPlatform::createVkImageFromExternal(
        ExternalImageHandleRef image, uint32_t, uint32_t) const {
    Platform::VulkanExternalImageData data;
    if (!image || !image->getVulkanImageData(data)) {
        return {};
    }
    ImageData result;
    result.internal.image = reinterpret_cast<VkImage>(data.image);
    result.internal.memory = VK_NULL_HANDLE;
    return result;
}

"""
    replace_once(
        vulkan_swapchain_cpp,
        """    if (!mHeadless && mTransitionSwapChainImageLayoutForPresent) {
        VulkanCommandBuffer& commands = mCommands->get();
        VkImageSubresourceRange const subresources{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
""",
        """    if (!mHeadless && mTransitionSwapChainImageLayoutForPresent) {
        VulkanCommandBuffer& commands = mCommands->get();
        if (mDepth) {
            VkImageSubresourceRange const depthSubresources{
                    .aspectMask = static_cast<VkImageAspectFlags>(
                            fvkutils::isVkStencilFormat(mDepth->getVkFormat())
                                    ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                    : VK_IMAGE_ASPECT_DEPTH_BIT),
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = mLayerCount,
            };
            mDepth->transitionLayout(&commands, depthSubresources,
                    VulkanLayout::DEPTH_STENCIL_ATTACHMENT);
        }
        VkImageSubresourceRange const subresources{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
""",
    )
    replace_once(
        vulkan_swapchain_cpp,
        """    if (imageSyncData.imageReadySemaphore != VK_NULL_HANDLE) {
        mCommands->injectDependency(imageSyncData.imageReadySemaphore,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    mAcquired = true;
""",
        """    if (imageSyncData.imageReadySemaphore != VK_NULL_HANDLE) {
        mCommands->injectDependency(imageSyncData.imageReadySemaphore,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    if (mDepth) {
        VkImageSubresourceRange const depthSubresources{
                .aspectMask = static_cast<VkImageAspectFlags>(
                        fvkutils::isVkStencilFormat(mDepth->getVkFormat())
                                ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                : VK_IMAGE_ASPECT_DEPTH_BIT),
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = mLayerCount,
        };
        mDepth->transitionLayout(&mCommands->get(), depthSubresources,
                VulkanLayout::DEPTH_STENCIL_ATTACHMENT);
    }
    mAcquired = true;
""",
    )
    vulkan_platform_cpp.write_text(
        cpp.replace(marker, helper + marker), encoding="utf-8", newline=""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
