# AMD ROCm OpenXR glow GPU path

Glow generation stays in torch on the AMD GPU. The small RGBA8 result is
transferred HIP -> host -> Vulkan into an optimal-tiling image. Direct HIP
writes into Vulkan-imported image memory make the AMD driver lose the Vulkan
device under sustained OpenXR compositing (Virtual Desktop's OpenXR log reports
`VkStatus failure [-4]` at `xrEndFrame`); the 320x180 glow payload is tiny, so
host staging avoids that driver fault while keeping the GPU computation path.
NVIDIA's implementation is unchanged.

## Ownership and synchronization

The former ring checked leases with `id(slot)`, while `acquire()` recorded
`id(slot.resource)`. This allowed the worker to overwrite images still held by
OpenXR. Selection, acquisition, and release now share a lock and resource keys.
The currently published image also stays reserved between frame acquisitions.

The worker waits for Vulkan completion before publishing, releases an image
from sampling before reusing it, and only overwrites slots not currently held
by OpenXR. Publication follows the Vulkan barrier to shader-read layout. See the
[Vulkan synchronization specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)
for the image access and layout dependency rules.

Shutdown joins the worker before destroying its images. Unclaimed ROCm prewarm
workers are also closed when OpenXR initialization fails.

## Validation

```powershell
src\python3\python.exe -m pytest -q tests/test_rocm_torch_glow_source.py
src\python3\python.exe src\tools\rocm_glow_smoke.py --seconds 120
```

On RX 9060 XT with bundled torch `2.10.0+rocm7.14.0a20260615`, a 20-second
smoke passed 12 exact patterned-image readbacks, 424 glow publications, and 46
checks that retained images stayed unchanged. Diagnostic readbacks exist only
in the smoke tool. Visual/headset validation is required to confirm the
previous device-loss failure no longer appears.

## OpenXR eye handoff

The AMD output adapter writes HIP results into exported Vulkan staging buffers,
then Vulkan runs `vkCmdCopyBufferToImage` so HIP never writes image memory
directly. This lasted longer in headset tests than both direct image copies and
host staging. `D2S_ROCM_EYE_HOST_STAGING=1` opts into pure host staging, but the
GPU buffer path is the default because it sustained ~70 FPS longer in testing.
Host staging passed a 36-frame real-GPU handoff with exact Vulkan readback.

`D2S_OPENXR_DISABLE_GLOW_DRAW=1` disables rendering regardless of upload success.
AMD clears this legacy diagnostic automatically and keeps an explicit AMD-only
`D2S_ROCM_DISABLE_GLOW_DRAW=1` switch; restart any launcher that inherited the
old setting.
