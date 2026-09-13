# OpenXR Vulkan 流水并行性能优化方案

**文档状态**：实施方案  
**更新时间**：2026-08-03  
**适用范围**：Vulkan Compute、Filament Vulkan、OpenXR Projection/Quad 提交路径

## 1. 目标与边界

本方案用于降低 OpenXR Presenter 热路径中的 CPU 等待、GPU 空泡和重复提交，使已经实测达到约 55 FPS 的高吞吐路径能够长时间稳定运行，并为后续继续优化提供顺序、指标和回退条件。

目标：

- 左右眼 GPU 工作连续入队，只在双眼全部提交后进行一次帧级完成同步。
- Vulkan Compute 与 OpenXR graphics 在资源和队列允许时并行执行。
- 正常帧不执行 CPU 像素回读、设备级空闲等待或逐眼 `flushAndWait()`。
- 所有优化保持现有 zero-copy、颜色空间、左右眼顺序和 OpenXR 行为不变。
- NVIDIA、AMD、Intel 共用同一 Vulkan 调度契约；厂商推理后端继续独立选择。

非目标：

- 不用两个 Python 线程分别调用 eye0/eye1 Filament 渲染。
- 不创建两套 Filament Engine。
- 不把 `xrWaitFrame`、初始化加载和关闭阶段的空闲等待误判为热路径问题。
- 不在缺少 A/B 数据时直接启用 multiview 或新的采样算法。

## 2. 根因与正确提交模型

### 2.0 当前架构修订

Filament worker 只渲染环境/手柄 GLB；Vulkan Projection Composer 负责虚拟屏幕、激光、Glow 和 Projection Layer 合成；FPS、指南、屏幕/键盘固定 2D 光圈和虚拟键盘使用 Quad Layer。Presenter 线程独占 OpenXR layer 生命周期，屏幕与 UI worker 通过 latest-frame/timeline 交付，不直接调用 OpenXR 或 Filament ABI。

旧串行路径每只眼 `endFrame()` 后分别调用 `engine->flushAndWait()`：

```text
xrWaitFrame
  -> acquire/wait eye0
  -> acquire/wait eye1
  -> render eye0
  -> flushAndWait
  -> render eye1
  -> flushAndWait
  -> release eye0/eye1
  -> xrEndFrame
```

这会让左眼 GPU 完全结束后才开始右眼。主要损耗不是 Python 循环，而是两次整 GPU 完成等待。原始证据位于 [`native/filament/bridge/bridge_eye.cpp`](../native/filament/bridge/bridge_eye.cpp)。

目标路径：

```text
xrWaitFrame
  -> acquire eye0 + eye1
  -> wait eye0 + eye1
  -> 更新一次公共场景状态
  -> enqueue eye0
  -> enqueue eye1
  -> frame-wide flushAndWait/completion
  -> source release barrier
  -> release eye0 + eye1
  -> xrEndFrame
```

同一个 Filament Engine 和 graphics queue 下，两个 Python 线程不能让 GPU 队列并行；反而会破坏 Filament 线程归属和 Vulkan `VkQueue` 外部同步。正确优化是在 Presenter 所在线程连续生成两眼命令，由原生 Bridge 在帧边界统一提交和同步。

## 3. 当前实施状态

| 优先级 | 优化项 | 当前状态 | 后续动作 |
|---|---|---|---|
| P0 | 双眼先 acquire、再分别 wait | 已实现 | 保留实机计时 |
| P0 | 控制器状态、共享灯光和 GLB 动画每帧只更新一次 | 已实现；右眼 Camera 更新不再改共享灯光 | 视觉回归确认双眼一致 |
| P0 | 双眼一次提交 + frame-wide completion | Vulkan Projection Composer 目标为 `array_size=2`；Filament 环境/手柄可先逐眼渲染 | 远程原生构建与长时间实机验收 |
| P0 | CUDA/Vulkan ready/release 跨 API timeline 同步 | 已实现；Projection Composer visible 信号与 source-ready/release timeline 分离 | 连续实机验收无卡死、无 device lost |
| P0 | Vulkan Compute RGB/Depth 输入环形槽 | 已实现 | CUDA 路径确认无 host wait；host fallback 只允许槽位复用时等待 |
| P0 | Vulkan Projection/Glow 共享 graphics queue 外部同步 | 规划中；屏幕/Glow 不再通过 Filament ABI 合成 | 实机确认无 device lost |
| P0 | graphics/compute/transfer 分离提交锁和帧索引 | 未实现 | 先确认实际 queue handle 拓扑 |
| P1 | Compute 移出 Presenter，交给 Output Worker | 未实现 | 必须在队列锁与资源所有权改造后进行 |
| P1 | 左右眼 transition 合并为一个命令缓冲和一次提交 | 未实现 | 与 Output Worker 一起实施 |
| P1 | Glow 合并到立体 Compute 后处理命令链 | 未实现 | 先测量独立 Glow 提交成本 |
| P2 | `array_size=2` projection swapchain | 规划中；创建或合成失败时回退双 swapchain 逐眼 | Runtime 能力、画质与稳定性验收 |
| P2 | layered MIP/SPD 一次处理双眼 | 未实现 | 先完成视觉回归和 GPU timestamp 基线 |
| P3 | Quad/MSDF 双缓冲或三缓冲异步更新 | 未实现 | 仅在 timeline wait 被测为瓶颈后实施 |

当前相关入口：

- OpenXR 帧循环与双眼提交：[`src/desktop2stereo/xr_viewer/core_openxr_vulkan.py`](../src/desktop2stereo/xr_viewer/core_openxr_vulkan.py)
- Filament 原生帧边界：[`native/filament/bridge/bridge_eye.cpp`](../native/filament/bridge/bridge_eye.cpp)
- Vulkan 提交和 FrameContext：[`src/desktop2stereo/viewer/vulkan_context.py`](../src/desktop2stereo/viewer/vulkan_context.py)
- Vulkan Compute 输入环：[`src/desktop2stereo/stereo_runtime/vulkan_backend.py`](../src/desktop2stereo/stereo_runtime/vulkan_backend.py)
- Runtime 输出调度：[`src/desktop2stereo/app_runtime/runtime_output.py`](../src/desktop2stereo/app_runtime/runtime_output.py)

## 4. 分阶段实施

### 阶段 0：稳定当前高吞吐路径

先以逐眼完成路径建立稳定基线，不同时引入其它架构变化。

要求：

1. 每 15 秒输出一次 `FPSBreakdown`，共 5 次，用于固定性能窗口。
2. 连续运行至少 10 分钟，用于稳定性验收。
3. 日志应满足 `eye0_finish_wait > 0`、`eye1_finish_wait > 0`、`eye0_deferred=0`、
   `eye1_deferred=0`，并持续消费每眼 Filament 完成信号。
4. 新旧 Bridge 均使用逐眼完成路径，直到 array swapchain + multiview 通过长时间实机验收。
5. 不得出现 `VK_ERROR_DEVICE_LOST`、access violation、30 Hz 锁定或 XR 输入帧率随 SBS 降低。

Virtual Desktop OpenXR 实机已经证明双 SwapChain deferred batch 不满足 Filament 帧
生命周期：两个 Renderer 同时 in-flight 会触发 `VK_ERROR_DEVICE_LOST`；改成共享一个
Renderer 后，连续对两个 SwapChain 执行 deferred frame 又会在固定运行时间后于
`finish_frame_batch()` 内发生原生 access violation。因此 capability ABI 不能作为默认启用
条件，Presenter 当前固定使用逐眼 `endFrame()` + `flushAndWait()`，完成一眼后才切换
SwapChain。真正的一次双眼提交必须建立在单个 array swapchain / multiview frame 上。

共享 Renderer 后仍需修复 Filament 1.76 Vulkan backend 的多 SwapChain 状态：
`VulkanDriver` 只保存一个 `mDefaultRenderTarget`，且 `isSwapchainBound()` 不区分左右眼
SwapChain，导致切换到 eye1 时仍绑定 eye0 的默认 RenderTarget。D2S backend patch 让
`VulkanRenderTarget` 记录当前绑定的 SwapChain/image，`acquireNextSwapchainImage()` 在
目标 image 变化时先 `releaseSwapchain()` 再 `bindSwapChain()`，从而消除双 SwapChain
的渲染目标污染和 device-lost。

Filament 的 `VulkanCommands`/semaphore pool 仍是全局共享的，但自定义 OpenXR
`Platform::present()` 不会像 `vkQueuePresentKHR` 一样自动消费 `finished_drawing` binary
semaphore。`flushAndWait()` 只保证 GPU 完成，不会把 binary semaphore 恢复为未信号状态；
若跳过消费，semaphore pool 复用该信号时会形成非法重复 signal 并最终触发 device-lost。
因此当前逐眼路径在两眼完成后读取各 ExternalSwapChain 记录的完成信号，用一次 Vulkan
completion drain 消费并转换为 timeline，再据此释放 zero-copy 源图；未来 batch 方案也必须
满足同一信号消费契约。

外部屏幕纹理原先共用同一个 `screen_material_instance`。第一次修复仅拆分 material
instance，但最终屏幕 Renderable、MIP copy Renderable 和 copy View 仍被左右眼复用；
eye0 deferred 后切到 eye1 仍可能在前一眼 GPU 工作未完成时改写共享对象。Bridge 现为
最终屏幕和 MIP copy 分别保存每眼独立 material instance、Renderable entity 和 copy
View，并用 layer mask 固定每个 View 只看到对应眼资源，不再调用
`setMaterialInstanceAt()` 切换共享 Renderable。新版 ABI 探针只在完整隔离 Bridge 上存在，
旧二进制自动逐眼 `flushAndWait()`，避免兼容路径重新暴露 device-lost。

屏幕资源隔离后仍出现的 descriptor validation 来自另一个隐藏共享写入：逐眼
`set_camera_look_at()` 除了更新各自 Camera，还会移动同一组控制器灯光。eye1 的相机更新
发生在 eye0 `end_frame_deferred()` 之后，因此会改写已入队帧使用的共享场景状态。Bridge
现只在 eye0 更新这组共享灯光；eye1 仍更新自己的 Camera，但不再触碰共享灯光，也不靠
关闭 external semaphore 或恢复逐眼等待规避问题。

CUDA 与 Vulkan 之间不再交替复用 binary ready/release semaphore。每个输出 slot/eye
改用独立 exportable timeline semaphore，并以单调递增 generation 表示 producer-ready
和 consumer-release；CUDA wait、图像拷贝及下一次 ready signal 保持在同一 stream 顺序中，
不增加 CPU `cudaStreamSynchronize()`。Vulkan barrier 后交给 Filament 的 visible semaphore
仍为 Vulkan-only binary，且每个新源帧只 signal/wait 一次。

### 阶段 1：VulkanContext 真正分队列

当前 `VulkanContext` 仍使用全局锁和共享 `_frame_index`，会串行化 graphics、compute、transfer 的主机提交。

改造要求：

- 每个实际 `VkQueue` handle 拥有独立锁、FrameContext 索引、命令池和 fence ring。
- 如果 graphics/compute/transfer 映射到同一个 `VkQueue`，这些角色必须共享同一把锁；不能只按角色名拆锁。
- timeline value 仍由设备级单调计数器分配，分配过程保持线程安全。
- 资源状态表的布局与 queue-family ownership 更新必须与提交顺序一致。
- 优先使用独立 compute queue；只有 queue family 不同才执行 ownership transfer。

验收：graphics 与 compute 主机提交可重叠，且 Vulkan validation layer 无 queue 外部同步、布局或 ownership 错误。

### 阶段 2：Compute 移到 Output Worker

目标线程结构：

```text
Capture
  -> Inference / stereo scheduling
  -> Output Worker: queue1 Compute + Glow + external-image prepare
  -> ready semaphore/timeline + latest completed frame
  -> OpenXR Presenter: queue0 Vulkan Projection Composer + swapchain + xrEndFrame
```

约束：

- Presenter 只消费已经提交并带 ready semaphore/timeline 的最新帧。
- 队列积压仍执行 latest-frame overwrite，不建立无界队列。
- ring slot 未完成时只能覆盖其它可用槽；槽位用尽才允许等待或丢弃新帧。
- Presenter 未及时消费时回到单帧背压，不允许覆盖正在被 Filament 采样的图像。
- Output Worker 不调用 Filament、OpenXR session 或 swapchain API。

### 阶段 3：合并 GPU 提交

按测量结果依次处理：

1. 在一个命令缓冲中同时 transition 左右眼外部图像。
2. 立体 Compute、遮挡、补洞、输出打包和 Glow 通过 timeline 串成一个命令链。
3. 左右眼 MIP/质量过滤改为 layered compute 或 SPD，一次处理两层。
4. Quad/MSDF 只批量更新变化资源，并使用双缓冲或三缓冲消除正常帧 CPU wait。

每次只启用一项，通过相同输入、相同头部姿态和相同配置做 A/B；任何颜色、左右眼、锐度或 UI 合成差异都视为失败。

### 阶段 4：Array Swapchain 与 Multiview

目标实现优先创建一个 `array_size=2` 的 Projection SwapChain，layer 0/1 分别对应左右眼。
Vulkan Projection Composer 使用 layered draw/dispatch 合成两层；Filament 环境/手柄输出可先逐眼，稳定后再启用 multiview；
OpenXR 仍提交两个 Projection View，但两者共享同一个 SwapChain handle，并分别引用 array
layer 0/1。左右眼源图 ready semaphore 在进入 Filament 前由同一 graphics queue submit
消费，Filament 完成后只读取并消费一个 render-finished semaphore。

以下情况不启用 array Projection，并保留原双 SwapChain 逐眼稳定路径：

- Runtime 不支持目标 Vulkan/OpenXR 特性。
- 两眼推荐尺寸或采样要求不同。
- Filament Bridge ABI 或 GPU 不支持所需的颜色/深度输出。

Quad Layer 的 `array_size=2` 单独实验，不作为 Projection Layer 成功的前置条件；Runtime 创建或提交失败时回退为 per-eye Quad swapchain。
- layered OpenXR 或 Filament 目标创建失败。

初始化失败只销毁临时 layered SwapChain，不销毁原左右眼 SwapChain；运行路径不调用
`end_frame_deferred()` 或 `finish_frame_batch()`。远程原生构建、视觉回归、性能和至少
10 分钟稳定性验收未通过前，不删除双 SwapChain 回退。

Multiview 不会减少两眼的全部像素着色量，不得按“两倍帧率”估算收益。

## 5. 诊断与验收指标

### CPU 时间

- `xr_wait`、`xr_begin`、`xr_end`
- `xr_acquire_pair`、`xr_wait_l`、`xr_wait_r`、`xr_release_pair`
- `xr_shared`
- `eye0_queue`、`eye1_queue`
- `eye0_finish_wait`、`eye1_finish_wait`
- `eye0_deferred`、`eye1_deferred`（当前必须为 0）
- `filament_drain`
- Compute slot wait、输入上传、输出转换和 Presenter command drain

### GPU 时间

使用 Vulkan timestamp query 分别记录：

- 左眼 Filament、右眼 Filament
- 立体 Compute 总时间及 warp/occlusion/fill 子阶段
- 左右眼 transition
- Glow
- MIP/质量过滤
- Projection 总 GPU 时间

CPU 提交耗时不能替代 GPU timestamp；`xrWaitFrame` 也不能计入应用 GPU 工作时间。

### 验收矩阵

| 类别 | 验收条件 |
|---|---|
| 性能 | 相同配置下比较 5 个固定窗口的中位数；目标恢复已观察到的 50-55 SBS FPS 区间 |
| 稳定性 | 至少连续运行 10 分钟，无卡死、device lost、access violation 或持续降至 30 Hz |
| 延迟 | `rt_pending_age`、source latency 和 Presenter queue age 不持续增长 |
| 同步 | 跨 API ready/release timeline generation 单调递增；每个新源帧的 Projection visible signal 只 signal/wait 一次 |
| 画质 | 固定输入视觉回归通过；左右眼顺序、颜色空间、屏幕清晰度和 Glow 不变 |
| 兼容 | 新 Bridge 在能力满足时走 array multiview，旧 Bridge 或初始化失败走逐眼回退；三平台远程构建通过 |

## 6. 风险与回退

- **Filament 线程归属**：所有 Filament API 继续由 Presenter 线程调用。
- **VkQueue 外部同步**：同一实际 queue handle 的主机访问必须串行，即使调用方使用不同角色或线程。
- **Timeline generation**：每个 slot/eye 的 ready/release 值必须单调递增，wait 必须使用对应帧发布的值。
- **Projection visible signal**：只允许每个新源帧 signal/wait 一次，静态复用帧不得重复等待旧信号。
- **源图生命周期**：显示帧、渲染帧和 pending 帧只能由现有幂等 lease/release 契约回收。
- **设备丢失**：锁存 device-lost 后只释放 CPU 租约，不再提交 Vulkan 命令或查询 fence。
- **运行时兼容**：任何新路径都必须有启动期 capability probe 和明确日志，不能静默切换。

每个阶段使用独立开关或能力探针完成 A/B；验收失败即回退上一阶段，不用多个未验证优化共同替换稳定路径。

## 7. 参考

- [OpenXR 1.1 Specification](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html)
- [XR_KHR_vulkan_enable concurrency](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XR_KHR_vulkan_enable-concurrency.html)
- [`docs/01-Realtime-2d-to-3d-specification.md`](01-Realtime-2d-to-3d-specification.md)
- [`docs/02-desktop2stereo-engineering-design-specification.md`](02-desktop2stereo-engineering-design-specification.md)
