# D2S 架构转向 Vulkan 技术实现报告

## 1. 引言

### 1.1 背景

D2S（Desktop to Stereo）项目是一个基于 OpenXR 的虚拟桌面应用，旨在将用户桌面画面实时转换为立体 SBS（Side-by-Side）画面，并在 VR 环境中显示。项目现有架构采用 **Panda3D (OpenGL) + CUDA 互操作 → D3D11 交换链** 的混合 API 方案，运行于 Windows + Virtual Desktop 环境。

### 1.2 迁移动因

现有架构存在以下核心问题，促使我们转向 Vulkan：

| 问题 | 现状 | 影响 |
|------|------|------|
| **多 API 碎片化** | OpenGL 渲染 + CUDA 计算 + D3D11 提交 | 维护三套互操作代码，复杂易错 |
| **跨厂商计算受限** | CUDA 仅支持 NVIDIA | AMD/Intel GPU 无法使用光效异步计算 |
| **跨平台不可行** | macOS 无 CUDA，OpenGL 4.1 已废弃 | 无法覆盖 Apple Silicon 用户 |
| **互操作不稳定** | WGL_NV_DX_interop2 为 NVIDIA 私有扩展 | 长期依赖单一厂商 |

**核心目标**：统一整个图形与计算管线到 **Vulkan**，实现 Windows/Linux/macOS 全平台兼容，同时保持现有异步架构的性能优势。

**语言原则**：项目继续采用 Python 作为应用、捕获、推理、Vulkan 编排、OpenXR 和输出层的实现语言。现有 WindowsCaptureCUDA、WindowsCaptureROCm、TensorRT/ROCm Provider 与调度优化不做 C++ 重写，并继续直接使用当前 Python 包/API，不新增项目自有绑定。Filament 因缺少满足项目要求的 Python接口，使用项目唯一的自有原生 DLL Bridge。

---

## 2. 目标架构总览

### 2.0 当前架构修订

Filament 仅负责环境与手柄 GLB 的加载、材质/动画和每眼颜色/深度渲染。Vulkan 统一负责虚拟屏幕采样、激光、Glow 以及 Projection Layer 最终合成；FPS 面板、操作指南、固定在虚拟屏幕和虚拟键盘上的两处 2D 光圈、虚拟键盘使用 Quad Layer。OpenXR Session、swapchain acquire/release、Projection/Quad layer 构建和 `xrEndFrame` 只能由 Presenter 线程执行；屏幕与 UI 生产线程通过有界 latest-frame 资源和 timeline 同步交付。

### 2.1 全 Vulkan 流程图

```
┌────────────────────────────────────────────────────────────────┐
│                       推理层 (厂商专用)                          │
│                                                                 │
│  Windows NVIDIA: CUDA + TensorRT                                │
│  Windows AMD:    ROCm + MIOpen (DirectML 备用)                  │
│  Linux AMD:      ROCm + MIOpen                                  │
│  macOS:          MPSGraph / CoreML                              │
│                                                                 │
│  输出：SBS 立体画面 (厂商私有纹理)                               │
└───────────────────────┬────────────────────────────────────────┘
                        │ external memory / 单次 GPU 拷贝
                        ▼
┌────────────────────────────────────────────────────────────────┐
│                    Vulkan 统一渲染与计算层                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Filament (Vulkan 后端)                                   │   │
│  │  - 加载房间.glb / 手柄.glb                                │   │
│  │  - 渲染 3D 场景 + 虚拟屏幕四边形                          │   │
│  │  - 直接输出到 OpenXR Vulkan 交换链                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  异步计算队列 (Vulkan 原生)                               │   │
│  │  - 降采样 + 模糊 → Glow 纹理                             │   │
│  │  - 取平均色 + 预计算掩码 → 墙面反射光斑纹理              │   │
│  │  - 消费旧帧，零阻塞                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  OpenXR 提交                                              │   │
│  │  - Projection Layer: 3D 场景 + 虚拟屏幕                  │   │
│  │  - Quad Layer: Glow 特效 / 文字面板 / 虚拟键盘          │   │
│  │  - xrEndFrame 直接提交 VkImage                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 与旧架构对比

| 维度 | 旧架构 (OpenGL + D3D11 桥接) | 新架构 (纯 Vulkan) |
|------|------------------------------|---------------------|
| **渲染后端** | Panda3D (OpenGL) | Filament (Vulkan) |
| **OpenXR Session** | D3D11 | Vulkan |
| **纹理共享** | WGL_NV_DX_interop2 (NVIDIA 私有) | 无需桥接，直接渲染到交换链 |
| **异步计算** | CUDA 流 (仅 NVIDIA) | Vulkan 计算队列 (全平台) |
| **跨厂商** | NVIDIA 专属 | NVIDIA / AMD / Intel / Apple Silicon |
| **跨平台** | Windows only | Windows / Linux / macOS |
| **代码复杂度** | 三套 API 互操作 | 单一 API，统一维护 |

---

## 3. 核心组件选型

### 3.1 渲染引擎：Filament

**选择理由**：
- 原生支持 Vulkan 后端，性能优秀。
- **glTF 2.0 支持度业界最高**，包括 PBR 材质、骨骼动画、KHR 扩展。
- Google 持续维护，社区活跃。
- 支持离屏渲染到外部提供的 `VkImage`。

**集成方式**：
- 使用同进程加载的 Filament DLL Bridge，这是项目唯一允许维护的 C/C++ 原生桥接层。
- Bridge 提供稳定、窄化的 C ABI；Python 使用 `ctypes`/`cffi` 调用，不引入独立 C++ Runtime。
- 提供 `load_glb(path)`、`update_scene(state)`、`bind_vulkan_target(image)` 和 `render(eye)` 等高层 API。
- Bridge 负责 Filament 对象生命周期和 3D 绘制，不负责 Capture、AI 推理、OpenXR 帧循环或产品状态机。
- 禁止使用 `subprocess`、共享内存或 CPU 图像传输连接 Filament。

### 3.2 推理层：保留厂商最优方案

**结论**：AI 推理不迁移到 Vulkan 计算着色器，保留厂商专用库。**AMD 平台优先使用 ROCm 方案，DirectML 作为备用**。

| 厂商 | 推理后端 | 优先级 | 性能基准 | 与 Vulkan 连接方式 |
|------|----------|--------|----------|-------------------|
| NVIDIA | CUDA + TensorRT | 首选 | 最优 (100%) | `cudaImportExternalMemory` 零拷贝 |
| AMD (Windows) | ROCm + MIOpen | **首选** | 良好 (80-95%) | `hipMemcpy` GPU 拷贝 或 `hipImportExternalMemory` |
| AMD (Windows) | DirectML | **备用** | 良好 (70-90%) | D3D12 纹理 → NT Handle → Vulkan |
| AMD (Linux) | ROCm + MIOpen | 首选 | 良好 (80-95%) | `hipImportExternalMemory` (FD) 或 `hipMemcpy` |
| Apple Silicon | MPSGraph / CoreML | 首选 | 最优 (含 ANE) | Metal 纹理 → Vulkan (MoltenVK 内部处理) |

#### 3.2.1 选择 ROCm 优先于 DirectML 的理由

| 考量维度 | ROCm + MIOpen | DirectML |
|----------|---------------|----------|
| **性能** | 原生 AMD 汇编级优化，接近 CUDA 水平 | 通用抽象层，存在额外开销 |
| **跨平台一致性** | Linux 主力支持，与 Linux AMD 路径统一 | Windows 专属，Linux 不可用 |
| **生态整合** | 与 PyTorch/TensorFlow 深度集成 | 需 ONNX Runtime 或单独适配 |
| **控制粒度** | 完全的 GPU 控制权，可精细调优 | 受限于 DirectML 抽象层 |
| **未来潜力** | AMD 持续投入，ROCm 6.x 大幅改进 Windows 支持 | 微软维护，但重心在 DirectX 生态 |

**DirectML 作为备用的场景**：
- ROCm 在特定 Windows 版本/驱动上出现兼容性问题。
- 用户显卡为 AMD Radeon 入门级（ROCm 主要针对高端 Radeon/Instinct 优化）。
- 需要快速部署且无法解决 ROCm 环境配置问题。

### 3.3 后处理：Vulkan 原生计算着色器

**所有光效统一使用 Vulkan 计算队列**：

| 计算任务 | 实现方式 | 复杂度 |
|----------|----------|--------|
| 降采样 | 计算着色器 separable downscale | 低 |
| 高斯模糊 | 计算着色器 separable blur | 低 |
| 光斑生成 | 取平均色 + 预计算掩码乘法 | 低 |
| 色彩校正 | 计算着色器 LUT 或矩阵 | 低 |

**优势**：
- 一套 GLSL/HLSL 着色器代码，全平台运行。
- 与图形队列共享 VkImage，零拷贝、零互操作。
- 延续"旧帧消费"异步模式，计算队列使用上一帧渲染结果，主队列永不等待。

---

## 4. 跨平台互操作具体方案

### 4.1 NVIDIA (Windows/Linux)

**CUDA → Vulkan 零拷贝路径**：

```
CUDA Tensor (AI 输出)
    ↓ cudaImportExternalMemory (导入 Vulkan 导出的 NT Handle / FD)
VkImage (直接在 CUDA 中可写)
    ↓ Filament 渲染时直接采样该 VkImage
OpenXR 交换链
```

- 使用 `VK_KHR_external_memory_win32` (Windows) 或 `VK_KHR_external_memory_fd` (Linux)。
- 推理输出与渲染输入的同一块 GPU 内存，零拷贝。

该路径不是“把裸句柄交给 Filament”就完成了。每个源 `VkImage` 必须在创建时建立一次 Filament 外部纹理，并保存格式、尺寸、当前 layout、producer/consumer queue family 及槽位 lease。完整闭环为：producer signal ready external semaphore -> Presenter graphics submit wait ready 并执行 `GENERAL -> VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` barrier -> 同一 submit signal device-local visible semaphore -> Bridge 将 visible semaphore 作为 Filament acquire wait -> Filament 采样完成 -> Presenter 执行 `SHADER_READ_ONLY_OPTIMAL -> GENERAL` release barrier 并 signal exportable consumer-release semaphore -> CUDA/HIP producer 在复用该 slot 前以 stream wait 消费 release semaphore。它与 OpenXR 输出 swapchain 的 acquire、render-finished 和 release 同步完全独立。

如果上述任一条件无法由当前 Vulkan Runtime、CUDA interop 或 Filament Bridge ABI 可靠表达，运行时必须选择一次 Vulkan GPU copy 的屏幕路径或 Quad Layer 回退，并记录能力缺失原因；不得把未同步的外部图像提交给 Filament，也不得退回 CPU 像素传输。

### 4.2 AMD (Windows) — ROCm 首选路径

```
ROCm HIP Tensor (AI 输出)
    ↓ hipMemcpy (GPU 内部拷贝)
Vulkan staging buffer → VkImage
    ↓ Filament 采样
OpenXR 交换链
```

- `hipImportExternalMemory` 理论上可实现零拷贝，但 Windows 上成熟度有限。
- **首选使用 `hipMemcpy` GPU 拷贝**：对于 4K SBS 纹理（约 32MB），拷贝耗时 < 0.1ms，在 90fps 帧预算中可忽略。
- 与 NVIDIA 路径的差异仅在于拷贝这一步，其余 Vulkan 渲染代码完全复用。

**备选 DirectML 路径**（当 ROCm 不可用时）：

```
DirectML 推理输出 → ID3D12Resource
    ↓ 导出 NT Handle
Vulkan 通过 VK_KHR_external_memory_win32 导入
    ↓
VkImage → Filament 采样
```

### 4.3 AMD (Linux) — ROCm 路径

```
ROCm HIP Tensor (AI 输出)
    ↓ hipImportExternalMemory (导入 Vulkan 导出的 FD)
VkImage (零拷贝或接近零拷贝)
    ↓ Filament 采样
OpenXR 交换链
```

- Linux 上 ROCm 的 Vulkan interop 支持优于 Windows。
- 优先尝试 `hipImportExternalMemory` 零拷贝；若不稳定则回退到 `hipMemcpy` GPU 拷贝。

### 4.4 macOS (Apple Silicon)

**MPSGraph → MoltenVK 路径**：

```
MPSGraph 推理 → MTLTexture
    ↓ (MoltenVK 内部翻译)
VkImage (MoltenVK 管理的 Metal 纹理)
    ↓ Filament 直接采样
```

- MoltenVK 负责 Metal ↔ Vulkan 翻译，应用层无感知。
- 推理仍使用 MPSGraph 以获得 ANE 加速和最佳性能。
- 无法做到完全零拷贝（Metal 和 Vulkan 内存池隔离），但 MoltenVK 优化了此路径。

---

## 5. 异步架构保留设计

转向 Vulkan 后，异步架构不仅保留，而且得到原生支持：

### 5.0 当前 Vulkan 纹理管理落地

Projection Layer 的运行时屏幕输出现已采用左右眼独立的三帧 Vulkan image ring。每个槽位的 external memory、GPU producer mapped array 和 Filament imported Texture 均可复用，帧循环只切换槽位和材质绑定，不再因为新帧销毁并重新导入纹理。输出槽位现在带有跨线程 producer lease：生产者领取空闲槽位，pending 被替换时立即释放，当前 Filament 屏幕/Quad 消费帧保持占用直到下一帧完成提交；因此 ring wrap 不会覆盖仍被 Filament 采样的 VkImage。Bridge 同时将左右眼提交合并到一次帧边界等待，避免旧实现每只眼睛一次 `flushAndWait`。

输出线程边界已进一步收紧：`VulkanRuntimeOutputConsumer` 只从推理队列取出最新原始结果并投递 `submit_runtime_result` 命令，不再创建、导入或释放 Vulkan 输出图像。CUDA/ROCm/HIP 到 Vulkan 的 image ring、external semaphore、屏幕光采样和输出槽位 lease 全部由 Presenter 线程执行；非 Vulkan sink 仍保留原兼容转换路径。这样后台线程只负责捕捉、文件读取和推理，Filament/Vulkan 资源的创建、使用和销毁集中在同一线程。

Presenter 每个 OpenXR tick 只消费最新一条原始输出命令。禁止在一次帧边界内连续执行多次 GPU producer/Vulkan 导入，否则可能在旧帧尚未完成提交和 lease 释放前耗尽有界 image ring，并使 Presenter 线程阻塞等待自身下一帧。

当前已增加 CUDA/ROCm/HIP/Vulkan/Filament external semaphore signal/wait ABI：支持路径不再在发布前执行 producer stream synchronization；source barrier submit 等待 producer-ready 并发出 Filament visible semaphore，Filament 完成采样后再发出 consumer-release，producer 复用 slot 前等待 release。CUDA external semaphore 现在默认开启，`D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE=0` 可用于回归测试和故障隔离；ROCm/HIP 则在后端自动识别后按 HIP runtime 能力启用，`D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE=0` 仅用于调试禁用。Filament v1.76.0 源码补丁 `native/filament/patches/apply_d2s_vulkan_external_image.py` 现已扩展 `VulkanPlatform` 的借用式外部 `VkImage` 接口，并由 GitHub Actions 远程构建三平台 Bridge；只有新 ABI 能力探针通过后才进入 Filament 直采样，旧 Bridge/stock SDK 继续安全回退 Vulkan GPU copy。layout、queue ownership、producer-ready 和 consumer-release 不由 Filament 猜测，仍由 Presenter 的 Vulkan 同步层明确管理。

控制器路径已按旧工程生命周期迁移到 Vulkan：Python Presenter 维护逐手 Grip/Aim 跟踪、移动时间、One Euro 位置滤波和四元数方向平滑；Filament Bridge 在共享 Projection Layer Scene 中加载左右手 GLB、更新 profile 校准姿态和按键动画，并通过独立 C ABI 控制模型与 3D 激光实体显隐。按键动画按 GLB `_value/_min/_max` 层级匹配，使用输入平滑、平移/缩放插值和四元数 SLERP；加载日志列出逐手动画节点与语义。静止 5 秒或 Grip 跟踪丢失时隐藏对应手柄和激光，重新移动后恢复；激光不再通过 Quad Layer 模拟，并恢复旧工程两张交叉锥形面及动态蓝至红彩虹渐变。

原生 Bridge 已从单一实现文件拆分为 Context、Eye、Scene、Controller、Laser、Screen、Material 和 Preview 模块。`filament_bridge.cpp` 只保留与 `filament_bridge.h` 一一对应的 C ABI 转发，因此 Python wrapper 无需变化；共享 Engine/Scene 和资源 ownership 仍集中在 `bridge_context`，模块拆分不复制 GLB、材质、纹理或 Shader。CMake 默认隐藏内部 C++ 符号，防止后续功能把内部实现扩展成不稳定 ABI。三平台二进制分别进入 `src/desktop2stereo/xr_viewer/native/windows`、`linux`、`macos`，不再平铺在 `native` 根目录。

### 5.0.1 Vulkan 1.4 Binding 与传输基准（未来目标）

后续建立独立 `d2s-vulkan-1.4` 实验分支，使用固定 Khronos Vulkan 1.4 Registry 输入在 GitHub Actions 远程生成 Python binding wheel，不改变当前生产环境的 `vulkan==1.3.275.1` 锁定。实验路径查询并显式启用 Vulkan 1.4 feature/property 链，同时保留 Vulkan 1.2/1.3 回退。

首轮只验证 `hostImageCopy` 与独立 Transfer Queue 对 FPS、键盘、帮助面板等 CPU 生成工具纹理的上传收益。CUDA/Vulkan 外部图像主链路不因该实验改回 CPU 上传。合入门槛为三平台 wheel/Bridge CI、Validation Layer、相同场景性能基准和目标实机长稳全部通过；没有可重复净收益时保留现状。

### 5.1 主图形队列（每帧必须完成）

```
主图形队列：
├── 导入 SBS 推理结果纹理 (已就绪)
├── 绘制虚拟屏幕四边形 (采样 SBS 纹理)
├── 绘制房间背景 (采样全景图 + 旧光斑纹理)
├── 绘制手柄模型
├── 绘制 Glow Quad (采样旧 Glow 纹理)
└── 提交 xrEndFrame

帧时间预估：2-4ms
```

### 5.2 异步计算队列（与主队列并行）

```
异步计算队列：
├── 接收当前帧渲染结果 VkImage
├── 降采样 → 模糊 → 生成新 Glow 纹理
├── 取平均色 + 预计算掩码 → 生成新光斑纹理
└── 原子更新 "最新光效纹理" 索引

允许滞后 1-3 帧，主队列永不等待
```

### 5.3 同步原语

- 主队列与计算队列通过 **Vulkan 信号量** 同步（仅在计算完成时通知，主队列不阻塞）。
- 主队列消费旧纹理，只读取 `latestReadyIndex` 原子变量，无等待。

**结果**：帧时间曲线平滑如直线，任何 GPU 波动不影响虚拟屏幕流畅度。

---

## 6. 实施路线图

### 第一阶段：技术验证（2-3 周）
- 编译 Filament，启用 Vulkan 后端。
- 创建最小 Python 封装：加载 `.glb`，渲染到离屏 `VkImage`。
- 验证 OpenXR Vulkan Session 创建成功，交换链纹理可正常提交。
- 测试一个简单计算着色器（降采样），验证异步队列可行。

### 第二阶段：核心迁移（4-6 周）
- 将现有场景逻辑（房间、手柄、虚拟屏幕）从 Panda3D 移植到 Filament。
- 实现 Glow 和墙面反射的计算着色器。
- 实现 **NVIDIA CUDA → Vulkan external memory 零拷贝路径**。
- 实现 **AMD ROCm → Vulkan GPU 拷贝路径**（Windows + Linux）。
- Windows 上联调通过 Virtual Desktop。

### 第三阶段：跨平台扩展（3-4 周）
- 实现 macOS (MPSGraph + MoltenVK) 的推理与渲染路径。
- 各平台性能调优与稳定性测试。
- DirectML 备用路径验证（仅在 ROCm Windows 兼容性问题时激活）。

### 第四阶段：优化与交付（2-3 周）
- 全平台 benchmark 测试。
- 遗留代码清理，移除 OpenGL/D3D11 互操作相关模块。
- 文档与发布。

**总预估**：约 12-16 周完成全面迁移。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Filament Python 绑定不成熟 | 集成困难 | 维护同进程 Filament DLL Bridge 和窄 C ABI，禁止子进程与共享内存整帧传输 |
| ROCm Windows 互操作不稳定 | AMD Windows 用户性能下降 | 回退至 `hipMemcpy` GPU 拷贝（<0.1ms），或启用 DirectML 备用路径 |
| macOS MoltenVK 性能瓶颈 | 帧率不足 | 降低计算着色器负载，依赖 MPSGraph 推理 |
| 现有用户依赖 OpenGL 路径 | 升级阻力 | 保留 OpenGL 路径作为兼容模式，新架构默认启用 |
| ROCm 安装配置复杂 | 用户上手门槛高 | 提供一键安装脚本，自动检测并配置 ROCm 环境 |

---

## 8. 总结

本次架构迁移将 D2S 项目从 **OpenGL + D3D11 + CUDA 多 API 碎片化架构**，统一到 **Vulkan 全平台架构**，核心收益：

1. **消除多 API 互操作层**：不再需要 WGL interop、CUDA-GL interop、D3D11 桥接。
2. **原生跨厂商异步计算**：Glow、光效等后处理用 Vulkan 计算着色器，全平台统一。
3. **真正的跨平台**：Windows / Linux / macOS 使用同一套 Vulkan 渲染代码。
4. **保留推理层最优性能**：AI 推理仍用 CUDA/ROCm/MPSGraph，AMD 平台优先 ROCm，DirectML 备用。
5. **架构更简洁**：单一 API 管理所有 GPU 资源，维护成本大幅降低。

**推荐优先实施此迁移方案，从根本上解决当前架构的扩展性瓶颈。**
