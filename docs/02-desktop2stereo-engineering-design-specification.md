# Desktop2Stereo Python Vulkan 工程设计规范

**文档版本**：4.1
**发布日期**：2026 年 8 月 24 日
**规范状态**：目标态工程设计
**上位规格**：`01-Realtime-2d-to-3d-specification.md`
**技术依据**：`01.D2S_Vulkan_Migration_Technical_Report.md`

---

## 1. 文档定位

本文规定 Desktop2Stereo Python Vulkan 运行时的工程实现方式，包括代码组织、模块接口、线程和队列、GPU资源、推理互操作、Filament DLL Bridge、OpenXR提交、配置、诊断、测试和交付。

`docs/01` 定义系统必须表现出的行为和验收结果；本文定义工程如何实现这些行为。二者冲突时以 `docs/01` 为准，并同步修订本文。

本文不是旧OpenGL/D3D11运行时的兼容模块地图，也不要求为迁移而复制旧架构。经过验证的Python Capture、Inference Provider、latest-frame调度和诊断代码属于可复用实现；图形与OpenXR路径按Vulkan目标重新组织。本文定义的OpenGL Fallback是隔离兼容后端，不等同于归档中的旧viewer、Panda3D、WGL或D3D11桥接路径。

### 1.1 工程目标

1. 应用、实时调度、Capture、Inference、Vulkan、OpenXR和Output统一使用Python源码实现。
2. Vulkan 负责默认主路径的图形、通用计算、资源管理和显示提交；OpenGL 只承担受限 Fallback。
3. OpenXR Session 默认使用 Vulkan Graphics Binding；兼容模式允许使用原生 OpenGL Graphics Binding。
4. AI 推理保留厂商最优后端，通过明确的 GPU 互操作接口接入。
5. 捕获、推理、立体合成、场景渲染和呈现使用固定容量资源池，不产生无界队列。
6. 正常帧循环不进行 CPU 像素回读，不调用全设备空闲等待。
7. 平台差异只存在于 Capture Adapter、Inference Adapter 和系统句柄层。
8. 所有关键 GPU 工作可测量、可验证、可故障定位。
9. 图形后端在启动探测阶段确定，运行中不静默切换；Fallback 必须通过受控重启进入。
10. 核心 XR 场景的项目自有原生代码仅允许存在于 Filament DLL Bridge；网络输出、Windows Desktop Duplication 和 Intel 推理/编码边界可以使用独立窄 ABI 原生组件，但不得拥有 OpenXR Session/交换链或场景资源。

### 1.2 明确排除

- 不把现有Python Capture、Inference Provider和OpenXR行为机械改写为C++。
- 不把 OpenGL 嵌入 Vulkan 资源图；OpenGL Fallback 必须作为独立 Graphics Backend 实现。
- 不恢复 D3D11 OpenXR binding、WGL 跨 API viewer 或旧 PBO uploader。网络编码器允许使用受控的 D3D11/oneVPL 或 CUDA-OpenGL 兼容桥，但必须与 XR 图形会话隔离并输出 copy/zero-copy 诊断。
- 不通过CPU NumPy数组、PIL Image或共享内存传递实时整帧像素；Python中的GPU tensor和GPU handle属于正式数据路径。
- 不保留旧类名、旧环境变量、旧 YAML 字段的长期运行时适配器。
- 不把 Filament 封装为通过 CPU 图像传输工作的独立渲染子进程。
- 不允许推理后端直接管理 OpenXR Session 或场景资源。
- 不为CUDA、ROCm、TensorRT、WindowsCapture或OpenXR新增项目自有Binding；继续直接使用其现有Python包/API。

---

## 2. 技术基线

### 2.1 语言与构建

| 项目 | 规定 |
|------|------|
| 核心语言 | Python；版本按平台锁定文件确定 |
| Python运行方式 | 源码直接执行，正式入口保持为 `python src/desktop2stereo/main.py` |
| 包管理 | 固定版本清单；依赖必须可校验 SHA-256 |
| 图形 API | 默认请求 Vulkan 1.4，按 OpenXR Runtime 协商，最低 Vulkan 1.2；OpenGL 4.3+ Fallback，macOS 为 OpenGL 4.1 受限模式 |
| XR API | OpenXR 1.1；主路径使用 `XR_KHR_vulkan_enable2`，Fallback 使用 Runtime 支持的 OpenGL Graphics Binding |
| 场景引擎 | Filament Vulkan Backend 为主；OpenGL 使用隔离的兼容 Scene Renderer |
| Shader | GLSL/HLSL 离线编译为 SPIR-V |
| 测试 | pytest；GPU/OpenXR集成测试使用独立marker和实机矩阵 |
| 自有原生代码 | Filament Bridge；可选网络/捕获/推理边界桥接；均由独立 CMake 工程预编译 |

Python产品代码必须直接支持Windows x86_64、Linux x86_64和macOS arm64。Filament DLL Bridge以及启用平台所需的可选网络/捕获/推理桥按平台预编译；不得因为 MoltenVK、Bridge 或某个可选桥能够加载就宣称 macOS OpenXR 可用。

### 2.2 依赖版本管理

所有第三方二进制依赖必须在版本清单中记录：版本、下载来源、平台、架构和 SHA-256。至少包括 Vulkan SDK、OpenXR Loader、Filament SDK、TensorRT/CUDA、ROCm/MIOpen、DirectML/ONNX Runtime、MoltenVK 和 CoreML 相关构建要求。

升级 Filament 或 OpenXR Loader 时，必须同时运行三类验证：ABI/链接验证、最小场景渲染验证、真实 OpenXR 交换链验证。只通过资产加载测试不能视为升级完成。

### 2.3 Filament DLL Bridge定位

当前 `native/filament/bridge` 已验证Filament gltfio加载和动画接口，但仍依赖调用方提供OpenGL context，尚未实现目标OpenXR/Vulkan swapchain绑定。

新工程允许重新实现该Bridge，使其支持Filament Vulkan Backend和Python提供的Vulkan/OpenXR目标。Bridge是核心 XR 场景的自有原生边界，只管理Filament对象和渲染调用；Capture、Inference、Vulkan资源图、OpenXR生命周期和产品状态机必须保留在Python。网络/捕获/推理边界的可选原生组件另行管理，不能越过边界拥有 OpenXR 或场景资源。所有桥接使用窄C ABI并由Python通过`ctypes`或`cffi`加载。

Filament Vulkan 外部源图像接口必须以源码扩展方式接入，不得把 release SDK 的通用 `Texture::Builder::import()` 猜测为 Vulkan `VkImage` 接口。当前版本锁定 Filament `v1.76.0`，补丁脚本为 `native/filament/patches/apply_d2s_vulkan_external_image.py`：它在 `VulkanPlatform` 中增加借用式 `VkImage` 外部句柄、格式/尺寸元数据和 `ExternalImage` 工厂，Bridge 通过 `D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE` 编译开关启用。补丁只包装调用方所有的 `VkImage`，不负责销毁图像、内存、OpenXR swapchain 或 producer semaphore；layout、queue family、producer-ready 和 consumer-release 仍由 Python Presenter 的 Vulkan 同步契约负责。

该 Filament 源码和 Bridge 必须由 GitHub Actions 在 Windows、Linux、macOS 三个平台远程构建并安装到临时构建前缀；本机不编译 C++/Filament。三平台二进制只有在 CI 构建完成、ABI 能力探针通过并下载回 `src/desktop2stereo/xr_viewer/native/<platform>/` 后，才允许实机启用外部源图像路径。旧 stock SDK 或旧 Bridge 的能力探针返回 false 时，必须保持 Vulkan GPU copy/Quad Layer 回退。

### 2.4 推荐的 Fallback 策略

```text
主路径：Vulkan
    ├── Windows: 原生 Vulkan
    ├── Linux: 原生 Vulkan
    └── macOS: MoltenVK (Vulkan → Metal)

Fallback 路径：OpenGL
    ├── Windows: 原生 OpenGL
    ├── Linux: 原生 OpenGL
    └── macOS: OpenGL 4.1 (功能受限，但可用于基本渲染)
```

触发 Fallback 的条件：

1. Vulkan 驱动不可用，例如老旧硬件、虚拟机或远程桌面环境。
2. macOS 上 MoltenVK 初始化失败，或能力探测/基准测试确认性能不可接受。
3. 用户显式选择兼容模式。

后端选择发生在正式 Runtime 初始化前。`auto` 模式先探测 Vulkan，失败后返回带原因的 OpenGL 重启建议；启动器根据策略创建新的兼容会话。已经创建 Vulkan Device、Filament Engine 或 OpenXR Session 后不得在原会话中切换 API。

OpenGL Fallback 只保证基本场景、虚拟屏幕和必要的 OpenXR/窗口呈现。Vulkan Compute、异步 Compute Queue、高级 Glow/Reflection、Timeline Semaphore 和 external-memory 零拷贝不属于其必需能力。Fallback 必须消费推理/合成后端已经生成的 GPU Left/Right Eye 或 SBS；无法建立 GPU 路径时明确失败，不启用 CPU 实时像素链路。

---

## 3. 目标代码布局

新工程保持当前项目已经使用并验证过的`src/`顶层目录和模块名称，使日志路径、异常栈、测试定位和开发习惯保持一致。除Filament Bridge源码和预编译库外，所有项目代码均为Python；Vulkan Shader位于`src/`并由Python工具按需编译。新工程不得为了架构形式重新命名已稳定的模块，目标目录如下：

```text
Desktop2Stereo/
  src/
    main.py                      # Keep the current program entrypoint
    main.bat
    settings.yaml
    requirements.txt
    requirements-cuda.txt
    requirements-rocm7.txt
    requirements-mps.txt
    app_runtime/                 # Lifecycle, queues and runtime assembly
    capture/
      backends/
      dxgi/
    gui/                         # Flet GUI
    stereo_runtime/
      providers/
        nvidia/
        amd/
        apple/
        intel/
        cpu/
      model_impl/
      vulkan_graph.py            # Python Vulkan stereo/compute graph
      vulkan_resources.py
    viewer/
      viewer.py
      viewer_runtime.py
      vulkan_renderer.py         # Default desktop renderer
      opengl_renderer.py         # Isolated Fallback
    xr_viewer/
      openxr_runtime.py
      openxr_frame_pipeline.py
      core_openxr_vulkan.py      # Python OpenXR Vulkan session
      core_openxr_opengl.py      # Python OpenXR OpenGL Fallback
      controllers/
      environments/
      gltf/
      native/
        filament_bridge/         # The only project-owned native source
        windows-x64/
        linux-x64/
        macos-arm64/
    streaming/
    tools/
      model_tooling/
      benchmarks/
      probe.py
    utils/
    shaders/
      preprocess/
      depth/
      stereo/
      effects/
      output/
  tests/
    test_*.py
    fixtures/
```

`src/`是产品发布边界。发行流程整体复制或打包`src/`；Filament对应平台的预编译库必须已放入`src/desktop2stereo/xr_viewer/native/<platform>/`，不得在用户启动时编译。产品启动方式继续保持当前习惯，只要求建立Python环境并执行`src/desktop2stereo/main.py`或`src/desktop2stereo/main.bat`，不要求CMake或C++编译器。

新项目不建立`migration_reference`运行目录。可以整文件复用的实现直接复制到与当前项目相同的模块路径；尚未确认可用的代码留在旧项目或文档归档，不进入新项目`src/`。目录一致用于降低排查成本，不代表复制旧兼容分支和废弃实现。

### 3.1 Python模块与唯一原生目标

| 模块/目标 | 类型 | 职责 |
|----------|------|------|
| `app_runtime` | Python package | 状态机、配置快照、队列、生命周期和运行时装配 |
| `capture` | Python package | 复用WindowsCaptureCUDA/ROCm及跨平台Capture实现 |
| `stereo_runtime` | Python package | 推理Provider、模型、深度、立体合成和Vulkan Compute编排 |
| `viewer.vulkan_renderer` | Python module | 默认Vulkan窗口输出、资源和同步 |
| `viewer.opengl_renderer` | Python module | 隔离的OpenGL Fallback |
| `xr_viewer.core_openxr_vulkan` | Python module | OpenXR Vulkan生命周期、交换链和帧提交 |
| `xr_viewer.core_openxr_opengl` | Python module | OpenXR OpenGL Fallback |
| `xr_viewer.gltf` | Python package | 场景contract、资产状态和虚拟屏幕 |
| `streaming` | Python package | 编码和网络输出 |
| `gui` | Python package | Flet控制界面 |
| `tools` / `utils` | Python package | 模型工具、探测、benchmark和通用辅助 |
| `filament_bridge` | shared library | 唯一CMake目标；Filament Vulkan/OpenGL调用 |

厂商后端通过Python环境锁定文件和运行时capability probe控制。OpenGL Fallback由配置控制，推理Provider不得隐式改变图形后端选择。Filament 及可选原生边界桥均使用 CMake feature option，且不能改变 OpenXR 图形后端选择。

### 3.1.1 Vulkan 1.4 Python Binding 未来目标

生产主路径继续使用已锁定的 `vulkan==1.3.275.1` binding，并按 Loader、OpenXR Runtime 和 Physical Device 能力在 Vulkan 1.2 至 1.4 之间协商。未来在独立 `d2s-vulkan-1.4` 分支中，基于固定版本的 Khronos Vulkan 1.4 `vk.xml` 与 Headers 远程生成项目自用 wheel；不得直接依赖浮动 Git master，也不得要求用户本地生成或编译绑定。

该实验分支首先补齐 `VkPhysicalDeviceVulkan14Features`/`VkPhysicalDeviceVulkan14Properties` 查询与显式启用，并保留 1.3 扩展别名和 1.2 最低回退。首批性能验证只覆盖与当前数据面相关的两条路径：`hostImageCopy` 和独立 Transfer Queue 的工具纹理上传。两者必须在相同分辨率、纹理数量和帧率下比较上传延迟、CPU 占用、GPU 队列重叠、显存和帧抖动；只有 Windows/Linux/macOS wheel 可复现、Validation Layer 通过且目标实机存在可重复净收益时，才允许提议合入主路径。

### 3.2 依赖方向

```text
main.py
 `- app_runtime
     |- capture
     |- stereo_runtime.providers
     |- stereo_runtime.vulkan_graph
     |- viewer.vulkan_renderer
     |- xr_viewer.core_openxr_vulkan
     |- xr_viewer.gltf -> xr_viewer.native.filament_bridge
     |- streaming
     `- utils / telemetry

viewer.opengl_renderer and xr_viewer.core_openxr_opengl
  <- compatibility mode only
```

禁止依赖反转：`stereo_runtime`不得引用`xr_viewer`场景对象，`xr_viewer`不得控制capture session，`gui`不得取得底层图形资源所有权。Vulkan与OpenGL实现不得相互导入或共享资源对象；公共契约放在对应现有包的backend-neutral模块中。

### 3.3 目录兼容映射

| 当前项目路径 | 新项目路径 | 处理方式 |
|-------------|-----------|----------|
| `src/desktop2stereo/main.py` / `src/desktop2stereo/main.bat` | 原路径保留 | 重写装配逻辑，启动习惯不变 |
| `src/desktop2stereo/app_runtime/` | 原路径保留 | 保留状态、队列和生命周期职责 |
| `src/desktop2stereo/capture/` | 原路径保留 | 优先整文件迁入已验证Capture实现 |
| `src/desktop2stereo/gui/` | 原路径保留 | 保留Flet界面和配置职责 |
| `src/desktop2stereo/stereo_runtime/` | 原路径保留 | 保留Provider、模型和立体算法，新增Vulkan Graph模块 |
| `src/desktop2stereo/viewer/` | 原路径保留 | 删除旧图形实现，在原目录加入Vulkan主路径和OpenGL Fallback |
| `src/desktop2stereo/xr_viewer/` | 原路径保留 | 用Python重写Vulkan OpenXR路径，保留控制器、环境和帧调度职责 |
| `src/desktop2stereo/xr_viewer/native/` | 原路径保留 | 仅存放Filament Bridge源码和平台预编译库 |
| `src/streaming/` | 原路径保留 | 接入新的GPU输出契约 |
| `src/tools/` / `src/desktop2stereo/utils/` | 原路径保留 | 放置probe、模型工具、benchmark和公共辅助 |

目录兼容只保证定位和职责连续性，不保证旧模块内部API兼容。新项目禁止通过`sys.path`指向旧仓库，也禁止从旧仓库动态导入模块；所有正式依赖必须实际存在于新项目同名路径中。

---

## 4. 运行时总体结构

### 4.1 进程模型

正式数据面运行在单个Python进程中，一次进程会话只加载一个Graphics Backend。Flet GUI、Capture、Inference、Vulkan、OpenXR和Telemetry可以作为同一Python应用内的模块协作；也允许GUI通过`multiprocessing`启动独立Python Runtime进程以提高故障隔离，但进程之间只传控制消息，不传整帧像素。

```text
python src/desktop2stereo/main.py
   |- Flet Control UI / CLI
   |- Python Capture Adapter
   |- Python Inference Provider
   |- Python Vulkan Stereo Graph
   |- Python OpenXR Presenter
   |- Python Filament Bridge wrapper -> filament_bridge DLL
   `- Python Telemetry
```

采用独立GUI/Runtime进程时，Windows使用Named Pipe，Linux/macOS使用Unix Domain Socket；单进程模式可直接使用有界Python队列。两种模式使用同一带版本号的结构化消息，图像和GPU handle不得进入跨进程控制通道。

### 4.2 生命周期状态机

```text
Created
  -> Probing
  -> Initializing
  -> Ready
  -> Running
  -> Reconfiguring -> Running
  -> Recovering    -> Running
  -> Stopping
  -> Stopped

Any state -> Failed
```

状态转换由 `RuntimeController` 串行执行。模块不得自行把全局状态改为 Running 或 Stopped。每次转换生成带原因、时间戳和配置版本的事件。

### 4.3 初始化顺序

```text
1. Parse and validate RuntimeConfig.
2. Probe OpenXR runtime and platform capabilities without creating a formal XR Session.
3. Select Vulkan main path or OpenGL Fallback before creating graphics resources.
4. Prepare model artifacts and initialize the Inference Adapter; TensorRT engine build or model download must finish here.
5. Start Capture Adapter only after inference loading succeeds and obtain the first real frame shape.
6. Execute the first inference/stereo result and shape-dependent kernel warmup, then publish `runtime_ready_event` together with the first renderable output.
7. Vulkan OpenXR path may now create XrInstance requirements, Vulkan Instance/Device, queues, Session and swapchains.
8. Initialize Filament Engine, Scene Renderer, scene resources and persistent output slots.
9. Preserve a renderable latest-frame output until the presenter reports initialized; the consumer must not dequeue and discard it during graphics startup, while the bounded producer queue may replace it with a newer completed frame.
10. Enter Ready and start presentation. OpenXR pose/controller frames continue independently after startup and may reuse the last-good screen image.
```

推理 engine 构建或模型下载必须发生在 Capture 启动前。运行时不得边捕获边编译 TensorRT engine，也不得在推理加载和首帧 shape-dependent warmup 完成前创建正式 OpenXR Session、Vulkan Device 或 Filament Engine。

Vulkan/OpenXR 本身没有等价于神经网络推理的通用“跑空数据 warmup”。本项目的图形预热定义为启动阶段完成 Device/队列、OpenXR swapchain、Filament 材质与资源、持久化输出槽的创建；实际 graphics pipeline 的首次提交发生在首个有效 XR frame，必须遵循 OpenXR `wait/begin/end` 帧协议，禁止脱离 Session 伪造提交。

### 4.5 Graphics Backend 选择

```python
class GraphicsBackendKind(StrEnum):
    AUTO = "auto"
    VULKAN = "vulkan"
    OPENGL = "opengl"

@dataclass(frozen=True, slots=True)
class GraphicsBackendSelection:
    selected: GraphicsBackendKind
    reason: str
    capabilities: GraphicsCapabilities
```

`GraphicsBackendProbe` 在启动前生成选择结果。`Auto` 优先 Vulkan；只有符合第 2.4 节条件时才能选择 OpenGL。选择结果、触发原因和禁用功能写入启动报告和 GUI 状态。

Vulkan 与 OpenGL 实现共同遵循 `IGraphicsBackend` 生命周期接口，但资源类型保持后端私有。公共层只能传递 backend-neutral handle、Frame ID、尺寸、颜色空间和同步状态，不得用 `void*` 在两个 API 之间偷渡原生资源。

### 4.4 关闭顺序

停止接收新 capture frame 后，依次停止推理提交、等待有限数量在途 Frame Context、结束 OpenXR Session、释放场景、推理槽和 Vulkan 资源。正常关闭最多等待配置定义的超时；超时必须报告具体未完成 timeline 值。

---

## 5. 线程与队列设计

### 5.1 线程职责

| 线程 | 职责 | 禁止事项 |
|------|------|----------|
| Control Thread | 状态机、配置、命令和恢复协调 | 不提交逐帧 GPU 工作 |
| XR/Render Thread | OpenXR frame loop、视图定位、Graphics Queue 提交 | 不等待新 capture frame |
| Capture Thread | 平台捕获回调和 latest-frame 发布 | 不执行推理和场景渲染 |
| Inference Thread | 厂商后端 enqueue、external semaphore 协调 | 不操作 OpenXR swapchain |
| Asset Thread | 文件读取、GLB 解码准备、shader/cache IO | 不销毁在用 GPU 资源 |
| Telemetry Thread | 日志落盘、统计聚合、控制面事件 | 不读取 GPU 图像内容 |

Vulkan queue submit 由 `GpuScheduler` 统一序列化或按队列外部同步规则保护。不得让多个模块无约束地并发调用同一个 `VkQueue`。

### 5.2 有界队列

| 队列 | 容量 | 满载策略 |
|------|-----:|----------|
| Capture latest slot | 1 | 新帧覆盖旧帧 |
| Inference pending | 1 | 保留最新尚未开始的帧 |
| Frame Context pool | 默认 9，允许通过 `D2S_OPENXR_VULKAN_FRAME_CONTEXTS` 有界覆盖 | 无可用上下文时跳过旧输入；启动诊断必须记录实际槽位数 |
| Effects pending | 1 | 新任务替换旧任务 |
| Control commands | 64 | 拒绝并报告过载，不丢停止命令 |
| Telemetry events | 固定 ring | 丢弃低级别采样，保留 warning/error |

任何实时队列都不得按运行时间增长。丢帧发生在未提交 GPU 工作之前，已提交工作不得通过破坏性方式取消。

### 5.3 帧调度

Render Thread 按 OpenXR 预测节奏持续更新头部和手柄姿态。没有新立体纹理时复用 last-good screen image，不能暂停 `xrWaitFrame/xrBeginFrame/xrEndFrame`。

Capture 与 Inference 以 latest-frame 推进；新 screen image 完成后以原子方式更新 `latest_screen_slot`。场景渲染只等待所选 screen slot 的完成信号，不等待下一张输入。

Effects Graph 消费最近完成的 screen slot，并发布 `latest_effect_slot`。Graphics Queue 从不等待指定 frame ID 的光效结果。

### 5.4 Vulkan 屏幕图像环与 Filament 外部采样

`VulkanZeroCopyOutputAdapter` 和 `CudaVulkanOutputAdapter` 共同负责拥有左右眼输出图像环，默认 `D2S_VULKAN_OUTPUT_RING_SIZE=3`。前者用于 Vulkan Compute 直接输出，后者用于 CUDA/ROCm/HIP producer；两者都必须：

1. 为每个环槽创建可导出、可采样的 device-local `VkImage`；
2. Vulkan Compute 槽位只创建一次 storage-capable external `VkImage`；CUDA/ROCm/HIP 槽位只注册一次 external memory 和 mapped array；
3. 按 `frame_id % ring_size` 选择槽位，并在输出元数据中记录 `vulkan_output_ring_slot`；
4. 让 `VulkanStereoOutputFrame` 持有该槽位的非拥有资源视图，不复制像素；
5. 在尺寸变化或关闭阶段等待有限的在途工作后统一释放整个环。

Vulkan Compute 直接输出的资源状态机为：

```text
GENERAL / producer-writable
  -> Compute shader write (storage image)
  -> timeline wait + graphics barrier
SHADER_READ_ONLY_OPTIMAL / Filament sampling
  -> Filament finished semaphore + release barrier
GENERAL / slot reusable
```

Presenter 线程创建 `VulkanStereoImageComputeBackend`，在其所属 Vulkan context 中提交
`d2s_stereo_layered_output.spv`。该 pass 的两个输出是环槽的 `VulkanExportableImage`；NVIDIA
CUDA float32 RGB/Depth 输入优先写入可导出的 Vulkan storage buffer，CUDA 通过 mapped external
buffer 和 ready semaphore 将 GPU 内拷贝完成点交给 Vulkan Compute，Compute 再等待该 semaphore。
外部 buffer 能力不可用时才使用 host-visible storage buffer，并在元数据中标记兼容回退。Compute
完成后只发布 timeline 和 per-eye visible semaphore，禁止读取 left/right 输出 buffer。因而当前
“zero-copy”是 Compute 输入（CUDA external buffer）和 Compute→`VkImage`→Filament 输出链路的
GPU 路径；CUDA Tensor 到 Vulkan buffer 仍是一次 GPU 内拷贝，不得误称为 CPU 零拷贝。

该环的图形所有权固定属于 Presenter 线程。Capture、Inference、文件读取和后台输出消费者只能通过有界命令队列提交原始结果或资源描述，不得直接创建、导入、转换、绑定、释放 `VkImage`/`VkSemaphore`，也不得调用 Filament C ABI。所有 source-image barrier、queue ownership transfer、外部纹理绑定、consumer-release 和 GPU copy/Quad Layer 回退都必须在 Presenter 的 OpenXR 帧边界内执行。任何后续零拷贝重构若绕过命令队列或恢复后台线程直接操作图形资源，均视为架构回归。

Filament Bridge 对每只眼维护 `VkImage -> Filament Texture` 缓存。`set_screen_image` 命中缓存时只切换材质参数，不能销毁旧 Texture 后重新 import。每个缓存槽必须同时保存以下状态：

```text
ScreenExternalImage {
  VkImage image
  VkFormat format
  VkExtent2D extent
  VkImageLayout layout
  uint32_t producer_queue_family
  uint32_t consumer_queue_family
  uint64_t producer_ready_value
  uint64_t consumer_release_value
  bool filament_sampling
}
```

Vulkan Compute 完成写入后，Presenter graphics submit 等待 Compute timeline，并在 source barrier 中把图像转换到 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`，同时 signal per-eye Filament visible semaphore。CUDA/Vulkan producer 完成写入时使用等价的 external binary/timeline semaphore。Bridge 把 visible semaphore 作为外部纹理 acquire wait，Filament 完成采样后，Presenter 再提交反向 barrier 到 `GENERAL`；槽位 release timeline 交给下一次 Compute submit 等待。屏幕源图像的 ready/visible/release 同步与 OpenXR 输出 swapchain 的 acquire/render-finished/release 同步必须分开管理。

直接外部采样是 Vulkan 主路径的首选：每张源 `VkImage` 只创建一次 Filament 外部纹理，稳态帧只切换槽位和材质绑定。不得把“裸 `VkImage` 可导入”当作完整同步；如果缺少格式/尺寸、layout、queue ownership 或 producer/consumer 同步信息，必须拒绝零拷贝绑定并选择一次 GPU copy 的屏幕路径或 Quad Layer 回退。回退不能使用 CPU 像素回读，也不能在已提交的 Filament 采样期间复用或销毁源图像。槽位扩容或尺寸变化属于受控重配置，不得发生在正常帧循环。

上述禁止 CPU 回读的要求针对主屏幕实时图像链。辉光能力回退必须与主屏幕隔离：从立体合成前的桌面源图低频生成不超过 `320x180` 的 RGBA sRGB 特效纹理，只用于继续提供 Surround、Glow 和 Veil 行为；不得把 CPU 回读或上传伪装成 GPU zero-copy。

NVIDIA/CUDA 的正式 Glow 纹理来源使用独立 GPU 路径：CUDA 张量复制到 Presenter 持有的 external storage buffer，并通过 CUDA external semaphore 通知同一 Vulkan queue family 的第二条 Compute queue；`d2s_glow_source.comp` 在线性光空间完成固定 `320x180` RGBA8_UNORM 特效源生成，Vulkan Projection Composer 在 LOD 0 采样。Compute 中的预过滤足迹保持现有效果语义：Veil 使用目标像素自身的缩小足迹，Glow 和 Surround 使用约 256 个源像素的宽范围；不得把该值再次乘以 320x180 降采样足迹。该路径使用至少三个有界槽位，只发布已完成并已在 graphics queue 排队完成 layout transition 的最新槽位；新任务未完成、资源预算不足或更新间隔未到时复用上一张已完成纹理，禁止让主 graphics queue、OpenXR 帧或主屏幕 zero-copy 等待 Glow。日志标记 `glow_source_path=vulkan_compute_external_image`，并分别报告 submit、reuse 和 budget-skip。ROCm/HIP、Vulkan Stereo producer、缺少第二队列或缺少 CUDA external memory/semaphore 时仍使用明确回退，不能静默破坏主屏幕输出。

Glow 外部图像和 Filament 外部纹理包装必须按严格生命周期销毁：先停止新帧并释放输出帧租约，等待 graphics/compute queue 空闲，销毁 Filament texture wrapper，最后销毁 Presenter 持有的 `VkImage`、buffer、semaphore 和 fence。Compute queue 与 Filament graphics queue 当前必须属于同一 queue family；不满足时禁用该 GPU Glow 路径，不允许用 `VK_QUEUE_FAMILY_IGNORED` 冒充跨 family ownership transfer。

源图像 producer 必须通过后端无关的 Vulkan interop contract 接入 Presenter：`GpuProducerAdapter` 负责导出/导入 external memory、提交 producer-ready semaphore 或 timeline、报告实际 image layout/queue family，并在收到 consumer-release 后回收槽位。CUDA、ROCm/HIP、MIGraphX、DirectML 或其它推理后端只能实现该适配器，不得把 CUDA API、HIP API 或厂商句柄泄漏到 Filament Bridge 和 `VulkanContext` 的策略接口。Windows/Linux 优先复用各后端已有 external memory；ROCm/HIP 或驱动不支持安全零拷贝时，必须降级为一次 GPU copy，不能降级为 CPU 像素往返。

Presenter 只能通过 `GpuProducerAdapter` 注册表创建具体 producer；不得在 OpenXR、Filament 或 `VulkanContext` 中直接实例化 CUDA、HIP 或其它厂商适配器。未注册或能力不足的后端必须明确进入 GPU copy/Quad Layer 回退路径，禁止将一个厂商的句柄或同步语义伪装成另一个厂商的实现。

`src/tools/probe.py` 的 capability report 必须输出 producer 自动选择结果、CUDA/HIP runtime 状态和显式覆盖标记；探测只读取运行时能力，不应为了生成报告而创建 Vulkan external memory 或导入厂商句柄。

当前已注册 `cuda`/`nvidia`、`rocm`/`hip` 与 `vulkan_zero_copy` producer。OpenXR runtime result 携带 `VulkanComputeRequest` 时，Presenter 必须选择 `vulkan_zero_copy`，不能误选 `vulkan_host` 回退；只有请求不满足（例如仍启用尚未迁移的时域状态）或能力探测失败时才进入兼容路径。ROCm 适配器延迟加载 `amdhip64`/`libamdhip64`，使用 Vulkan 导出的 Win32 handle 或 FD；HIP external memory/semaphore 不可用时，不得启用直接采样实验路径，必须保留 Vulkan GPU copy 回退。`D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE=0` 仅作为调试禁用开关，不是正常运行前提。

适配器发现或加载失败属于能力缺失，不得传播为 Presenter 线程异常；必须限频记录原因、保持 OpenXR 生命周期运行，并在后续输出帧重新尝试创建适配器。若当前 producer 没有可用 Vulkan import/copy 能力，则该帧只能丢弃，禁止退回 CPU 像素往返。

当前实现状态：图像环、Filament 多槽缓存、Vulkan Compute 直接写图像、Compute timeline、source barrier、Filament visible semaphore、consumer-release barrier、Filament per-eye render-finished ABI 和 CUDA/ROCm/HIP producer wait 已接入；Filament v1.76.0 源码补丁和三平台 CI 远程构建入口已接入。直接外部采样请求默认开启，但只有带 `D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE` 的新 Bridge 才会报告 `filament_bridge_vulkan_external_image_abi_available=true` 并实际包装外部 `VkImage`；旧 stock SDK/Bridge 仍报告 false，Presenter 自动回退到显式标记的 Vulkan GPU copy。`D2S_ENABLE_FILAMENT_SCREEN_IMAGE=0` 仍可用于回归测试和故障隔离。不得把通用 `import()` 强行当作 Vulkan 外部纹理接口，也不得移除 consumer-release。

### 5.5 Projection Layer 异步提交

XR/Render Thread 对一帧执行：

```text
acquire + wait(left), acquire + wait(right)
-> set active ring slot and camera for each eye
-> begin/end Filament frame for left and right
-> one frame-wide completion wait
-> release(left), release(right)
-> xrEndFrame
```

Bridge 的 `end_frame` 只提交当前眼睛的命令，不等待 GPU。`wait_for_idle` 只作为当前 external synchronization ABI 的帧边界兼容实现；external semaphore 完成后应替换为按槽位 completion point 回收，不能恢复每眼 `flushAndWait`。

---

## 6. 核心数据契约

### 6.1 基础类型

```python
FrameId = int

@dataclass(frozen=True, slots=True)
class Extent2D:
    width: int
    height: int

@dataclass(frozen=True, slots=True)
class SyncPoint:
    semaphore: object
    value: int
    stage_mask: int

@dataclass(frozen=True, slots=True)
class GpuImageView:
    image: object
    view: object
    format: int
    extent: Extent2D
    layout: int
    queue_family: int
```

`GpuImageView`是非拥有视图。拥有资源由`GpuImage`显式管理，并提供幂等`close()`和context manager；不得依赖Python垃圾回收时机释放Vulkan对象。

### 6.2 输出同步契约

```python
OutputFrame {
  frame_id
  ring_slot
  left_eye
  right_eye
  color_space = "srgb"
  image_origin = "top_left"
  producer_ready: SyncPoint | None
  consumer_release: SyncPoint | None
}
```

`producer_ready` 未提供时，生产者必须在发布前完成等价的 CUDA/GPU 同步，并将实际降级方式写入 metadata。任何消费者不得假设图像可立即复用；槽位回收必须由 producer 和 graphics consumer 的完成点共同决定。

### 6.3 CaptureFrame

```python
@dataclass(frozen=True, slots=True)
class CaptureFrame:
    id: FrameId
    timestamp_ns: int
    capture_size: Extent2D
    pixel_format: PixelFormat
    color_space: ColorSpace
    image: ExternalImageHandle
    ready: SyncPoint
```

`ExternalImageHandle` 是 tagged union，平台实现可以承载 Vulkan image、Win32 handle、DMA-BUF/FD、IOSurface 或受控 CPU 测试帧。实时模式下不接受普通 CPU pointer。

### 6.4 InferenceResult

```python
@dataclass(frozen=True, slots=True)
class InferenceResult:
    frame_id: FrameId
    relative_depth: object
    metadata: DepthMetadata
    ready: SyncPoint
    stale: bool
```

`DepthMetadata` 必须包含模型 ID、backend、precision、near/far direction、normalization、模型输入尺寸和 GPU timing。下游不得猜测深度方向。

### 6.5 StereoFrame

```python
@dataclass(frozen=True, slots=True)
class StereoFrame:
    frame_id: FrameId
    left_eye: GpuImageView
    right_eye: GpuImageView
    packed_sbs: GpuImageView | None
    ready: SyncPoint
    config_version: int
    color_space: str = "srgb"
    image_origin: str = "top_left"
```

Left/Right Eye 是标准输出；只有虚拟屏幕材质或输出目标需要时才生成 `packed_sbs`。未生成时使用空 view，不分配无意义图像。输出帧契约明确声明 `color_space=srgb` 和 `image_origin=top_left`；任何 OpenXR、Preview 或编码后端需要改变目标坐标原点时，必须在边界处显式适配，不能修改源图像语义。

### 6.6 FrameContext

每个 Frame Context 独占 command pool、command buffer、descriptor arena、timestamp query 范围和中间图像索引。Frame Context 只能在其最终 timeline 值完成后复用。

---

## 7. Vulkan 基础层

### 7.1 VulkanContext

`VulkanContext` 负责 Instance、Device、Physical Device、queue family、allocator、pipeline cache 和 debug messenger。它不管理 OpenXR frame loop，也不加载 glTF。

必须提供：

```python
class VulkanContext:
    @classmethod
    def create(cls, requirements: VulkanRequirements) -> "VulkanContext": ...
    def close(self) -> None: ...
    @property
    def instance(self) -> object: ...
    @property
    def physical_device(self) -> object: ...
    @property
    def device(self) -> object: ...
    @property
    def graphics_queue(self) -> QueueHandle: ...
    @property
    def compute_queue(self) -> QueueHandle: ...
    @property
    def transfer_queue(self) -> QueueHandle: ...
```

`VulkanRequirements` 由 OpenXR、Filament、Compute Graph 和推理互操作能力合并生成。缺失必需扩展时在创建 Device 前失败。

Vulkan 版本和设备 Feature 规则：

- OpenXR Vulkan 初始化默认请求 Vulkan 1.4；实际版本必须限制在 `xrGetVulkanGraphicsRequirements2KHR` 返回的最小/最大范围内，主路径最低保证 Vulkan 1.2。
- Vulkan 1.4 只是默认请求上限，不要求所有 Runtime、驱动或 GPU 都实际创建 1.4 Device；Runtime 协商到 1.2 或 1.3 时，只要能力探测通过即可继续。
- 创建设备前必须使用 `vkGetPhysicalDeviceFeatures2` 查询 `timelineSemaphore`。
- 当 Timeline Semaphore 可用时，必须将 `VkPhysicalDeviceTimelineSemaphoreFeatures` 追加到 `VkDeviceCreateInfo.pNext`，并将 `timelineSemaphore` 设置为 `VK_TRUE`。请求 Vulkan 版本不会自动启用该 Feature。
- Timeline Semaphore 不支持或启用失败时，Vulkan OpenXR 路径必须在创建正式 Device/Session 前失败并给出原因；由启动器按策略受控重启进入 OpenGL Fallback。

### 7.2 资源分配

- 所有长期图像通过统一 `GpuAllocator` 创建。
- transient image 按尺寸、格式和 usage 建立可复用池。
- external-memory image 单独分配，不与普通 transient memory alias。
- OpenXR swapchain image 不绑定应用内存，不进入 allocator 销毁流程。
- 内存预算来自 `VK_EXT_memory_budget` 时必须采集并上报。

### 7.3 图像状态追踪

`ImageStateTracker` 记录每个受管图像的 layout、stage、access 和 queue family。Compute Graph 编译阶段产生 barrier plan，执行阶段使用 Synchronization2 提交。

不得使用 `VK_IMAGE_LAYOUT_GENERAL` 规避全部状态管理。External-memory 交界可按后端要求使用 GENERAL，但进入场景采样前必须转换为正确只读布局。

### 7.4 Descriptor

- Descriptor layout 在 shader 构建时反射并生成稳定 binding metadata。
- 静态资源使用 persistent set；帧资源使用 Frame Context arena。
- 不更新仍可能被 GPU 读取的 descriptor。
- Descriptor Indexing 只用于资源数组，不允许动态越界。

### 7.5 Pipeline 与 Shader

Shader 源码进入版本控制，SPIR-V 由构建系统生成。每个 shader 必须有：入口、workgroup size、descriptor binding、push constant 大小和精度要求的机器可读 manifest。

Pipeline Cache key 至少包含 GPU UUID、驱动版本、Filament 版本、shader hash 和 build type。缓存无效时重建，不把 pipeline 创建放入稳态帧循环。

### 7.6 同步规则

内部依赖使用 Timeline Semaphore 和 `vkQueueSubmit2`。正常帧循环禁止调用：

```text
vkDeviceWaitIdle
vkQueueWaitIdle
CPU polling loop on fence status
implicit queue ownership assumptions
```

只有初始化失败清理、Session 销毁和进程关闭可执行受控全局等待。

Timeline 值必须由统一的 `FrameContext` 同步协议管理。NVIDIA、AMD、Intel 和 MoltenVK 路径均不得假设该 Feature 默认开启；能力探测、Device 创建 pNext 链和启动诊断必须记录实际支持与启用状态。

---

## 8. Capture 工程设计

### 8.1 接口

```python
class CaptureAdapter(Protocol):
    def start(self, config: CaptureConfig, sink: CaptureSink) -> None: ...
    def stop(self) -> None: ...
    def capabilities(self) -> CaptureCapabilities: ...
```

`CaptureSink::publish()` 只发布句柄和 metadata，不进行图像转换。Capture Adapter 对平台捕获对象负责，Gpu Importer 对 Vulkan 导入负责。

### 8.2 平台路径

| 平台 | 首选捕获 | Vulkan 接入目标 |
|------|----------|-----------------|
| Windows | Windows Graphics Capture / DXGI | 共享 handle 导入或 GPU copy 到 Vulkan image |
| Linux | PipeWire/DMABUF 或 DRM 路径 | DMA-BUF/FD 导入 Vulkan |
| macOS | ScreenCaptureKit | IOSurface/Metal 纹理桥接到 MoltenVK 路径 |

平台路径必须在运行报告中标明`native_import`、`gpu_copy`或`cpu_test_input`。Windows首先复用现有`WindowsCaptureCUDA`、`WindowsCaptureROCm`和对应测试，不因Vulkan迁移重写捕获算法。正式实时运行不得把CPU路径命名为零拷贝。

### 8.3 尺寸与颜色

Capture Adapter 报告真实 `capture_size`、pixel format、transfer function、primaries、range 和 `color_space`。任何未知色彩空间必须告警并阻止静默推断。SDR sRGB 输入在显示输出路径中保持显示参考语义；RGB preprocess 可以为模型或线性工作 Pass 建立明确的线性副本，但不得在没有声明的情况下对原始显示 RGB 执行 gamma、tone mapping 或曝光。

WindowsCaptureCUDA 在回调入口执行软件 pacing gate，目标来自 `fps_provider`，`D2S_WGC_SOFTWARE_THROTTLE` 只作为诊断覆盖。Auto 模式由 `AdaptiveCaptureRate` 按稳定 SBS 消费能力在有界评估窗口内调整为测得峰值加 5 FPS，并受配置/显示刷新上限约束；节流发生在 latest-frame 发布前，不得扩大 capture queue。

正式输出契约固定为 `color_space=srgb`、RGBA8、`image_origin=top_left`。OpenXR Projection swapchain 必须使用 Vulkan sRGB 格式，不得回退到 UNORM。Filament 主场景使用线性 Rec.709/D65 和明确配置的 tone mapping；场景曝光必须进入 ColorGrading，不得乘改 glTF 材质颜色。

手柄 GLB 的加载和材质处理以 WebXR Input Profiles 官方 Viewer 为基准：模型加载后必须保留 glTF 原始 PBR 参数和纹理，不得在 Bridge 中统一覆盖 roughness、specular、metallic 或 base color。房间和前景必须使用不同的 Filament Scene：房间 Scene 承载 GLB 与房间全局 `IndirectLight`；前景 Scene 只承载手柄、屏幕、激光和 UI，并共享同一 Engine、Camera 和 OpenXR 输出目标。这样房间环境光不能污染手柄和虚拟屏幕。`controller_hdr_lighting=false` 的 3D 房间模式关闭前景间接光，使用旧工程校准的 `env_head_light_color`、跟随头部主光、顶部补光和始终开启的屏幕光；`controller_hdr_lighting=true` 的 HDR 图片模式只给前景 Scene 挂载独立的 controller IBL，且必须使用与当前 HDR 匹配的预过滤 reflection cubemap 和 irradiance。当前 Bridge 的 HDR 资源仍记录为 `hdr_ibl_pending_profile_fallback`，不得伪报为完整 IBL；后续应由三平台 CI 预生成 Filament 可加载的 KTX IBL 资源并通过稳定 C ABI 接入。手柄 profile 的 `ambient_light_multiplier` 只能缩放前景 controller IBL，不得修改房间全局光、GLB 材质、屏幕光或直接补光，浅色 PICO/Quest 默认保持 `1.0`。

控制器按键引导不得使用固定品牌坐标或控制器根节点。运行时必须从当前右手柄 GLB 的 B 键 `_pressed_value` 动画枢轴解析模型局部锚点，再应用与控制器渲染一致的 profile 旋转、偏移和 Grip 世界变换；切换手柄品牌或替换模型后必须立即清除缓存并重新解析，确保引导端点跟随实际 B 键位置。

虚拟屏幕光与上述环境模式正交，HDR 开关不得关闭或控制屏幕光。运行时每 250 ms 在 CUDA 流上异步下采样左右眼显示用 sRGB 图像，按标准 sRGB EOTF 转为线性平均色；颜色继续沿用旧工程的 `82%` 屏幕采样色与 `18%` profile 中性色混合。Filament 在屏幕中心创建只作用于控制器 light channel 1 的无阴影前向聚光源，方向跟随屏幕法线、衰减距离使用屏幕对角线，并由 `screen_light_intensity` 控制强度。该补充采样不得阻塞输出路径，失败时必须保留 last-good 屏幕光颜色。

虚拟屏幕、UI 和手柄激光属于显示参考 LDR 内容。传统 Quad Layer/直接 sRGB 纹理路径使用 `VK_FORMAT_R8G8B8A8_SRGB`，由 Filament 只解码一次；Presenter-owned Vulkan Compute 的 storage image 受 Vulkan storage-image 格式约束使用 `VK_FORMAT_R8G8B8A8_UNORM`，Compute 必须先把输入 sRGB 解码为线性光再写入，Filament 以线性 `RGBA8` 采样。两条路径在输出契约上都保持 `color_space=srgb`，最终只由 sRGB OpenXR 目标完成一次编码。每只眼使用一个房间 View 和一个前景 View：房间 View 渲染 GLB，前景 View 在同一 Renderer 帧内随后合成手柄、屏幕/UI 与激光；两者共享 Camera、交换链、深度附件和 Engine，但不共享 Scene 级间接光。手柄 PBR 先写入前景深度，激光以 `depthCulling=true` 渲染，必须自然被外壳遮挡。两个 View 的 ColorGrading 使用 `LINEAR` tone mapping，不使用 ACES 或其他场景曲线，但保持后处理启用以在最终 sRGB 目标上完成唯一一次输出编码。禁止重复 Engine、重复 controller asset 或 CPU 像素往返；默认颜色控制必须为恒等变换，非中性颜色控制只能来自明确的用户配置。

窗口尺寸或 HDR 状态改变时发布 format-change event。资源重建在 Frame Boundary 进行，capture callback 不直接重建 Vulkan 资源。

---

## 9. 推理适配器设计

### 9.1 接口

```python
class InferenceProvider(Protocol):
    def initialize(
        self, config: InferenceConfig, interop: InteropContext
    ) -> InferenceCapabilities: ...
    def submit(self, submission: InferenceSubmission) -> SyncPoint: ...
    def reset(self) -> None: ...
```

`InferenceSubmission`引用预分配input/output slot。现有TensorRT、PyTorch CUDA/ROCm、MIGraphX和其他Python Provider应保留优化实现，只调整统一接口和Vulkan输出契约；不得每帧创建engine、分配大块device memory或重新注册external resource。

### 9.2 Slot 注册

初始化阶段由 Vulkan 创建 N 组可导出资源，Inference Adapter 一次性导入并保存后端对象。每帧只选择 slot、等待 timeline、enqueue 和 signal。

```text
Vulkan creates exportable input/depth slots
-> exports OS memory handles
-> inference adapter imports handles once
-> frame loop reuses registered slots
```

Win32/FD handle 在成功导入后按对应 API 所有权规则关闭。句柄泄漏测试必须覆盖重复初始化和 Session 恢复。

### 9.3 NVIDIA

NVIDIA 后端使用 TensorRT + CUDA。优先由 CUDA 直接访问 Vulkan 导出的 input/output memory，并使用 external semaphore 同步。

验收要求：主路径无 host memcpy、无 `.cpu()`/NumPy 往返、无每帧 external-memory import。TensorRT enqueue 时间和 GPU 完成时间分别记录。

#### 9.3.1 双 TensorRT context 并发实现

Native TensorRT provider 在 engine 创建后必须尝试创建两个 execution context。每个 slot 独占以下资源：

- CUDA stream；
- TensorRT execution context；
- 输入 tensor 引用和输出 buffer；
- CUDA Graph（启用时）；
- slot 可复用 completion event。

`RuntimePipelineLoop` 使用最多两个 Python depth worker 仅调用 `StereoRuntime.predict_openxr_depth()`。worker 完成后返回 `DepthProfileResult`；主 runtime 按 `frame_id` 取最早完成且顺序正确的结果，并把该 profile 传给 `process_openxr_frame()` 执行既有立体合成、补洞和输出准备。不得让 worker 调用 Vulkan submit、Filament C ABI、OpenXR swapchain 或 `xrEndFrame`。

TensorRT engine 可能在首帧输入尺寸确定后才完成创建，因此 pipeline 初始化和首帧 RGB 准备后都必须检查 `pipeline_slot_count`。当开关开启、slot 数至少为 2 且 `profile_sync=false` 时，延迟创建 worker；避免启动期因 provider 尚未实例化 engine 而永久误降级为单 worker。关闭开关、slot 数不足或 profile-sync 启用时保持原单路路径。

worker 队列上限为 2。任务对象必须携带 `frame_id`、capture timestamp、RGB tensor、depth future 和 render/capture metadata；超过上限时仅取消尚未运行的任务。运行循环退出时必须关闭 executor；后续扩展的暂停、重建或 source/尺寸切换不得遗留 worker 对已释放 GPU 资源的访问。

诊断必须同时包含：`parallel_inference_enabled`、`parallel_inference_workers`、`parallel_inference_pending`、`parallel_inference_dropped`、`runtime_depth_execution_slot` 和 `runtime_depth_execution_slot_count`。`FPSBreakdown` 只在 `rt_parallel_workers=2`、`rt_pending_limit=2` 且 slot 日志交替为 `0/2`、`1/2` 时报告真实双路状态；只看到 GUI 开关或 slot_count=2 不足以证明调度器已运行。

Admission 还必须读取 Presenter 反压状态：Presenter 正在合成或 SBS/latest-frame 待消费时，将在途深度任务限制为 1；Presenter 缺帧且无待消费资源时，才恢复到配置允许的并发上限。`present_fps` 只统计 Presenter 实际消费的唯一立体帧，不能把重复显示或 OpenXR refresh 当作新处理帧。

RTX 2060 动态内容实机对比中，该实现较单路增加约 6~10 FPS 的处理/SBS 吞吐。该数值仅为当前模型、分辨率和补洞配置下的验证记录，不构成所有 GPU 或所有质量档位的性能承诺；`rt_gpu_synth_fill`、立体合成、Vulkan/Presenter 同步仍可能限制最终帧率。

### 9.4 AMD

Windows/Linux 优先 ROCm/HIP + MIOpen。可靠 external memory 可用时使用零拷贝；否则允许一次明确的 GPU 内拷贝。

Windows ROCm 不可用时，DirectML Adapter 可作为独立推理后端。D3D12 resource 仅存在于该 Adapter 内部，导出到 Vulkan 后不扩散为 D3D 渲染架构。

### 9.5 Apple Silicon

CoreML/MPSGraph Adapter 管理 MTLTexture/IOSurface 和模型执行。MoltenVK 与 Metal 的资源连接必须通过原型验证确定；不能假设 MoltenVK 自动消除所有拷贝。

### 9.6 失败策略

Adapter 初始化失败时终止本次启动并返回 capability report。运行中单帧失败可丢弃；连续失败达到阈值后进入 Failed 或受控重建。不得静默切换 CPU 推理。

---

## 10. Stereo Compute Graph

### 10.1 Graph 结构

```text
Capture Import
-> RGB Normalize/Resize
-> Inference Bridge
-> Depth Normalize
-> Edge-aware Depth Filter
-> Parallax Field
-> Left/Right Warp
-> Occlusion Mask
-> Directional Hole Fill
-> Temporal Stabilization
-> Optional SBS Pack
```

Graph 在 `render_size` 和质量配置确定后编译。Pass、资源和 barrier 在编译时固定，执行时只更新 descriptor、push constant 和 slot index。

### 10.2 Pass 接口

```python
class ComputePass(Protocol):
    def declare(self) -> PassDeclaration: ...
    def create_pipeline(self, shaders: ShaderLibrary) -> None: ...
    def record(
        self,
        command_buffer: object,
        resources: PassResources,
        parameters: FrameParameters,
    ) -> None: ...
```

Pass 不得自行提交 queue，不得持有 OpenXR swapchain，不得执行文件 IO。

### 10.3 参数上传

每帧参数写入 host-visible ring buffer 或 push constant。参数快照带 `config_version`，一帧所有 Pass 必须使用同一版本。

#### 10.3.1 补洞模式 ABI 与 shader 实现

补洞模式使用跨后端稳定枚举：`VULKAN_HOLE_FILL_BALANCED=0`、`VULKAN_HOLE_FILL_QUALITY=1`、`VULKAN_HOLE_FILL_NONE=2`。Python 运行时负责把唯一的产品模式 `balanced/quality/none` 解析为枚举，并同步解析固定参数：均衡为 radius 1、strength 0.6；增强/高质量为 radius 3、strength 1.0；关闭为 radius 0、strength 0.0。`soft_low_ghost` 和 `sharp_test` 已删除，不建立旧值兼容表。

以下 shader 必须接受并执行同一模式语义：

- `d2s_stereo_fused.comp`：普通 SBS `fast_plus` Vulkan fallback。
- `d2s_stereo_layered.comp`：`quality_4k/hq_4k` 通用 layered fallback。
- `d2s_stereo_layered_tiled.comp`：tiled reference/优化对照路径。
- `d2s_stereo_layered_output.comp`：OpenXR presenter-owned storage image zero-copy 路径。

`quality` 的公式顺序固定为：生成 feather mask；以左右深度差或位移差判断方向可靠性；沿可靠背景方向对 step 1..3 采样并平均；可靠时使用 `directional * 0.75 + blurred * 0.25`，否则使用 blurred；最后应用通道均值亮度边缘保护与深度边缘保护。窗口边界按固定分母零填充，方向采样按边界复制。颜色运算必须服从该输出路径既有的 sRGB/线性契约，不得加入 tone mapping。

`none` 的短路必须位于 occlusion dilation、feather 和 box/directional neighborhood 之前；通用 buffer 路径写零 mask，OpenXR image 路径不执行 mask 计算。Debug 至少包含 `vulkan_hole_fill_mode` 和 `vulkan_hole_fill_backend`。Push constant ABI 变化必须同步更新 Python pack 格式、pipeline size、SPIR-V 和 ABI 测试。

GUI 与 canonical runtime preset 必须一致：电影 `balanced/1/0.6`，游戏 `none/0/0.0`，图片 `quality/3/1.0`。Tooltip 仅展示“关闭 / 不补洞、均衡 / 标准、增强 / 高质量”三档及上述默认映射。

视差核心语义保持：

```text
disparity_px = depth_response(relative_depth, convergence)
             * max_disparity_px
             * depth_strength
```

IPD、Stereo Scale、Max Shift Ratio 不进入 normalized-depth shader contract。

### 10.4 Temporal 资源

Temporal history 每个 eye 至少包含 color、confidence 和有效性标记。资源按 ping-pong 方式复用。尺寸、捕获源、模型、深度方向或关键曲线改变时，`TemporalController` 在下一帧前清除有效性。

### 10.5 Shader 测试

每个 Compute Pass 必须有 CPU reference 或固定 golden fixture。GPU 测试比较误差、边界像素、NaN 输入、极端尺寸和非 16:9 输入。

---

## 11. Filament 场景系统

### 11.1 Python SceneRenderer接口

```python
class FilamentSceneRenderer:
    def initialize(self, vulkan: VulkanContext, config: SceneConfig) -> None: ...
    def load_glb(self, path: Path) -> SceneHandle: ...
    def activate_scene(self, handle: SceneHandle) -> None: ...
    def update(self, state: SceneFrameState) -> None: ...
    def render(self, target: SceneRenderTarget, eye: EyeView) -> None: ...
```

`FilamentSceneRenderer`是Python封装，内部通过`ctypes`/`cffi`调用Filament DLL Bridge。DLL管理Filament Engine、Renderer、Scene、View、Camera、AssetLoader、ResourceLoader和Animator；Python只持有不透明整数handle，不直接持有Filament entity或C++对象地址。

### 11.2 Graphics Backend

Filament DLL Bridge默认创建Filament Vulkan Backend，并实现经过版本控制的Vulkan Render Target接入。OpenGL Fallback可由同一Bridge创建独立Filament OpenGL Backend或使用Python兼容Renderer，但不得复用Vulkan会话资源。旧Bridge的glTF加载和动画经验可以复用，旧OpenGL-only ABI必须由新版窄C ABI取代。

OpenXR swapchain 的接入必须通过经过验证的 Filament external render target/Vulkan platform integration 实现。若特定 Filament 版本不能安全直接绑定 acquired `VkImage`，允许在 Vulkan 内渲染到应用 color image 后执行一次 Vulkan copy/blit 到 swapchain；不得改用 D3D11/GL 桥接。

OpenGL Fallback 直接渲染到 OpenGL OpenXR swapchain 或 OpenGL window framebuffer，不经过 Vulkan、D3D11 或 WGL 跨 API 桥接。Windows/Linux 目标 OpenGL 4.3 及以上；macOS OpenGL 4.1 仅提供基本网格、材质、虚拟屏幕和 UI，不启用依赖 Compute Shader 的效果。

### 11.2.1 Filament 颜色管线

本节旧版“前景 View 合成屏幕/UI/激光”的描述由 11.2.2 当前架构修订覆盖；Filament 只输出环境与手柄 GLB 的颜色/深度，屏幕采样、激光、Glow 和 2D 光圈由 Vulkan 合成。

Filament Bridge 必须将 GLB 放入房间 Scene，将原始 PBR 手柄、显示参考屏幕/UI 和激光放入前景 Scene。房间 View 先渲染，前景 View 随后在同一帧合成；两个 View 使用同一 Camera、Renderer、交换链目标和深度附件，前景 View 不清理已有颜色/深度。手柄使用原始 PBR 材质并写前景深度；激光使用不透明、`depthWrite=true`、`depthCulling=true` 的材质和控制器相同的 layer，因此外壳在几何上遮挡激光。房间 Scene 的 IndirectLight 不得挂到前景 Scene；HDR 模式的 controller IBL 必须只挂到前景 Scene。左右眼共享的 Renderer 必须在创建时显式设置 `ClearOptions.clear=true`、黑色 clear color 和 `discard=true`。由于 Filament 1.76 Vulkan backend 的默认 RenderTarget 是单例，Bridge 必须通过 D2S backend patch 让默认 RenderTarget 记录当前绑定的 SwapChain/image，切换左右眼时先释放旧绑定再绑定当前眼，避免两个 OpenXR SwapChain 互相污染并触发 `VK_ERROR_DEVICE_LOST`。两个 View 使用 ColorGrading `LINEAR` tone mapping，不应用 ACES/其他场景曲线；保留后处理仅用于最终目标的单次 sRGB 编码。AssetLoader 只加载一个 controller asset，不得为遮挡创建副本、修改同一 Renderable 的材质绑定或创建第二个 Engine。屏幕源格式必须与其采样语义匹配：sRGB 源使用 `VK_FORMAT_R8G8B8A8_SRGB`/`SRGB8_A8`，Compute 输出的 `VK_FORMAT_R8G8B8A8_UNORM` 必须在 shader 中先完成 sRGB→线性转换并以 Filament `RGBA8` 采样；禁止把 sRGB 数值直接写入 UNORM 后再当线性纹理使用。

### 11.2.3 Bridge 接口边界与完整性清单

### 11.2.2 当前架构修订（覆盖旧的屏幕/前景 Filament 描述）

Filament Bridge 仅负责环境/手柄 GLB 的加载、材质、动画和每眼颜色/深度输出。虚拟屏幕采样、激光、Glow、屏幕光圈和键盘光圈全部由 Vulkan Projection/Quad 路径负责；不得新增屏幕纹理、激光或 Glow 的 Filament 材质 ABI。Projection Layer 由 Vulkan Composer 统一合成，FPS、操作指南、两处固定 2D 光圈和虚拟键盘使用 Quad Layer。

本修订覆盖第 5.4、8.3、11.2.1、11.4、11.5、13.1 和 13.5 节中将屏幕、激光或 Glow 绑定到 Filament 的旧实现描述；旧接口名仅保留为迁移历史，不构成新架构要求。

OpenXR Presenter 线程独占 session、swapchain acquire/release、layer 构建和 `xrEndFrame`。Filament 渲染、虚拟屏幕生产和 UI 纹理生产可以由不同 worker 执行，但只能通过有界 latest-frame 资源、timeline semaphore 和命令队列交付，后台线程不得直接调用 Presenter 的 OpenXR API。


补充约束：前景 View 必须关闭后处理；房间主 View 是唯一执行 LINEAR tone mapping 和最终颜色编码的 View，避免前景合成再次处理已完成的房间颜色。

Bridge 采用窄 C ABI，按产品功能补充接口，不做 Filament 全量 API 的 Python 化。C++ 内部管理 Engine、Renderer、Scene、View、Camera、AssetLoader、ResourceLoader、Animator、Material 和 Texture 等对象；Python 通过 `ctypes` 使用不透明 handle、标量、矩阵、资源字节和结构化状态。

Bridge 接口必须覆盖以下功能域：

1. ABI/Filament 版本、平台能力、错误码和最近错误。
2. Vulkan Bridge 与 Preview 的创建、初始化、销毁和资源释放。
3. Python/OpenXR Vulkan 对象借用、swapchain image 注册、acquired image 绑定、帧开始/结束和 swapchain 重建。
4. GLB 加载、纹理/PBR/透明材质资源准备、场景切换、卸载和 last-good scene 保留。
5. 场景对象的可见性、变换、虚拟屏幕、手柄和场景根节点状态更新。
6. 预览相机、左右眼相机、look-at、视图矩阵、投影/frustum、near/far 和 viewport。
7. Preview window、Vulkan Render Target、OpenXR Projection Layer 目标的渲染提交，以及 per-eye Filament render-finished semaphore 的借用查询。
8. 动画枚举、名称/时长、选择、播放、暂停、循环和由 Python 驱动的动画时间应用。
9. 场景主体亮度/曝光、天空盒亮度、方向光、填充光及其颜色和方向的独立控制。
10. 材质参数、亮度、对比度、饱和度、Gamma、色温和色调控制。
11. Glow、星光闪烁、平均色、墙面反射及其他后处理/特效的资源设置、启停、参数更新和时间推进。
12. Preview resize、surface/window 变化、渲染目标尺寸变化和资源重建。
13. 加载、上传、渲染、同步和 GPU 资源统计，以及 capability report 所需的诊断数据。

每个功能域必须完成以下闭环才算实现：

```text
功能需求确认
    -> filament_bridge.h C ABI 声明
    -> filament_bridge.cpp 稳定 ABI 转发
    -> bridge_*.cpp / preview_bridge.cpp 内部实现
    -> Python ctypes wrapper
    -> Windows/Linux/macOS CI 编译
    -> 单元/集成/头显或预览运行验证
    -> 接口清单标记完成
```

原生 Bridge 固定采用以下内部布局，Python 不感知内部文件数量：

```text
native/filament/bridge/
  filament_bridge.cpp          # Stable C ABI forwarding only
  filament_bridge.h            # Stable Python ABI
  bridge_context.cpp/.h        # Engine, Scene, shared resources
  bridge_eye.cpp/.h            # Per-eye View, Camera, Swapchain
  bridge_scene.cpp/.h          # Environment GLB loading
  bridge_controller.cpp/.h     # Controller models and input animation
  bridge_laser.cpp/.h          # Laser material and geometry
  bridge_screen.cpp/.h         # Virtual screen and external VkImage
  bridge_material.cpp/.h       # Color, exposure, material parameters
  preview_bridge.cpp/.h        # Desktop preview implementation
  bridge_internal.h            # Internal shared types
```

`filament_bridge.h` 是唯一面向 Python 的原生契约。内部模块不得直接导出 C++ 类型；Engine、Scene、GLB、材质、纹理和 Shader 的 ownership 归 `bridge_context`，各功能模块只通过 `bridge_internal.h` 访问共享状态。Linux/macOS 构建默认隐藏内部符号，只有带 `FILAMENT_BRIDGE_API` 的 C ABI 可见。

三平台二进制固定存放在 `src/desktop2stereo/xr_viewer/native/windows`、`src/desktop2stereo/xr_viewer/native/linux` 和 `src/desktop2stereo/xr_viewer/native/macos`；运行时、能力探测、CMake 和 GitHub Actions 必须共用该路径契约，根目录不得保留兼容副本。

接口清单以本节为功能基线，另在实现任务中记录“已实现、待实现、暂不需要、验证证据”。新增能力不得只修改 C++ 而遗漏 Python wrapper、错误处理、CI 或测试；也不得为了所谓完整性导出 Filament 未使用的模板、内部类型、Entity 管理器或逐资源底层 API。

### 11.3 资产加载

- GLB 文件 IO 和解析准备在 Asset Thread 完成。
- GPU 上传由 Render Thread 在受控阶段执行。
- 新场景全部资源 ready 后，在帧边界原子切换。
- 加载失败保留当前有效场景，首次加载失败则启动失败。
- 资产路径必须位于允许的资源根目录，禁止从配置执行任意脚本。

### 11.4 虚拟屏幕

MIP/LOD、EASU/Lanczos2/RCAS 属于格式分流前的 StereoRuntime 公共输出质量阶段；Vulkan Projection Composer 只负责处理后 OpenXR 眼图的屏幕投影。以下历史 Bridge 参数名仅作为迁移记录，不得作为新实现边界。

虚拟屏幕材质采样 Left/Right Eye 或 SBS 对应区域。每眼 View 必须选择正确 eye texture/UV，不通过 cross-eyed 旧兼容参数猜测顺序。

运行时从所有模式共用的 `XR Headset Model` 解析 2K/4K/8K 应用采样档位，默认型号为 Pico 4 / 4 Ultra。输入尺寸取立体生成完成、格式打包前的实际每眼纹理，并按最长边近似归入 1K/2K/4K；矩阵固定为 `1K→2K`、`2K→4K`、`4K→8K`，有效档位取 GUI 头显档位与矩阵推荐档位的较小值。非 16:9 只影响档位归类，源图宽高比保持不变。公共阶段先按基础缩放 LOD、GUI Min/Max LOD 与 MIP Bias 进行三线性 MIP 预滤，再执行 EASU 或 Lanczos2，最后按 GUI sharpness 执行可选 RCAS；本地、OpenXR 和流媒体随后消费同一语义的处理结果。OpenXR metadata 标记公共质量阶段已执行后，Projection Composer 必须使用中性 sampler，禁止二次 MIP/LOD、缩放或锐化。

实现必须保留以下运行时数据边界：

1. `headset_model` 使用 GUI 保存的稳定型号 key，不使用显示字符串反向猜测实际分辨率；型号表中的 `resolution_tier_k` 是应用采样策略字段。
2. 立体生成后的 eye image 尺寸用于判断输入是 1K、2K 还是 4K；`capture_size` 只保留为捕捉诊断，不得覆盖实际参加质量处理的纹理尺寸。输入尺寸变化必须使采样策略重新计算。
3. 采样计划至少记录 `input_tier_k`、`headset_tier_k`、`recommended_headset_tier_k`、`effective_tier_k`、`filter_scale`、`upscale_scale` 和 `mode`，并在这些值变化时记录一次诊断日志，禁止每帧刷屏。
4. 公共阶段先执行 MIP/LOD 预滤；`upscale_scale>1` 再选择 EASU → RCAS，`upscale_scale=1` 且 `filter_scale=1` 保持原生尺寸，`filter_scale>1` 选择 Lanczos2 → RCAS。SBS/TAB 等格式打包只能发生在该阶段之后；OpenXR 不再重复质量处理。所有路径保持颜色契约，不加入 tone mapping。
5. Bridge 不支持任一采样 ABI 时必须保留兼容路径并报告不可用状态；新 ABI 必须通过三平台远程构建后才可实机验收为 active。

#### 11.4.1 公共动态 MIP/LOD

桌面源图像持续变化时，每帧在 StereoRuntime 公共质量阶段直接处理 CUDA 眼图。RTMP 与本地路径没有最终投影的隐式纹理导数，因此统一请求值为 `max(log2(源尺寸/目标尺寸), Max LOD) + MIP Bias`，再限制在 `Min LOD` 与 `Max LOD` 之间；这使 Max LOD 在同尺寸输入时仍可直接控制过滤强度。相邻层使用三线性混合并恢复到原眼图尺寸，之后才进入 EASU/Lanczos2 和 RCAS。该结果由本地、OpenXR 与流媒体共同消费，SBS/TAB 打包前不得再次生成模式专属版本。

- 正常路径不得把眼图回读 CPU；PyTorch/Triton 操作必须停留在当前 GPU device。
- 运行时诊断必须记录最终 `output_quality_mip_lod` 和实际 backend，便于确认热更新已生效。
- OpenXR metadata 中 `output_quality_applied=1` 时，Vulkan Projection sampler 必须切换为 `minLod=0`、`maxLod=0`、`mipLodBias=0`、`RCAS=0`，只完成投影呈现。
- 正常运行不得为视觉诊断执行 GPU readback 或逐帧 PNG 写盘；画质回归使用离线 artifact。

屏幕变换、距离、曲率和可见性由 `SceneFrameState` 提供。交互状态与渲染资源分离，手柄拖动只更新下一帧 transform snapshot。

手柄状态同样由 Python OpenXR Presenter 管理，Bridge 只消费每帧快照。Presenter 必须分别维护左右手 Grip/Aim 有效性、上一帧姿态和最后移动时间；Grip 跟踪无效时立即隐藏对应模型和激光，连续 5 秒无位置或方向变化时自动隐藏，恢复移动后重新显示。激光使用 Aim 负 Z、旧工程标定偏移和稳定滤波，在共享 Filament Scene 的 Projection Layer 中绘制，不得提交为 Quad Layer；视觉契约为两张交叉锥形面和沿射线方向流动的蓝至红循环渐变。模型显隐、激光显隐、按键动画和姿态更新必须逐手独立；按键动画使用 GLB `_value/_min/_max` 节点、输入平滑和四元数旋转插值，任一可选 ABI 缺失不得阻断基础控制器 GLB 加载。`_value/_min/_max` 是 WebXR Input Profiles 的控制器变换契约，不得因节点本身不可渲染而丢弃；Bridge 应先枚举资产节点，再以打包 GLB 的完整节点名集合执行 `getFirstEntityByName` 回退，并逐型号测试所有完整三元组。摇杆、触控板和 thumbrest 的 touch 状态必须从 OpenXR action 传入动画层；为保持稳定 C ABI，可使用 `button_mask` 的版本化空闲位承载布尔状态，但不得把 touch 错当成 click。

### 11.5 相机和裁剪面

每帧使用 OpenXR `pose` 和 `fov` 更新 Filament Camera。Near/Far Clip 来自场景配置，并满足 `far > near`。大型 GLB 场景必须通过实际包围盒和头显视觉验证，不能依赖固定 100 m 默认值。

---

## 12. 异步光效系统

Effects Graph 使用 Vulkan Compute 处理最近完成的 screen image：

```text
Downsample -> Horizontal Blur -> Vertical Blur -> Glow
           -> Color Reduction -> Reflection Mask Composite
```

预计算掩码在场景或尺寸变化时更新，不得每帧从 CPU 生成。颜色归约结果若仅供 GPU 材质使用，不回读 CPU。

效果 slot 发布采用 timeline + 原子索引。SceneRenderer 使用最新 ready slot；效果缺帧或失败时继续使用上一结果。Graphics Queue 不等待与当前 screen frame 同 ID 的效果任务。

---

## 13. OpenXR Presenter

### 13.1 职责

`OpenXrPresenter` 负责 XrInstance、System、Session、Space、Action、Swapchain、事件和 Composition Layer。它不执行深度推理或立体算法。

Presenter 线程同时拥有 Vulkan context、Filament Engine/Scene/资源和输出图像环。Capture、Inference 与文件读取线程不得直接调用 Filament C ABI，也不得操作 Presenter 的 Vulkan 资源。运行时结果通过有界命令队列交给 Presenter；Presenter 在 OpenXR 帧边界消费命令，完成 CUDA/Vulkan 图像导入、外部同步、屏幕材质绑定和 Projection Layer 提交。队列只保留有限的新帧，覆盖或关闭时释放对应输出槽位，避免跨线程释放仍被 Filament 采样的图像。

### 13.1.1 Worker 与 Presenter 边界

Presenter 线程独占 OpenXR session、Projection/Quad swapchain acquire/release、CompositionBuilder 和 `xrEndFrame`。虚拟屏幕 worker 只生产最新 Vulkan 源图像；效果/UI worker 只生产 Glow、FPS、指南、两处 2D 光圈和键盘纹理。Filament worker 只渲染环境/手柄 GLB。所有 worker 通过有界 ring、timeline semaphore 和命令队列交付，禁止后台线程直接调用 OpenXR API、Filament C ABI 或修改 Presenter-owned Vulkan 资源。

### 13.2 Vulkan Session 创建

必须通过 `xrGetVulkanGraphicsRequirements2KHR`、`xrCreateVulkanInstanceKHR`/等效规范路径和 `XrGraphicsBindingVulkan2KHR` 建立 Session。Physical Device 选择服从 OpenXR Runtime 要求。

禁止创建 D3D11 Session 后导入 Vulkan 结果。

### 13.3 Frame Loop

```text
xrPollEvent
-> xrWaitFrame
-> xrBeginFrame
-> xrLocateViews / xrSyncActions
-> acquire and wait swapchain images
-> update scene snapshots
-> record and submit scene render
-> release swapchain images
-> xrEndFrame
```

即使没有新 capture frame，也继续更新 pose、controller 和场景，并复用 last-good screen image。只有 Runtime 指示 `shouldRender=false` 时跳过图像渲染，但仍按 OpenXR 规则结束本帧。

### 13.4 Swapchain

- 格式从 Runtime 支持列表按色彩语义选择。
- 每眼 swapchain image 建立非拥有 `GpuImageView`。
- acquire/wait/release 必须成对，异常路径也要归还已 acquire 图像。
- swapchain 重建前等待引用它的 Frame Context 完成。
- 应用不得销毁 OpenXR 创建的 `VkImage`。

### 13.5 Layers

主房间、手柄、虚拟屏幕、激光和默认 Glow 使用 Vulkan 合成后的 Projection Layer。FPS 面板、操作指南、固定在虚拟屏幕上的光圈、固定在虚拟键盘上的光圈、虚拟键盘和其它独立 2D UI 使用 Quad Layer。Layer 数量和顺序由单一 `CompositionBuilder` 生成，模块不能各自调用 `xrEndFrame`。

FPS 面板、操作指南和其它工具 Quad Layer 必须区分“内容更新”和“姿态提交”：纹理内容由 Python 策略层按旧工程的 content key 缓存，静态指南只生成/上传一次，FPS/延迟等统计按约 1 秒快照更新；没有内容变化时，Presenter 每帧只更新 layer pose 和 composition 结构，不得重新栅格化字体、map staging 或 acquire/wait/release 图像。工具纹理的更新仍由 Presenter 命令队列线程执行，不能由后台线程直接操作 Vulkan 或 Filament 资源。

手柄激光属于与控制器同坐标系的 3D 场景几何体，随左右眼 View 分别投影；激光 View 必须使用与主控制器共享 GLB 资源但 Renderable 独立的外壳实例作为仅写深度的遮挡体，使外壳遮挡光束根部，同时控制器颜色仍仅由 HDR 主 View 的原始 PBR 实例输出。两实例必须同步姿态和动画，材质绑定在加载后保持不变。它不是平面 UI，不得占用 OpenXR Quad Layer。

---

## 14. 非 XR 输出

### 14.1 Vulkan Window

桌面预览默认使用独立 Vulkan surface/swapchain。兼容模式创建独立 OpenGL window/context；它不复用 OpenXR frame loop，也不与 Vulkan surface 共同存在。

### 14.2 Headless/Encoder

Headless 输出将 Left/Right/SBS Vulkan image 提供给编码适配器。支持 external-memory 的编码器直接导入；否则允许明确的 GPU 格式转换和 GPU copy。

文件截图、离线测试可以回读 CPU，但必须使用独立命令和日志标签，不能进入实时执行路径。

### 14.3 高级网络推流边界

网络推流使用统一的 `RTMP Streamer` 运行模式（GUI 文案为“高级网络推流”）；历史 GPU 推流名称只在配置迁移层归一化。它与本地/XR 输出共享 StereoRuntime 的公共画质阶段和 latest-frame 语义，但拥有独立的 `NetworkStreamSession`，负责视频编码、音频、MediaMTX/信令、校准统计、重连和生命周期。编码器只接收压缩包或明确的 GPU surface 描述，不接收 CPU/IPC 原始 4K 眼图。

Windows/Intel 的工程契约如下：

1. Vulkan 眼图输出到 D3D11 共享 BGRA8 SBS surface，再由同一 Adapter 转换为 NV12 并交给 oneVPL；共享资源必须携带 Win32 memory handle、格式/尺寸/分配大小、producer-ready 同步句柄和 Adapter LUID。
2. D3D11 texture 由消费侧创建并以 Vulkan import 方向接入；禁止把 `OPAQUE_WIN32` memory handle 直接当作 D3D11 texture handle 传给 `OpenSharedResource1`。
3. 缺少 LUID、格式不为 BGRA8、同步句柄无效或能力探测失败时，禁用该桥并回退到已验证的 QSV/D3D11 或 FFmpeg 路径；不得终止输出线程。
4. 每次会话必须记录 `gpu_to_cpu`、`zero_copy`、`gpu_copy_count`、实际编码后端和回退原因。Intel 当前实现的 Vulkan→D3D11→NV12 路径允许明确的 GPU copy；未完成的 fused shader 不得宣称 zero-copy。

OpenGL/CUDA 兼容编码路径同样不属于 XR OpenGL Fallback：GPU 眼图路径必须记录实际 GPU copy 次数，CPU 捕获帧直接 host upload，禁止 CPU→OpenGL→CPU 往返；OpenGL interop 错误只允许触发一次会话级 handoff 到稳定 FFmpeg 编码器。编码失败不能改变核心 Vulkan/OpenXR 图形后端。

### 14.4 捕获与推理原生边界

Windows Desktop Duplication 可把借用的 D3D11 frame 同时提供给 OpenVINO GPU RemoteTensor；若原生能力不完整，受控 staging BGR readback 的诊断值为 `gpu_to_cpu=True zero_copy=False gpu_copy_count=1`，不能伪报零拷贝。OpenVINO、oneVPL、D3D11 surface 和 Vulkan 编码桥均使用同一 Adapter LUID 校验，并通过独立 capability probe 决定是否启用。

---

## 15. 配置系统

### 15.1 RuntimeConfig

核心运行时只接受规范化配置，不解析 GUI 文案或旧 `settings.yaml` 字段：

```python
@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    schema_version: int
    graphics: GraphicsConfig
    capture: CaptureConfig
    inference: InferenceConfig
    stereo: StereoConfig
    scene: SceneConfig
    openxr: OpenXrConfig
    output: OutputConfig
    telemetry: TelemetryConfig
```

配置文件采用明确 schema version。GUI 负责把用户选择转换成新 schema；一次性迁移工具负责读取旧 YAML 并输出新配置，核心运行时不包含旧字段 alias。

### 15.2 快照与热更新

`RuntimeConfigSnapshot` 不可变并带单调 `version`。Control Thread 校验差异后生成：

| 变更类别 | 示例 | 动作 |
|----------|------|------|
| Uniform Update | depth strength、convergence、effect strength | 下一帧生效 |
| Temporal Reset | depth response、temporal enabled | 下一帧清 history |
| Graph Rebuild | render scale、hole-fill quality tier | drain 有关 Frame Context 并重建图 |
| Adapter Rebuild | capture source、model、inference backend | 停止对应 adapter 后重建 |
| Session Rebuild | XR swapchain format、关键 XR 能力 | 重建 Session/Swapchain |
| Process Restart | Graphics Backend、Vulkan device、核心 feature set | 返回 restart-required；新进程重新 probe |

一帧只能使用一个完整配置版本。模块不得在帧中途读取可变全局配置。

### 15.3 配置验证

启动前验证平台、GPU、模型、格式、尺寸、场景路径和输出组合。未知字段默认报错；可选扩展字段必须位于命名 namespace，避免拼写错误被忽略。

---

## 16. 控制面与 GUI 边界

GUI 可继续使用 Flet/Python，但其职责收敛为：设备探测结果展示、配置编辑、启动停止、状态和日志展示、错误报告导出。

控制协议至少支持：

```text
hello(protocol_version)
probe()
start(config)
stop(reason)
apply_config(snapshot)
load_scene(path)
get_status()
subscribe_events()
```

所有命令返回 request ID 和结构化结果。停止命令具有最高优先级。GUI 进程退出时，Runtime 根据启动模式决定自动停止或继续 headless，行为必须显式配置。

独立进程模式下GUI不读取Runtime内部对象，也不解析自由格式日志来判断状态；单进程模式通过稳定的Python控制接口访问状态。两种模式都不得取得Vulkan、OpenXR或Filament原生对象的所有权。

---

## 17. 错误模型与恢复

### 17.1 Status

```python
class ErrorDomain(StrEnum):
    CONFIG = "config"
    PLATFORM = "platform"
    CAPTURE = "capture"
    GRAPHICS = "graphics"
    VULKAN = "vulkan"
    OPENGL = "opengl"
    INFERENCE = "inference"
    STEREO = "stereo"
    FILAMENT = "filament"
    OPENXR = "openxr"
    OUTPUT = "output"
    INTERNAL = "internal"

@dataclass(frozen=True, slots=True)
class Status:
    domain: ErrorDomain
    code: int
    severity: Severity
    message: str
    context: str
    retryable: bool
```

模块边界捕获并转换异常为`Status`或领域异常；主循环不得泄漏未处理异常。GPU资源必须显式`close()`，清理方法应幂等且不得抛出未处理异常。

### 17.2 恢复策略

| 故障 | 策略 |
|------|------|
| Capture source closed | 进入等待或停止，取决于配置 |
| 单帧推理失败 | 丢帧并计数，连续失败后重建 Adapter |
| External import 不可用 | 平台允许时选择一次 GPU copy，启动报告固定记录 |
| OpenXR Session loss | 停止 XR 提交，重建 Session 和 swapchain |
| Scene asset reload 失败 | 保留 last-good scene |
| Shader/Pipeline 失败 | 启动失败或保持旧 pipeline，不带病切换 |
| Vulkan Device lost | 收集 fault 信息并终止 Device，不原地继续提交 |
| Vulkan 启动探测失败 | 返回 OpenGL Fallback 建议和原因，由启动器受控重启 |
| OpenGL Fallback 初始化失败 | 输出缺失版本、扩展或 OpenXR binding，并终止启动 |

任何降级路径必须在初始化能力选择时确定。运行中不得从 Vulkan 静默切换到 OpenGL；OpenGL 只能通过受控重启进入。不得切换到旧图形桥接 API 或 CPU 实时渲染。

---

## 18. 日志、指标与诊断

### 18.1 结构化日志

日志事件至少包含 timestamp、severity、domain、event name、thread、frame ID、config version 和 key/value context。控制台可渲染为文本，文件保存 JSON Lines。

密钥、访问令牌、完整用户文件内容和系统私密路径不得写入报告。配置快照输出前执行字段级脱敏。

### 18.2 GPU Timing

Vulkan 阶段使用 timestamp query；CUDA/HIP 使用后端 event；OpenXR wait/submit 使用 CPU monotonic clock。以下指标必须分开：

```text
capture_fps
inference_fps
stereo_fps
scene_submit_fps
present_fps
capture_to_present_ms
gpu_preprocess_ms
gpu_inference_ms
gpu_stereo_ms
gpu_scene_ms
gpu_effects_ms
xr_wait_ms
xr_submit_ms
capture_overwrite_count
frames_in_flight
```

不得把 `xrWaitFrame` 或 present interval 计入 Stereo Compute GPU 耗时，也不得用 CPU enqueue 时长代表 GPU 完成时长。

### 18.3 Diagnostic Bundle

错误报告包含版本、commit、平台、GPU、驱动、Vulkan/OpenXR capability、配置摘要、最近日志、GPU timing、validation 摘要和 device fault 信息。默认只生成本地文件，上传必须经过用户确认。

---

## 19. 测试设计

### 19.1 单元测试

- 配置 schema、差异分类和状态机转换。
- Render Size、Parallax Budget 和视差参数。
- Frame slot、latest-frame 覆盖和 timeline 递增。
- Resource ownership、RAII move 和异常清理。
- glTF asset handle、scene switch 和 animation time。
- OpenXR event 到状态转换的纯逻辑部分。

### 19.2 Vulkan GPU 测试

- Headless Vulkan device 上执行每个 Compute Pass。
- Validation Layer 下检查 layout、access、queue ownership 和 descriptor lifetime。
- 比较 CPU reference/golden fixture。
- 覆盖 720p、1080p、1440p、4K、超宽和竖屏。
- 覆盖 dedicated compute queue 与 single universal queue。
- 运行资源重建、尺寸切换和 pipeline cache 冷热启动。

### 19.3 推理互操作测试

- external memory 一次导入、多帧复用。
- external semaphore wait/signal 次序。
- 重复初始化、失败清理和 handle 泄漏。
- 输入输出无 CPU readback 的工具级证明。
- GPU copy fallback 的次数和方向符合平台规格。

### 19.4 Filament 测试

- 静态 GLB、动画、PBR texture、透明材质和大场景。
- Scene load failure 保留 last-good scene。
- 每眼 camera pose/FOV、near/far 和 screen UV 正确。
- Vulkan render target 输出非空且颜色空间正确。

### 19.5 OpenXR 测试

- Mock/头less 可覆盖的 lifecycle 与 swapchain 顺序测试。
- Windows 至少两个真实 OpenXR Runtime 实机测试。
- Session loss、headset disconnect/reconnect 和 swapchain rebuild。
- 无新 screen frame 时 pose/controller 仍按显示帧率更新。
- Projection/Quad Layer 数量、顺序和 image release 正确。

### 19.6 OpenGL Fallback 测试

- Vulkan 不可用、用户显式兼容模式和 macOS MoltenVK 失败三类触发条件可复现。
- Windows/Linux 原生 OpenGL 和 macOS OpenGL 4.1 capability report 正确。
- 同一会话不加载 Vulkan Device、D3D11、WGL/CUDA-GL bridge 或旧 viewer。
- 基本场景、虚拟屏幕、Left/Right Eye 或 SBS 呈现正确。
- 不支持的 Compute/Glow/Reflection 功能在 UI 和日志中明确禁用。
- GPU stereo input 不可用时明确失败，不发生 CPU 实时像素回读。

### 19.7 长稳测试

正式候选版本至少运行：

| 场景 | 时长 | 通过标准 |
|------|-----:|----------|
| 1080p/90 Hz XR | 2 小时 | 无 validation error、无队列增长 |
| 4K scale XR | 1 小时 | 内存稳定、P95 延迟达标 |
| 连续 resize/source switch | 500 次 | 无泄漏、无 device lost |
| Scene hot swap | 200 次 | 无悬空资源、last-good 生效 |
| Session reconnect | 100 次 | Session 可恢复或明确失败 |

---

## 20. CI 与构建交付

### 20.1 CI Pipeline

```text
format/lint
-> Python import/compile checks
-> pytest unit tests
-> shader compile + reflection validation
-> Filament Bridge build matrix when bridge source changes
-> headless Vulkan GPU tests where runner supports
-> package
-> dependency manifest and SHA verification
```

真实 GPU/OpenXR 测试由专用硬件 runner 执行，不用普通 CI 成功替代实机验收。

### 20.1.1 Filament Bridge 三平台构建策略

Filament Bridge 的正式构建由 GitHub Actions 负责。三平台必须使用对应平台 runner 编译和链接 Filament 1.76 及其运行库，构建结果通过 Actions Artifact 或 GitHub Release 交付给 Python 运行时。

```text
native/filament/bridge source change
    -> Windows runner -> filament_bridge.dll
    -> Linux runner   -> libfilament_bridge.so
    -> macOS runner   -> libfilament_bridge.dylib
    -> ABI/link/runtime checks
    -> upload artifact or release asset
    -> download into src/desktop2stereo/xr_viewer/native/<platform>/
```

必须遵守以下规则：

- Bridge 源码、C ABI、Filament 版本或 CMake/平台构建配置变更时，CI 必须执行三平台构建矩阵；只修改 Python、GUI、配置或资源时不重新编译 Bridge。
- CI 产物必须记录平台、架构、Filament 版本、Git commit、构建类型和 SHA-256，并随产物提供依赖清单。
- 发布包使用 CI 产物，不把本地 `build/`、临时 SDK 解压目录或未验证的本地 DLL/SO/DYLIB 作为正式交付物。
- 本地开发默认下载与当前平台匹配的 CI 产物。只有在 CI 故障、调试原生崩溃或验证未提交的 Bridge 修改时，才允许本地编译。
- 本地编译结果只能用于诊断和开发验证；合并和发布前仍必须通过三平台 CI，且本地编译不得替代 Linux/macOS 平台验证。
- Python 通过 `ctypes` 加载 `src/desktop2stereo/xr_viewer/native/` 下对应平台的 Bridge。加载前检查文件存在性、架构、依赖库可解析性和 ABI 版本；检查失败必须在 capability report 中明确报告。
- Bridge 二进制与 Filament 运行库必须作为同一版本构建包管理，禁止混用不同 Filament 版本的库文件。

### 20.2 构建产物

发布包至少包含：

```text
Python source packages and entrypoints
platform Python environment lock/requirements
Filament Bridge and Filament runtime libraries
OpenGL compatibility backend when enabled
OpenXR loader where platform packaging requires
enabled Python inference packages/providers
compiled SPIR-V and shader manifest
default assets and config schema
dependency/version manifest
licenses
```

不得包含旧OpenGL/D3D11 viewer DLL、Panda3D runtime或仅供开发的build directory。发布包可包含新版 Filament DLL Bridge，以及启用网络/Intel 平台时对应的、带 manifest/SHA-256 的 D3D11/OpenVINO/oneVPL/Vulkan 编码桥；这些可选桥不得被当作 XR 图形组件，也不得缺少能力探针和明确回退。启用 OpenGL Fallback 时包含 Python `viewer.opengl_renderer` 和 `xr_viewer.core_openxr_opengl` 模块及其明确依赖。

推荐的运行时目录：

```text
src/desktop2stereo/xr_viewer/native/
├── windows/filament_bridge.dll
├── linux/libfilament_bridge.so
└── macos/libfilament_bridge.dylib
```

上述目录中的文件来自对应 GitHub Actions 构建产物或 Release 资产。源码修改后先提交并等待 CI 生成新产物，再更新本地运行目录；不要求每次 Python 代码修改都重新编译 Filament。

### 20.3 启动探测

`python src/tools/probe.py`必须可独立输出JSON capability report，包括GPU、Vulkan、OpenGL、MoltenVK、external memory、OpenXR Graphics Binding、swapchain format、Filament Bridge初始化和推理Provider可用性。GUI只根据该报告启用可选项。

---

## 21. 性能与资源预算

### 21.1 关键路径预算

90 Hz 目标遵循 `docs/01`：应用 GPU 关键路径不高于 10 ms，P95/P99 单独报告。工程层必须能分解 preprocess、inference、stereo、scene 和 submit。

### 21.2 内存预算

资源预算按格式和 Frame Context 数量计算并在启动时输出。4K 模式必须覆盖 capture、linear RGB、depth、disparity、mask、双眼、temporal、effects 和 swapchain 的峰值。

当预计使用量超过可用 device-local budget 的安全比例时，启动失败或要求降低 render tier；不得等到分配失败后静默降质。

### 21.3 CPU 预算

稳态帧中 CPU 只处理命令构建、状态快照和提交。禁止逐帧像素循环、逐帧大对象分配和自由格式日志拼接。Telemetry 采样使用预分配 ring。

---

## 22. 安全与健壮性

- GLB、配置和 shader cache 输入必须验证尺寸上限和路径范围。
- 不从资产或配置执行脚本、命令或动态代码。
- 本地控制通道限制为当前用户，并验证 protocol/schema version。
- 外部内存句柄只在受信进程边界内传递，并严格遵循所有权规则。
- 下载模型和 SDK 必须验证摘要；运行时不自动执行未验证二进制。
- Validation/diagnostic 信息不得泄露密钥或完整私密路径。

---

## 23. 实施顺序

### Phase 1：工程骨架与 Vulkan/OpenXR

1. 建立以`src/`为产品发布边界的Python package和平台依赖锁定文件。
2. 在`src/tools/probe.py`、`src/desktop2stereo/viewer/vulkan_renderer.py`和`src/desktop2stereo/xr_viewer/core_openxr_vulkan.py`中实现能力探测、`VulkanContext`、`GpuAllocator`和`GpuScheduler`。
3. 使用Python OpenXR代码建立Vulkan Session和每眼清屏闭环。
4. 接入Validation、timestamp、结构化日志和显式资源清理测试。

完成标准：真实头显稳定显示 Vulkan Projection Layer，运行 30 分钟无 validation error。

### Phase 2：Filament Vulkan 场景

1. 重新实现唯一的Filament DLL Bridge窄C ABI，保留已验证的GLB/Animator能力。
2. 在Python `FilamentSceneRenderer`中实现Bridge加载、handle生命周期、每眼Camera、虚拟屏幕和手柄状态更新。
3. 验证Vulkan Backend直接swapchain target；不支持时由Bridge执行一次Vulkan内copy。

完成标准：正式房间/手柄在两个OpenXR Runtime正确显示；项目中除新版Filament DLL Bridge外没有其他自有原生运行代码。

### Phase 2B：OpenGL Fallback

1. 实现Python `GraphicsBackend`协议以及`viewer.opengl_renderer`、`xr_viewer.core_openxr_opengl`隔离模块。
2. 使用Python OpenGL API实现Windows/Linux与macOS OpenGL 4.1基本渲染。
3. 使用Python OpenXR代码实现OpenGL Graphics Binding、窗口输出和受限功能报告。
4. 验证受控重启、GPU stereo input 和禁止 CPU 实时回读。

完成标准：三类 Fallback 触发条件通过，兼容会话不加载 Vulkan/D3D11/WGL 旧桥接，基本场景和虚拟屏幕可用。

### Phase 3：Stereo Compute Graph

1. 建立 shader manifest、Graph compiler 和固定资源池。
2. 实现 preprocess、depth postprocess、parallax、warp、fill、temporal、pack。
3. 完成 CPU reference、golden 和多尺寸 GPU 测试。

完成标准：`docs/01` 算法正确性和 Vulkan 正确性验收通过。

### Phase 4：厂商推理互操作

1. 原样迁入并验证现有WindowsCaptureCUDA、TensorRT和PyTorch CUDA Python实现，再接入Vulkan输出契约。
2. 原样迁入并验证现有WindowsCaptureROCm、PyTorch ROCm/MIGraphX Python实现，再接入external-memory或一次GPU copy路径。
3. DirectML与Apple Python Provider分平台验证；不得通过C++重写替代迁移。

完成标准：实时主路径无 CPU readback，句柄和同步长稳测试通过。

### Phase 5：异步光效、输出与控制面

1. 接入 Glow、颜色归约和墙面反射 Compute Graph。
2. 完成 Vulkan window/headless output。
3. 接入控制协议、GUI、配置快照和 diagnostic bundle。

完成标准：完整产品流程、热更新、故障恢复和 2 小时长稳通过。

### Phase 6：旧架构删除

保留并整理有效的Python Capture、Inference、调度和诊断代码；删除旧OpenGL/D3D11 viewer、Panda3D、WGL/CUDA-GL bridge、CPU实时fallback和历史兼容配置。旧Filament OpenGL-only Bridge由新版Vulkan/OpenGL Bridge取代。更新`src/desktop2stereo/main.py`、`src/desktop2stereo/main.bat`、依赖、CI和发布清单，保持当前启动方式并在启动阶段选择后端。

完成标准：发布包和运行依赖中不存在旧图形桥接后端；OpenGL只通过Python `GraphicsBackend`、`viewer.opengl_renderer`和`xr_viewer.core_openxr_opengl`出现；Filament Bridge是核心 XR 场景唯一自有原生组件，网络/捕获/推理边界桥接必须有独立 manifest、能力探针和回退验收。

---

## 24. 完成定义

Vulkan 工程迁移只有同时满足以下条件才算完成：

1. `docs/01` 与本文所有必需验收项通过。
2. OpenXR Session、场景和交换链全部使用 Vulkan。
3. 默认主路径使用 Filament Vulkan Backend；OpenGL 只在符合条件的兼容会话中启用。
4. 立体合成和光效使用 Vulkan Compute。
5. 正式实时路径每帧 CPU 图像回读为零。
6. GPU 任务、资源池和队列全部有界。
7. NVIDIA 主路径为 external-memory 零拷贝；其他平台不超过规范允许的一次 GPU copy。
8. OpenXR 无新输入时仍保持 pose、controller 和 `xrEndFrame` 节奏。
9. Validation、长稳、Session 恢复和 Device Lost 诊断通过。
10. 发布包不包含旧OpenGL viewer、旧 D3D11 viewer 或 Panda3D；允许包含经过 manifest/探针校验的网络/捕获/推理 D3D11 桥。Python是正式运行时，禁止的是 CPU NumPy/PIL 逐帧像素往返。
11. Capture、Inference、Vulkan、OpenXR和核心 Output 编排均由Python源码实现；Filament DLL Bridge负责 XR 场景，网络/捕获/推理边界可使用经过 CI 交付的窄 ABI 原生桥接。

本文自Python Vulkan Runtime开发开始生效。后续工程实现、代码审查和发布验收均以本文和`docs/01`为唯一目标架构依据。

## 25. 全量符合性追踪

本文与 `docs/01` 的所有要求统一由 [`docs/requirements-matrix.md`](requirements-matrix.md) 追踪。矩阵按架构、捕捉、推理、Vulkan、Compute Graph、Filament、OpenXR、输出、配置、GUI、错误恢复、诊断、性能、测试、平台、CI 和安全领域登记要求，且每条要求必须关联代码映射和测试/实机验收记录。

日常开发使用 `src/tools/check_compliance.py` 检查矩阵结构；发布候选版本使用 `--strict`，只允许 `verified` 或 `accepted` 条目，并额外要求 pytest、三平台 Bridge CI 和专用 GPU/OpenXR 实机验收通过。
