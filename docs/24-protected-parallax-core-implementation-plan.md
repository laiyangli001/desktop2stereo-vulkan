# Desktop2Stereo 视差核心保护与授权解密实施计划

## 1. 目标与原则

在继续遵守第三方开源许可证、保留普通源代码可读性以及兼容 GitHub 更新和构建流程的前提下，保护 Desktop2Stereo 最核心的 3D 视差计算资产。

本方案只保护直接决定 3D 视差效果的核心公式、关键参数和相关实现，不加密 GUI、配置、普通业务逻辑、开源依赖或授权界面。

保护目标是提高逆向和破解成本，并使“跳过登录界面后直接运行”无法获得正常 3D 视差效果。由于算法最终仍需在用户设备上执行，客户端保护不能保证绝对不可破解；服务端授权、签名私钥、在线租约和设备密钥仍是最终安全边界。

## 2. 当前代码边界

当前视差逻辑主要分布在以下运行时路径：

- `stereo_runtime/parallax.py`：视差预算、深度响应、视差版本和关键参数表；
- `stereo_runtime/baseline_shift.py`：深度到水平位移的转换及基础 warp；
- `stereo_runtime/synthesis.py`：多种后端的立体合成和 warp 调度；
- `stereo_runtime/warp_composite_triton.py`、Vulkan/Metal 路径：实际像素重映射和 GPU 执行；
- `stereo_runtime/adapter.py`：GUI 配置到运行时视差参数的适配。

实施前必须先完成依赖图和数值回归基线，明确哪些逻辑属于项目自有的专有视差资产，哪些属于第三方或已公开的通用实现，避免把第三方代码整体加密或违反许可证义务。

## 3. 核心保护架构

### 3.1 公开适配层与私有核心分离

保留公开的配置结构、类型定义、调用协议和调试接口。公开适配层不得保留完整视差公式、关键常量组合或可直接复原核心算法的备用实现。

公开层提供稳定接口，例如：

```python
core = load_protected_parallax_core(auth_context)
budget = core.resolve_budget(render_width, render_height, preset, convergence)
shift = core.compute_shift(depth, budget, runtime_parameters)
```

所有 CUDA、Triton、Vulkan、Metal 和 CPU 路径必须通过统一核心接口获取视差结果，不能存在绕过保护核心的隐式 fallback。

### 3.2 原生保护模块

新增跨平台原生保护模块：

- Windows：DLL；
- Linux：SO；
- macOS：DYLIB。

原生模块负责：

- 校验加密资源版本、产品标识和 SHA-256；
- 校验服务端授权签名；
- 使用 AES-256-GCM 解密算法资源；
- 拒绝被替换、截断或篡改的资源；
- 仅暴露必要的受控 C ABI；
- 不携带服务端私钥，不输出算法明文到磁盘。

加密资源采用版本化封装，至少包含：核心版本、资源标识、资源哈希、nonce、ciphertext 和 authentication tag。由于当前项目仓库是公开仓库，完整核心资产统一放入单独的私有核心仓库。

### 3.3 私有核心仓库与本地临时目录

采用双仓库方案：

- 公开仓库 `laiyangli001/desktop2stereo-vulkan`：维护开源代码、公开适配层、授权接口、构建接口和文档；
- 私有核心仓库 `laiyangli001/desktop2stereo-vulkan-core`：维护完整视差公式、加密资源、密钥文件、资源清单和核心构建文件。

项目管理人员通过私有核心仓库的 GitHub 成员权限下载完整文件。公式和密钥等关键资产必须在私有仓库中进行 Git 追踪，不能只保存在个人电脑上。

在私有核心仓库完成压缩加密前，视差公式明文暂时保存在：

```text
private/parallax-core/plaintext/
```

约定如下：

- 该目录仅用于私有核心仓库本地整理、压缩、加密和测试明文视差公式；
- 公式明文、压缩前文件和可直接使用的原始密钥只允许提交到私有核心仓库；
- 公开仓库不得出现 `private/parallax-core/` 下的任何核心资产；
- 完成加密后，将加密公式包和加密密钥包放入私有核心仓库的 `private/parallax-core/encrypted/`；
- 计划中的原生加载器只读取加密资源，不直接读取该明文暂存目录；
- 加密完成后应删除或转移不必要的本地明文副本，并分别检查两个仓库的 Git 状态和发布包清单，确认没有交叉泄露。

私有核心仓库中的完整关键文件必须纳入 GitHub 追踪，允许项目管理人员下载，包括公式、密钥、加密资源和构建文件。至少包括：

- `private/parallax-core/` 下的视差公式文件；
- `private/parallax-core/encrypted/` 下的加密核心资源；
- 核心授权密钥、资源加密密钥和密钥封装文件；
- 加密后的公式压缩包或核心资源包；
- 加密资源清单、核心版本和资源 SHA-256；
- 原生保护模块源码、构建配置和 ABI 定义；
- 发布包中使用的资源路径和版本映射。

每次更新公式、加密核心或密钥文件时，必须同时更新资源清单和哈希，并通过私有仓库 GitHub Actions 校验资源存在、哈希匹配、版本一致和发布包包含该资源。私有仓库访问权限仅授予项目管理人员和授权维护者，并启用分支保护和审计。

推荐的本地工作结构为：

```text
private/parallax-core/
├── plaintext/       # plaintext formula files, private repository only
├── work/            # compression/encryption workspace, private repository only
├── encrypted/       # tracked encrypted formula and key resources
└── manifest/        # tracked versions, hashes, and release mappings
```

## 4. 授权与密钥流程

继续兼容现有在线、离线和永久授权模式。

### 4.1 在线授权

1. `--runtime` 子进程重新执行 `validate_saved_authentication()`。
2. 在线许可证启动 `RuntimeLease`，首次申请运行租约。
3. 服务端签发绑定用户、license、设备、产品、核心版本和资源哈希的核心授权包。
4. 原生模块验证授权包并获得短期核心解密授权。
5. 运行期间持续发送租约心跳。
6. 租约失效后停止核心视差输出，并关闭或降级 3D 运行路径。

授权包必须包含或能够验证：

- 用户/会话标识；
- license ID；
- 设备指纹；
- 产品标识；
- 核心版本；
- 加密资源 SHA-256；
- 生效时间和过期时间；
- 授权模式；
- 服务端签名。

### 4.2 离线授权

离线授权继续使用现有 ES256 签名凭证，并增加核心版本、资源哈希和核心解密授权信息。

客户端必须验证：

- ES256 签名和公钥 ID；
- 产品标识和 license ID；
- 设备指纹；
- 生效时间和到期时间；
- 核心版本；
- 加密资源哈希。

离线解密密钥不得以明文直接写入 JWS、配置文件或 Python 源码。应使用设备安全存储中的设备密钥进行包装。设备私钥通过现有安全存储抽象保存，不进入普通配置文件。

### 4.3 永久授权

永久授权仍必须依赖合法服务端签名凭证，不在客户端内置永久解密密钥。凭证仍需绑定设备、产品、核心版本和资源哈希，并支持服务端吊销核心版本或资源版本。

## 5. 服务端接口扩展

在 `desktop2stereo-site` 授权服务中扩展授权响应，保持现有登录、许可证状态、在线心跳和离线签发接口兼容。

核心授权包至少应支持以下字段：

```json
{
  "core_id": "parallax-core",
  "core_version": 1,
  "resource_sha256": "...",
  "grant_expires_at": 0,
  "wrapped_core_key": "...",
  "signature": "..."
}
```

服务端必须：

- 仅向有效许可证签发核心授权；
- 校验产品、授权模式、设备和许可证状态；
- 在线模式限制授权包和租约有效期；
- 离线模式限制签发周期；
- 支持核心版本和资源版本吊销；
- 记录必要的授权审计信息；
- 不保存或记录客户端设备私钥。

服务端签名私钥只允许存在于服务端安全环境，客户端只内置对应公钥。

## 6. 启动链路与失败行为

继续保持现有两层启动门禁：

- GUI 启动前执行 `require_authentication()`；
- 运行子进程 `--runtime` 再次执行 `validate_saved_authentication()`；
- 在线许可证启动 `RuntimeLease`；
- 视差核心加载前完成原生模块授权校验；
- 未完成核心解密时不得进入正常 3D 合成路径。

因此，关闭登录窗口、直接启动 `Desktop2Stereo.exe`、直接启动 `main.py --runtime`、删除本地登录缓存或替换加密资源，都不能获得正常 3D 视差。

视差核心加载失败、授权失效、签名错误、资源篡改、设备不匹配或原生模块缺失时：

- 静默禁用 3D 视差或降级为 2D；
- 状态栏使用红字显示 `3D 功能未开启`；
- 写入通用诊断代码；
- 不在界面或日志中输出密钥、完整授权凭证、算法明文和详细校验过程。

建议诊断代码包括：

```text
CORE_AUTH_REQUIRED
CORE_RESOURCE_INVALID
CORE_LICENSE_EXPIRED
CORE_DEVICE_MISMATCH
CORE_DECRYPT_FAILED
CORE_NATIVE_MODULE_MISSING
```

## 7. 构建、发布与开源合规

- 普通 Python 源码、GUI、配置和开源依赖继续在公开 GitHub 仓库维护；
- 完整视差公式、加密资源、密钥文件和核心构建资产统一在私有核心仓库维护；
- 保留并更新全部第三方许可证和版权声明；
- 算法明文、可直接使用的私钥、资源生成原始密钥和生产授权私钥只允许进入私有核心仓库；
- 公开仓库禁止出现 `private/parallax-core/` 下的任何核心资产；
- 私有核心仓库允许追踪完整公式、加密资源、密钥文件、资源清单、版本信息和哈希；
- 私有核心仓库的项目管理人员必须拥有仓库访问权限，并通过分支保护和审计避免误删或遗漏；
- 原生保护模块纳入 Windows、Linux、macOS GitHub Actions 构建；
- 构建时检查资源哈希、原生导出符号和发布包完整性；
- 保持现有 GitHub 构建、二进制回传和本地升级流程；
- 不修改用户本地 `settings.yaml` 等运行时配置。

私有核心仓库中的完整资源由受控流程生成和更新。私有 GitHub Actions 使用私有仓库权限获取公式、加密资源和密钥文件，构建完成的发布包只向授权人员或发布渠道提供。公开仓库的 CI 不得下载或缓存私有核心仓库中的敏感文件。用户完成本地压缩加密后，应先验证公式、密钥和密文资源的哈希及发布包内容，再提交到私有核心仓库。

## 8. 测试计划

### 授权测试

- 无登录状态启动 GUI；
- 无登录状态直接启动 `--runtime`；
- 登录状态过期；
- license 被撤销；
- 设备指纹不匹配；
- 在线租约申请失败；
- 在线心跳失败并超过租约有效期；
- 离线凭证过期；
- 离线凭证签名错误；
- 核心版本不匹配；
- 资源哈希不匹配。

### 核心保护测试

- 加密资源正常解密并产生与当前实现一致的视差；
- 资源被替换或截断；
- nonce、tag、ciphertext 被修改；
- 解密密钥错误；
- 原生模块缺失或 ABI 不匹配；
- 公开源码中不存在完整视差公式和关键私有参数；
- 所有运行后端均无法绕过保护核心。

### 回归测试

- 2D 降级行为和红色状态栏提示；
- CUDA、Triton、Vulkan、Metal 和 CPU 路径；
- OpenXR、本地查看器和流媒体模式；
- Windows、Linux 和 macOS 构建；
- GitHub 发布包启动；
- 现有登录、语言切换和设置功能。

## 9. 验收标准

1. 普通开源代码仍可阅读、升级和构建。
2. 视差核心明文不出现在公开源码和最终发布包中。
3. 没有合法授权时，运行核心无法获得正常 3D 视差。
4. 在线授权失效后，运行中的 3D 功能会停止或降级。
5. 离线授权只有在签名、设备、版本和有效期全部正确时才能解密。
6. 核心资源被篡改时不会正常产生 3D 画面。
7. 用户只看到“3D 功能未开启”，不会看到保护实现细节。
8. 不影响现有 GitHub 更新、构建和跨平台运行流程。
9. 测试覆盖跳过登录界面、直接启动运行子进程和篡改核心资源等场景。

## 10. GitHub 私有核心集成配置

公共仓库的保护核心 CI 默认允许在没有私有仓库令牌时运行公开契约测试；这不代表私有核心已经下载或完成原生集成验证。
正式集成验证必须在公共仓库配置以下 GitHub Actions Secret：

- Secret 名称：`D2S_CORE_REPO_TOKEN`；
- 权限范围：仅允许读取 `laiyangli001/desktop2stereo-vulkan-core` 的 Contents；
- 保存位置：公开仓库 Settings → Secrets and variables → Actions；
- 不得把令牌写入源码、工作流文件、日志或发布包。

在 GitHub Actions 手动运行 `Verify protected core integration` 时，保持 `require_private_core=true`。
该模式会下载私有仓库、编译当前平台原生模块、暂存加密资源并执行真实原生 ABI 解密测试；令牌缺失时必须失败。
公开推送触发的默认检查可以在令牌缺失时跳过私有资产步骤，但工作流会明确报告跳过原因。
