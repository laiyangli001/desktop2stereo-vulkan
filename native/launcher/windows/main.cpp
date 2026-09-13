#define NOMINMAX
#include <windows.h>
#include <wincodec.h>
#include <shellapi.h>
#include "resource.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "shell32.lib")

namespace {
constexpr wchar_t kWindowClass[] = L"Desktop2StereoNativeSplash";
constexpr UINT_PTR kTimerId = 1;
constexpr UINT kPollMs = 100;
constexpr DWORD kTimeoutMs = 60000;

HWND g_window = nullptr;
HANDLE g_process = nullptr;
DWORD g_startedAt = 0;
HBITMAP g_bitmap = nullptr;
int g_width = 0;
int g_height = 0;
std::filesystem::path g_root;

std::filesystem::path ResolveProjectRoot(std::filesystem::path executableDir) {
    if (executableDir.filename() == L"src") executableDir = executableDir.parent_path();
    return executableDir;
}

std::filesystem::path FindAsset(const std::filesystem::path& root) {
    const std::filesystem::path candidates[] = {
        root / L"src" / L"desktop2stereo" / L"d2s_blur.png",
        root / L"d2s_blur.png",
        root / L"resources" / L"d2s_blur.png",
    };
    for (const auto& path : candidates) {
        if (std::filesystem::is_regular_file(path)) return path;
    }
    return {};
}

bool LoadPng(const std::filesystem::path& path, int width, int height) {
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    IWICBitmapScaler* scaler = nullptr;
    bool ok = false;
    do {
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                    CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) break;
        if (FAILED(factory->CreateDecoderFromFilename(path.c_str(), nullptr,
                    GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder))) break;
        if (FAILED(decoder->GetFrame(0, &frame))) break;
        if (FAILED(factory->CreateFormatConverter(&converter))) break;
        if (FAILED(converter->Initialize(frame, GUID_WICPixelFormat32bppPBGRA,
                    WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom))) break;
        if (FAILED(factory->CreateBitmapScaler(&scaler))) break;
        if (FAILED(scaler->Initialize(converter, width, height,
                    WICBitmapInterpolationModeFant))) break;

        const UINT stride = static_cast<UINT>(width * 4);
        std::vector<BYTE> pixels(static_cast<size_t>(stride) * height);
        if (FAILED(scaler->CopyPixels(nullptr, stride, static_cast<UINT>(pixels.size()), pixels.data()))) break;

        BITMAPINFO info{};
        info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        info.bmiHeader.biWidth = width;
        info.bmiHeader.biHeight = -height;
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB;
        void* bits = nullptr;
        HDC dc = GetDC(nullptr);
        HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
        ReleaseDC(nullptr, dc);
        if (!bitmap || !bits) {
            if (bitmap) DeleteObject(bitmap);
            break;
        }
        memcpy(bits, pixels.data(), pixels.size());
        g_bitmap = bitmap;
        ok = true;
    } while (false);
    if (scaler) scaler->Release();
    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    return ok;
}

void PresentBitmap() {
    HDC screen = GetDC(nullptr);
    HDC memory = CreateCompatibleDC(screen);
    HGDIOBJ old = SelectObject(memory, g_bitmap);
    POINT position{
        (GetSystemMetrics(SM_CXSCREEN) - g_width) / 2,
        (GetSystemMetrics(SM_CYSCREEN) - g_height) / 2,
    };
    SIZE size{g_width, g_height};
    POINT source{0, 0};
    BLENDFUNCTION blend{AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    UpdateLayeredWindow(g_window, screen, &position, &size, memory, &source,
                        0, &blend, ULW_ALPHA);
    SelectObject(memory, old);
    DeleteDC(memory);
    ReleaseDC(nullptr, screen);
}

bool ReadyFileExists() {
    const auto logs = g_root / L"src" / L"desktop2stereo" / L"logs";
    return std::filesystem::is_regular_file(logs / L"auth_ready.flag") ||
           std::filesystem::is_regular_file(logs / L"gui_ready.flag");
}

void ShowFailure(const wchar_t* message) {
    if (g_window) DestroyWindow(g_window);
    MessageBoxW(nullptr, message, L"Desktop2Stereo", MB_OK | MB_ICONERROR);
}

void StopChildProcess() {
    if (!g_process) return;
    if (WaitForSingleObject(g_process, 0) == WAIT_TIMEOUT) {
        TerminateProcess(g_process, 1);
        WaitForSingleObject(g_process, 2000);
    }
    CloseHandle(g_process);
    g_process = nullptr;
}

void DetachChildProcess() {
    if (!g_process) return;
    CloseHandle(g_process);
    g_process = nullptr;
}

bool StartPython() {
    const auto python = g_root / L"src" / L"python3" / L"python.exe";
    const auto script = g_root / L"src" / L"desktop2stereo" / L"main.py";
    if (!std::filesystem::is_regular_file(python) || !std::filesystem::is_regular_file(script)) {
        ShowFailure(L"Desktop2Stereo Python runtime or main.py was not found.");
        return false;
    }
    const auto logDir = g_root / L"src" / L"desktop2stereo" / L"logs";
    std::error_code ec;
    std::filesystem::create_directories(logDir, ec);
    std::filesystem::remove(logDir / L"auth_ready.flag", ec);
    std::filesystem::remove(logDir / L"gui_ready.flag", ec);
    const auto pythonPath = g_root / L"src";
    SetEnvironmentVariableW(L"PYTHONPATH", pythonPath.c_str());
    std::wstring command = L"\"" + python.wstring() + L"\" \"" + script.wstring() + L"\"";
    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(python.c_str(), mutableCommand.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, g_root.c_str(), &startup, &process)) {
        ShowFailure(L"Failed to start the Desktop2Stereo Python runtime.");
        return false;
    }
    CloseHandle(process.hThread);
    g_process = process.hProcess;
    g_startedAt = GetTickCount();
    return true;
}

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    if (message == WM_NCHITTEST) return HTTRANSPARENT;
    if (message == WM_TIMER && wParam == kTimerId) {
        if (ReadyFileExists()) {
            KillTimer(window, kTimerId);
            DetachChildProcess();
            DestroyWindow(window);
            return 0;
        }
        if (g_process && WaitForSingleObject(g_process, 0) == WAIT_OBJECT_0) {
            KillTimer(window, kTimerId);
            ShowFailure(L"Desktop2Stereo exited before the GUI became ready.\nSee logs/launcher_stderr.log for details.");
            StopChildProcess();
            PostQuitMessage(1);
            return 0;
        }
        if (GetTickCount() - g_startedAt > kTimeoutMs) {
            KillTimer(window, kTimerId);
            ShowFailure(L"Desktop2Stereo did not become ready within 60 seconds.");
            StopChildProcess();
            PostQuitMessage(2);
            return 0;
        }
        return 0;
    }
    if (message == WM_DESTROY) {
        if (g_bitmap) {
            DeleteObject(g_bitmap);
            g_bitmap = nullptr;
        }
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, wParam, lParam);
}
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    wchar_t modulePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, modulePath, MAX_PATH);
    g_root = ResolveProjectRoot(std::filesystem::path(modulePath).parent_path());
    const auto image = FindAsset(g_root);
    if (image.empty()) {
        ShowFailure(L"d2s_blur.png was not found.");
        CoUninitialize();
        return 3;
    }
    const int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    const int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    const double aspect = 1672.0 / 941.0;
    const double targetArea = static_cast<double>(screenWidth) * screenHeight * 0.25;
    g_width = std::max(240, static_cast<int>(std::sqrt(targetArea * aspect)));
    g_height = std::max(135, static_cast<int>(std::round(g_width / aspect)));
    if (g_width > screenWidth * 0.9 || g_height > screenHeight * 0.9) {
        const double scale = std::min(screenWidth * 0.9 / g_width, screenHeight * 0.9 / g_height);
        g_width = static_cast<int>(g_width * scale);
        g_height = static_cast<int>(g_height * scale);
    }
    if (!LoadPng(image, g_width, g_height)) {
        ShowFailure(L"Failed to load d2s_blur.png.");
        CoUninitialize();
        return 4;
    }
    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.hInstance = instance;
    windowClass.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(IDI_DESKTOP2STEREO));
    windowClass.hIconSm = windowClass.hIcon;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.lpszClassName = kWindowClass;
    RegisterClassExW(&windowClass);
    g_window = CreateWindowExW(WS_EX_LAYERED | WS_EX_TOOLWINDOW |
        WS_EX_NOACTIVATE | WS_EX_TOPMOST,
        kWindowClass, L"Desktop2Stereo", WS_POPUP,
        0, 0, g_width, g_height, nullptr, nullptr, instance, nullptr);
    if (!g_window) {
        ShowFailure(L"Failed to create the native startup window.");
        CoUninitialize();
        return 5;
    }
    PresentBitmap();
    ShowWindow(g_window, SW_SHOWNOACTIVATE);
    // Keep the startup artwork above other applications without activating
    // it or stealing keyboard focus from the user's current window.
    SetWindowPos(g_window, HWND_TOPMOST, 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    UpdateWindow(g_window);
    if (!StartPython()) {
        CoUninitialize();
        return 6;
    }
    SetTimer(g_window, kTimerId, kPollMs, nullptr);
    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    StopChildProcess();
    CoUninitialize();
    return static_cast<int>(message.wParam);
}
