#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <png.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct Image {
    int width{};
    int height{};
    std::vector<std::uint32_t> pixels;
};

static fs::path executable_dir() {
    char buffer[4096]{};
    const auto length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) return fs::current_path();
    buffer[length] = '\0';
    return fs::path(buffer).parent_path();
}

static fs::path project_root(fs::path executable_directory) {
    if (executable_directory.filename() == "src") return executable_directory.parent_path();
    return executable_directory;
}

static fs::path find_asset(const fs::path& root) {
    for (const auto& candidate : {root / "src/desktop2stereo/d2s_blur.png",
                                  root / "d2s_blur.png",
                                  root / "resources/d2s_blur.png"}) {
        if (fs::is_regular_file(candidate)) return candidate;
    }
    return {};
}

static bool load_png(const fs::path& path, int target_width, int target_height, Image& out) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return false;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info || setjmp(png_jmpbuf(png))) {
        if (png) png_destroy_read_struct(&png, &info, nullptr);
        fclose(file);
        return false;
    }
    png_init_io(png, file);
    png_read_info(png, info);
    const auto width = png_get_image_width(png, info);
    const auto height = png_get_image_height(png, info);
    const auto color = png_get_color_type(png, info);
    const auto depth = png_get_bit_depth(png, info);
    if (depth == 16) png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color == PNG_COLOR_TYPE_RGB || color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xff, PNG_FILLER_AFTER);
    if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    png_read_update_info(png, info);
    const auto stride = png_get_rowbytes(png, info);
    std::vector<std::uint8_t> source(stride * height);
    std::vector<png_bytep> rows(height);
    for (png_uint_32 y = 0; y < height; ++y) rows[y] = source.data() + y * stride;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(file);

    out.width = target_width;
    out.height = target_height;
    out.pixels.resize(static_cast<size_t>(target_width) * target_height);
    for (int y = 0; y < target_height; ++y) {
        const int sy = std::min<int>(height - 1, y * static_cast<int>(height) / target_height);
        for (int x = 0; x < target_width; ++x) {
            const int sx = std::min<int>(width - 1, x * static_cast<int>(width) / target_width);
            const auto* px = source.data() + sy * stride + sx * 4;
            const std::uint32_t a = px[3];
            const std::uint32_t r = px[0] * a / 255;
            const std::uint32_t g = px[1] * a / 255;
            const std::uint32_t b = px[2] * a / 255;
            out.pixels[static_cast<size_t>(y) * target_width + x] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
    return true;
}

static bool ready(const fs::path& auth_path, const fs::path& gui_path) {
    return fs::is_regular_file(auth_path) || fs::is_regular_file(gui_path);
}

static void stop_child(pid_t child) {
    if (child <= 0) return;
    if (kill(child, SIGTERM) == 0) {
        int status = 0;
        for (int attempt = 0; attempt < 20; ++attempt) {
            if (waitpid(child, &status, WNOHANG) == child) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    if (kill(child, SIGKILL) == 0) {
        int status = 0;
        waitpid(child, &status, 0);
    }
}

int main() {
    const auto root = project_root(executable_dir());
    const auto app = root / "src/desktop2stereo";
    const auto python = root / "src/python3/bin/python";
    const auto main_script = app / "main.py";
    const auto log_dir = app / "logs";
    const auto ready_file = log_dir / "gui_ready.flag";
    const auto auth_ready_file = log_dir / "auth_ready.flag";
    const auto image_path = find_asset(root);
    if (!fs::is_regular_file(python) || !fs::is_regular_file(main_script) || image_path.empty()) {
        std::cerr << "Desktop2Stereo launcher: runtime, main.py, or d2s_blur.png is missing\n";
        return 2;
    }
    std::error_code ec;
    fs::create_directories(log_dir, ec);
    fs::remove(ready_file, ec);
    fs::remove(auth_ready_file, ec);

    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        std::cerr << "Desktop2Stereo launcher: X11 display is unavailable\n";
        return 3;
    }
    const int screen = DefaultScreen(display);
    const int screen_width = DisplayWidth(display, screen);
    const int screen_height = DisplayHeight(display, screen);
    const double aspect = 1672.0 / 941.0;
    const double area = static_cast<double>(screen_width) * screen_height * 0.25;
    const int width = std::max(240, static_cast<int>(std::sqrt(area * aspect)));
    const int height = std::max(135, static_cast<int>(std::round(width / aspect)));
    Image image;
    if (!load_png(image_path, width, height, image)) {
        XCloseDisplay(display);
        std::cerr << "Desktop2Stereo launcher: failed to decode d2s_blur.png\n";
        return 4;
    }

    XVisualInfo visual_info{};
    const bool has_argb_visual = XMatchVisualInfo(display, screen, 32, TrueColor, &visual_info) != 0;
    Visual* visual = has_argb_visual ? visual_info.visual : DefaultVisual(display, screen);
    const int depth = has_argb_visual ? visual_info.depth : DefaultDepth(display, screen);
    Colormap colormap = XCreateColormap(display, RootWindow(display, screen), visual, AllocNone);
    XSetWindowAttributes attributes{};
    attributes.override_redirect = True;
    attributes.colormap = colormap;
    attributes.background_pixel = 0;
    Window window = XCreateWindow(display, RootWindow(display, screen),
                                  (screen_width - width) / 2, (screen_height - height) / 2,
                                  width, height, 0, depth, InputOutput, visual,
                                  CWOverrideRedirect | CWColormap | CWBackPixel, &attributes);
    XSelectInput(display, window, ExposureMask | StructureNotifyMask);
    XMapRaised(display, window);
    XFlush(display);
    XImage* ximage = XCreateImage(display, visual, depth,
                                  ZPixmap, 0, reinterpret_cast<char*>(image.pixels.data()), width, height, 32, 0);
    if (ximage) {
        GC gc = XCreateGC(display, window, 0, nullptr);
        XPutImage(display, window, gc, ximage, 0, 0, 0, 0, width, height);
        XFreeGC(display, gc);
        ximage->data = nullptr;
        XDestroyImage(ximage);
    }
    XFlush(display);

    const pid_t child = fork();
    if (child == 0) {
        const auto python_path = root / "src";
        setenv("PYTHONPATH", python_path.c_str(), 1);
        chdir(app.c_str());
        execl(python.c_str(), python.c_str(), main_script.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    if (child < 0) {
        XDestroyWindow(display, window);
        XCloseDisplay(display);
        return 5;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (ready(auth_ready_file, ready_file)) break;
        if (waitpid(child, &status, WNOHANG) == child) {
            std::cerr << "Desktop2Stereo launcher: GUI exited before ready\n";
            stop_child(child);
            XDestroyWindow(display, window);
            XCloseDisplay(display);
            return WIFEXITED(status) ? WEXITSTATUS(status) : 6;
        }
        while (XPending(display)) { XEvent event{}; XNextEvent(display, &event); }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const bool is_ready = ready(auth_ready_file, ready_file);
    if (!is_ready) {
        std::cerr << "Desktop2Stereo launcher: GUI ready timeout\n";
        stop_child(child);
    }
    XDestroyWindow(display, window);
    XFreeColormap(display, colormap);
    XCloseDisplay(display);
    return is_ready ? 0 : 7;
}
