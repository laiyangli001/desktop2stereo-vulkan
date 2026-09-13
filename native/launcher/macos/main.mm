#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>

@interface D2SDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSWindow* splashWindow;
@property(nonatomic, strong) NSTask* task;
@property(nonatomic, strong) NSTimer* timer;
@property(nonatomic, copy) NSString* readyPath;
@property(nonatomic, copy) NSString* authReadyPath;
@property(nonatomic, strong) NSDate* deadline;
@end

@implementation D2SDelegate

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    (void)notification;
    NSString* executable = [[[NSBundle mainBundle] executablePath] stringByStandardizingPath];
    NSString* location = [executable stringByDeletingLastPathComponent];
    NSString* root = location;
    // Release packages contain the standalone executable directly in src/.
    // Keep accepting an older app bundle when users launch one that is already
    // installed, but never create or package a bundle for new builds.
    if ([location.lastPathComponent caseInsensitiveCompare:@"MacOS"] == NSOrderedSame) {
        root = [[location stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
        root = [root stringByDeletingLastPathComponent];
    } else if ([location.lastPathComponent caseInsensitiveCompare:@"src"] == NSOrderedSame) {
        root = [location stringByDeletingLastPathComponent];
    }
    NSString* app = [root stringByAppendingPathComponent:@"src/desktop2stereo"];
    NSString* python = [root stringByAppendingPathComponent:@"src/python3/bin/python"];
    NSString* script = [app stringByAppendingPathComponent:@"main.py"];
    NSString* imagePath = [app stringByAppendingPathComponent:@"d2s_blur.png"];
    NSString* iconPath = [app stringByAppendingPathComponent:@"icon/icon-256x256.png"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:imagePath]) imagePath = [root stringByAppendingPathComponent:@"d2s_blur.png"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:iconPath]) iconPath = [root stringByAppendingPathComponent:@"src/desktop2stereo/icon/icon-256x256.png"];
    self.readyPath = [app stringByAppendingPathComponent:@"logs/gui_ready.flag"];
    self.authReadyPath = [app stringByAppendingPathComponent:@"logs/auth_ready.flag"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:python] || ![[NSFileManager defaultManager] fileExistsAtPath:script] || ![[NSFileManager defaultManager] fileExistsAtPath:imagePath]) {
        [[NSAlert alertWithMessageText:@"Desktop2Stereo 启动失败" defaultButton:@"确定" alternateButton:nil otherButton:nil informativeTextWithFormat:@"运行时、main.py 或启动图片缺失。"] runModal];
        [NSApp terminate:nil];
        return;
    }
    NSImage* applicationIcon = [[NSImage alloc] initWithContentsOfFile:iconPath];
    if (applicationIcon != nil) [NSApp setApplicationIconImage:applicationIcon];
    [[NSFileManager defaultManager] removeItemAtPath:self.readyPath error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:self.authReadyPath error:nil];
    NSScreen* screen = [NSScreen mainScreen];
    NSRect visible = screen.visibleFrame;
    const double aspect = 1672.0 / 941.0;
    const double area = visible.size.width * visible.size.height * 0.25;
    CGFloat width = std::max<CGFloat>(240, std::sqrt(area * aspect));
    CGFloat height = std::max<CGFloat>(135, std::round(width / aspect));
    NSImage* image = [[NSImage alloc] initWithContentsOfFile:imagePath];
    NSWindowStyleMask style = NSWindowStyleMaskBorderless;
    self.splashWindow = [[NSWindow alloc] initWithContentRect:NSMakeRect(visible.origin.x + (visible.size.width - width) / 2,
        visible.origin.y + (visible.size.height - height) / 2, width, height) styleMask:style backing:NSBackingStoreBuffered defer:NO];
    self.splashWindow.backgroundColor = [NSColor clearColor];
    self.splashWindow.opaque = NO;
    self.splashWindow.level = NSFloatingWindowLevel;
    self.splashWindow.ignoresMouseEvents = YES;
    NSImageView* view = [[NSImageView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
    view.image = image;
    view.imageScaling = NSImageScaleProportionallyUpOrDown;
    [self.splashWindow.contentView addSubview:view];
    [self.splashWindow orderFrontRegardless];

    self.task = [[NSTask alloc] init];
    self.task.launchPath = python;
    self.task.arguments = @[script];
    self.task.currentDirectoryPath = app;
    NSMutableDictionary* environment = [NSMutableDictionary dictionaryWithDictionary:NSProcessInfo.processInfo.environment];
    environment[@"PYTHONPATH"] = [root stringByAppendingPathComponent:@"src"];
    self.task.environment = environment;
    [self.task launch];
    self.deadline = [NSDate dateWithTimeIntervalSinceNow:60.0];
    self.timer = [NSTimer scheduledTimerWithTimeInterval:0.1 target:self selector:@selector(poll:) userInfo:nil repeats:YES];
}

- (void)poll:(NSTimer*)timer {
    (void)timer;
    if ([[NSFileManager defaultManager] fileExistsAtPath:self.readyPath] ||
        [[NSFileManager defaultManager] fileExistsAtPath:self.authReadyPath]) {
        [self.timer invalidate];
        [self.splashWindow orderOut:nil];
        [self.splashWindow close];
        // The GUI process is intentionally independent; only the bootstrap exits.
        [NSApp terminate:nil];
        return;
    }
    if (!self.task.isRunning || [self.deadline timeIntervalSinceNow] <= 0) {
        [self.timer invalidate];
        [self.splashWindow orderOut:nil];
        [self.splashWindow close];
        if (self.task.isRunning) [self.task terminate];
        [[NSAlert alertWithMessageText:@"Desktop2Stereo 启动失败" defaultButton:@"确定" alternateButton:nil otherButton:nil informativeTextWithFormat:@"GUI 未在 60 秒内完成初始化。"] runModal];
        [NSApp terminate:nil];
    }
}
@end

int main() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
        D2SDelegate* delegate = [D2SDelegate new];
        NSApp.delegate = delegate;
        [NSApp run];
    }
    return 0;
}
