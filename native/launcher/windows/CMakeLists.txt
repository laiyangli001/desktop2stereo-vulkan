cmake_minimum_required(VERSION 3.20)
project(Desktop2StereoLauncher LANGUAGES CXX)

add_executable(Desktop2Stereo WIN32 main.cpp resource.rc)
target_compile_features(Desktop2Stereo PRIVATE cxx_std_17)
target_link_libraries(Desktop2Stereo PRIVATE windowscodecs user32 gdi32 shell32)
target_compile_definitions(Desktop2Stereo PRIVATE UNICODE _UNICODE)
set_target_properties(Desktop2Stereo PROPERTIES OUTPUT_NAME "Desktop2Stereo")
