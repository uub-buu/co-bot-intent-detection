cmake_minimum_required(VERSION 3.15)
project(cobid LANGUAGES CXX)

# Install OpenCV is through "sudo apt install libopencv-dev"
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Qualcomm AI Runtime SDK (aka SNPE, aka QNN
set(
  QAIRT_ROOT "${CMAKE_SOURCE_DIR}/libs/qairt/2.50.0.260828"
  CACHE PATH "Qualcomm AI Runtime SDK location")

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "inc/*.h" "inc/*.hpp")

add_executable(cobid ${SOURCES} ${HEADERS})

target_include_directories(
  cobid
  PRIVATE
    inc
    ${OpenCV_INCLUDE_DIRS}
    ${QAIRT_ROOT}/include/SNPE
)

# TODO: Maybe check the GCC version?
target_link_directories(
  cobid
  PRIVATE
    ${QAIRT_ROOT}/lib/aarch64-ubuntu-gcc9.4)

target_link_libraries(
  cobid
  PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    SNPE
)
