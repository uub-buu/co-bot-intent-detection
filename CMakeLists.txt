cmake_minimum_required(VERSION 3.15)
project(cobid LANGUAGES CXX)

# Install OpenCV is through "sudo apt install libopencv-dev"
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Qualcomm AI Runtime SDK (aka SNPE, aka QNN)
set(QAIRT_VER "2.50.0.260828")

set(
  QAIRT_ROOT "${CMAKE_SOURCE_DIR}/libs/qairt/${QAIRT_VER}"
  CACHE PATH "Qualcomm AI Runtime SDK location")

if(NOT DEFINED QAIRT_LIB_DIR)
  set(QAIRT_LIB_DIR "${QAIRT_ROOT}/lib/aarch64-ubuntu-gcc9.4")
endif()

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "inc/*.h" "inc/*.hpp")

add_executable(cobid ${SOURCES} ${HEADERS})
add_executable(cobid_with_debug ${SOURCES} ${HEADERS})

target_compile_definitions(cobid_with_debug PRIVATE DEBUG)
#target_compile_definitions(cobid_with_debug PRIVATE DEBUG_SNPE)

foreach(target_name cobid cobid_with_debug)
  target_include_directories(
    ${target_name}
      PRIVATE
        inc
        ${OpenCV_INCLUDE_DIRS}
        ${QAIRT_ROOT}/include/SNPE
  )

  target_link_directories(
    ${target_name}
      PRIVATE
        ${QAIRT_LIB_DIR}
  )
  target_link_options(${target_name} PRIVATE "-Wl,--allow-shlib-undefined")

  target_link_libraries(
    ${target_name}
      PRIVATE
        ${OpenCV_LIBS}
        Threads::Threads
        SNPE
  )

  set_target_properties(
    ${target_name}
      PROPERTIES
        BUILD_RPATH "${QAIRT_LIB_DIR}"
  )
endforeach()
