cmake_minimum_required(VERSION 3.26)
project(pygwtf LANGUAGES CXX)

# GPU backend selection: AUTO picks CUDA (NVIDIA) if nvcc is found, else HIP
# (AMD/ROCm) if hipcc is found, else builds the CPU backend only.
set(GWTF_GPU "AUTO" CACHE STRING "GPU backend to build: AUTO, CUDA, HIP or OFF")
set_property(CACHE GWTF_GPU PROPERTY STRINGS AUTO CUDA HIP OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(GWTF_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include)
set(GWTF_BINDINGS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/bindings/module.cpp)
set(GWTF_GPU_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/gpu/gpu_ops.hip)

# ---------------------------------------------------------------- CPU backend
nanobind_add_module(_core_cpu NB_STATIC NB_DOMAIN pygwtf_cpu ${GWTF_BINDINGS} src/cpp/cpu/cpu_ops.cpp)
target_include_directories(_core_cpu PRIVATE ${GWTF_INCLUDE})
install(TARGETS _core_cpu LIBRARY DESTINATION pygwtf)

# ---------------------------------------------------------------- GPU backend
set(GWTF_GPU_LANG "")
if(GWTF_GPU STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(GWTF_GPU_LANG CUDA)
  else()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      set(GWTF_GPU_LANG HIP)
    endif()
  endif()
elseif(GWTF_GPU STREQUAL "CUDA" OR GWTF_GPU STREQUAL "HIP")
  set(GWTF_GPU_LANG ${GWTF_GPU})
endif()

if(GWTF_GPU_LANG STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(gwtf_gpu_ops STATIC ${GWTF_GPU_SOURCE})
  set_source_files_properties(${GWTF_GPU_SOURCE} PROPERTIES LANGUAGE CUDA)
  target_compile_options(gwtf_gpu_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
  target_link_libraries(gwtf_gpu_ops PUBLIC CUDA::cudart_static)
elseif(GWTF_GPU_LANG STREQUAL "HIP")
  enable_language(HIP)
  find_package(hip REQUIRED)
  add_library(gwtf_gpu_ops STATIC ${GWTF_GPU_SOURCE})
  set_source_files_properties(${GWTF_GPU_SOURCE} PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gwtf_gpu_ops PUBLIC GWTF_HIP)
  target_link_libraries(gwtf_gpu_ops PUBLIC hip::host)
endif()

if(GWTF_GPU_LANG)
  message(STATUS "pygwtf: building GPU backend with ${GWTF_GPU_LANG}")
  set_target_properties(gwtf_gpu_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gwtf_gpu_ops PUBLIC ${GWTF_INCLUDE})
  nanobind_add_module(_core_gpu NB_STATIC NB_DOMAIN pygwtf_gpu ${GWTF_BINDINGS})
  target_compile_definitions(_core_gpu PRIVATE GWTF_GPU_MODULE)
  target_link_libraries(_core_gpu PRIVATE gwtf_gpu_ops)
  install(TARGETS _core_gpu LIBRARY DESTINATION pygwtf)
else()
  message(STATUS "pygwtf: GPU backend disabled (CPU backend only)")
endif()
