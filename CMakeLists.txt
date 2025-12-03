cmake_minimum_required(VERSION 3.23)

project(neural_k
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Backend options
option(NEURAL_K_OPENMP "Enable Kokkos OpenMP backend" OFF)
option(NEURAL_K_CUDA "Enable Kokkos CUDA backend" OFF)

include(FetchContent)

# Kokkos configuration
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos SERIAL backend" FORCE)

if(NEURAL_K_OPENMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OPENMP backend" FORCE)
else()
    set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable Kokkos OPENMP backend" FORCE)
endif()

if(NEURAL_K_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable Kokkos CUDA lambdas" FORCE)
else()
    set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Disable Kokkos CUDA backend" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA OFF CACHE BOOL "Disable Kokkos CUDA lambdas" FORCE)
endif()

# Fetch Kokkos
FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 4.5.00
)
FetchContent_MakeAvailable(kokkos)

# KokkosKernels configuration
set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF CACHE BOOL "" FORCE)
set(KokkosKernels_ENABLE_COMPONENT_BLAS ON CACHE BOOL "" FORCE)
set(KokkosKernels_ENABLE_COMPONENT_SPARSE ON CACHE BOOL "" FORCE)

# Fetch KokkosKernels
FetchContent_Declare(
    kokkoskernels
    GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
    GIT_TAG 4.5.00
)
FetchContent_MakeAvailable(kokkoskernels)

# Main executable
add_executable(nk
    src/main.cpp
    src/activations.cpp
    src/optimizers.cpp
    src/layers.cpp
    src/network.cpp
    src/training.cpp
    src/sparsity_controller.cpp
    src/sparsity_strategies.cpp
)

target_link_libraries(nk
    Kokkos::kokkos
    Kokkos::kokkoskernels
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Neural K Configuration ===")
message(STATUS "  OpenMP backend: ${NEURAL_K_OPENMP}")
message(STATUS "  CUDA backend:   ${NEURAL_K_CUDA}")
message(STATUS "==============================")
message(STATUS "")
