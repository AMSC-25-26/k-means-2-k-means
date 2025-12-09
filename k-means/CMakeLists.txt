cmake_minimum_required(VERSION 4.0)
project(k_means)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Needed libraries
include(FetchContent)
FetchContent_Declare(
        rapidcsv
        GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git
        GIT_TAG v8.90
)
FetchContent_MakeAvailable(rapidcsv)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.x
)
FetchContent_MakeAvailable(spdlog)

add_executable(
        k_means
        src/main.cpp
        src/KMeansClassifier.cpp
        src/CentroidInitializer.cpp
        src/clustering_evaluator.cpp
        src/data_io.cpp
)

target_include_directories(k_means PRIVATE ${rapidcsv_SOURCE_DIR}/src)
target_include_directories(k_means PRIVATE ${spdlog_SOURCE_DIR}/include)

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Find OpenMP and link to the target
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(k_means PRIVATE OpenMP::OpenMP_CXX)
endif ()

# Link your executable to the DataFrame target provided by the project
target_link_libraries(k_means PRIVATE spdlog::spdlog stdc++exp)
