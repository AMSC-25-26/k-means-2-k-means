cmake_minimum_required(VERSION 4.0)
project(k_means)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
        rapidcsv
        GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git
        GIT_TAG v8.90
)
FetchContent_MakeAvailable(rapidcsv)

add_executable(
        k_means
        main.cpp
)

target_include_directories(k_means PRIVATE ${rapidcsv_SOURCE_DIR}/src)

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Link your executable to the DataFrame target provided by the project
target_link_libraries(k_means PRIVATE stdc++exp)
