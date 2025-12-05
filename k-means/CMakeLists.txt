cmake_minimum_required(VERSION 4.0)
project(k_means)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
        DataFrame
        GIT_REPOSITORY https://github.com/hosseinmoein/DataFrame.git
        GIT_TAG master
)
# Make the fetched project available and create its targets
FetchContent_MakeAvailable(DataFrame)

add_executable(
        k_means
        main.cpp
)

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Link your executable to the DataFrame target provided by the project
target_link_libraries(k_means PRIVATE stdc++exp)
target_link_libraries(k_means PRIVATE DataFrame::DataFrame)