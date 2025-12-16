cmake_minimum_required(VERSION 3.28.3)

execute_process(COMMAND which mpic++ OUTPUT_VARIABLE _MPICXX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(_MPICXX)
    set(CMAKE_CXX_COMPILER ${_MPICXX} CACHE FILEPATH "MPI C++ compiler" FORCE)
endif()

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
)

target_include_directories(k_means PRIVATE ${rapidcsv_SOURCE_DIR}/src)
target_include_directories(k_means PRIVATE ${spdlog_SOURCE_DIR}/include)
# cmake
if (APPLE)
    # Prova prima il find_package standard
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(k_means PRIVATE OpenMP::OpenMP_CXX)
    else ()
        # Usa Homebrew prefix per libomp se find_package fallisce
        execute_process(
                COMMAND brew --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
        )
        if (LIBOMP_PREFIX)
            find_library(LIBOMP_LIBRARY NAMES omp libomp PATHS ${LIBOMP_PREFIX}/lib NO_DEFAULT_PATH)
            if (LIBOMP_LIBRARY)
                target_include_directories(k_means PRIVATE ${LIBOMP_PREFIX}/include)
                target_link_libraries(k_means PRIVATE ${LIBOMP_LIBRARY})
            else ()
                message(WARNING "libomp not found in ${LIBOMP_PREFIX}/lib. Set LDFLAGS/CPPFLAGS or install libomp")
            endif ()
        else ()
            message(WARNING "Homebrew did not give any prefix for libomp. Install libomp or set flag manually")
        endif ()
    endif ()
else ()
    find_package(OpenMP REQUIRED)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(k_means PRIVATE OpenMP::OpenMP_CXX)
    endif ()

    find_package(MPI REQUIRED)
    if (MPI_CXX_FOUND)
        target_link_libraries(k_means PRIVATE MPI::MPI_CXX)
    endif ()

    target_link_libraries(k_means PRIVATE spdlog::spdlog stdc++exp)
endif ()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

