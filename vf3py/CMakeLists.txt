cmake_minimum_required(VERSION 3.4...3.18)
project(vf3py)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD 17)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

find_package(pybind11 REQUIRED)

if (WIN32)
add_definitions(-DWIN32=1)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(PROJ_SRC_FILES vf3/main.cpp)

# Release flags
set(VF3_CFLAGS -std=c++17 -Wno-deprecated -pedantic -DBOOST_UBLAS_NDEBUG -O3 -D NDEBUG -flto)

# Debug flags
# set(VF3_CFLAGS -std=c++17 -Wno-deprecated -pedantic -g -Og -fsanitize=address -static-libasan -fno-omit-frame-pointer)
# set(VF3_LDFLAGS -lasan -static-libasan)
# add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES)

# Basic VF3
pybind11_add_module(vf3py_base ${PROJ_SRC_FILES})
target_include_directories(vf3py_base PRIVATE vf3/)

target_compile_definitions(vf3py_base PUBLIC -DVF3=1)
target_compile_options(vf3py_base PRIVATE ${VF3_CFLAGS})
target_link_options(vf3py_base PRIVATE ${VF3_CFLAGS} ${VF3_LDFLAGS})
target_link_libraries(vf3py_base PRIVATE ${VF3_LDFLAGS})

# VF3L
pybind11_add_module(vf3py_vf3l ${PROJ_SRC_FILES})
target_include_directories(vf3py_vf3l PRIVATE vf3/)

target_compile_definitions(vf3py_vf3l PUBLIC -DVF3L=1)
target_compile_options(vf3py_vf3l PRIVATE ${VF3_CFLAGS})
target_link_options(vf3py_vf3l PRIVATE ${VF3_CFLAGS} ${VF3_LDFLAGS})
target_link_libraries(vf3py_vf3l PRIVATE ${VF3_LDFLAGS})

# VF3P
pybind11_add_module(vf3py_vf3p ${PROJ_SRC_FILES})
target_include_directories(vf3py_vf3p PRIVATE vf3/)

target_compile_definitions(vf3py_vf3p PUBLIC -DVF3P=1)
target_compile_options(vf3py_vf3p PRIVATE ${VF3_CFLAGS})
target_link_options(vf3py_vf3p PRIVATE ${VF3_CFLAGS} ${VF3_LDFLAGS} -lpthread -latomic)
target_link_libraries(vf3py_vf3p PRIVATE ${VF3_LDFLAGS} -lpthread -latomic)
