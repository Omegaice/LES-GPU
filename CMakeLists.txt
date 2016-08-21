cmake_minimum_required (VERSION 2.8)
project (YIP)
enable_language (Fortran)

# Default to Release
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

option( BUILD_FORTRAN "Build Fortran code" OFF)
if (BUILD_FORTRAN)
  # Require MPI
  find_package(MPI REQUIRED)
  set( CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})

  # Setup Compile Flags
  set (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -i4 -r8")
  set (CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -i4 -r8")

  # Build Library
  add_library (fft STATIC "fft.f")

  # Build Executable
  set_source_files_properties( "les.F" COMPILE_FLAGS "-assume byterecl")

  add_executable (lesmpi.a "les.F")
  target_link_libraries (lesmpi.a fft)

  install ( TARGETS lesmpi.a RUNTIME DESTINATION "bin")
endif (BUILD_FORTRAN)

# CUDA
option( BUILD_CUDA "Build CUDA code" OFF)
if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
endif(BUILD_CUDA)

option( BUILD_TESTS "Build LES tests" ON)
if (BUILD_TESTS)
  # Download and unpack googletest at configure time
  configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest )
  execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest )

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  # Add googletest directly to our build. This adds
  # the following targets: gtest, gtest_main, gmock
  # and gmock_main
  add_subdirectory(test/google ${CMAKE_BINARY_DIR}/googletest-build)

  # The gtest/gmock targets carry header search path
  # dependencies automatically when using CMake 2.8.11 or
  # later. Otherwise we have to add them here ourselves.
  if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
  endif()

  include_directories("test/")

  if (BUILD_CUDA)
    cuda_include_directories("${gtest_SOURCE_DIR}/include" "test/")
    cuda_add_executable (les-test "test/main.cpp" "test/utility.cpp" "test/reference.cpp" "test/cuda.cu")
  else (BUILD_CUDA)
    add_executable (les-test "test/main.cpp" "test/utility.cpp" "test/reference.cpp")
  endif (BUILD_CUDA)
  target_link_libraries (les-test gtest)
endif (BUILD_TESTS)