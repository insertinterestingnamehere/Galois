# if(("${CMAKE_VERSION}" VERSION_GREATER 3.8.0) OR ("${CMAKE_VERSION}" VERSION_EQUAL 3.8.0))
  app(grids_gen grids_gen.cpp)
  set_property(TARGET grids_gen PROPERTY CXX_STANDARD 17)

  set(google_segystack_src
        google-segystack/mmap_file.cc
        google-segystack/segy_file.cc
        google-segystack/float_util.cc
     )
  add_library(google-segystack STATIC ${google_segystack_src})
  target_link_libraries(google-segystack PUBLIC galois_shmem)

  add_library(numpy-cpp STATIC numpy/cnpy.cpp)
  # target_link_libraries(numpy-cpp PUBLIC galois_shmem)
  set_property(TARGET numpy-cpp APPEND PROPERTY COMPILE_FLAGS " -Wno-unused ")

  app(fmm fmm.cpp speed_function.cpp boundary.cpp)
  set_property(TARGET fmm PROPERTY CXX_STANDARD 17)
  set_property(TARGET fmm APPEND PROPERTY COMPILE_FLAGS " -Werror=unused ")
  target_link_libraries(fmm PRIVATE google-segystack PRIVATE numpy-cpp)
  # find_package(Eigen3 CONFIG)
  # if(Eigen3_FOUND)
  #   target_link_libraries(fmm PRIVATE Eigen3::Eigen)
  #   target_compile_definitions(fmm PRIVATE -DHAS_EIGEN)
  #   # if("${Eigen3_VERSION}" VERSION_LESS 3.3.90)
  #   #   target_compile_definitions(fmm PRIVATE -DEIGEN_OLD_ITERATOR)
  #   # endif()
  # endif()

  app(fastmarchingmethod-triangles-2d fmm_triangles_2d.cpp speed_function.cpp boundary.cpp)
  set_property(TARGET fastmarchingmethod-triangles-2d PROPERTY CXX_STANDARD 17)
  set_property(TARGET fastmarchingmethod-triangles-2d APPEND PROPERTY COMPILE_FLAGS " -Werror=unused ")
# else()
#   message(WARNING "Cannot configure sweeps or other C++17 apps with a CMake version older than 3.8. Skipping.")
# endif()

