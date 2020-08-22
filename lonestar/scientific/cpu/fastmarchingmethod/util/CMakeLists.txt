if (NOT TARGET google-segystack)
  set(google_segystack_src
        google-segystack/mmap_file.cc
        google-segystack/segy_file.cc
        google-segystack/float_util.cc
     )
  add_library(google-segystack STATIC ${google_segystack_src})
  target_link_libraries(google-segystack PUBLIC galois_shmem)
endif()

if (NOT TARGET numpy-cpp)
  add_library(numpy-cpp STATIC numpy/cnpy.cpp)
  # target_link_libraries(numpy-cpp PUBLIC galois_shmem)
  set_property(TARGET numpy-cpp APPEND PROPERTY COMPILE_FLAGS " -Wno-unused ")
endif()

app(grids_gen grids_gen.cpp)
set_property(TARGET grids_gen PROPERTY CXX_STANDARD 17)

