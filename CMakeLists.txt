cmake_minimum_required(VERSION 3.20)
project(axionFEM VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(AXF_BUILD_PYTHON "Build the nanobind Python module" ON)
option(AXF_BUILD_MESHER "Build axf-mesher (links the gmsh SDK; separate GPL program)" ON)
option(AXF_BUILD_TESTS "Build C++ unit tests" ON)

# MPI drives the 3D engine (cpp/src/model3d*.cpp are the only files that use MPI
# or hypre).  AUTO follows the MFEM build, so a serial MFEM automatically gives
# a serial axionFEM with the 3D stub (cpp/src/model3d_nompi.cpp).
set(AXF_MPI "AUTO" CACHE STRING "3D engine with MPI + hypre: ON, OFF, or AUTO (follow MFEM)")
set_property(CACHE AXF_MPI PROPERTY STRINGS AUTO ON OFF)

# OpenMP parallelises assembly in the axisymmetric engine.  AUTO uses it when
# present; OFF (or a toolchain without it, e.g. Apple clang with no libomp)
# builds single-threaded through cpp/include/axionfem/omp_shim.hpp.
set(AXF_OPENMP "AUTO" CACHE STRING "OpenMP in the assembly loops: ON, OFF, or AUTO")
set_property(CACHE AXF_OPENMP PROPERTY STRINGS AUTO ON OFF)

# GPU: makes a backend of the MFEM build available; $AXF_DEVICE turns it on at
# run time (see cpp/src/device.cpp and docs/install.md).
set(AXF_GPU "OFF" CACHE STRING "GPU backend of the MFEM build: OFF, CUDA, or HIP")
set_property(CACHE AXF_GPU PROPERTY STRINGS OFF CUDA HIP)

# Wheels: copy the third-party shared libraries beside the extension module and
# point the rpaths at them, so the wheel needs nothing from the build machine.
# auditwheel (Linux) and delocate (macOS) then only have to add system libraries.
if(SKBUILD)
  set(_axf_bundle_default ON)
else()
  set(_axf_bundle_default OFF)
endif()
option(AXF_BUNDLE_LIBS "Install the third-party shared libraries into the package" ${_axf_bundle_default})

set(AXF_DEPS "$ENV{AXIONFEM_DEPS}" CACHE PATH "third-party install prefix")
if(NOT AXF_DEPS)
  set(AXF_DEPS "${CMAKE_SOURCE_DIR}/third_party/install")
endif()
# Wheels also leave the build machine's paths out of the compiled code: __FILE__ (MFEM_VERIFY messages, ours and those
# in MFEM's headers) is written relative to the source tree and to the dependency prefix.
if(SKBUILD)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=" AXF_HAVE_FILE_PREFIX_MAP)
  if(AXF_HAVE_FILE_PREFIX_MAP)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=>"
                        "$<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${AXF_DEPS}/=>")
  endif()
endif()
list(PREPEND CMAKE_PREFIX_PATH "${AXF_DEPS}")

set(AXF_GMSH_ROOT "" CACHE PATH "gmsh SDK or PyPI gmsh wheel prefix (holds include/gmsh.h and lib/libgmsh.*)")

# --------------------------------------------------------------- dependencies
find_package(MFEM CONFIG)
if(NOT MFEM_FOUND)
  message(FATAL_ERROR
    "MFEM was not found (searched AXF_DEPS=${AXF_DEPS} and CMAKE_PREFIX_PATH for MFEMConfig.cmake).\n"
    "Build the third-party stack first and point AXF_DEPS at its prefix:\n"
    "  MPI build (3D engine):  bash scripts/fetch_deps.sh && bash scripts/build_deps.sh   (prefix third_party/install)\n"
    "  sequential build:       bash scripts/build_deps_wheel.sh                           (prefix third_party/install-seq)\n"
    "then configure with -DAXF_DEPS=<prefix> (or export AXIONFEM_DEPS=<prefix>). See docs/guides/install.md.")
endif()
foreach(_axf_hdr "eigen3/Eigen/Core" "nlohmann/json.hpp")
  if(NOT EXISTS "${AXF_DEPS}/include/${_axf_hdr}")
    message(FATAL_ERROR
      "${AXF_DEPS}/include/${_axf_hdr} is missing: the Eigen and nlohmann/json headers are installed into AXF_DEPS "
      "by scripts/build_deps.sh (MPI) and scripts/build_deps_wheel.sh (sequential). Run the script for this prefix "
      "or set AXF_DEPS to the prefix it installed into.")
  endif()
endforeach()
# macOS: Apple clang has no OpenMP of its own; use Homebrew's keg-only libomp when it is installed, in CMake builds
# and in `pip install` source builds alike. Not in the wheel builds (cibuildwheel sets CIBUILDWHEEL): Homebrew's
# bottles do not support the wheels' deployment target. -DOpenMP_ROOT=... or -DAXF_OPENMP=OFF override this.
if(APPLE AND NOT DEFINED ENV{CIBUILDWHEEL} AND NOT AXF_OPENMP STREQUAL "OFF" AND NOT DEFINED OpenMP_ROOT
   AND NOT DEFINED ENV{OpenMP_ROOT})
  foreach(_axf_omp "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
    if(EXISTS "${_axf_omp}/include/omp.h")
      set(OpenMP_ROOT "${_axf_omp}")
      message(STATUS "OpenMP: using Homebrew's libomp in ${_axf_omp}")
      break()
    endif()
  endforeach()
endif()

if(AXF_MPI STREQUAL "AUTO")
  if(MFEM_USE_MPI)
    set(AXF_MPI ON)
  else()
    set(AXF_MPI OFF)
  endif()
  message(STATUS "AXF_MPI=AUTO -> ${AXF_MPI} (MFEM_USE_MPI=${MFEM_USE_MPI})")
endif()
if(AXF_MPI AND NOT MFEM_USE_MPI)
  message(FATAL_ERROR
    "AXF_MPI=ON but this MFEM (${MFEM_DIR}) was built with MFEM_USE_MPI=NO.\n"
    "The 3D engine needs mfem::ParMesh and hypre. Either build the MPI stack "
    "(scripts/build_deps.sh) and point AXF_DEPS at it, or configure with "
    "-DAXF_MPI=OFF to build the serial library with the 3D stub.")
endif()
if(AXF_MPI)
  find_package(MPI COMPONENTS C CXX)
  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR
      "MPI was not found, and this MFEM (${MFEM_DIR}) was built with MPI.\n"
      "Install MPI with its compiler wrappers (Debian/Ubuntu: libopenmpi-dev openmpi-bin; "
      "Fedora/RHEL: openmpi-devel or mpich-devel, then 'module load mpi/openmpi-x86_64' or put its bin directory on "
      "PATH; macOS: brew install open-mpi), use the MPI that built MFEM, and either have mpicxx on PATH, pass "
      "-DMPI_CXX_COMPILER=/path/to/mpicxx, or configure with the wrapper as the compiler "
      "(-DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc).")
  endif()
elseif(MFEM_USE_MPI)
  # A serial axionFEM (3D stub) against an MPI MFEM still needs MPI to *compile*: mfem.hpp includes
  # <mpi.h> and the parallel classes unconditionally.  Without this the build fails on the first
  # translation unit.
  find_package(MPI COMPONENTS C CXX)
  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR
      "AXF_MPI=OFF with an MFEM built with MFEM_USE_MPI=YES (${MFEM_DIR}), and no MPI was found.\n"
      "mfem.hpp includes <mpi.h>, so even the serial library needs the MPI headers. Either load an "
      "MPI module (the stack built by scripts/build_deps.sh uses OpenMPI), build a serial MFEM, or "
      "configure with -DAXF_MPI=ON to get the 3D engine.")
  endif()
  message(STATUS "AXF_MPI=OFF against an MPI MFEM: linking MPI for the mfem.hpp headers only "
                 "(the 3D engine is the stub in cpp/src/model3d_nompi.cpp)")
endif()

if(AXF_OPENMP STREQUAL "AUTO")
  find_package(OpenMP COMPONENTS CXX)
  set(AXF_OPENMP ${OpenMP_CXX_FOUND})
  message(STATUS "AXF_OPENMP=AUTO -> ${AXF_OPENMP}")
elseif(AXF_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# GPU: the backend must be in the MFEM build; axionFEM itself has no device code.
if(NOT AXF_GPU STREQUAL "OFF")
  string(TOUPPER "${AXF_GPU}" AXF_GPU)
  if(AXF_GPU STREQUAL "CUDA")
    if(NOT MFEM_USE_CUDA)
      message(FATAL_ERROR
        "-DAXF_GPU=CUDA needs an MFEM built with MFEM_USE_CUDA=YES; ${MFEM_DIR} has "
        "MFEM_USE_CUDA=${MFEM_USE_CUDA}. Build the CUDA stack with "
        "scripts/build_deps_gpu.sh and set AXF_DEPS to its prefix (docs/install.md).")
    endif()
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
    message(STATUS "AXF_GPU=CUDA (architectures ${CMAKE_CUDA_ARCHITECTURES})")
  elseif(AXF_GPU STREQUAL "HIP")
    if(NOT MFEM_USE_HIP)
      message(FATAL_ERROR
        "-DAXF_GPU=HIP needs an MFEM built with MFEM_USE_HIP=YES; ${MFEM_DIR} has "
        "MFEM_USE_HIP=${MFEM_USE_HIP}. See docs/install.md, \"GPU builds\".")
    endif()
    find_package(hip REQUIRED)
    message(STATUS "AXF_GPU=HIP")
  else()
    message(FATAL_ERROR "AXF_GPU must be OFF, CUDA or HIP; got '${AXF_GPU}'")
  endif()
  if(NOT AXF_MPI)
    message(WARNING
      "AXF_GPU=${AXF_GPU} with AXF_MPI=OFF: the only solver that can use a GPU is the 3D "
      "engine's hypre AMS/BoomerAMG, which needs MPI, so this build has no GPU path. "
      "The axisymmetric engine is a CPU sparse direct solve.")
  endif()
endif()

# --------------------------------------------------------------- rpath helpers
if(APPLE)
  set(AXF_ORIGIN "@loader_path")
else()
  set(AXF_ORIGIN "$ORIGIN")
endif()
set(CMAKE_INSTALL_RPATH "${AXF_DEPS}/lib;${AXF_DEPS}/lib64;${AXF_DEPS}/gmsh-sdk/lib")
set(CMAKE_BUILD_RPATH "${AXF_DEPS}/lib;${AXF_DEPS}/lib64;${AXF_DEPS}/gmsh-sdk/lib")
if(SKBUILD)
  # A wheel must not keep absolute paths from the build machine.
  set(CMAKE_INSTALL_RPATH "${AXF_ORIGIN};${AXF_ORIGIN}/_libs")
  set(CMAKE_SKIP_BUILD_RPATH OFF)
endif()

# ------------------------------------------------------------------ library
file(GLOB AXF_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/*.cpp)
# Exactly one of the 3D engine (model3d.cpp, model3d_fullwave.cpp, model3d_fw_indicator.cpp, model3d_surface_losses.cpp,
# model3d_electro.cpp, model3d_mech.cpp) and model3d_nompi.cpp (the stub).
if(AXF_MPI)
  list(REMOVE_ITEM AXF_SOURCES ${CMAKE_SOURCE_DIR}/cpp/src/model3d_nompi.cpp)
else()
  list(REMOVE_ITEM AXF_SOURCES ${CMAKE_SOURCE_DIR}/cpp/src/model3d.cpp ${CMAKE_SOURCE_DIR}/cpp/src/model3d_fullwave.cpp
                              ${CMAKE_SOURCE_DIR}/cpp/src/model3d_electro.cpp ${CMAKE_SOURCE_DIR}/cpp/src/model3d_mech.cpp
                              ${CMAKE_SOURCE_DIR}/cpp/src/model3d_fw_indicator.cpp
                              ${CMAKE_SOURCE_DIR}/cpp/src/model3d_surface_losses.cpp)
endif()

# chemistry module (beta): its own folder, sequential mfem::Mesh
file(GLOB AXF_CHEM_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/chem/*.cpp)
list(APPEND AXF_SOURCES ${AXF_CHEM_SOURCES})

add_library(axionfem SHARED ${AXF_SOURCES})
target_include_directories(axionfem PUBLIC
  ${CMAKE_SOURCE_DIR}/cpp/include
  ${AXF_DEPS}/include
  ${AXF_DEPS}/include/eigen3)
target_link_libraries(axionfem PUBLIC mfem)
# Build provenance: the git commit of the source tree and whether it differed from that commit, written by
# cmake/axf_gitinfo.cmake into a generated header on EVERY build (not when CMake configures), so a rebuild after a
# commit reports the new commit without re-running CMake. libaxionfem (cpp/src/build_provenance.cpp) and the Python
# module (bindings.cpp, build_info()) both compile it in; a rebuild of one without the other shows as two commits.
set(AXF_GITINFO_DIR "${CMAKE_BINARY_DIR}/generated")
add_custom_target(axf_gitinfo ALL
  COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DOUT=${AXF_GITINFO_DIR}/axionfem/build_git.hpp -P ${CMAKE_SOURCE_DIR}/cmake/axf_gitinfo.cmake
  BYPRODUCTS ${AXF_GITINFO_DIR}/axionfem/build_git.hpp
  COMMENT "axionFEM build provenance (git sha, dirty flags)" VERBATIM)
add_dependencies(axionfem axf_gitinfo)
target_include_directories(axionfem PRIVATE ${AXF_GITINFO_DIR})
target_link_libraries(axionfem PRIVATE ${CMAKE_DL_LIBS})   # dladdr: where the loaded library is (glibc < 2.34)
# optional: STRUMPACK (BSD) for large axisymmetric systems (scripts/build_strumpack.sh)
option(AXF_USE_STRUMPACK "Use STRUMPACK from AXF_DEPS when it is installed" ON)
# A distributed-memory STRUMPACK (MPI + ScaLAPACK, scripts/build_strumpack_mpi.sh) in its own prefix replaces the
# serial one: its static library also holds the sequential solver, and the 3D direct full-wave path then factorizes
# across the ranks (AXF_STRUMPACK_MPI) instead of on rank 0.
set(AXF_STRUMPACK_PREFIX "" CACHE PATH "Prefix of an MPI STRUMPACK build (empty: the serial STRUMPACK in AXF_DEPS)")
# find_package uses a cached STRUMPACK_DIR before it searches PATHS: drop one that belongs to the other prefix, so that
# switching an existing build tree between the serial and the MPI STRUMPACK does not link the wrong one
function(axf_forget_strumpack_dir prefix)
  if(STRUMPACK_DIR)
    get_filename_component(_dir "${STRUMPACK_DIR}" REALPATH)
    get_filename_component(_pre "${prefix}" REALPATH)
    string(FIND "${_dir}/" "${_pre}/" _at)
    if(NOT _at EQUAL 0)
      message(STATUS "axionFEM: dropping the cached STRUMPACK_DIR ${STRUMPACK_DIR} (not under ${prefix})")
      unset(STRUMPACK_DIR CACHE)
    endif()
  endif()
endfunction()
# The guard around STRUMPACK's process-wide state (cpp/src/sym_solver.cpp) has two modes: "concurrent", validated on
# STRUMPACK 8.0.x, and "serialized" (one STRUMPACK call at a time in the process, frequency-parallel STRUMPACK sweeps
# run serially), which uses only STRUMPACK's public interface. Other versions always get the serialized mode; this
# option forces it on 8.0.x as well (to test it, or to rule the concurrent mode out).
option(AXF_STRUMPACK_SERIALIZE "Serialize every STRUMPACK call, also with STRUMPACK 8.0.x" OFF)
if(AXF_USE_STRUMPACK)
  if(AXF_STRUMPACK_PREFIX)
    if(NOT AXF_MPI)
      message(FATAL_ERROR "AXF_STRUMPACK_PREFIX (an MPI STRUMPACK) needs the MPI build (AXF_MPI)")
    endif()
    if(NOT EXISTS "${AXF_STRUMPACK_PREFIX}/include/StrumpackConfig.h")
      message(FATAL_ERROR "no STRUMPACK under AXF_STRUMPACK_PREFIX=${AXF_STRUMPACK_PREFIX} "
                          "(build it with scripts/build_strumpack_mpi.sh)")
    endif()
    file(STRINGS "${AXF_STRUMPACK_PREFIX}/include/StrumpackConfig.h" _axf_spk_mpi REGEX "define STRUMPACK_USE_MPI")
    if(NOT _axf_spk_mpi)
      message(FATAL_ERROR "STRUMPACK in ${AXF_STRUMPACK_PREFIX} was not built with MPI")
    endif()
    axf_forget_strumpack_dir("${AXF_STRUMPACK_PREFIX}")
    find_package(STRUMPACK CONFIG REQUIRED PATHS ${AXF_STRUMPACK_PREFIX} NO_DEFAULT_PATH)
    # its headers (StrumpackConfig.h with STRUMPACK_USE_MPI) must shadow the serial ones in ${AXF_DEPS}/include
    target_include_directories(axionfem BEFORE PUBLIC ${AXF_STRUMPACK_PREFIX}/include)
    target_link_libraries(axionfem PUBLIC STRUMPACK::strumpack)
    target_compile_definitions(axionfem PUBLIC AXF_STRUMPACK AXF_STRUMPACK_MPI)
    if(NOT APPLE)
      # keep the static STRUMPACK and ScaLAPACK symbols private to libaxionfem (no interposition with other copies)
      target_link_options(axionfem PRIVATE "LINKER:--exclude-libs,libstrumpack.a:libscalapack.a")
    endif()
    message(STATUS "axionFEM: STRUMPACK ${STRUMPACK_VERSION} with MPI from ${AXF_STRUMPACK_PREFIX}")
  else()
    axf_forget_strumpack_dir("${AXF_DEPS}")
    find_package(STRUMPACK CONFIG QUIET PATHS ${AXF_DEPS} NO_DEFAULT_PATH)
    if(STRUMPACK_FOUND)
      target_link_libraries(axionfem PUBLIC STRUMPACK::strumpack)
      target_compile_definitions(axionfem PUBLIC AXF_STRUMPACK)
      message(STATUS "axionFEM: STRUMPACK ${STRUMPACK_VERSION} for large axisymmetric systems")
    endif()
  endif()
endif()
if(TARGET STRUMPACK::strumpack)
  if(AXF_STRUMPACK_SERIALIZE)
    target_compile_definitions(axionfem PRIVATE AXF_STRUMPACK_SERIALIZE)
    message(STATUS "axionFEM: AXF_STRUMPACK_SERIALIZE=ON: every STRUMPACK call runs alone (serialized guard)")
  elseif(NOT STRUMPACK_VERSION MATCHES "^8\\.0(\\.|$)")
    message(WARNING "axionFEM: STRUMPACK ${STRUMPACK_VERSION}: the concurrent STRUMPACK guard is validated on 8.0.x "
                    "only, so this build serializes every STRUMPACK call and runs frequency-parallel STRUMPACK sweeps "
                    "one frequency at a time (slower there). scripts/build_strumpack.sh installs 8.0.0.")
  endif()
endif()
# METIS: nested-dissection ordering for the large sparse Cholesky factorizations
# (Eigen/MetisSupport).  Optional: the sequential PyPI wheels are built without METIS, and the
# solver then always uses Eigen's own AMD ordering.
find_library(AXF_METIS_LIB metis PATHS ${AXF_DEPS}/lib ${AXF_DEPS}/lib64 NO_DEFAULT_PATH)
if(AXF_METIS_LIB)
  target_link_libraries(axionfem PUBLIC ${AXF_METIS_LIB})
  target_compile_definitions(axionfem PRIVATE AXF_HAVE_METIS=1)
  message(STATUS "METIS found (${AXF_METIS_LIB}): nested-dissection ordering available for large 2D solves")
else()
  message(STATUS "METIS not found under ${AXF_DEPS}/lib: the sparse direct solver will always use the "
                 "AMD ordering (scripts/build_deps.sh and scripts/build_deps_wheel.sh install METIS)")
endif()
# The source directory: the C++ tests read their data files through it, and a source build's mesher search falls back
# to <source>/build/axf-mesher (cpp/src/axi_mesh.cpp). A wheel must not carry the path it was built in, so there the
# library itself does not get it (INTERFACE: only targets that link axionfem, such as the tests, do).
if(SKBUILD)
  target_compile_definitions(axionfem INTERFACE AXF_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
else()
  target_compile_definitions(axionfem PUBLIC AXF_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
endif()
if(AXF_MPI)
  target_link_libraries(axionfem PUBLIC MPI::MPI_CXX)
else()
  target_compile_definitions(axionfem PUBLIC AXF_NO_MPI=1)
  if(MFEM_USE_MPI)
    target_link_libraries(axionfem PUBLIC MPI::MPI_CXX)  # headers only: mfem.hpp includes <mpi.h>
  endif()
endif()
if(AXF_OPENMP)
  target_link_libraries(axionfem PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(axionfem PUBLIC AXF_HAVE_OPENMP=1)
endif()
if(AXF_GPU STREQUAL "CUDA")
  target_compile_definitions(axionfem PUBLIC AXF_GPU_CUDA=1)
elseif(AXF_GPU STREQUAL "HIP")
  target_compile_definitions(axionfem PUBLIC AXF_GPU_HIP=1)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(axionfem PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy)
  if(NOT AXF_OPENMP)
    target_compile_options(axionfem PRIVATE -Wno-unknown-pragmas)
  endif()
endif()

# ------------------------------------------------------------------ apps
add_executable(axf cpp/apps/axf.cpp)
target_link_libraries(axf PRIVATE axionfem)
if(SKBUILD)  # installed one level deeper, in axionfem/bin
  set_target_properties(axf PROPERTIES INSTALL_RPATH "${AXF_ORIGIN}/..;${AXF_ORIGIN}/../_libs")
endif()

# fluids module (beta): its own library on top of libaxionfem,
# ParMesh-based, MPI build only; libaxionfem itself is unchanged.
if(AXF_MPI)
  # STATIC (position-independent): it is linked into _core and must not appear as a second libaxionfem*.so in the
  # provenance of a run.
  file(GLOB AXF_FLUIDS_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/fluids/*.cpp
       ${CMAKE_SOURCE_DIR}/cpp/src/numerics/transport/*.cpp)
  add_library(axionfem_fluids STATIC ${AXF_FLUIDS_SOURCES})
  set_target_properties(axionfem_fluids PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(axionfem_fluids PUBLIC axionfem)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(axionfem_fluids PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy)
  endif()
endif()

# gmsh (GPL) is only ever linked by this separate program, so that libaxionfem and _core stay BSD-licensed.
# It is found either in the pinned SDK (third_party/install/gmsh-sdk) or in a
# prefix where the PyPI `gmsh` wheel installed include/gmsh.h and
# lib/libgmsh.so.<soname> -- the same upstream binaries, which is what the
# wheels use so that no GPL library is shipped by us (docs/install.md).
if(AXF_BUILD_MESHER)
  set(AXF_GMSH_HINTS "")
  if(AXF_GMSH_ROOT)
    list(APPEND AXF_GMSH_HINTS "${AXF_GMSH_ROOT}")
  endif()
  if(DEFINED ENV{AXF_GMSH_ROOT})
    list(APPEND AXF_GMSH_HINTS "$ENV{AXF_GMSH_ROOT}")
  endif()
  list(APPEND AXF_GMSH_HINTS "${AXF_DEPS}/gmsh-sdk" "${AXF_DEPS}")
  if(AXF_BUILD_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
    if(Python_Interpreter_FOUND)
      # The gmsh wheel installs its data files into the interpreter's data prefix, or, in an isolated build
      # environment (pip install . with build isolation), into the prefix that holds its gmsh.py
      # (<prefix>/lib/pythonX.Y/site-packages/gmsh.py); find_spec locates it without importing gmsh.
      execute_process(COMMAND "${Python_EXECUTABLE}" -c
        "import os,sys,sysconfig,importlib.util as u;s=u.find_spec('gmsh');print(sysconfig.get_paths()['data']);print(sys.base_prefix);print(os.path.normpath(os.path.join(os.path.dirname(s.origin),'..','..','..')) if s and s.origin else '')"
        OUTPUT_VARIABLE _axf_py_dirs OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      string(REPLACE "\n" ";" _axf_py_dirs "${_axf_py_dirs}")
      list(APPEND AXF_GMSH_HINTS ${_axf_py_dirs})
    endif()
  endif()
  list(REMOVE_DUPLICATES AXF_GMSH_HINTS)

  find_path(AXF_GMSH_INCLUDE_DIR gmsh.h PATHS ${AXF_GMSH_HINTS} PATH_SUFFIXES include NO_DEFAULT_PATH)
  find_library(AXF_GMSH_LIB NAMES gmsh PATHS ${AXF_GMSH_HINTS} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
  if(NOT AXF_GMSH_LIB)
    # The PyPI wheel ships only the SONAME-versioned file, with no libgmsh.so symlink.
    foreach(h ${AXF_GMSH_HINTS})
      file(GLOB _axf_gmsh_cand "${h}/lib/libgmsh.so.*" "${h}/lib64/libgmsh.so.*"
                               "${h}/lib/libgmsh.*.dylib" "${h}/lib/libgmsh.dylib")
      if(_axf_gmsh_cand)
        list(SORT _axf_gmsh_cand)
        list(GET _axf_gmsh_cand 0 AXF_GMSH_LIB)
        break()
      endif()
    endforeach()
  endif()

  if(AXF_GMSH_LIB AND AXF_GMSH_INCLUDE_DIR)
    message(STATUS "axf-mesher: gmsh ${AXF_GMSH_LIB} (headers ${AXF_GMSH_INCLUDE_DIR})")
    add_executable(axf-mesher cpp/apps/axf_mesher.cpp)
    target_include_directories(axf-mesher PRIVATE ${AXF_GMSH_INCLUDE_DIR} ${AXF_DEPS}/include)
    target_link_libraries(axf-mesher PRIVATE ${AXF_GMSH_LIB})
    get_filename_component(_axf_gmsh_libdir "${AXF_GMSH_LIB}" DIRECTORY)
    if(SKBUILD)
      # libgmsh is NOT bundled (GPL): python/axionfem/_paths.py puts the gmsh
      # wheel's library directory on the loader path for this child process.
      set_target_properties(axf-mesher PROPERTIES INSTALL_RPATH
        "${AXF_ORIGIN}/..;${AXF_ORIGIN}/../_libs;${AXF_ORIGIN}/../../../..;${AXF_ORIGIN}/../../../../..")
    else()
      set_target_properties(axf-mesher PROPERTIES INSTALL_RPATH "${_axf_gmsh_libdir}"
                                                  BUILD_RPATH "${_axf_gmsh_libdir}")
    endif()
  else()
    message(WARNING
      "gmsh not found (looked for include/gmsh.h and lib/libgmsh.* under: ${AXF_GMSH_HINTS}); "
      "axf-mesher is not built, so meshing will not work. Install the pinned SDK "
      "(scripts/fetch_deps.sh, then scripts/build_deps.sh), or `pip install \"gmsh>=4.15,<4.16\"` into the Python "
      "that CMake uses, or set AXF_GMSH_ROOT to a prefix with include/gmsh.h and lib/libgmsh.*.")
  endif()
endif()

# ------------------------------------------------------------------ tests
if(AXF_BUILD_TESTS)
  enable_testing()
  file(GLOB AXF_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/tests/test_*.cpp)
  # chemistry unit tests live next to the chemistry sources
  file(GLOB AXF_CHEM_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/chem/tests/test_*.cpp)
  list(APPEND AXF_TESTS ${AXF_CHEM_TESTS})
  foreach(t ${AXF_TESTS})
    get_filename_component(n ${t} NAME_WE)
    if(NOT AXF_MPI AND n MATCHES "^test_fluids_")
      continue()  # the fluids module is built with MPI only
    endif()
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE axionfem)
    if(n MATCHES "^test_fluids_")
      target_link_libraries(${n} PRIVATE axionfem_fluids)
    endif()
    add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(${n} PROPERTIES ENVIRONMENT "AXF_MESHER=$<TARGET_FILE_DIR:axf>/axf-mesher")
  endforeach()
  # one OpenMP thread when STRUMPACK loads: its timer vector then has one list
  if(TEST test_strumpack_threads)
    set_tests_properties(test_strumpack_threads PROPERTIES ENVIRONMENT
                         "AXF_MESHER=$<TARGET_FILE_DIR:axf>/axf-mesher;OMP_NUM_THREADS=1")
  endif()
endif()

# ------------------------------------------------------------------ python
if(AXF_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT RESULT_VARIABLE nb_res)
    if(nb_res EQUAL 0)
      find_package(nanobind CONFIG REQUIRED)
      nanobind_add_module(_core NB_STATIC cpp/python/bindings.cpp cpp/src/chem/python/bindings_chem.cpp)
      target_link_libraries(_core PRIVATE axionfem)
      # build provenance (cmake/axf_gitinfo.cmake, the target axf_gitinfo above): build_info() reads it from the
      # generated header
      add_dependencies(_core axf_gitinfo)
      target_include_directories(_core PRIVATE ${AXF_GITINFO_DIR})
      set_target_properties(_core PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${AXF_ORIGIN}"
                                             BUILD_RPATH "${CMAKE_BUILD_RPATH};${CMAKE_BINARY_DIR}")
      if(NOT SKBUILD)
        # In-place development build: python/axionfem/_core*.so.  Point
        # AXF_INPLACE_PYTHON_DIR elsewhere when several build trees (MPI,
        # serial, GPU) must coexist without overwriting each other's module.
        set(AXF_INPLACE_PYTHON_DIR "${CMAKE_SOURCE_DIR}/python/axionfem" CACHE PATH
            "Directory a non-wheel build writes _core into")
        file(MAKE_DIRECTORY "${AXF_INPLACE_PYTHON_DIR}")
        set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${AXF_INPLACE_PYTHON_DIR}")
      endif()
      # fluids Python boundary (beta): the submodule axionfem._core._fluids, registered by cpp/python/bindings.cpp
      # and imported by axionfem.fluids. Every MPI build has it, the in-place CMake build and `pip install` alike;
      # the wheels are serial and do not.
      if(AXF_MPI)
        target_sources(_core PRIVATE cpp/src/fluids/python/fluids_bindings.cpp
          cpp/src/fluids/python/fluids_bindings3d.cpp)
        target_link_libraries(_core PRIVATE axionfem_fluids)
        target_compile_definitions(_core PRIVATE AXF_WITH_FLUIDS)
      endif()
      if(SKBUILD)
        install(TARGETS _core axionfem LIBRARY DESTINATION axionfem RUNTIME DESTINATION axionfem)
        install(TARGETS axf RUNTIME DESTINATION axionfem/bin)
        if(TARGET axf-mesher)
          install(TARGETS axf-mesher RUNTIME DESTINATION axionfem/bin)
        endif()
        # The viewer assets (three.js, viewer.js, template.html) are looked up as
        # axionfem/_viewer by python/axionfem/viewer.py.
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/viewer/
                DESTINATION axionfem/_viewer
                PATTERN "*.pyc" EXCLUDE
                PATTERN "__pycache__" EXCLUDE)
        if(AXF_BUNDLE_LIBS)
          # Third-party shared libraries, next to the extension module.
          # $ORIGIN/_libs is in the rpath of _core, libaxionfem, axf and axf-mesher.
          set(_axf_bundle "")
          # MFEM_DIR is <prefix>/lib/cmake/mfem, so its grandparent is the library dir.
          get_filename_component(_axf_mfem_dir "${MFEM_DIR}/../.." ABSOLUTE)
          foreach(d "${_axf_mfem_dir}" "${AXF_DEPS}/lib" "${AXF_DEPS}/lib64")
            file(GLOB _axf_found
                 "${d}/libmfem.so*" "${d}/libmfem*.dylib"
                 "${d}/libHYPRE.so*" "${d}/libHYPRE*.dylib"
                 "${d}/libmetis.so*" "${d}/libmetis*.dylib"
                 # STRUMPACK (BSD) when the wheel is built against a stack that has it; its own
                 # dependencies (BLAS/LAPACK, and MPI if it was built with it) are system libraries
                 # that auditwheel/delocate add.
                 "${d}/libstrumpack.so*" "${d}/libstrumpack*.dylib")
            list(APPEND _axf_bundle ${_axf_found})
          endforeach()
          list(REMOVE_DUPLICATES _axf_bundle)
          # Keep only real files.  install(FILES) dereferences symlinks, so the
          # unversioned development link (libmfem.so -> libmfem.so.4.10.0) would
          # become a second full copy and double the wheel.  It is not needed at
          # run time: DT_NEEDED records the SONAME (libmfem.so.4.10.0), and
          # nothing links against the bundled copy.  A build whose SONAME *is*
          # unversioned has that name as the real file, so it is kept.
          # The copy takes the name the loader asks for: the SONAME, which CMake-built libraries carry as the
          # middle link of libX.so -> libX.so.<soversion> -> libX.so.<version> (hypre: libHYPRE.so.233 ->
          # libHYPRE.so.2.33.0; macOS: libX.<soversion>.dylib). MFEM's SONAME is its full version, the real file.
          set(_axf_real "")
          foreach(f ${_axf_bundle})
            if(NOT IS_SYMLINK "${f}")
              list(APPEND _axf_real "${f}")
              get_filename_component(_axf_name "${f}" NAME)
              get_filename_component(_axf_freal "${f}" REALPATH)
              foreach(l ${_axf_bundle})
                get_filename_component(_axf_lname "${l}" NAME)
                if(IS_SYMLINK "${l}" AND NOT _axf_lname MATCHES "^lib[^.]+\\.(so|dylib)$")
                  get_filename_component(_axf_lreal "${l}" REALPATH)
                  if(_axf_lreal STREQUAL _axf_freal)
                    set(_axf_name "${_axf_lname}")
                  endif()
                endif()
              endforeach()
              install(FILES "${f}" DESTINATION axionfem/_libs RENAME "${_axf_name}")
            endif()
          endforeach()
          set(_axf_bundle ${_axf_real})
          if(_axf_bundle)
            message(STATUS "bundling into axionfem/_libs: ${_axf_bundle}")
          else()
            message(WARNING "AXF_BUNDLE_LIBS is ON but no third-party shared libraries were "
                            "found under ${AXF_DEPS}; the wheel will depend on the build machine.")
          endif()
        endif()
      endif()
    else()
      message(WARNING "nanobind not found for ${Python_EXECUTABLE}; the Python module is not built. Install it into "
                      "that interpreter (pip install \"nanobind>=2.0\") or point CMake at the interpreter that has "
                      "it with -DPython_EXECUTABLE=/path/to/python.")
    endif()
  else()
    message(WARNING "Python >= 3.10 with development headers was not found; the Python module is not built. "
                    "Install the headers (Debian/Ubuntu: python3-dev; Fedora/RHEL: python3-devel or python3.12-devel) "
                    "or pass -DPython_EXECUTABLE=/path/to/python; -DAXF_BUILD_PYTHON=OFF silences this.")
  endif()
endif()

# ------------------------------------------------------------------ summary
if(STRUMPACK_FOUND)
  set(_axf_sp "${STRUMPACK_VERSION}")
else()
  set(_axf_sp "no")
endif()
if(AXF_METIS_LIB)
  set(_axf_metis "yes")
else()
  set(_axf_metis "no")
endif()
message(STATUS "axionFEM ${PROJECT_VERSION}: MPI=${AXF_MPI} (3D engine ${AXF_MPI}) "
               "OpenMP=${AXF_OPENMP} GPU=${AXF_GPU} STRUMPACK=${_axf_sp} METIS=${_axf_metis} "
               "deps=${AXF_DEPS}")
