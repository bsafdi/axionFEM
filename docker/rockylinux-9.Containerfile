# A clean Rocky Linux 9 (or another RHEL 9 rebuild) with the prerequisites of docs/guides/install.md ("RHEL 9 and its
# rebuilds"): exactly the packages listed there (tests/test_packaging.py checks that the two lists agree). Nothing of
# axionFEM is built into the image; follow the guide's build steps inside it (with python3.12 for the venv).
#
#   podman build -t axionfem-rocky -f docker/rockylinux-9.Containerfile docker
#   podman run --rm -it --shm-size=2g axionfem-rocky
#
# MPICH instead of OpenMPI: --build-arg MPI_PACKAGES=mpich-devel --build-arg MPI_BIN=/usr/lib64/mpich/bin
# The optional STRUMPACK builds: --build-arg EXTRA_PACKAGES=gcc-gfortran
# The distribution's MPI lives under /usr/lib64/<mpi>; its bin directory is put on PATH below, which is what
# `module load mpi/openmpi-x86_64` (or mpi/mpich-x86_64) does on a host with environment modules.
FROM docker.io/library/rockylinux:9
ARG MPI_PACKAGES="openmpi-devel"
ARG MPI_BIN=/usr/lib64/openmpi/bin
ARG EXTRA_PACKAGES=""
RUN dnf install -y epel-release dnf-plugins-core && dnf config-manager --set-enabled crb \
 && dnf install -y gcc gcc-c++ make cmake ninja-build pkgconf-pkg-config git tar gzip \
      python3.12 python3.12-devel $MPI_PACKAGES openblas-devel $EXTRA_PACKAGES \
 && dnf clean all
ENV PATH=$MPI_BIN:$PATH
RUN useradd -m -s /bin/bash axf
USER axf
WORKDIR /home/axf
CMD ["/bin/bash"]
