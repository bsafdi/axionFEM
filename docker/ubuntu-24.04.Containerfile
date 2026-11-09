# A clean Ubuntu 24.04 with the prerequisites of docs/guides/install.md ("Debian/Ubuntu"): exactly the packages
# listed there (tests/test_packaging.py checks that the two lists agree). Nothing of axionFEM is built into the
# image; follow the guide's build steps inside it.
#
#   podman build -t axionfem-ubuntu -f docker/ubuntu-24.04.Containerfile docker
#   podman run --rm -it --shm-size=2g axionfem-ubuntu
#
# (docker works the same way.) Variants through build arguments:
#   --build-arg MPI_PACKAGES= --build-arg BLAS_PACKAGES= \
#   --build-arg EXTRA_PACKAGES="libglu1-mesa libgl1 libxcursor1 libxft2 libxinerama1"
#                                              the sequential build with the gmsh PyPI wheel
#   --build-arg EXTRA_PACKAGES=gfortran        the optional STRUMPACK builds
FROM docker.io/library/ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG MPI_PACKAGES="libopenmpi-dev openmpi-bin"
ARG BLAS_PACKAGES="libopenblas-dev liblapack-dev"
ARG EXTRA_PACKAGES=""
RUN apt-get update \
 && apt-get install -y build-essential cmake ninja-build pkg-config git curl ca-certificates \
      python3-dev python3-venv $MPI_PACKAGES $BLAS_PACKAGES $EXTRA_PACKAGES \
 && rm -rf /var/lib/apt/lists/*
# OpenMPI refuses to run as root; build and run as an ordinary user.
RUN useradd -m -s /bin/bash axf
USER axf
WORKDIR /home/axf
CMD ["/bin/bash"]
