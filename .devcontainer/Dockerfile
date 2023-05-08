# syntax=docker/dockerfile-upstream:master-labs
FROM mcr.microsoft.com/devcontainers/cpp:0-debian-11

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>


ADD https://github.com/catchorg/Catch2.git#v3.2.1 /catch2

WORKDIR /catch2

RUN cmake -Bbuild -H/catch2 \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_CXX_STANDARD=17 \
              -DCMAKE_CXX_STANDARD_REQUIRED=ON \
              -DCMAKE_CXX_EXTENSIONS=OFF \
              -DCATCH_DEVELOPMENT_BUILD=ON \
              -G Ninja && \
    cmake --build build --target install

ARG USERNAME

RUN adduser --disabled-password --gecos '' ${USERNAME} && \
    adduser ${USERNAME} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN apt update && \
    apt install -y lsb-release gdb python3 pip pkg-config libeigen3-dev && \
    pip install "pybind11[global]"