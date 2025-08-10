FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12, pip, and necessary build tools and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gpg wget \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    libssl-dev \
    libssl3 \
    cmake \
    bash \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspaces

# Default command
CMD ["bash"]
