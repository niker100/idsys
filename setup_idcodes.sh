#!/bin/bash
# Setup idcodes library for Ubuntu 24.04

set -e  # Exit immediately if a command fails

echo "===== Setting up idcodes library ====="

# Install required Boost dependencies for idcodeslibrary
apt-get update

# Install the Python wheel for idcodes (x86_64 version)
# echo "Installing idcodes x86_64 wheel..."
# pip install --break-system-packages idcodes-0.2.7-cp312-cp312-linux_x86_64.whl

# Install the .deb package for idcodeslibrary (x86_64 version)
echo "Installing idcodeslibrary .deb package..."
# dpkg -i idcodeslibrary-0.2.7-x86_64-Ubuntu-24.04.deb || apt-get install -f -y
dpkg -i idcodeslibrary_0.2.8_amd64.deb || apt-get install -f -y  

echo "===== idcodes setup completed for Ubuntu 24.04 ====="